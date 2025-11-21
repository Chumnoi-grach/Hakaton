import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')


def load_data():
    print("--- Загрузка данных ---")
    # Укажи правильные пути к файлам
    try:
        data = pd.read_csv('../../kaggle/input/nstu-hach-ai-track-education-case/data.csv')
        marking = pd.read_csv('../../kaggle/input/nstu-hach-ai-track-education-case/marking.csv')
        df = data.merge(marking, left_on='PK', right_on='ИД', how='left')
        return df
    except FileNotFoundError:
        print("ОШИБКА: Файлы не найдены. Проверь пути к data.csv и marking.csv")
        return None


def calculate_trend(series):
    """
    Считает тренд успеваемости.
    Сравниваем средний балл второй половины оценок со средним баллом первой половины.
    Если положительное - студент исправляется, отрицательное - скатывается.
    """
    grades = pd.to_numeric(series, errors='coerce').dropna()
    if len(grades) < 2:
        return 0

    mid = len(grades) // 2
    first_half = grades.iloc[:mid].mean()
    second_half = grades.iloc[mid:].mean()

    return second_half - first_half


def extract_features(df):
    # Предварительная обработка оценок
    df['BALLS_NUM'] = pd.to_numeric(df['BALLS'], errors='coerce')

    # Группируем по студенту
    grp = df.groupby('PK')

    # 1. Базовые агрегации (быстро)
    features = grp.agg({
        'Факультет': 'first',
        'Направление': 'first',
        'год поступления': 'first',
        'BALLS_NUM': ['mean', 'max', 'min', 'std', 'count', 'last'],
        'выпуск': 'first'
    })

    # Выпрямляем названия колонок
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.rename(columns={
        'Факультет_first': 'faculty',
        'Направление_first': 'direction',
        'год поступления_first': 'admission_year',
        'BALLS_NUM_mean': 'avg_grade',
        'BALLS_NUM_max': 'max_grade',
        'BALLS_NUM_min': 'min_grade',
        'BALLS_NUM_std': 'std_grade',  # Стабильность (чем меньше, тем стабильнее)
        'BALLS_NUM_count': 'total_grades',
        'BALLS_NUM_last': 'last_grade',
        'выпуск_first': 'target_raw'
    })

    # 2. Сложные фичи
    def advanced_stats(x):
        d = {}
        # Считаем двойки
        grades = pd.to_numeric(x['BALLS'], errors='coerce')
        d['fail_count'] = (grades < 3).sum()
        d['low_grade_count'] = (grades == 3).sum()  # Тройки тоже зона риска
        d['excellent_count'] = (grades == 5).sum()

        # Статусы
        statuses = x['Unnamed: 5'].value_counts()
        d['expelled_history'] = statuses.get('отчислен', 0)
        d['academic_leave'] = statuses.get('академ', 0)

        # Тренд успеваемости
        if len(grades) > 1:
            mid = len(grades) // 2
            d['grade_trend'] = grades.iloc[mid:].mean() - grades.iloc[:mid].mean()
        else:
            d['grade_trend'] = 0

        return pd.Series(d)

    stats_df = grp.apply(advanced_stats)

    # Объединяем
    full_features = pd.concat([features, stats_df], axis=1).reset_index()

    # Обработка целевой переменной
    full_features['target'] = full_features['target_raw'].apply(
        lambda x: 1 if x == 'выпустился' else (0 if pd.notna(x) else np.nan)
    )

    # Заполнение пропусков (std может быть nan если 1 оценка)
    full_features['std_grade'] = full_features['std_grade'].fillna(0)
    full_features['grade_trend'] = full_features['grade_trend'].fillna(0)

    return full_features


def add_target_encoding(train_df, test_df, cat_cols):
    """
    Добавляет 'Сложность факультета/направления'.
    Считает средний target (процент выпускников) по категории в TRAIN
    и мапит это значение в TRAIN и TEST.
    """
    print("Сложность факультетов")

    for col in cat_cols:
        # Считаем вероятность выпуска для каждой категории (факультета/направления)
        # Используем только TRAIN чтобы избежать утечки данных (leakage)
        encoding_map = train_df.groupby(col)['target'].mean()

        new_col_name = f'{col}_success_rate'

        # Мапим значения
        train_df[new_col_name] = train_df[col].map(encoding_map)
        test_df[new_col_name] = test_df[col].map(encoding_map)

        # Если в тесте есть факультет, которого не было в трейне, заполняем общим средним
        global_mean = train_df['target'].mean()
        test_df[new_col_name] = test_df[new_col_name].fillna(global_mean)

    return train_df, test_df


def train_and_predict(df):
    # Разделение на Train и Test (где target NaN - это то, что надо предсказать)
    train_df = df[df['target'].notna()].copy()
    test_df = df[df['target'].isna()].copy()

    print(f"Обучающая выборка: {len(train_df)}")
    print(f"Тестовая выборка: {len(test_df)}")

    # Обработка категорий для CatBoost
    cat_features = ['faculty', 'direction']
    # Приводим к строкам
    for col in cat_features:
        train_df[col] = train_df[col].astype(str).fillna('Unknown')
        test_df[col] = test_df[col].astype(str).fillna('Unknown')

    # Сложность факультетов
    train_df, test_df = add_target_encoding(train_df, test_df, cat_features)

    # Формируем X и y
    drop_cols = ['PK', 'target', 'target_raw']
    X = train_df.drop(columns=drop_cols)
    y = train_df['target']
    X_test = test_df.drop(columns=drop_cols)

    # Убеждаемся в качестве пере отправкой
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                              stratify=y)

    # НАСТРОЙКИ CATBOOST
    model = CatBoostClassifier(
        iterations=2000,  # Много итераций
        learning_rate=0.02,  # Медленно, но верно спускаемся к минимуму ошибки
        depth=6,  # Оптимальная глубина
        l2_leaf_reg=4,  # Регуляризация от переобучения
        loss_function='Logloss',
        eval_metric='Accuracy',
        cat_features=cat_features,
        early_stopping_rounds=200,  # Остановить, если не улучшается 200 эпох
        verbose=100,
        random_seed=42,
        auto_class_weights='Balanced'  # Важно для несбалансированных классов
    )

    print("--- Начало обучения ---")
    model.fit(
        X_train_split, y_train_split,
        eval_set=(X_val_split, y_val_split),
        use_best_model=True
    )

    # Оценка на валидации
    val_preds = model.predict(X_val_split)
    acc = accuracy_score(y_val_split, val_preds)
    print(f"\n=== VALIDATION ACCURACY: {acc:.4f} ===")
    print(classification_report(y_val_split, val_preds))

    # Обучение на ПОЛНОМ датасете перед финальным предсказанием (опционально, но круто для Kaggle)
    print("--- Финальное переобучение на всем датасете ---")
    final_model = CatBoostClassifier(
        iterations=model.best_iteration_,  # Используем оптимальное число итераций
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=4,
        cat_features=cat_features,
        random_seed=42,
        auto_class_weights='Balanced',
        verbose=0
    )
    final_model.fit(X, y)

    # Важность признаков (чтобы ты понимал, что сработало)
    print("\nТоп-10 важнейших признаков:")
    print(final_model.get_feature_importance(prettified=True).head(10))

    # Предсказание
    test_predictions = final_model.predict(X_test)

    submission = pd.DataFrame({
        'PK': test_df['PK'],
        'выпуск': test_predictions.astype(int)
    })

    submission.to_csv('submission 84.80%.csv', index=False)
    print(f"\nФайл submission 84.80%.csv сохранен! Предсказано выпускников: {sum(test_predictions)}")


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        full_features = extract_features(df)
        train_and_predict(full_features)