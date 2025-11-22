import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

data = pd.read_csv('../../kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('../../kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

df = data.merge(marking, left_on='PK', right_on='ИД', how='left')

def create_features_fast(df):
    # Группируем один раз
    grp = df.groupby('PK')

    # Базовые агрегации
    features = grp.agg({
        'Факультет': 'first',
        'Направление': 'first',
        'год поступления': 'first',
        'BALLS': ['mean', 'max', 'min', 'std', 'count'],  # std показывает стабильность
        'выпуск': 'first'  # Таргет
    })

    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.rename(columns={
        'Факультет_first': 'faculty',
        'Направление_first': 'direction',
        'год поступления_first': 'admission_year',
        'BALLS_mean': 'avg_grade',
        'BALLS_max': 'max_grade',
        'BALLS_min': 'min_grade',
        'BALLS_std': 'std_grade',  # Важная фича: разброс оценок
        'BALLS_count': 'total_grades',
        'выпуск_first': 'target_raw'
    })

    # Дополнительные фичи через apply (чуть медленнее, но гибко)
    # Считаем долги и отличные оценки
    def complex_stats(x):
        d = {}
        d['zach_count'] = (x['TYPE'] == 'зач').sum()
        d['exam_count'] = (x['TYPE'] == 'экз').sum()

        # количество плохих оценок
        grades = pd.to_numeric(x['BALLS'], errors='coerce')
        d['failed_count'] = (grades < 3).sum()  # Двойки
        d['low_grade_count'] = (grades == 3).sum()  # Тройки
        d['excellent_count'] = (grades == 5).sum()  # Пятерки

        # Статусы
        statuses = x['Unnamed: 5'].value_counts()
        d['expelled_count'] = statuses.get('отчислен', 0)
        d['academic_count'] = statuses.get('академ', 0)
        return pd.Series(d)

    print("Считаем сложную статистику...")
    stats_df = grp.apply(complex_stats)

    full_features = pd.concat([features, stats_df], axis=1).reset_index()

    # Обработка таргета
    # Если в данных есть target (для трейна)
    full_features['target'] = full_features['target_raw'].apply(
        lambda x: 1 if x == 'выпустился' else (0 if pd.notna(x) else np.nan)
    )

    return full_features.drop(columns=['target_raw'])


student_features = create_features_fast(df)
print(f"Фичи готовы. Размер: {student_features.shape}")

# Подготовка к обучению
train_df = student_features[student_features['target'].notna()].copy()
test_df = student_features[student_features['target'].isna()].copy()

# Заполняем пропуски (std может быть NaN если была 1 оценка)
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)

# Указываем категориальные фичи (CatBoost съест их сам, без LabelEncoder!)
cat_features = ['faculty', 'direction']
# Приводим к строкам, чтобы CatBoost не ругался на смешанные типы
for col in cat_features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

X = train_df.drop(columns=['PK', 'target'])
y = train_df['target']
X_test = test_df.drop(columns=['PK', 'target'])

model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='Accuracy',
    cat_features=cat_features,
    verbose=100,
    random_seed=42,
    auto_class_weights='Balanced'  # Помогает, если отчисленных сильно больше/меньше
)

# Обучаем
model.fit(X, y)

# Смотрим качество
print("Feature Importance:")
print(model.get_feature_importance(prettified=True).head(10))

# Предикт
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PK': test_df['PK'],
    'выпуск': test_predictions
})

submission.to_csv('submission_catboost.csv', index=False)
print("Готово!")