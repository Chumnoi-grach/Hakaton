import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

# Объединение данных
df = data.merge(marking, left_on='PK', right_on='ИД', how='left')


def create_features_xgboost(df):
    # Группируем по студентам
    grp = df.groupby('PK')

    # Базовые агрегации
    features = grp.agg({
        'Факультет': 'first',
        'Направление': 'first',
        'год поступления': 'first',
        'BALLS': ['mean', 'max', 'min', 'std', 'count'],
        'выпуск': 'first'
    })

    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.rename(columns={
        'Факультет_first': 'faculty',
        'Направление_first': 'direction',
        'год поступления_first': 'admission_year',
        'BALLS_mean': 'avg_grade',
        'BALLS_max': 'max_grade',
        'BALLS_min': 'min_grade',
        'BALLS_std': 'std_grade',
        'BALLS_count': 'total_grades',
        'выпуск_first': 'target_raw'
    })

    # Дополнительные фичи
    def complex_stats(x):
        d = {}
        d['zach_count'] = (x['TYPE'] == 'зач').sum()
        d['exam_count'] = (x['TYPE'] == 'экз').sum()

        # Анализ оценок
        grades = pd.to_numeric(x['BALLS'], errors='coerce')
        d['failed_count'] = (grades < 3).sum()
        d['low_grade_count'] = (grades == 3).sum()
        d['good_grade_count'] = (grades == 4).sum()
        d['excellent_count'] = (grades == 5).sum()
        d['grade_variance'] = grades.var()  # Дисперсия оценок

        # Статусы
        statuses = x['Unnamed: 5'].value_counts()
        d['studying_count'] = statuses.get('учится', 0)
        d['expelled_count'] = statuses.get('отчислен', 0)
        d['academic_count'] = statuses.get('академ', 0)

        # Бинарные признаки
        d['ever_expelled'] = 1 if d['expelled_count'] > 0 else 0
        d['ever_academic'] = 1 if d['academic_count'] > 0 else 0

        return pd.Series(d)

    print("Считаем сложную статистику...")
    stats_df = grp.apply(complex_stats)

    full_features = pd.concat([features, stats_df], axis=1).reset_index()

    # Обработка таргета
    full_features['target'] = full_features['target_raw'].apply(
        lambda x: 1 if x == 'выпустился' else (0 if pd.notna(x) else np.nan)
    )

    return full_features.drop(columns=['target_raw'])


# Создаем фичи
student_features = create_features_xgboost(df)
print(f"Фичи готовы. Размер: {student_features.shape}")

# Разделяем на train/test
train_df = student_features[student_features['target'].notna()].copy()
test_df = student_features[student_features['target'].isna()].copy()

print(f"Обучающая выборка: {len(train_df)} студентов")
print(f"Тестовая выборка: {len(test_df)} студентов")

# Кодируем категориальные фичи для XGBoost
le_faculty = LabelEncoder()
le_direction = LabelEncoder()

# Объединяем train и test для кодирования
all_faculties = pd.concat([train_df['faculty'], test_df['faculty']]).fillna('Unknown')
all_directions = pd.concat([train_df['direction'], test_df['direction']]).fillna('Unknown')

le_faculty.fit(all_faculties)
le_direction.fit(all_directions)

train_df['faculty_encoded'] = le_faculty.transform(train_df['faculty'].fillna('Unknown'))
test_df['faculty_encoded'] = le_faculty.transform(test_df['faculty'].fillna('Unknown'))

train_df['direction_encoded'] = le_direction.transform(train_df['direction'].fillna('Unknown'))
test_df['direction_encoded'] = le_direction.transform(test_df['direction'].fillna('Unknown'))

# Список фичей для модели
feature_columns = [
    'admission_year', 'avg_grade', 'max_grade', 'min_grade', 'std_grade',
    'total_grades', 'zach_count', 'exam_count', 'failed_count',
    'low_grade_count', 'good_grade_count', 'excellent_count', 'grade_variance',
    'studying_count', 'expelled_count', 'academic_count',
    'ever_expelled', 'ever_academic', 'faculty_encoded', 'direction_encoded'
]

# Подготовка данных
X_train = train_df[feature_columns].fillna(-1)
y_train = train_df['target']
X_test = test_df[feature_columns].fillna(-1)

# Баланс классов для подбора весов
class_balance = y_train.value_counts()
scale_pos_weight = class_balance[0] / class_balance[1] if 1 in class_balance else 1
print(f"Баланс классов: {class_balance.to_dict()}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Модель XGBoost с правильными параметрами
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos_weight,  # Балансировка классов
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,  # Параметр ранней остановки здесь!
    use_label_encoder=False
)

# Обучение (без early_stopping_rounds в fit)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    verbose=50
)

# Предсказания на train
train_predictions = model.predict(X_train)
accuracy = accuracy_score(y_train, train_predictions)
print(f"Accuracy на обучающей выборке: {accuracy:.3f}")
print(classification_report(y_train, train_predictions))

# Важность фич
print("\nВажность фич:")
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Предсказания на test
test_predictions = model.predict(X_test)

# Создание submission файла
submission = pd.DataFrame({
    'PK': test_df['PK'],
    'выпуск': test_predictions
})

submission.to_csv('submission_xgboost.csv', index=False)
print(f"\nСоздан файл submission_xgboost.csv с {len(submission)} предсказаниями")
print(f"Выпускников предсказано: {test_predictions.sum()} из {len(test_predictions)}")
print(f"Отчисленных предсказано: {len(test_predictions) - test_predictions.sum()} из {len(test_predictions)}")