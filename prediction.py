import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


def calculate_grade_trend(student_data):
    """Тренд успеваемости по семестрам"""
    if len(student_data) < 2:
        return 0

    semester_grades = student_data.groupby('SEMESTER')['BALLS'].mean().sort_index()

    if len(semester_grades) < 2:
        return 0

    # Рассчитываем линейный тренд (коэффициент наклона)
    x = np.arange(len(semester_grades))
    y = semester_grades.values
    slope = np.polyfit(x, y, 1)[0]

    return slope


def calculate_grade_stability(student_data):
    """Стабильность оценок (стандартное отклонение)"""
    grades = student_data['BALLS'].dropna()

    if len(grades) < 2:
        return 0

    # Стандартное отклонение оценок
    std_dev = grades.std()

    return std_dev


def calculate_max_grade_drop(student_data):
    """Максимальное падение оценки между последовательными предметами"""
    if len(student_data) < 2:
        return 0

    if 'EXAM_DATE' in student_data.columns and student_data['EXAM_DATE'].notna().any():
        sorted_data = student_data.dropna(subset=['EXAM_DATE']).sort_values('EXAM_DATE')
    else:
        sorted_data = student_data.sort_values(['SEMESTER', 'DNAME'])

    if len(sorted_data) < 2:
        return 0

    grades = sorted_data['BALLS'].values

    drops = []
    for i in range(1, len(grades)):
        drop = grades[i - 1] - grades[i]
        if drop > 0:
            drops.append(drop)

    if drops:
        return max(drops)
    else:
        return 0


def calculate_grade_distribution(student_data):
    """Доли оценок разных категорий"""
    grades = student_data['BALLS'].dropna()

    if len(grades) == 0:
        return {
            'excellent_ratio': 0,
            'good_ratio': 0,
            'satisfactory_ratio': 0,
            'fail_ratio': 0,
            'critical_fail_ratio': 0
        }

    total_grades = len(grades)

    excellent_ratio = (grades > 85).sum() / total_grades

    good_ratio = ((grades >= 70) & (grades <= 85)).sum() / total_grades

    satisfactory_ratio = ((grades >= 60) & (grades < 70)).sum() / total_grades

    fail_ratio = (grades < 60).sum() / total_grades

    critical_fail_ratio = (grades < 40).sum() / total_grades

    return {
        'excellent_ratio': excellent_ratio,
        'good_ratio': good_ratio,
        'satisfactory_ratio': satisfactory_ratio,
        'fail_ratio': fail_ratio,
        'critical_fail_ratio': critical_fail_ratio
    }

def create_student_features(df):
    features_list = []

    for student_id in df['PK'].unique():
        student_data = df[df['PK'] == student_id]

        features = {'student_id': student_id}

        first_row = student_data.iloc[0]
        features['faculty'] = first_row['Факультет']
        features['direction'] = first_row['Направление']
        features['admission_year'] = first_row['год поступления']

        grades = student_data['BALLS'].dropna()
        features['avg_grade'] = grades.mean()
        features['max_grade'] = grades.max()
        features['min_grade'] = grades.min()
        features['total_subjects'] = len(student_data)
        features['total_grades'] = len(grades)

        features['grade_trend'] = calculate_grade_trend(student_data)
        features['grade_stability'] = calculate_grade_stability(student_data)
        features['max_grade_drop'] = calculate_max_grade_drop(student_data)

        grade_distribution = calculate_grade_distribution(student_data)
        features.update(grade_distribution)

        features['zach_count'] = (student_data['TYPE'] == 'зач').sum()
        features['exam_count'] = (student_data['TYPE'] == 'экз').sum()

        status_counts = student_data['Unnamed: 5'].value_counts()
        features['studying_count'] = status_counts.get('учится', 0)
        features['expelled_count'] = status_counts.get('отчислен', 0)
        features['academic_count'] = status_counts.get('академ', 0)

        features['ever_expelled'] = 1 if features['expelled_count'] > 0 else 0
        features['ever_academic'] = 1 if features['academic_count'] > 0 else 0

        if 'выпуск' in student_data.columns and pd.notna(first_row['выпуск']):
            features['target'] = 1 if first_row['выпуск'] == 'выпустился' else 0

        features_list.append(features)

    return pd.DataFrame(features_list)

data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

df = data.merge(marking, left_on='PK', right_on='ИД', how='left')

student_features = create_student_features(df)
print(f"Создано профилей: {len(student_features)}")

train_df = student_features[student_features['target'].notna()].copy()
test_df = student_features[student_features['target'].isna()].copy()

print(f"Обучающая выборка: {len(train_df)} студентов")
print(f"Тестовая выборка: {len(test_df)} студентов")

le_faculty = LabelEncoder()
le_direction = LabelEncoder()

all_faculties = pd.concat([train_df['faculty'], test_df['faculty']]).fillna('Unknown')
all_directions = pd.concat([train_df['direction'], test_df['direction']]).fillna('Unknown')

le_faculty.fit(all_faculties)
le_direction.fit(all_directions)

train_df['faculty_encoded'] = le_faculty.transform(train_df['faculty'].fillna('Unknown'))
test_df['faculty_encoded'] = le_faculty.transform(test_df['faculty'].fillna('Unknown'))

train_df['direction_encoded'] = le_direction.transform(train_df['direction'].fillna('Unknown'))
test_df['direction_encoded'] = le_direction.transform(test_df['direction'].fillna('Unknown'))

feature_columns = [
    'admission_year', 'avg_grade', 'max_grade', 'min_grade', 'grade_trend',
    #'grade_stability', 'max_grade_drop',
    #'excellent_ratio', 'good_ratio', 'satisfactory_ratio', 'fail_ratio', 'critical_fail_ratio',
    'total_subjects', 'total_grades', 'zach_count', 'exam_count',
    'studying_count', 'expelled_count', 'academic_count',
    'ever_expelled', 'ever_academic',
    'faculty_encoded', 'direction_encoded'
]

X_train = train_df[feature_columns].fillna(0)
y_train = train_df['target']
X_test = test_df[feature_columns].fillna(0)

#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'ID': test_df['student_id'],
    'выпуск': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"Создан файл submission.csv с {len(submission)} предсказаниями")
print(f"Выпускников предсказано: {test_predictions.sum()} из {len(test_predictions)}")