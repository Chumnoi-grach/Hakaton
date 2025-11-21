import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

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
    'admission_year', 'avg_grade', 'max_grade', 'min_grade',
    'total_subjects', 'total_grades', 'zach_count', 'exam_count',
    'studying_count', 'expelled_count', 'academic_count',
    'ever_expelled', 'ever_academic',
    'faculty_encoded', 'direction_encoded'
]

X_train = train_df[feature_columns].fillna(0)
y_train = train_df['target']
X_test = test_df[feature_columns].fillna(0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
accuracy = accuracy_score(y_train, train_predictions)
print(f"Accuracy на обучающей выборке: {accuracy:.3f}")
print(classification_report(y_train, train_predictions))

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PK': test_df['student_id'],
    'выпуск': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"Создан файл submission.csv с {len(submission)} предсказаниями")
print(f"Выпускников предсказано: {test_predictions.sum()} из {len(test_predictions)}")