import numpy as np
import pandas as pd
import os


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
            features['target'] = 1 if first_row['выпуск'] == 'выпустился' else 0  # 🎓 1=выпустился, 0=отчислен

        features_list.append(features)

    return pd.DataFrame(features_list)

data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

ds = data.merge(marking, left_on='PK', right_on='ИД', how='left')
ds.drop('ИД', axis=1, inplace=True)

train_df = df[df['выпуск'].notna()]
test_df = df[df['выпуск'].isna()]

print(f"Обучающая выборка: {len(train_df)} записей")
print(f"Тестовая выборка: {len(test_df)} записей")

