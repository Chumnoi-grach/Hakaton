import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import numpy as np

def create_student_features(df):
    features_list = []

    for student_id in df['PK'].unique():
        student_data = df[df['PK'] == student_id]

        student_data = student_data.drop_duplicates(subset=['SEMESTER', 'BALLS', 'TYPE', 'DNAME', 'MARK'])

        features = {'student_id': student_id}

        first_row = student_data.iloc[0]
        features['faculty'] = first_row['Факультет']
        features['direction'] = first_row['Направление']
        features['admission_year'] = first_row['год поступления']

        grades = student_data['BALLS'].dropna()
        features['avg_grade'] = grades.mean()
        features['total_subjects'] = len(student_data)
        features['total_grades'] = len(grades)

        for i in range(1, 5):
            sem_grades = student_data[student_data['SEMESTER'] == i]['BALLS'].dropna()
            features[f'sem_{i}_avg'] = sem_grades.mean() if len(sem_grades) > 0 else 0

        features['course_1_avg'] = student_data[student_data['SEMESTER'].isin([1, 2])]['BALLS'].dropna().mean()
        features['course_2_avg'] = student_data[student_data['SEMESTER'].isin([3, 4])]['BALLS'].dropna().mean()

        for i in range(1, 4):
            features[f'sharp_drop_{i}-{i+1}'] = 0 if abs(features[f'sem_{i + 1}_avg'] - features[f'sem_{i}_avg']) >= 1 else 1

        features['zach_count'] = (student_data['TYPE'] == 'зач').sum()
        features['exam_count'] = (student_data['TYPE'] == 'экз').sum()

        status_counts = student_data['Unnamed: 5'].value_counts()
        features['studying_count'] = status_counts.get('учится', 0)
        features['expelled_count'] = status_counts.get('отчислен', 0)

        features['ever_expelled'] = 1 if features['expelled_count'] > 0 else 0

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
    'admission_year', 'avg_grade',
    'total_subjects', 'total_grades', 'zach_count', 'exam_count',
    'studying_count', 'expelled_count',
    'ever_expelled',
    'faculty_encoded', 'direction_encoded',
    'sem_1_avg', 'sem_2_avg', 'sem_3_avg', 'sem_4_avg',
    'course_1_avg', 'course_2_avg',
    'sharp_drop_1-2', 'sharp_drop_2-3',
    'sharp_drop_3-4'
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

os.makedirs('analysis_plots', exist_ok=True)

feature_imp_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Визуализация важности признаков
plt.figure(figsize=(12, 8))
plt.barh(feature_imp_df['feature'][:15], feature_imp_df['importance'][:15])
plt.xlabel('Важность')
plt.title('Топ-15 самых важных признаков (Random Forest)')
plt.tight_layout()
plt.savefig('analysis_plots/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Инициализация SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_train)

if hasattr(shap_values, 'values'):
    shap_values_array = shap_values.values
    base_value = explainer.expected_value
else:
    shap_values_array = shap_values
    base_value = explainer.expected_value

if len(shap_values_array.shape) == 3:
    shap_values_array = shap_values_array[:, :, 1]
    if isinstance(base_value, list):
        base_value = base_value[1]

# Детальный summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_array, X_train, show=False)
plt.title("SHAP Summary Plot - Влияние признаков на прогноз")
plt.tight_layout()
plt.savefig('analysis_plots/shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()