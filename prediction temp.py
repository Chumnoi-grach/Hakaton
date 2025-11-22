import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
import numpy as np

#data preparation
def remove_balls(df):
    mask_to_remove = (df['BALLS'] == 0.0) & (df['MARK'].isin(['ня', 'нд', 'з']))
    df = df[~mask_to_remove]

def get_random_ball(mark):
    if mark == '3':
        return 55
    elif mark == '4':
        return 75
    elif mark == '5':
        return 88
    else:
        return 0

def fill_balls(df):
    mask_missing_balls = df['BALLS'].isna() & df['MARK'].isin(['2', '3', '4', '5'])
    df.loc[mask_missing_balls, 'BALLS'] = df.loc[mask_missing_balls, 'MARK'].apply(get_random_ball)


#make features
def avg_feature(student_data, features):
    grades = student_data['BALLS'].dropna()
    features['avg_grade'] = grades.mean()

def sem_avg_feature(student_data, features):
    for i in range(1, 5):
        sem_grades = student_data[student_data['SEMESTER'] == i]['BALLS'].dropna()
        features[f'sem_{i}_avg'] = sem_grades.mean() if len(sem_grades) > 0 else 0

def correlation_sem_feature(student_data, features):
    time_points = [1, 2, 3, 4]
    sem_avgs = [features[f'sem_{i}_avg'] for i in [1, 2, 3, 4]]
    if len(set(sem_avgs)) == 1:
        features['correlation_sem'] = 0.0
        features['p_value_sem'] = 1.0
    else:
        correlation, p_value = stats.spearmanr(time_points, sem_avgs)
        features['correlation_sem'] = correlation
        features['p_value_sem'] = p_value

def cource_avg_feature(student_data, features):
    features['course_1_avg'] = student_data[student_data['SEMESTER'].isin([1, 2])]['BALLS'].dropna().mean()
    features['course_2_avg'] = student_data[student_data['SEMESTER'].isin([3, 4])]['BALLS'].dropna().mean()

def correlation_cource_feature(student_data, features):
    time_points = [1, 2]
    cource_avgs = [features['course_1_avg'], features['course_2_avg']]
    if (len(set(cource_avgs)) == 1):
        features['correlation_cource'] = 0.0
        features['p_value_cource'] = 1.0
    else:
        correlation, p_value = stats.spearmanr(time_points, cource_avgs)
        features['correlation_cource'] = correlation
        features['p_value_cource'] = p_value

def gpa_drops_feature(student_data, features):
    features['gpa_drop'] = features['course_1_avg'] - features['course_2_avg']
    features['gpa_drop_bin'] = 1 if features['gpa_drop'] < 0 else 0

def sem_drops_feature(student_data, features):
    for i in range(1, 4):
        features[f'sem{i + 1}-{i}_drop'] = features[f'sem_{i + 1}_avg'] - features[f'sem_{i}_avg']
        features[f'sem{i + 1}-{i}_drop_bin'] = 1 if features[f'sem{i + 1}-{i}_drop'] < 0 else 0

def debts_feature(student_data, features):
    debt_mask = (student_data['BALLS'].isna()) | (student_data['MARK'] == 'з')
    debts_data = student_data[debt_mask]
    features['dolgi'] = len(debts_data)
    for sem in range(1, 5):
        features[f'dolgi_sem{sem}'] = len(debts_data[debts_data['SEMESTER'] == sem])

def closed_debts_feature(student_data, features):
    closed_debt_mask = (student_data['BALLS'].notna()) | (student_data['MARK'] == 'з')
    closed_debts_data = student_data[closed_debt_mask]
    features['closed_dolgi'] = len(closed_debts_data)
    for sem in range(1, 5):
        features[f'closed_dolgi_sem{sem}'] = len(closed_debts_data[closed_debts_data['SEMESTER'] == sem])

def zach_feature(student_data, features):
    features['zach_count'] = (student_data['TYPE'] == 'зач').sum()

def exam_feature(student_data, features):
    features['exam_count'] = (student_data['TYPE'] == 'экз').sum()

def status_counts_feature(student_data, features):
    status_counts = student_data['Unnamed: 5'].value_counts()
    features['studying_count'] = status_counts.get('учится', 0)
    features['expelled_count'] = status_counts.get('отчислен', 0)
    features['ever_expelled'] = 1 if features['expelled_count'] > 0 else 0

def target_feature(student_data, features):
    first_row = student_data.iloc[0]
    if 'выпуск' in student_data.columns and pd.notna(first_row['выпуск']):
        features['target'] = 1 if first_row['выпуск'] == 'выпустился' else 0

def create_student_features(df):
    features_list = []
    for student_id in df['PK'].unique():
        student_data = df[df['PK'] == student_id]
        features = {'student_id': student_id}
        avg_feature(student_data, features)
        sem_avg_feature(student_data, features)
        correlation_sem_feature(student_data, features)
        cource_avg_feature(student_data, features)
        correlation_cource_feature(student_data, features)
        gpa_drops_feature(student_data, features)
        sem_drops_feature(student_data, features)
        debts_feature(student_data, features)
        closed_debts_feature(student_data, features)
        zach_feature(student_data, features)
        exam_feature(student_data, features)
        status_counts_feature(student_data, features)
        target_feature(student_data, features)
        features_list.append(features)
    return pd.DataFrame(features_list)

data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

df = data.merge(marking, left_on='PK', right_on='ИД', how='left')
df = df.drop_duplicates(subset=['PK', 'SEMESTER', 'BALLS', 'TYPE', 'DNAME', 'MARK'])

remove_balls(df)
fill_balls(df)

df.to_csv('cleaned_data.csv', index=False)

student_features = create_student_features(df)
print(f"Создано профилей: {len(student_features)}")

train_df = student_features[student_features['target'].notna()].copy()
test_df = student_features[student_features['target'].isna()].copy()

print(f"Обучающая выборка: {len(train_df)} студентов")
print(f"Тестовая выборка: {len(test_df)} студентов")

feature_columns = [
    'avg_grade',
    'zach_count', 'exam_count',
    #'studying_count', 'expelled_count',
    #'ever_expelled',
    'sem_1_avg', 'sem_2_avg', 'sem_3_avg', 'sem_4_avg',
    'course_1_avg', 'course_2_avg',
    'gpa_drop', 'gpa_drop_bin',
    'sem4-3_drop', 'sem4-3_drop_bin',
    'sem3-2_drop', 'sem3-2_drop_bin',
    'sem2-1_drop', 'sem2-1_drop_bin',
    'correlation_sem', 'p_value_sem',
    'correlation_cource', 'p_value_cource',
    'dolgi', 'dolgi_sem1', 'dolgi_sem2', 'dolgi_sem3', 'dolgi_sem4',
    'closed_dolgi', 'closed_dolgi_sem1', 'closed_dolgi_sem2', 'closed_dolgi_sem3', 'closed_dolgi_sem4'
]

X_train = train_df[feature_columns].fillna(0)
y_train = train_df['target']
X_test = test_df[feature_columns].fillna(0)

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

print(feature_imp_df.head(15))

plt.figure(figsize=(12, 8))
plt.barh(feature_imp_df['feature'][:15], feature_imp_df['importance'][:15])
plt.xlabel('Важность')
plt.title('Топ-15 самых важных признаков (Random Forest)')
plt.tight_layout()
plt.savefig('analysis_plots/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

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

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_array, X_train, show=False)
plt.title("SHAP Summary Plot - Влияние признаков на прогноз")
plt.tight_layout()
plt.savefig('analysis_plots/shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Анализ распределения вероятностей
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
y_train_proba = model.predict_proba(X_train)[:, 1]
plt.hist(y_train_proba[y_train == 0], bins=30, alpha=0.7, label='Не выпустились', color='red')
plt.hist(y_train_proba[y_train == 1], bins=30, alpha=0.7, label='Выпустились', color='green')
plt.xlabel('Вероятность выпуска')
plt.ylabel('Количество студентов')
plt.legend()
plt.title('Распределение вероятностей')

plt.subplot(1, 2, 2)
# Анализ порогов классификации
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_train, y_train_proba)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Порог')
plt.ylabel('Score')
plt.legend()
plt.title('Precision-Recall vs Threshold')

plt.tight_layout()
plt.savefig('analysis_plots/probability_analysis.png', dpi=300)
plt.show()