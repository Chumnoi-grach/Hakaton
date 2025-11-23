import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap
import scipy.stats as stats
import numpy as np

#data preparation
def remove_balls(df):
    mask_to_fix = (df['BALLS'] == 0.0) & (df['MARK'].isin(['ня', 'нд', 'з']))
    df.loc[mask_to_fix, 'BALLS'] = np.nan
    return df

def fill_balls_with_mean(df):
    mean_balls_by_mark = df.groupby('MARK')['BALLS'].mean().to_dict()
    mask_missing_balls = df['BALLS'].isna() & df['MARK'].isin(['2', '3', '4', '5'])

    def get_mean_ball(mark):
        return mean_balls_by_mark.get(mark, 0)

    df.loc[mask_missing_balls, 'BALLS'] = df.loc[mask_missing_balls, 'MARK'].apply(get_mean_ball)
    return df

def round_balls_to_int(df):
    df['BALLS'] = df['BALLS'].apply(lambda x: int(round(x)) if pd.notna(x) else x)
    return df


def fill_balls_from_grade(df):
    mask = (df['MARK'] == 'з') & (df['BALLS'].isna()) & (df['GRADE'].notna())
    grade_medians = df.groupby('GRADE')['BALLS'].median().to_dict()

    def get_median_by_grade(grade):
        return grade_medians.get(grade, None)

    df.loc[mask, 'BALLS'] = df.loc[mask, 'GRADE'].apply(get_median_by_grade)
    return df


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

def zachet_feature(student_data, features):
    debt_mask = (student_data['BALLS'].isna()) | (student_data['MARK'] == 'з')
    zachet_data = student_data[debt_mask]
    features['zachet'] = len(zachet_data)
    for sem in range(1, 5):
        features[f'zachet_sem{sem}'] = len(zachet_data[zachet_data['SEMESTER'] == sem])

def closed_zachet_feature(student_data, features):
    closed_debt_mask = (student_data['BALLS'].notna()) | (student_data['MARK'] == 'з')
    closed_zachet_data = student_data[closed_debt_mask]
    features['closed_zachet'] = len(closed_zachet_data)
    for sem in range(1, 5):
        features[f'closed_zachet_sem{sem}'] = len(closed_zachet_data[closed_zachet_data['SEMESTER'] == sem])

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

def nia_feature(student_data, features):
    features['nia'] = (student_data['MARK'] == 'ня').sum()

def nd_feature(student_data, features):
    features['nd'] = (student_data['MARK'] == 'нд').sum()

def nz_feature(student_data, features):
    features['nz'] = (student_data['MARK'] == 'нз').sum()

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
        zachet_feature(student_data, features)
        closed_zachet_feature(student_data, features)
        zach_feature(student_data, features)
        exam_feature(student_data, features)
        status_counts_feature(student_data, features)
        target_feature(student_data, features)
        nia_feature(student_data, features)
        nd_feature(student_data, features)
        nz_feature(student_data, features)
        features_list.append(features)
    return pd.DataFrame(features_list)

data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

df = data.merge(marking, left_on='PK', right_on='ИД', how='left')
df = df.drop_duplicates(subset=['PK', 'SEMESTER', 'BALLS', 'TYPE', 'DNAME', 'MARK'])

df = remove_balls(df)
df = fill_balls_with_mean(df)
df = round_balls_to_int(df)
df = fill_balls_from_grade(df)

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
    'sem_1_avg', 'sem_2_avg', 'sem_3_avg', 'sem_4_avg',
    'course_1_avg', 'course_2_avg',
    'gpa_drop', 'gpa_drop_bin',
    'sem4-3_drop', 'sem4-3_drop_bin',
    'sem3-2_drop', 'sem3-2_drop_bin',
    'sem2-1_drop', 'sem2-1_drop_bin',
    'correlation_sem', 'p_value_sem',
    'correlation_cource', 'p_value_cource',
    'zachet', 'zachet_sem1', 'zachet_sem2', 'zachet_sem3', 'zachet_sem4',
    'closed_zachet', 'closed_zachet_sem1', 'closed_zachet_sem2', 'closed_zachet_sem3', 'closed_zachet_sem4',
    'nia', 'nd', 'nz'
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

#Топ самых важных признаков
plt.figure(figsize=(12, 8))
plt.barh(feature_imp_df['feature'][:15], feature_imp_df['importance'][:15])
plt.xlabel('Важность')
plt.title('Топ-15 самых важных признаков (Random Forest)')
plt.tight_layout()
plt.savefig('analysis_plots/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Исследования по snap
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

# Распределение признаков по классам
top_features = feature_imp_df['feature'].head(5).tolist()

n_cols = 3
n_rows = (len(top_features) + n_cols - 1) // n_cols

plt.figure(figsize=(15, 5 * n_rows))
for i, feature in enumerate(top_features, 1):
    plt.subplot(n_rows, n_cols, i)

    # Данные для класса 0 и 1
    feature_class_0 = X_train[y_train == 0][feature]
    feature_class_1 = X_train[y_train == 1][feature]

    plt.hist(feature_class_0, bins=30, alpha=0.7, label='Не выпустились', color='red', density=True)
    plt.hist(feature_class_1, bins=30, alpha=0.7, label='Выпустились', color='green', density=True)

    plt.xlabel(feature)
    plt.ylabel('Плотность')
    plt.legend()
    plt.title(f'Распределение {feature}')

plt.tight_layout()
plt.savefig('analysis_plots/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#ROC-кривая
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

plt.figure(figsize=(10, 8))

y_train_proba = model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая (Random Forest)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

gmeans = np.sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
optimal_threshold = thresholds[ix]
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Optimal threshold: {optimal_threshold:.3f}')

plt.legend()
plt.tight_layout()
plt.savefig('analysis_plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Оптимальный порог: {optimal_threshold:.4f}")
print(f"При оптимальном пороге:")
print(f"  True Positive Rate: {tpr[ix]:.4f}")
print(f"  False Positive Rate: {fpr[ix]:.4f}")

#Precision-Recall кривая
from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(10, 8))

#Вычисляем Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_train, y_train_proba)
average_precision = average_precision_score(y_train, y_train_proba)

plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR curve (AP = {average_precision:.3f})')

baseline = len(y_train[y_train == 1]) / len(y_train)
plt.axhline(y=baseline, color='red', linestyle='--',
            label=f'Random classifier (AP = {baseline:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Полнота)')
plt.ylabel('Precision (Точность)')
plt.title('Precision-Recall кривая (Random Forest)')
plt.legend(loc="upper right")
plt.grid(alpha=0.3)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
ix = np.argmax(f1_scores)
optimal_threshold_pr = thresholds[ix] if ix < len(thresholds) else thresholds[-1]

plt.scatter(recall[ix], precision[ix], marker='o', color='black',
           label=f'Optimal F1: P={precision[ix]:.3f}, R={recall[ix]:.3f}')

plt.legend()
plt.tight_layout()
plt.savefig('analysis_plots/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Average Precision (AP): {average_precision:.4f}")
print(f"Базовая линия (случайный классификатор): {baseline:.4f}")
print(f"Оптимальная точка по F1-score:")
print(f"  Precision: {precision[ix]:.4f}")
print(f"  Recall: {recall[ix]:.4f}")
print(f"  F1-score: {f1_scores[ix]:.4f}")
print(f"  Порог: {optimal_threshold_pr:.4f}")

#Дисперсия топ-10 признаков
plt.figure(figsize=(12, 6))
top_10_features = feature_imp_df['feature'].head(10).tolist()

variances = []
for feature in top_10_features:
    variance = X_train[feature].var()
    variances.append(variance)

plt.bar(top_10_features, variances, color='skyblue', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Дисперсия')
plt.title('Дисперсия топ-10 самых важных признаков')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_plots/top_features_variance.png', dpi=300, bbox_inches='tight')
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
plt.tight_layout()
plt.savefig('analysis_plots/probability_analysis.png', dpi=300)
plt.show()