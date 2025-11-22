import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap

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

data = pd.read_csv('../../kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('../../kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

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

os.makedirs('../../analysis_plots', exist_ok=True)

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



# --------------------


import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, RocCurveDisplay

# --- БЛОК 1: Настройка визуализации ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- БЛОК 2: Корреляция признаков с целевой переменной ---
# Помогает понять линейные зависимости
plt.figure(figsize=(14, 12))
corr_matrix = train_df[feature_columns + ['target']].corr()
# Рисуем только нижний треугольник для чистоты
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.savefig('analysis_plots/correlation_matrix.png', dpi=300)
plt.close()

# --- БЛОК 3: Сравнение распределений (Отчислен vs Выпустился) ---
# Выбираем топ-4 самых важных признака из модели
top_features = feature_imp_df['feature'][:4].values

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    # Используем Violin plot для отображения плотности распределения
    sns.violinplot(data=train_df, x='target', y=col, ax=axes[i], palette="muted", split=True)
    axes[i].set_title(f'Распределение: {col}')
    axes[i].set_xlabel('Статус (0 - Отчислен, 1 - Выпуск)')

plt.tight_layout()
plt.savefig('analysis_plots/distributions_top_features.png', dpi=300)
plt.close()

# --- БЛОК 4: Partial Dependence Plots (PDP) ---
# Показывает, как меняется вероятность выпуска при изменении значения признака
print("Генерация Partial Dependence Plots...")
common_features = [col for col in ['avg_grade', 'course_1_avg', 'studying_count', 'exam_count'] if
                   col in X_train.columns]

if common_features:
    fig, ax = plt.subplots(figsize=(14, 10))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        common_features,
        kind="average",  # Показывает среднее влияние
        ax=ax
    )
    plt.suptitle('Влияние изменения значения признака на вероятность выпуска', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('analysis_plots/partial_dependence.png', dpi=300)
    plt.close()

# --- БЛОК 5: SHAP Dependence Plots (Взаимодействия) ---
# Показывает, как один признак меняет влияние другого
# Например: как влияет средний балл (x) на вывод модели (y), и зависит ли это от количества "удовлетворительно" (цвет)
if 'avg_grade' in X_train.columns:
    plt.figure(figsize=(10, 7))
    # shap_values_array мы берем из твоего кода выше (где ты обрабатывал shap_values)
    # Нужно найти индекс признака avg_grade
    feature_idx = X_train.columns.get_loc("avg_grade")

    shap.dependence_plot(
        "avg_grade",
        shap_values_array,
        X_train,
        interaction_index="course_1_avg",  # Смотрим взаимодействие с 1 курсом
        show=False
    )
    plt.title("Влияние среднего балла с учетом успеваемости на 1 курсе")
    plt.tight_layout()
    plt.savefig('analysis_plots/shap_dependence_avg_grade.png', dpi=300)
    plt.close()

# --- БЛОК 6: Confusion Matrix на обучающей выборке (через кросс-валидацию) ---
# Чтобы честно оценить, где модель ошибается
y_pred_cv = cross_val_predict(model, X_train, y_train, cv=5)
conf_mat = confusion_matrix(y_train, y_pred_cv)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Отчислен (Предсказано)', 'Выпуск (Предсказано)'],
            yticklabels=['Отчислен (Факт)', 'Выпуск (Факт)'])
plt.title('Матрица ошибок (Confusion Matrix) - Cross Validation')
plt.ylabel('Истина')
plt.xlabel('Предсказание')
plt.savefig('analysis_plots/confusion_matrix.png', dpi=300)
plt.close()

print("Все графики успешно сохранены в папку 'analysis_plots'")