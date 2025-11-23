import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')
data['EXAM_DATE'] = pd.to_datetime(data['EXAM_DATE'])

def clean_numeric_column(series):
    series = series.astype(str).str.replace(',', '.')
    series = pd.to_numeric(series, errors='coerce')
    return series

data['BALLS'] = clean_numeric_column(data['BALLS'])
data['SEMESTER'] = clean_numeric_column(data['SEMESTER'])

def prepare_features(data, marking):
    merged_data = []

    for student_id in data['PK'].unique():
        student_data = data[data['PK'] == student_id].copy()
        student_marking = marking[marking['ИД'] == student_id].copy()

        if len(student_marking) == 0:
            continue

        admission_year = student_marking['год поступления'].iloc[0]
        faculty = student_marking['Факультет'].iloc[0]
        admission_date = pd.Timestamp(f"{int(admission_year)}-09-01")
        total_exams = len(student_data)
        unique_dates = student_data['EXAM_DATE'].nunique()

        avg_balls = student_data['BALLS'].mean() if not student_data['BALLS'].isna().all() else 0

        exams_before_admission = len(student_data[student_data['EXAM_DATE'] < admission_date])

        semester_order_violation = 0
        if len(student_data) > 1:
            try:
                semester_stats = student_data.groupby('SEMESTER')['EXAM_DATE'].agg(['min', 'max']).reset_index()
                if len(semester_stats) > 1:
                    sorted_by_date = semester_stats.sort_values('min')
                    expected_order = sorted(semester_stats['SEMESTER'])
                    actual_order = sorted_by_date['SEMESTER'].tolist()
                    semester_order_violation = 1 if expected_order != actual_order else 0
            except:
                semester_order_violation = 0

        study_period = (student_data['EXAM_DATE'].max() - student_data['EXAM_DATE'].min()).days

        semester_date_anomalies = 0
        try:
            for semester in student_data['SEMESTER'].unique():
                sem_data = student_data[student_data['SEMESTER'] == semester]
                if len(sem_data) > 1:
                    sem_duration = (sem_data['EXAM_DATE'].max() - sem_data['EXAM_DATE'].min()).days
                    if sem_duration > 180:
                        semester_date_anomalies = 1
                        break
        except:
            semester_date_anomalies = 0

        try:
            date_counts = student_data['EXAM_DATE'].value_counts()
            count_max_exams_one_day = date_counts.max() if len(date_counts) > 0 else 0
        except:
            count_max_exams_one_day = 0

        max_subject_deviation = 0

        if not student_data['BALLS'].isna().all():
            try:
                semester_means = student_data.groupby('SEMESTER')['BALLS'].mean()

                all_deviations = []

                for semester, sem_avg in semester_means.items():
                    if not np.isnan(sem_avg):
                        semester_subjects = student_data[student_data['SEMESTER'] == semester]

                        subject_deviations = abs(semester_subjects['BALLS'] - sem_avg)
                        all_deviations.extend(subject_deviations.tolist())

                max_subject_deviation = max(all_deviations) if all_deviations else 0

            except:
                max_subject_deviation = 0

        features = {
            'PK': student_id,
            'admission_year': admission_year,
            'faculty': faculty,
            'total_exams': total_exams,
            'unique_exam_dates': unique_dates,
            'avg_balls': avg_balls if not np.isnan(avg_balls) else 0,
            'exams_before_admission': exams_before_admission,
            'semester_order_violation': semester_order_violation,
            'study_period_days': study_period,
            'semester_date_anomalies': semester_date_anomalies,
            'count_max_exams_one_day': count_max_exams_one_day,
            'max_subject_deviation': max_subject_deviation
        }

        merged_data.append(features)

    return pd.DataFrame(merged_data)


def detect_anomalies_with_isolation_forest(features_df):
    numeric_features = ['total_exams', 'unique_exam_dates', 'avg_balls',
                        'exams_before_admission', 'study_period_days', 'count_max_exams_one_day',
                        'max_subject_deviation']

    for feature in numeric_features:
        features_df[feature] = pd.to_numeric(features_df[feature], errors='coerce')

    features_df[numeric_features] = features_df[numeric_features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[numeric_features])

    model = IsolationForest(
        contamination='auto',
        random_state=42,
        n_estimators=100
    )

    features_df['is_anomaly'] = model.fit_predict(X_scaled)
    features_df['is_anomaly'] = features_df['is_anomaly'].map({1: 0, -1: 1})

    features_df['anomaly_score'] = model.decision_function(X_scaled)

    return features_df, model, scaler


def interpret_anomalies(features_df):
    anomalies_df = pd.DataFrame(columns=['PK', 'Экзамены до поступления',
                                         'Нарушение хронологии семестров',
                                         'Много экзаменов в один день',
                                         'Слишком долгий период обучения',
                                         'Подозрительные даты в экзаменах семестров',
                                         'Резкий скачок в успеваемости предмета'])

    for _, student in features_df.iterrows():
        many_exams_flag = 1 if student['count_max_exams_one_day'] > 5 else 0

        record = {
            'PK': student['PK'],
            'Экзамены до поступления': 1 if student['exams_before_admission'] > 0 else 0,
            'Нарушение хронологии семестров': student['semester_order_violation'],
            'Много экзаменов в один день': many_exams_flag,  # ИСПРАВЛЕНО
            'Слишком долгий период обучения': 1 if student['study_period_days'] > 800 else 0,
            'Подозрительные даты в экзаменах семестров': student['semester_date_anomalies'],
            'Резкий скачок в успеваемости предмета': 1 if student['max_subject_deviation'] > 2.0 else 0
        }

        anomalies_df = pd.concat([anomalies_df, pd.DataFrame([record])], ignore_index=True)

    return anomalies_df


features_df = prepare_features(data, marking)

features_with_anomalies, model, scaler = detect_anomalies_with_isolation_forest(features_df)

anomalies_df = interpret_anomalies(features_with_anomalies)

# Сохранение результатов
anomalies_df.to_csv('ml_anomalies_detection.csv', index=False)
features_with_anomalies.to_csv('ml_student_features.csv', index=False)

print(f"Обнаружено аномальных студентов: {features_with_anomalies['is_anomaly'].sum()}")
print(f"Всего студентов: {len(features_with_anomalies)}")

anomalies_with_faculty = anomalies_df.merge(features_with_anomalies[['PK', 'faculty']], on='PK', how='left')

faculty_anomalies = anomalies_with_faculty.groupby('faculty').sum().reset_index()
anomaly_columns = ['Экзамены до поступления', 'Нарушение хронологии семестров',
                   'Много экзаменов в один день', 'Слишком долгий период обучения',
                   'Подозрительные даты в экзаменах семестров', 'Резкий скачок в успеваемости предмета']
faculty_anomalies = faculty_anomalies[['faculty'] + anomaly_columns]


faculty_totals = faculty_anomalies[anomaly_columns].sum(axis=1)
faculty_anomalies['total_anomalies'] = faculty_totals
faculty_anomalies_sorted = faculty_anomalies.sort_values('total_anomalies', ascending=False)
faculty_order = faculty_anomalies_sorted['faculty']


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

faculty_anomalies_melted = faculty_anomalies.melt(id_vars=['faculty'],
                                                  value_vars=anomaly_columns,
                                                  var_name='Тип аномалии',
                                                  value_name='Количество')

pivot_data = faculty_anomalies_melted.pivot_table(index='faculty',
                                                  columns='Тип аномалии',
                                                  values='Количество').reindex(faculty_order)

pivot_data = pivot_data.apply(pd.to_numeric, errors='coerce').fillna(0)

pivot_data.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_title('Распределение типов аномалий по факультетам', fontsize=14, fontweight='bold')
ax1.set_xlabel('Факультет')
ax1.set_ylabel('Количество аномалий')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Тип аномалии')

faculty_student_counts = features_with_anomalies['faculty'].value_counts()
heatmap_data = faculty_anomalies.set_index('faculty')[anomaly_columns]
heatmap_data = heatmap_data.astype(float)

heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

for faculty in heatmap_data.index:
    if faculty in faculty_student_counts:
        total_students = faculty_student_counts[faculty]
        if total_students > 0:
            heatmap_data.loc[faculty] = ((heatmap_data.loc[faculty] / total_students) * 100)

heatmap_data = heatmap_data.reindex(faculty_order)

heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce').fillna(0)

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': 'Процент студентов'}, ax=ax2)
ax2.set_title('Процент студентов с аномалиями по факультетам (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Тип аномалии')
ax2.set_ylabel('Факультет')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()