import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
data['exam date'] = pd.to_datetime(data['EXAM_DATE'])

marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

def get_data_for_anomalies(data, marking):
    merged_data = []

    for student_id in data['PK'].unique():
        student_data = data[data['PK'] == student_id].copy()

        student_marking = marking[marking['ИД'] == student_id].copy()

        if len(student_marking) == 0:
            continue

        admission_year = student_marking['год поступления'].iloc[0]
        faculty = student_marking['Факультет'].iloc[0]

        admission_date = pd.Timestamp(f"{int(admission_year)}-09-01")

        for idx, row in student_data.iterrows():
            merged_row = {
                'id': student_id,
                'semester': row['SEMESTER'],
                'subject name': row['DNAME'],
                'type of subject': row['TYPE'],
                'mark': row['MARK'],
                'grade': row['GRADE'],
                'balls': row['BALLS'],
                'exam date': row['exam date'],
                'faculty': faculty,
                'admission_date': admission_date
            }
            merged_data.append(merged_row)

    merged_df = pd.DataFrame(merged_data)
    return merged_df

def detect_anomalies(merged_df):
    columns = [
        'id',
        'exams before admission',
        'violation of semester chronology',
        'many exams in one day',
        'interrupted his studies',
        'strange exam dates'
    ]
    anomalies = pd.DataFrame(columns=columns)

    for student_id in merged_df['id'].unique():
        student_data = merged_df[merged_df['id'] == student_id].sort_values('exam date')
        record = {
            'id': student_id,
            'exams before admission': 0,
            'violation of semester chronology': 0,
            'many exams in one day': 0,
            'interrupted his studies': 0,
            'strange exam dates': 0
        }

        if len(student_data) == 0:
            continue

        admission_date = student_data['admission_date'].iloc[0]

        # экзамены до поступления
        exams_before_admission = student_data[student_data['exam date'] < admission_date]
        if len(exams_before_admission) > 0:
            record['exams before admission'] = 1

        # нарушение хронологии семестров
        semester_analysis = student_data.groupby('semester').agg({
            'exam date': ['min', 'max', 'count']
        }).reset_index()

        semester_analysis.columns = ['semester', 'min_date', 'max_date', 'exam_count']

        # неправильный порядок семестров
        if len(semester_analysis) > 1:
            semester_analysis = semester_analysis.sort_values('min_date')
            expected_order = sorted(semester_analysis['semester'].unique())
            actual_order = semester_analysis['semester'].tolist()

            if expected_order != actual_order:
                record['violation of semester chronology'] = 1

        # много экзаменов в один день
        unique_dates = student_data['exam date'].nunique()
        total_exams = len(student_data)
        if unique_dates == 1 and total_exams > 2:
            record['many exams in one day'] = 1

        # студент прерывал свое обучение
        study_period = student_data['exam date'].max() - student_data['exam date'].min()
        if study_period > timedelta(days=(366 * 2)):
            record['interrupted his studies'] = 1

        # странные даты экзаменов
        for semester in student_data['semester'].unique():
            sem_data = student_data[student_data['semester'] == semester]
            sem_duration = sem_data['exam date'].max() - sem_data['exam date'].min()

            if sem_duration > timedelta(days=180):
                record['strange exam dates'] = 1
                break

        anomalies = pd.concat([anomalies, pd.DataFrame([record])], ignore_index=True)

    return anomalies

print("анализ аномалий начался")
merged_data = get_data_for_anomalies(data, marking)
anomalies_df = detect_anomalies(merged_data)

anomalies_df.to_csv('anomalies.csv', index=False)

anomalies_with_faculty = anomalies_df.merge(
    merged_data[['id', 'faculty']].drop_duplicates(),
    on='id',
    how='left'
)

faculty_anomalies = anomalies_with_faculty.groupby('faculty').agg({
    'exams before admission': 'sum',
    'violation of semester chronology': 'sum',
    'many exams in one day': 'sum',
    'interrupted his studies': 'sum',
    'strange exam dates': 'sum'
}).reset_index()

numeric_columns = [
    'exams before admission',
    'violation of semester chronology',
    'many exams in one day',
    'interrupted his studies',
    'strange exam dates'
]

faculty_anomalies['total anomalies'] = faculty_anomalies[numeric_columns].sum(axis=1)

faculty_anomalies = faculty_anomalies.sort_values('total anomalies', ascending=False)

ax = faculty_anomalies.set_index('faculty')[numeric_columns].plot(
    kind='bar',
    stacked=True,
    figsize=(14, 8),
    color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
)

plt.title('распределение аномалий по факультетам', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('факультет', fontsize=12)
plt.ylabel('количество аномалий', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='типы аномалий', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

for i, (idx, row) in enumerate(faculty_anomalies.iterrows()):
    plt.text(i, row['total anomalies'] + 0.5, str(int(row['total anomalies'])),
             ha='center', va='bottom', fontweight='bold')

plt.savefig('anomalies_by_faculty.png', dpi=300, bbox_inches='tight')
print('распределение аномалий по факультетам сохранено в anomalies_by_faculty.png')

students_with_anomalies = anomalies_with_faculty.groupby('faculty')['id'].nunique()
total_students_by_faculty = merged_data.groupby('faculty')['id'].nunique()

plt.figure(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(faculty_anomalies)))
plt.pie(faculty_anomalies['total anomalies'],
        labels=faculty_anomalies['faculty'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90)
plt.title('доля аномалий по факультетам в процентах от общего числа аномалий', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('anomalies_pie_chart.png', dpi=300, bbox_inches='tight')
print('доля аномалий по факультетам в процентах от общего числа аномалий сохранено в anomalies_pie_chart.png')
