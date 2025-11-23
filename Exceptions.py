import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/data.csv')
marking = pd.read_csv('kaggle/input/nstu-hach-ai-track-education-case/marking.csv')

# Преобразование дат
data['EXAM_DATE'] = pd.to_datetime(data['EXAM_DATE'])
marking['дата изменения'] = pd.to_timedelta(marking['дата изменения'].str.replace(' days', '').astype(float), unit='D')

def integrate_and_analyze_anomalies(data, marking):
    """Интеграция таблиц и комплексный анализ аномалий"""

    # Создаем объединенную таблицу
    merged_data = []

    for student_id in data['PK'].unique():
        # Данные об успеваемости
        student_data = data[data['PK'] == student_id].copy()

        # Данные о статусах
        student_marking = marking[marking['ИД'] == student_id].copy()

        if len(student_marking) == 0:
            continue

        # Базовая информация о студенте
        admission_year = student_marking['год поступления'].iloc[0]
        faculty = student_marking['Факультет'].iloc[0]
        direction = student_marking['Направление'].iloc[0]
        final_graduate = student_marking['выпуск'].iloc[0] if 'выпуск' in student_marking.columns else None

        # Рассчитываем предполагаемые даты семестров
        admission_date = pd.Timestamp(f"{int(admission_year)}-09-01")

        for idx, row in student_data.iterrows():
            merged_row = {
                'PK': student_id,
                'SEMESTER': row['SEMESTER'],
                'DNAME': row['DNAME'],
                'TYPE': row['TYPE'],
                'MARK': row['MARK'],
                'GRADE': row['GRADE'],
                'BALLS': row['BALLS'],
                'EXAM_DATE': row['EXAM_DATE'],
                'Факультет': faculty,
                'Направление': direction,
                'год_поступления': admission_year,
                'выпуск': final_graduate,
                'admission_date': admission_date
            }
            merged_data.append(merged_row)

    merged_df = pd.DataFrame(merged_data)
    return merged_df

def detect_complex_anomalies(merged_df):
    """Обнаружение комплексных аномалий"""
    columns = ['PK', 'Экзамены до поступления',
               'Нарушение хронологии семестров',
               'Много экзаменов в один день',
               'Слишком долгий период обучения',
               'Подозрительные даты в экзаменах семестров']
    anomalies = pd.DataFrame(columns=columns)

    for student_id in merged_df['PK'].unique():
        student_data = merged_df[merged_df['PK'] == student_id].sort_values('EXAM_DATE')
        # Обнуление данных
        record = {
            'PK': student_id,
            'Экзамены до поступления': 0,
            'Нарушение хронологии семестров': 0,
            'Много экзаменов в один день': 0,
            'Слишком долгий период обучения': 0,
            'Подозрительные даты в экзаменах семестров': 0
        }

        if len(student_data) == 0:
            continue

        # Базовые данные студента
        admission_year = student_data['год_поступления'].iloc[0]
        admission_date = student_data['admission_date'].iloc[0]
        faculty = student_data['Факультет'].iloc[0]

        # 1. Аномалия: Экзамены до поступления
        exams_before_admission = student_data[student_data['EXAM_DATE'] < admission_date]
        if len(exams_before_admission) > 0:
            record['Экзамены до поступления'] = 1

        # 2. Аномалия: Нарушение хронологии семестров
        semester_analysis = student_data.groupby('SEMESTER').agg({
            'EXAM_DATE': ['min', 'max', 'count']
        }).reset_index()

        semester_analysis.columns = ['SEMESTER', 'min_date', 'max_date', 'exam_count']

        # Проверяем порядок семестров
        if len(semester_analysis) > 1:
            semester_analysis = semester_analysis.sort_values('min_date')
            expected_order = sorted(semester_analysis['SEMESTER'].unique())
            actual_order = semester_analysis['SEMESTER'].tolist()

            if expected_order != actual_order:
                record['Нарушение хронологии семестров'] = 1

        # 3. Аномалия: Все экзамены в один день (5)
        unique_dates = student_data['EXAM_DATE'].nunique()
        total_exams = len(student_data)
        if unique_dates == 1 and total_exams > 2:
            record['Много экзаменов в один день'] = 1

        # 4. Аномалия: Слишком долгий период обучения за 2 курса
        study_period = student_data['EXAM_DATE'].max() - student_data['EXAM_DATE'].min()
        if study_period > timedelta(days=800):  # Более 2.5 лет за 2 курса
            record['Слишком долгий период обучения'] = 1

        # 5. Аномалия: Подозрительные даты в экзаменах семестров
        for semester in student_data['SEMESTER'].unique():
            sem_data = student_data[student_data['SEMESTER'] == semester]
            sem_duration = sem_data['EXAM_DATE'].max() - sem_data['EXAM_DATE'].min()

            if sem_duration > timedelta(days=180):
                record['Подозрительные даты в экзаменах семестров'] = 1
                break

        # Исправленная строка конкатенации
        anomalies = pd.concat([anomalies, pd.DataFrame([record])], ignore_index=True)

    return anomalies

# Запуск анализа
print("НАЧИНАЕМ КОМПЛЕКСНЫЙ АНАЛИЗ АНОМАЛИЙ...")
merged_data = integrate_and_analyze_anomalies(data, marking)
anomalies_df = detect_complex_anomalies(merged_data)

# Сохранение в файл
anomalies_df.to_csv('anomalies.csv', index=False)
print("Файл anomalies.csv успешно создан!")

# Построение диаграммы распределения аномалий по факультетам
print("\nСТРОИМ ДИАГРАММУ РАСПРЕДЕЛЕНИЯ АНОМАЛИЙ ПО ФАКУЛЬТЕТАМ...")

# Объединяем данные об аномалиях с информацией о факультетах
anomalies_with_faculty = anomalies_df.merge(
    merged_data[['PK', 'Факультет']].drop_duplicates(),
    on='PK',
    how='left'
)

# Считаем общее количество аномалий на каждом факультете
faculty_anomalies = anomalies_with_faculty.groupby('Факультет').agg({
    'Экзамены до поступления': 'sum',
    'Нарушение хронологии семестров': 'sum',
    'Много экзаменов в один день': 'sum',
    'Слишком долгий период обучения': 'sum',
    'Подозрительные даты в экзаменах семестров': 'sum'
}).reset_index()

# ИСПРАВЛЕНИЕ: Явно указываем числовые столбцы для суммирования
numeric_columns = ['Экзамены до поступления', 'Нарушение хронологии семестров',
                   'Много экзаменов в один день', 'Слишком долгий период обучения',
                   'Подозрительные даты в экзаменах семестров']

faculty_anomalies['Всего_аномалий'] = faculty_anomalies[numeric_columns].sum(axis=1)

# Сортируем по убыванию общего количества аномалий
faculty_anomalies = faculty_anomalies.sort_values('Всего_аномалий', ascending=False)

# Создаем график
plt.figure(figsize=(14, 10))

# Столбчатая диаграмма с разбивкой по типам аномалий
ax = faculty_anomalies.set_index('Факультет')[numeric_columns].plot(
    kind='bar',
    stacked=True,
    figsize=(14, 8),
    color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
)

plt.title('РАСПРЕДЕЛЕНИЕ АНОМАЛИЙ ПО ФАКУЛЬТЕТАМ', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Факультет', fontsize=12)
plt.ylabel('Количество аномалий', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Типы аномалий', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Добавляем подписи с общим количеством аномалий над столбцами
for i, (idx, row) in enumerate(faculty_anomalies.iterrows()):
    plt.text(i, row['Всего_аномалий'] + 0.5, str(int(row['Всего_аномалий'])),
             ha='center', va='bottom', fontweight='bold')

plt.savefig('anomalies_by_faculty.png', dpi=300, bbox_inches='tight')
plt.show()

# Дополнительная статистика
print("\nСТАТИСТИКА ПО ФАКУЛЬТЕТАМ:")
print("="*50)

# Количество студентов с аномалиями по факультетам
students_with_anomalies = anomalies_with_faculty.groupby('Факультет')['PK'].nunique()
total_students_by_faculty = merged_data.groupby('Факультет')['PK'].nunique()

# Создаем таблицу статистики
stats_df = pd.DataFrame({
    'Всего студентов': total_students_by_faculty,
    'Студентов с аномалиями': students_with_anomalies,
    'Всего аномалий': faculty_anomalies.set_index('Факультет')['Всего_аномалий']
})

# ИСПРАВЛЕНИЕ: Преобразуем столбцы в числовой формат
stats_df = stats_df.apply(pd.to_numeric, errors='coerce')

# Теперь можем безопасно выполнять вычисления
stats_df['% студентов с аномалиями'] = (stats_df['Студентов с аномалиями'] / stats_df['Всего студентов'] * 100).round(1)
stats_df['Аномалий на студента'] = (stats_df['Всего аномалий'] / stats_df['Студентов с аномалиями']).round(2)

print(stats_df.sort_values('Всего аномалий', ascending=False))

# Круговая диаграмма распределения общего количества аномалий по факультетам
plt.figure(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(faculty_anomalies)))
plt.pie(faculty_anomalies['Всего_аномалий'],
        labels=faculty_anomalies['Факультет'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90)
plt.title('ДОЛЯ АНОМАЛИЙ ПО ФАКУЛЬТЕТАМ (В % ОТ ОБЩЕГО КОЛИЧЕСТВА)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('anomalies_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nСозданы файлы:")
print("- anomalies_by_faculty.png - Столбчатая диаграмма распределения аномалий")
print("- anomalies_pie_chart.png - Круговая диаграмма долей аномалий")
print(f"\nВсего обнаружено аномалий: {anomalies_df.drop('PK', axis=1).sum().sum()}")
print(f"Студентов с аномалиями: {len(anomalies_df)}")

# Дополнительная информация для отладки
print("\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
print(f"Количество факультетов: {len(faculty_anomalies)}")
print(f"Типы данных в faculty_anomalies:")
print(faculty_anomalies.dtypes)
print(f"\nПервые 5 строк faculty_anomalies:")
print(faculty_anomalies.head())