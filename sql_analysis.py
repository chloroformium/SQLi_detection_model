import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# чтение .csv файла и вывод размеров датасета
data_path='C:\\Users\\thism\\OneDrive\\Desktop\\dih\\db\\Modified_SQL_Dataset.csv'
df = pd.read_csv(data_path) 

print(f"Dataset size: {df.shape}")

# проверка наличия пропущенных значений
print("\nПропущенные значения:")
print(df.isnull().sum())

# распределение классов (SQLi/безопасный запрос)
print("\nРаспределение классов:")
class_distribution = df['Label'].value_counts()
print(class_distribution)


# визуализация распределения классов
plt.figure(figsize=(8, 5))
sns.countplot(x='Label', data=df)
plt.title('Распределение классов (0 - безопасный, 1 - SQL-инъекция)')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.show()


# удаление дубликатов
df_cleaned = df.drop_duplicates()
print(f"размер после удаления дубликатов: {df_cleaned.shape}")

# удаление пустых запросов 
df_cleaned = df_cleaned[df_cleaned['Query'].str.strip() != '']
print(f"Размер после удаления пустых запросов: {df_cleaned.shape}")

# Проверка наличия пропущенных значений и заполнение
if df_cleaned['Query'].isnull().sum() > 0:
    df_cleaned['Query'] = df_cleaned['Query'].fillna('')

# проверка корректности меток (только 0 и 1)
print("\nУникальные значения меток:")
print(df_cleaned['Label'].unique())


# базовые признаки

# длина запроса
df_cleaned['length'] = df_cleaned['Query'].apply(len)

# количество спецсимволов и ключевых слов
df_cleaned['quotes_count'] = df_cleaned['Query'].str.count("'")   # количество "'"
df_cleaned['double_quotes'] = df_cleaned['Query'].str.count('"')  # количество '"'
df_cleaned['semicolons'] = df_cleaned['Query'].str.count(';')     # количество ';'

# количество SQL-комментариев
df_cleaned['comment_dash'] = df_cleaned['Query'].str.count('--')  # комментарии типа --
df_cleaned['comment_hash'] = df_cleaned['Query'].str.count('#')   # комментарии типа #
df_cleaned['comment_c'] = df_cleaned['Query'].str.count('/\*')    # комментарии типа /* */

# опасные ключевые слова
df_cleaned['has_union'] = df_cleaned['Query'].str.contains('UNION', case=False).astype(int)
df_cleaned['has_or'] = df_cleaned['Query'].str.contains('OR 1=1', case=False).astype(int)
df_cleaned['has_drop'] = df_cleaned['Query'].str.contains('DROP', case=False).astype(int)
df_cleaned['has_exec'] = df_cleaned['Query'].str.contains('EXEC', case=False).astype(int)
df_cleaned['has_xp'] = df_cleaned['Query'].str.contains('XP_', case=False).astype(int)

# сложные признаки
import re

# поиск шаблонов типа "OR 1=1", "= 1=1" и т.д.
pattern = re.compile(
    r'('
    r'\b(OR|AND)\s+[\w\'"]+\s*=\s*[\w\'"]+\s*(--|#|/\*)|'  # OR 1=1 --
    r'=\s*[\w\'"]+\s*\(|'                                   # =SUBSTRING(...)
    r'=\s*(SELECT|UNION|NULL|IF|EXEC)|'                     # =SELECT
    r'=\s*\d+\s*=\s*\d+'                                    # =1=1
    r')',
    re.IGNORECASE
)

df_cleaned['has_sqli_pattern'] = df_cleaned['Query'].apply(
    lambda x: bool(re.search(pattern, str(x)) if pd.notna(x) else False)
)


# тест на корректное отображение, не входит в итоговый проект
'''
print("\nпервые 5 строк признаков:")
print(df_cleaned[['Query', 'Label', 'length', 'quotes_count', 'has_union', 'has_sqli_pattern']].head())
'''

# Проанализируем связь между признаками и меткой класса
print("\nСредние значения признаков по классам:")
print(df_cleaned.groupby('Label')[['length', 'quotes_count', 'semicolons', 'has_union', 'has_sqli_pattern']].mean())


# Распределение длины запросов по классам
plt.figure(figsize=(10, 6))
sns.histplot(data=df_cleaned, x='length', hue='Label', bins=50, element='step')
plt.title('Распределение длины запросов')
plt.xlabel('Длина запроса')
plt.ylabel('Количество')
plt.legend(['Безопасный', 'SQL-инъекция'])
plt.show()

# Корреляция между числовыми признаками
numeric_features = ['length', 'quotes_count', 'double_quotes', 'semicolons', 
 'comment_dash', 'comment_hash', 'comment_c', 
'has_union', 'has_or', 'has_drop', 'has_exec', 'has_xp', 
'has_sqli_pattern', 'Label']

plt.figure(figsize=(12, 10))
correlation_matrix = df_cleaned[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция между признаками')
plt.show()





