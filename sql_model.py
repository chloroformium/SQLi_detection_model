
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

data_path = 'C:\\Users\\thism\\OneDrive\\Desktop\\dih\\db\\Modified_SQL_Dataset.csv'
# путь сохранения модели и векторизатора
model_dir = 'C:\\Users\\thism\\OneDrive\\Desktop\dih'
# путь к файлам модели и векторизатора
model_path = os.path.join(model_dir, 'sql_injection_detector_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл не найден")

# загрузка данных 
Kaggle = pd.read_csv(data_path)

# удаление дубликатов и пустых запросов
print(f"Размер датасета до удаления дубликатов: {Kaggle.shape}")
Kaggle_cleaned = Kaggle.drop_duplicates()
Kaggle_cleaned = Kaggle_cleaned[Kaggle_cleaned['Query'].str.strip() != '']
print(f"Размер датасета после удаления дубликатов: {Kaggle_cleaned.shape}")

# проверка баланса классов
print("\nРаспределение классов:")
class_counts = Kaggle_cleaned['Label'].value_counts()
print(class_counts)
print(f"Процент SQL-инъекций: {class_counts.get(1, 0) / len(Kaggle_cleaned) * 100:.2f}%")

# разделение на признаки и целевую переменную
X = Kaggle_cleaned['Query']  # признаком являются сами запросы
y = Kaggle_cleaned['Label']  # 0-нормальный запрос, 1-SQL-инъекция

# разделение на выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# Преобразование текстовых запросов в числовые векторы с помощью TF-IDF
print("Векторизация запросов...")
vectorizer = TfidfVectorizer(
    stop_words=None,       # не удаляем стоп-слова
    ngram_range=(1, 2),    # используем 1-граммы и 2-граммы для учета контекста
    lowercase=True         # преобразование к нижнему регистру
)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print(f"Размер векторизованных данных: {X_train_vect.shape}")

#выбираем модель Radnom Forest с автобалансировкой классов 
print("Начато обучение модели Random Forest")
model = RandomForestClassifier(
    n_estimators=200,      # кол-во деревьев
    random_state=42,       # сид для воспроизводимости результатов
    class_weight='balanced'
)

# обучение модели
model.fit(X_train_vect, y_train)

# предсказание на тестовой выборке
y_pred = model.predict(X_test_vect)

# определение точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(conf_matrix)

# Вывод метрик качества классификации
print("\nРезультаты классификации:")
report = classification_report(y_test, y_pred)
print(report)


# тестовые SQL-запросы для проверки
test_queries = [
    "SELECT username FROM users WHERE id = 1",  # безопасный
    "SELECT * FROM products ORDER BY price DESC",  # безопасный
    "admin' --",  # SQL-инъекция 
    "SELECT * FROM users WHERE username = 'admin' OR '1'='1'",  # SQL-инъекция
    "DROP TABLE users",  # SQL-инъекция 
    "INSERT INTO orders (id, product) VALUES (10, 'Book')",  # безопасный
    "UNION SELECT null, username, password FROM users",  # SQL-инъекция 
    "SELECT COUNT(*) FROM transactions WHERE amount > 100",  # безопасный
    "' OR 1=1 --",  # SQL-инъекция 
    "UPDATE users SET password='newpassword' WHERE username='admin'",  # безопасный
    "SELECT * FROM customers WHERE email LIKE '%gmail.com'",  # безопасный
    "EXEC xp_cmdshell('dir')",  # SQL-инъекция 
    "SELECT * FROM accounts WHERE user_id IN (SELECT id FROM admins)",  # безопасный
    "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",  # безопасный
    "admin' UNION SELECT password FROM users --" # SQL-инъекция 
]
# векторизация тестовых запросов
test_queries_vect = vectorizer.transform(test_queries)

# предсказание на тестовых запросах
print("\nРезультаты тестирования на примерах:")
predictions = model.predict(test_queries_vect)
probabilities = model.predict_proba(test_queries_vect)[:, 1] 

# результаты
for query, pred, prob in zip(test_queries, predictions, probabilities):
    print(f"Запрос: {query}")
    print(f"Предсказание: {'SQL-инъекция' if pred == 1 else 'Безопасный запрос'}")
    print(f"Вероятность SQL-инъекции: {prob:.4f}")
    print("---")

# определение наиболее важных признаков для модели
feature_importances = np.argsort(model.feature_importances_)[::-1] 
feature_names = vectorizer.get_feature_names_out()

# выводим 10 наиважнейних
print("\n10 признаков с наибольшим влиянием:")
for i in range(10):
    if i < len(feature_importances):
        idx = feature_importances[i]
        if idx < len(feature_names):
            print(f"{i+1}. {feature_names[idx]} (важность: {model.feature_importances_[idx]:.4f})")

# сохранение модели и векторизатора
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
print("Модель и векторизатор были сохранены")

print("\nОбучение модели успешно завершено! ")
