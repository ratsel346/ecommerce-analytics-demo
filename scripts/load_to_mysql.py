import os      
from pathlib import Path   
from urllib.parse import quote_plus     
import pandas as pd
from sqlalchemy import (       
    create_engine,
    text,
    VARCHAR,
    DATE,
    FLOAT,
    INTEGER,
    SMALLINT,
    DECIMAL,
)


# Путь к папке + проверка на существование очищенных файлов
CLEAN_DIR = Path("data/clean")

customers_path = CLEAN_DIR / "customers.csv"
orders_path = CLEAN_DIR / "orders.csv"

if not customers_path.exists():
    raise FileNotFoundError(    
        "Не найден файл data/clean/customers.csv. "
        "Сначала запусти скрипт clean_analyze.py"
    )

if not orders_path.exists():
    raise FileNotFoundError(
        "Не найден файл data/clean/orders.csv. "
        "Сначала запусти скрипт clean_analyze.py"
    )



# 1. ЧИТАЕМ ОЧИЩЕННЫЕ CSV
print("Читаем очищенные CSV...")

customers = pd.read_csv(
    customers_path,
    parse_dates=["registration_date"]  
)

orders = pd.read_csv(
    orders_path,
    parse_dates=["order_date"]
)

# ID ключи = текст/строка
customers["customer_id"] = customers["customer_id"].astype(str)
orders["order_id"] = orders["order_id"].astype(str)
orders["customer_id"] = orders["customer_id"].astype(str)

print("customers shape:", customers.shape)
print("orders shape:", orders.shape)



# 2. ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ К MYSQL
db_user = os.getenv("MYSQL_USER", "root")   

db_password = os.getenv("MYSQL_PASSWORD", "MYSQL2026")

db_host = os.getenv("MYSQL_HOST", "127.0.0.1")

db_port = os.getenv("MYSQL_PORT", "3307")

db_name = os.getenv("MYSQL_DB", "demo_project")

# Кодируем логин и пароль на случай спецсимволов
safe_user = quote_plus(db_user)
safe_password = quote_plus(db_password)

print(f"\nПодключаемся к MySQL: {db_host}:{db_port}")
print(f"Пользователь: {db_user}")
print(f"База данных: {db_name}")



# 3. СОЗДАЕМ БАЗУ ДАННЫХ
# Подключаемся к серверу без указания базы, чтобы выполнить CREATE DATABASE.
server_engine = create_engine(
    f"mysql+pymysql://{safe_user}:{safe_password}@{db_host}:{db_port}"
)

with server_engine.connect() as conn:
    conn.execute(text(
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
        f"CHARACTER SET utf8mb4 "
        f"COLLATE utf8mb4_unicode_ci"
    ))
    conn.commit()

print("База данных готова.")




# 4. ПОДКЛЮЧАЕМСЯ К БАЗЕ И ЗАГРУЖАЕМ ТАБЛИЦЫ
# строим новый мост с конкретной базой данных /{db_name} в конце адреса, которую выше создали
engine = create_engine(
    f"mysql+pymysql://{safe_user}:{safe_password}@{db_host}:{db_port}/{db_name}"
    f"?charset=utf8mb4"
)

customers.to_sql(       
    "customers",       
    engine,              
    if_exists="replace",    
    index=False,        
    dtype={         
        "customer_id": VARCHAR(20),
        "registration_date": DATE,
        "country": VARCHAR(50),
        "city": VARCHAR(100),
        "age": INTEGER,
        "signup_channel": VARCHAR(30),
        "marketing_consent": SMALLINT,
        "age_missing": SMALLINT,
    }
)
print("Таблица customers загружена.")

orders.to_sql(
    "orders",
    engine,
    if_exists="replace",
    index=False,
    dtype={
        "order_id": VARCHAR(20),
        "customer_id": VARCHAR(20),
        "order_date": DATE,
        "status": VARCHAR(20),
        "category": VARCHAR(30),
        "payment_method": VARCHAR(20),
        "quantity": INTEGER,
        "discount_percent": INTEGER,
        "amount_raw": DECIMAL(12, 2),
        "delivery_days": FLOAT,
        "rating": FLOAT,
        "amount_clean": DECIMAL(12, 2),
        "is_amount_outlier": SMALLINT,
        "is_successful": SMALLINT,
    }
)
print("Таблица orders загружена.")



# 5. СОЗДАЕМ ПЕРВИЧНЫЕ КЛЮЧИ И ИНДЕКСЫ
with engine.begin() as conn:
    conn.execute(text(
        "ALTER TABLE customers " 
        "ADD PRIMARY KEY (customer_id)"
    ))

    conn.execute(text(
        "ALTER TABLE orders "
        "ADD PRIMARY KEY (order_id)"
    ))

    conn.execute(text(
        "CREATE INDEX idx_orders_customer_id "
        "ON orders (customer_id)"
    ))

    conn.execute(text(
        "CREATE INDEX idx_orders_order_date "
        "ON orders (order_date)"
    ))
# PRIMARY KEY и индексы для ускорения JOIN и фильтрации по датам
print("Первичные ключи и индексы созданы.")



# 6. ПРОВЕРЯЕМ ЗАГРУЗКУ
check_queries = {
    "customers_count": "SELECT COUNT(*) FROM customers",
    "orders_count": "SELECT COUNT(*) FROM orders",
    "successful_orders_count": "SELECT SUM(is_successful) FROM orders",
    "min_order_date": "SELECT MIN(order_date) FROM orders",
    "max_order_date": "SELECT MAX(order_date) FROM orders",
}

print("\nПроверка загруженных данных:")

with engine.connect() as conn:
    for name, query in check_queries.items():
        result = conn.execute(text(query)).scalar()
        print(f"{name}: {result}")
print("\nЗагрузка в MySQL завершена.")
