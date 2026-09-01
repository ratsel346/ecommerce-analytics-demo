# Импортируем библиотеки.
# numpy нужен для генерации случайных чисел.
# pandas нужен для таблиц и сохранения CSV.
# Path нужен для работы с папками и файлами.
import numpy as np
import pandas as pd
from pathlib import Path


# Фиксируем random seed.
# Это значит, что каждый раз данные будут генерироваться одинаково.
SEED = 42

# Размер данных.
N_CUSTOMERS = 2000
N_ORDERS = 10000

# Папка для сырых файлов.
RAW_DIR = Path("data/raw")

# Создаем папку, если ее нет.
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Создаем генератор случайных чисел.
rng = np.random.default_rng(SEED)


# ============================================================
# 1) ГЕНЕРАЦИЯ КЛИЕНТОВ
# ============================================================

# Создаем идентификаторы клиентов вида C00001, C00002, ...
customer_ids = np.array([f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)])

# Период регистрации клиентов.
start_reg = pd.Timestamp("2024-01-01")
end_reg = pd.Timestamp("2026-06-30")

# Количество дней в периоде.
days_reg = (end_reg - start_reg).days

# Генерируем случайные даты регистрации.
registration_dates = start_reg + pd.to_timedelta(
    rng.integers(0, days_reg + 1, N_CUSTOMERS),
    unit="D"
)

# Справочник стран и городов.
country_city = {
    "Russia": [
        "Moscow",
        "Saint Petersburg",
        "Kazan",
        "Novosibirsk"
    ],
    "Belarus": [
        "Minsk",
        "Gomel"
    ],
    "Kazakhstan": [
        "Almaty",
        "Astana"
    ],
    "Armenia": [
        "Yerevan"
    ],
    "Georgia": [
        "Tbilisi"
    ],
}

# Список стран.
countries = np.array(list(country_city.keys()))

# Генерируем страны с разными вероятностями.
country = rng.choice(
    countries,
    N_CUSTOMERS,
    p=[0.60, 0.10, 0.15, 0.075, 0.075]
)

# Для каждой страны выбираем город.
city = np.array([
    rng.choice(country_city[str(c)])
    for c in country
])

# Каналы привлечения клиентов.
signup_channel = rng.choice(
    ["organic", "ads", "referral", "email"],
    N_CUSTOMERS,
    p=[0.35, 0.35, 0.20, 0.10]
).astype(object)

# Добавляем немного пропусков в канале привлечения.
signup_channel[rng.random(N_CUSTOMERS) < 0.03] = None

# Генерируем возраст.
age = rng.integers(18, 71, N_CUSTOMERS).astype(float)

# Добавляем пропуски возраста.
age[rng.random(N_CUSTOMERS) < 0.04] = np.nan

# Согласие на маркетинг.
# 0 — нет согласия,
# 1 — есть согласие.
marketing_consent = rng.choice(
    [0, 1],
    N_CUSTOMERS,
    p=[0.25, 0.75]
).astype(float)

# Немного пропусков в marketing_consent.
marketing_consent[rng.random(N_CUSTOMERS) < 0.02] = np.nan

# Собираем таблицу клиентов.
customers = pd.DataFrame({
    "customer_id": customer_ids,
    "registration_date": registration_dates,
    "country": country,
    "city": city,
    "age": age,
    "signup_channel": signup_channel,
    "marketing_consent": marketing_consent,
})


# ============================================================
# 2) ГЕНЕРАЦИЯ ЗАКАЗОВ
# ============================================================

# Идентификаторы заказов.
order_ids = np.array([
    f"O{i:06d}"
    for i in range(100001, 100001 + N_ORDERS)
])

# Для каждого заказа случайно выбирааем клиента.
customer_idx = rng.integers(0, N_CUSTOMERS, N_ORDERS)

# customer_id для заказов.
order_customer_id = customer_ids[customer_idx].copy()

# Дата регистрации выбранного клиента.
order_registration_dates = registration_dates[customer_idx]

# Заказ может произойти в течение 365 дней после регистрации клиента.
offsets = pd.to_timedelta(
    rng.integers(0, 365, N_ORDERS),
    unit="D"
)

# Дата заказа = дата регистрации + случайное смещение.
order_dates = order_registration_dates + offsets

# Ограничиваем дату заказа конечной датой анализа.
end_orders = pd.Timestamp("2026-07-31")
order_dates = pd.Series(order_dates).clip(upper=end_orders)

# Статусы заказов.
status = rng.choice(
    ["completed", "delivered", "cancelled", "processing"],
    N_ORDERS,
    p=[0.45, 0.30, 0.15, 0.10]
).astype(object)

# Добавляем немного грязных статусов.
dirty_status_idx = rng.choice(N_ORDERS, size=150, replace=False)

status[dirty_status_idx[:50]] = "Completed"
status[dirty_status_idx[50:100]] = "CANCELED"
status[dirty_status_idx[100:]] = "Processing "

# Категории товаров.
category = rng.choice(
    ["electronics", "clothing", "home", "beauty", "toys"],
    N_ORDERS,
    p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

# Способы оплаты.
payment_method = rng.choice(
    ["card", "cash", "sbp", "wallet"],
    N_ORDERS,
    p=[0.60, 0.15, 0.15, 0.10]
)

# Количество товаров в заказе.
quantity = rng.integers(1, 6, N_ORDERS)

# Цена единицы товара.
# Логнормальное распределение хорошо подходит для денежных сумм.
unit_price = rng.lognormal(mean=6.2, sigma=0.8, size=N_ORDERS)

# Сумма заказа = цена * количество.
amount = np.round(unit_price * quantity, 2)

# Минимальная сумма заказа.
amount = np.maximum(amount, 150)

# Добавляем немного откровенно битых сумм.
bad_amount_idx = rng.choice(N_ORDERS, size=30, replace=False)

# Отрицательные суммы.
amount[bad_amount_idx[:10]] = -amount[bad_amount_idx[:10]]

# Нулевые суммы.
amount[bad_amount_idx[10:20]] = 0

# Аномально большие суммы.
amount[bad_amount_idx[20:]] = 999999.99

# Скидка в процентах.
discount_percent = rng.integers(0, 31, N_ORDERS).astype(float)

# Немного пропусков в скидке.
discount_percent[rng.random(N_ORDERS) < 0.03] = np.nan

# Срок доставки в днях.
delivery_days = rng.integers(1, 21, N_ORDERS).astype(float)

# Добавляем выбросы в доставке.
# 999 дней — явная аномалия.
outlier_delivery_idx = rng.choice(N_ORDERS, size=20, replace=False)
delivery_days[outlier_delivery_idx] = 999

# Приводим статусы к нижнему регистру для дальнейшей обработки.
status_lower = pd.Series(status).str.lower().str.strip()

# Для отмененных заказов доставка обычно неактуальна.
delivery_days[status_lower.isin(["cancelled", "canceled"])] = np.nan

# Рейтинг заказа.
rating = np.full(N_ORDERS, np.nan)

# Рейтинг оставляем только для успешных заказов, где есть доставка.
rate_mask = (
    status_lower.isin(["completed", "delivered"]).to_numpy()
    & ~np.isnan(delivery_days)
)

# Шум, чтобы связь не была идеальной.
noise = rng.normal(0, 0.6, N_ORDERS)

# Базовая формула рейтинга.
base_rating = 5 - 0.18 * delivery_days + noise

# Заполняем рейтинг.
rating[rate_mask] = np.clip(
    np.round(base_rating[rate_mask]),
    1,
    5
)

# Немного пропусков рейтинга.
rating[rate_mask & (rng.random(N_ORDERS) < 0.05)] = np.nan

# Превращаем customer_id в Series, чтобы удобно добавлять пропуски.
order_customer_id = pd.Series(order_customer_id)

# Примерно 1% заказов будут с пропущенным customer_id.
missing_customer_idx = rng.choice(
    N_ORDERS,
    size=int(N_ORDERS * 0.01),
    replace=False
)

order_customer_id.loc[missing_customer_idx] = None

# Примерно 0.5% заказов будут ссылаться на несуществующего клиента.
invalid_customer_idx = rng.choice(
    N_ORDERS,
    size=int(N_ORDERS * 0.005),
    replace=False
)

order_customer_id.loc[invalid_customer_idx] = "C99999"

# Собираем таблицу заказов.
orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": order_customer_id,
    "order_date": order_dates,
    "status": status,
    "category": category,
    "payment_method": payment_method,
    "quantity": quantity,
    "discount_percent": discount_percent,
    "amount": amount,
    "delivery_days": delivery_days,
    "rating": rating,
})


# ============================================================
# 3) ДОБАВЛЯЕМ ДУБЛИКАТЫ И ГРЯЗНЫЕ ДАТЫ
# ============================================================

# Дубликаты клиентов.
duplicate_customers = customers.sample(15, random_state=SEED)
customers = pd.concat(
    [customers, duplicate_customers],
    ignore_index=True
)

# Переводим дату регистрации в строку.
customers["registration_date"] = (
    customers["registration_date"].dt.strftime("%Y-%m-%d")
)

# Исправляем несколько дат на битые.
bad_reg_idx = rng.choice(customers.index, size=10, replace=False)

customers.loc[bad_reg_idx[:5], "registration_date"] = "2024-02-30"
customers.loc[bad_reg_idx[5:], "registration_date"] = ""

# Добавляем несколько значений Unknown в канал привлечения.
unknown_channel_idx = customers.sample(5, random_state=SEED).index
customers.loc[unknown_channel_idx, "signup_channel"] = "Unknown"

# Дубликаты заказов.
duplicate_orders = orders.sample(25, random_state=SEED)
orders = pd.concat(
    [orders, duplicate_orders],
    ignore_index=True
)

# Переводим дату заказа в строку.
orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")

# Исправляем несколько дат заказов на битые.
bad_order_date_idx = rng.choice(orders.index, size=15, replace=False)

orders.loc[bad_order_date_idx[:5], "order_date"] = "2026-13-01"
orders.loc[bad_order_date_idx[5:10], "order_date"] = "2025-02-30"
orders.loc[bad_order_date_idx[10:], "order_date"] = ""

# Сохраняем CSV.
customers.to_csv(RAW_DIR / "customers.csv", index=False)
orders.to_csv(RAW_DIR / "orders.csv", index=False)

print("Созданы файлы:")
print(RAW_DIR / "customers.csv")
print(RAW_DIR / "orders.csv")
print("customers shape:", customers.shape)
print("orders shape:", orders.shape)