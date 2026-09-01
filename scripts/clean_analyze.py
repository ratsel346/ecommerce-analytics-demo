from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, ttest_ind




# ПОДГОТОВИТЕЛЬНЫЙ ЭТАП ДЛЯ ОСНОВНОЙ РАБОТЫ
# Пути к папкам внутри моего проекта
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
REPORT_DIR = Path("reports")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)    
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Опорная дата для отсечения даты из будущего при очистке
REFERENCE_DATE = pd.Timestamp("2026-08-14")

ANALYSIS_END_DATE = pd.Timestamp("2026-07-31")

ALPHA = 0.05





# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
def print_overview(df: pd.DataFrame, name: str, key_column: str) -> None:
    """
    Печатает краткий обзор таблицы:
    - размер
    - типы данных
    - пропуски
    - полные дубликаты
    - дубликаты по ключу
    """

    print("\n" + "=" * 70)
    print(f"Обзор таблицы: {name}")
    print("=" * 70)

    print("\nРазмер таблицы:")
    print(df.shape) 

    print("\nТипы данных:")
    print(df.dtypes)

    print("\nДоля пропусков, %:")
    missing_percent = df.isna().mean() * 100
    print(missing_percent.round(2).sort_values(ascending=False))

    print("\nПолные дубликаты строк:")
    print(df.duplicated().sum())

    print(f"\nДубликаты по ключу {key_column}:")
    print(df.duplicated(subset=[key_column]).sum())






# ОЧИСТКА ТАБЛИЦЫ КЛИЕНТОВ
def clean_customers(raw_customers: pd.DataFrame):
    """
    Очищает таблицу клиентов
    """

    df = raw_customers.copy()

    df.columns = df.columns.str.lower().str.strip()


    # Очистика колонки "customer_id"
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    
    df.loc[
        df["customer_id"].isin(["", "nan", "None", "NULL", "null"]),
        "customer_id"
    ] = np.nan
    
    df = df.dropna(subset=["customer_id"])

    n_before_duplicates = len(df)
    df = df.drop_duplicates(subset=["customer_id"], keep="first") # keep="first" - оставляет первую запись из таблицы
    removed_duplicates = n_before_duplicates - len(df)


    # Очистка даты регистрации
    df["registration_date"] = pd.to_datetime(
        df["registration_date"],
        errors="coerce"
    )
    
    df.loc[
        df["registration_date"] > REFERENCE_DATE,
        "registration_date"
    ] = pd.NaT

    n_before_bad_date = len(df)
    df = df.dropna(subset=["registration_date"])
    removed_bad_dates = n_before_bad_date - len(df)


    # Очистка возраста
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    invalid_age_mask = (
        (df["age"] < 14) |
        (df["age"] > 100)
    )

    invalid_age_count = int(invalid_age_mask.sum()) 
    
    df.loc[invalid_age_mask, "age"] = np.nan

    # Флаг для дальнейшего заполнения их медианным возрастом
    df["age_missing"] = df["age"].isna().astype(int)

    # Заполняем пропуски возраста медианой
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age).astype(int)
    

    # Унифицируем канал привлечения
    df["signup_channel"] = (
        df["signup_channel"]
        .fillna("unknown")   
        .astype(str)       
        .str.lower()        
        .str.strip()        
    )

    df.loc[
        df["signup_channel"].isin(["", "nan", "none", "unknown"]),
        "signup_channel"
    ] = "unknown"


    # marketing_consent: пропуск считаем отсутствием согласия
    df["marketing_consent"] = (
        pd.to_numeric(df["marketing_consent"], errors="coerce")     # значения превращаем в числа, если нет, то пропуск
        .fillna(0)      
        .astype(int)  
    )

    # Унифицируем country и city
    for col in ["country", "city"]:
        df[col] = (
            df[col]
            .fillna("unknown")      
            .astype(str)       
            .str.strip()       
        )

        df.loc[
            df[col].isin(["", "nan", "None"]),
            col
        ] = "unknown"       



    data_quality = {
        "table": "customers",
        "start_rows": n_before_duplicates,
        "final_rows": len(df),
        "removed_duplicates": removed_duplicates,
        "removed_bad_registration_dates": removed_bad_dates,
        "invalid_age_values": invalid_age_count,
        "missing_age_filled": int(df["age_missing"].sum()),
    }

    return df, data_quality






# ОЧИСТКА ТАБЛИЦЫ ЗАКАЗОВ
def clean_orders(raw_orders: pd.DataFrame, valid_customer_ids: pd.Series):
    """
    Очищает таблицу заказов
    """

    df = raw_orders.copy()

    df.columns = df.columns.str.lower().str.strip()

    # Очистка "order_id"
    df["order_id"] = df["order_id"].astype(str).str.strip()

    # Очистка "customer_id"
    df["customer_id"] = df["customer_id"].astype(str).str.strip()

    df.loc[
        df["customer_id"].isin(["", "nan", "None", "NULL", "null"]),
        "customer_id"
    ] = np.nan 

    # Удаляем дубликаты по "order_id"
    n_before_duplicates = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")      # удаляем дубликаты, оставляя только первых записи по колонке order_id
    removed_duplicates = n_before_duplicates - len(df)


    # Унификация даты заказа
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )
    
    # Будущие даты = ошибка
    df.loc[
        df["order_date"] > REFERENCE_DATE,
        "order_date"
    ] = pd.NaT

   
    # Унифицируем статус
    df["status"] = (
        df["status"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["status"] = df["status"].replace({
        "canceled": "cancelled"
    })

    valid_statuses = [
        "completed",
        "delivered",
        "cancelled",
        "processing"
    ]

    df.loc[
        ~df["status"].isin(valid_statuses),
        "status"
    ] = "unknown"
    

    # Приводим числовые колонки к числовому типу
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["discount_percent"] = pd.to_numeric(df["discount_percent"], errors="coerce")
    df["delivery_days"] = pd.to_numeric(df["delivery_days"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")


    # Чистим "amount"
    # Техническими ошибками считаем:
    # - пропуски
    # - отрицательные суммы
    # - нулевые суммы
    # - суммы больше 100000

    bad_amount_mask = (
        df["amount"].isna() |
        (df["amount"] <= 0) |
        (df["amount"] > 100000)
    )

    removed_bad_amount = int(bad_amount_mask.sum())

    df = df[~bad_amount_mask]

    # Сохраняем очищенную сумму как amount_raw
    # В проекте 2 вида amount: amount_raw (для выручки) и amount_clean (для устойчивого среднего чека)
    df["amount_raw"] = df["amount"]

    n_before_keys = len(df)
    df = df.dropna(subset=["order_date", "customer_id"])
    removed_missing_keys = n_before_keys - len(df)

    # Оставляем только заказы существующих клиентов
    n_before_orphans = len(df)
    df = df[df["customer_id"].isin(valid_customer_ids)]
    removed_orphan_orders = n_before_orphans - len(df)

    # Оставляем только период анализа
    n_before_period = len(df)
    df = df[df["order_date"] <= ANALYSIS_END_DATE]
    removed_out_of_period = n_before_period - len(df)

    # Чистим "quantity"
    df.loc[
        (df["quantity"].isna()) | (df["quantity"] < 1),
        "quantity"
    ] = 1

    df["quantity"] = df["quantity"].astype(int)

    # Чистим "discount_percent"
    # Пропуск считаем отсутствием скидки, то есть 0
    df.loc[
        (df["discount_percent"].isna()) |      
        (df["discount_percent"] < 0) |          
        (df["discount_percent"] > 100),         
        "discount_percent"
    ] = 0

    df["discount_percent"] = df["discount_percent"].astype(int)    

    # Чистим "delivery_days"
    # Нормальный срок доставки - от 1 до 60 дней
    df.loc[
        (df["delivery_days"].isna()) |
        (df["delivery_days"] < 1) |
        (df["delivery_days"] > 60),
        "delivery_days"
    ] = np.nan

    # Чистим "rating"
    # Рейтинг - от 1 до 5
    df.loc[
        (df["rating"].isna()) |
        (df["rating"] < 1) |
        (df["rating"] > 5),
        "rating"
    ] = np.nan

    # ДЛЯ СТАТИСТИКИ
    # Ищем статистические ВЫБРОСЫ по сумме заказа через IQR
    q1 = df["amount_raw"].quantile(0.25)
    q3 = df["amount_raw"].quantile(0.75)
    iqr = q3 - q1

    upper_fence = q3 + 1.5 * iqr

    # Флаг выброса
    df["is_amount_outlier"] = (
        df["amount_raw"] > upper_fence
    ).astype(int)

    # Создаем amount_clean
    # Выбросы не удаляем полностью, а ограничиваем сверху
    df["amount_clean"] = (
        df["amount_raw"]
        .clip(upper=upper_fence)
        .round(2)
    )


    
    # Успешные заказы
    df["is_successful"] = df["status"].isin([
        "completed",
        "delivered"
    ]).astype(int)


    data_quality = {
        "table": "orders",
        "start_rows": n_before_duplicates,      # кол-во строк до
        "final_rows": len(df),      # кол-во строк после
        "removed_duplicates": removed_duplicates,   # удаленные дубликаты
        "removed_bad_amount": removed_bad_amount,   # кол-во плохих сумм
        "removed_missing_keys": removed_missing_keys,   # удалено без ключей
        "removed_orphan_orders": removed_orphan_orders,     # удалено заказов без клиента
        "removed_out_of_period": removed_out_of_period,     # кол-во заказов вне периода 
        "amount_outliers_flagged": int(df["is_amount_outlier"].sum()),      # кол-во выбросов по сумме
    }

    return df, data_quality





# ПРОВЕРКА ГИПОТЕЗ
def run_hypothesis_tests(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Запускает статистические тесты по двум гипотезам.
    """

    results = []

    
    # ГИПОТЕЗА 1
    # Средний чек успешных заказов клиентов из каналов ads и organic различается
    # Сравниваем amount_clean для ads и organic
    h1_data = merged[
        (merged["is_successful"] == 1) &
        (merged["signup_channel"].isin(["ads", "organic"]))
    ].copy()

    ads = h1_data.loc[
        h1_data["signup_channel"] == "ads",     
        "amount_clean"    
    ]

    organic = h1_data.loc[
        h1_data["signup_channel"] == "organic",
        "amount_clean"
    ]
    
    print("\n" + "=" * 70)
    print("Гипотеза 1: средний чек ads vs organic")
    print("=" * 70)

    print("\nОписательная статистика ads:")
    print(ads.describe())


    print("\nОписательная статистика organic:")
    print(organic.describe())

    if len(ads) > 10 and len(organic) > 10:     

        # Mann-Whitney U test
        u_stat, u_p_value = mannwhitneyu(
            ads,
            organic,
            alternative="two-sided"    
        )


        # Дополнительно проверяем через Welch t-test
        # на логарифмированных суммах, что сделает распределение чуть спокойнее 
        t_stat, t_p_value = ttest_ind(
            np.log(ads + 1),        
            np.log(organic + 1),
            equal_var=False         # предполагаем, что дисперсии в группах не одинаковые. Это более безопасный вариант
        )

        median_diff = ads.median() - organic.median()
        mean_diff = ads.mean() - organic.mean()

        results.append({
            "hypothesis": "amount_clean ads vs organic",
            "test": "Mann-Whitney U",
            "statistic": u_stat,
            "p_value": u_p_value,
            "median_ads": ads.median(),
            "median_organic": organic.median(),
            "median_difference": median_diff,
            "mean_ads": ads.mean(),
            "mean_organic": organic.mean(),
            "mean_difference": mean_diff,
            "significant": u_p_value < ALPHA,
        })

        results.append({
            "hypothesis": "log(amount_clean+1) ads vs organic",
            "test": "Welch t-test",
            "statistic": t_stat,
            "p_value": t_p_value,
            "median_ads": ads.median(),
            "median_organic": organic.median(),
            "median_difference": median_diff,
            "mean_ads": ads.mean(),
            "mean_organic": organic.mean(),
            "mean_difference": mean_diff,
            "significant": t_p_value < ALPHA,
        })

        print("\nMann-Whitney U p-value:")
        print(u_p_value)

        print("\nWelch t-test p-value:")
        print(t_p_value)

    else:
        print("Недостаточно данных для теста по гипотезе 1.")




    # ГИПТОЗА 2
    # Заказы с доставкой дольше 7 дней имеют более низкий рейтинг, чем заказы с доставкой до 7 дней включительно
    # Сравниваем рейтинг для доставки <= 7 дней и > 7 дней
    h2_data = merged[
        (merged["is_successful"] == 1) &        
        (merged["delivery_days"].notna()) &     
        (merged["rating"].notna())
    ].copy()

    # Делим заказы на 2 группы: с доставкой больше 7 дней и меньше 7 дней
    late = h2_data.loc[
        h2_data["delivery_days"] > 7,
        "rating"
    ]

    on_time = h2_data.loc[
        h2_data["delivery_days"] <= 7,
        "rating"
    ]

    print("\n" + "=" * 70)
    print("Гипотеза 2: рейтинг при долгой доставке")
    print("=" * 70)

    print("\nОписательная статистика рейтинга для доставки > 7 дней:")
    print(late.describe())

    print("\nОписательная статистика рейтинга для доставки <= 7 дней:")
    print(on_time.describe())

    if len(late) > 10 and len(on_time) > 10:

        # Mann-Whitney U test
        u_stat, u_p_value = mannwhitneyu(
            late,
            on_time,
            alternative="two-sided"
        )

        median_diff = late.median() - on_time.median()
        mean_diff = late.mean() - on_time.mean()

        results.append({
            "hypothesis": "rating late delivery vs on-time delivery",
            "test": "Mann-Whitney U",
            "statistic": u_stat,
            "p_value": u_p_value,
            "median_late": late.median(),
            "median_on_time": on_time.median(),
            "median_difference": median_diff,
            "mean_late": late.mean(),
            "mean_on_time": on_time.mean(),
            "mean_difference": mean_diff,
            "significant": u_p_value < ALPHA,
        })

        print("\nMann-Whitney U p-value:")
        print(u_p_value)

    else:
        print("Недостаточно данных для теста по гипотезе 2.")

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df.to_csv(REPORT_DIR / "hypotheses.csv", index=False)       

    return results_df
# True - есть статистически значимое различие в рейтингах между сравниваемыми группами








# ОСНОВНОЙ СЦЕНАРИЙ
def main():

    
    # 1. Загружаем сырые данные
    raw_customers = pd.read_csv(RAW_DIR / "customers.csv")
    raw_orders = pd.read_csv(RAW_DIR / "orders.csv")

    print("Сырые данные загружены.")

    print_overview(raw_customers, "raw_customers", "customer_id")
    print_overview(raw_orders, "raw_orders", "order_id")

    
    # 2. Чистим клиентов
    customers, customers_quality = clean_customers(raw_customers)

    print("\nОчищенная таблица клиентов:")
    print(customers.head())
    print(customers_quality)

   
    # 3. Чистим заказы
    orders, orders_quality = clean_orders(
        raw_orders,
        valid_customer_ids=customers["customer_id"]
    )

    print("\nОчищенная таблица заказов:")
    print(orders.head())
    print(orders_quality)

   
    # 4. Сохраняем отчет о качестве данных
    quality_report = pd.DataFrame([
        customers_quality,
        orders_quality
    ])

    quality_report.to_csv(REPORT_DIR / "data_quality.csv", index=False)

    print("\nОтчет о качестве данных сохранен:")
    print(REPORT_DIR / "data_quality.csv")


    # 5. Сохраняем очищенные таблицы
    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

    print("\nОчищенные CSV сохранены:")
    print(CLEAN_DIR / "customers.csv")
    print(CLEAN_DIR / "orders.csv")

    
    # 6. Делаем объединенную таблицу для анализа
    merged = orders.merge(
        customers,
        on="customer_id",
        how="left",
        suffixes=("", "_customer")
    )

    merged.to_csv(CLEAN_DIR / "merged_analysis.csv", index=False)

    print("\nОбъединенная таблица сохранена:")
    print(CLEAN_DIR / "merged_analysis.csv")

    
    # 7. Считаем базовые бизнес-метрики
    successful_orders = merged[merged["is_successful"] == 1]

    print("\nБазовые метрики по успешным заказам:")

    base_metrics = {
        "successful_orders_count": successful_orders.shape[0],
        "customers_with_successful_orders": successful_orders["customer_id"].nunique(),
        "revenue_raw": successful_orders["amount_raw"].sum(),
        "revenue_clean": successful_orders["amount_clean"].sum(),
        "avg_check_raw": successful_orders["amount_raw"].mean(),
        "avg_check_clean": successful_orders["amount_clean"].mean(),
        "median_check_clean": successful_orders["amount_clean"].median(),
        "avg_rating": successful_orders["rating"].mean(),
        "avg_delivery_days": successful_orders["delivery_days"].mean(),
    }

    base_metrics_df = pd.DataFrame([base_metrics])
    base_metrics_df.to_csv(REPORT_DIR / "base_metrics.csv", index=False)

    print(base_metrics_df)

   
    # 8. Групповые метрики
    print("\nВыручка по каналам привлечения:")
    channel_revenue = (
        successful_orders
        .groupby("signup_channel")
        .agg(
            orders_count=("order_id", "count"),
            customers_count=("customer_id", "nunique"),
            revenue_raw=("amount_raw", "sum"),
            revenue_clean=("amount_clean", "sum"),
            avg_check_clean=("amount_clean", "mean"),
            avg_rating=("rating", "mean"),
        )
        .round(2)
        .sort_values("revenue_clean", ascending=False)
    )

    print(channel_revenue)

    print("\nВыручка по категориям:")
    category_revenue = (
        successful_orders
        .groupby("category")
        .agg(
            orders_count=("order_id", "count"),
            revenue_raw=("amount_raw", "sum"),
            revenue_clean=("amount_clean", "sum"),
            avg_check_clean=("amount_clean", "mean"),
        )
        .round(2)
        .sort_values("revenue_clean", ascending=False)
    )

    print(category_revenue)

    print("\nРейтинг по срокам доставки:")
    delivery_quality = (
        successful_orders[
            successful_orders["delivery_days"].notna() &
            successful_orders["rating"].notna()
        ]
        .assign(
            delivery_bucket=lambda x: pd.cut(
                x["delivery_days"],
                bins=[0, 3, 7, 14, 60],
                labels=["1-3", "4-7", "8-14", "15+"]
            )
        )
        .groupby("delivery_bucket", observed=True)
        .agg(
            orders_count=("order_id", "count"),
            avg_rating=("rating", "mean"),
            bad_rating_rate=("rating", lambda s: (s <= 3).mean()),
        )
        .round(3)
    )

    print(delivery_quality)

   
    # 9. Запускаем статистические тесты
    hypothesis_results = run_hypothesis_tests(merged)

    if not hypothesis_results.empty:
        print("\nРезультаты гипотез:")
        print(hypothesis_results)


# Точка входа в скрипт
if __name__ == "__main__":
    main()