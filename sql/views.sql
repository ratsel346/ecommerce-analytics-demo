-- SQL-скрипт для демо-проекта
-- Используется MySQL 

USE demo_analytics;


-- 1. JOINED ЗАКАЗЫ (заказы + клиенты)
CREATE OR REPLACE VIEW v_orders_joined AS
SELECT 
		o.order_id, o.customer_id, o.order_date, o.status, o.category, o.payment_method, o.quantity, o.discount_percent, o.delivery_days, o.rating, o.amount_raw, o.amount_clean, o.is_amount_outlier, o.is_successful,
		c.registration_date, c.country, c.city, c.age, c.signup_channel, c.marketing_consent, c.age_missing,
		CASE 
			WHEN c.customer_id IS NULL THEN 1
			ELSE 0
		END AS is_orphan_order

FROM orders AS o
LEFT JOIN customers AS c
ON o.customer_id = c.customer_id 





-- 2. ПРОДАЖИ ПО ДНЯМ
CREATE OR REPLACE VIEW v_daily_sales AS
SELECT order_date,
		COUNT(DISTINCT(order_id))       AS order_count,
        COUNT(DISTINCT(customer_id))    AS customer_count,
        SUM(amount_raw)                 AS revenue_raw,
        SUM(amount_clean)               AS revenue_clean,
        ROUND(AVG(amount_clean),2)      AS avg_check_clean
FROM v_orders_joined
WHERE is_successful = 1
GROUP BY order_date
ORDER BY order_date ASC






-- 3. ПРОДАЖИ ПО МЕСЯЦАМ С:
-- - агрегацией по месяцам
-- - running total
-- - скользящим средним за 3 месяца
-- - выручкой за прошлый месяц
-- - month-over-month growth
CREATE OR REPLACE VIEW v_monthly_sales_running AS
WITH mounthly AS (
		SELECT 
			CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE) 		AS month,
            SUM(amount_raw) 										AS revenue_raw,
            SUM(amount_clean) 										AS revenue_clean,
            COUNT(DISTINCT order_id) 								AS orders_count,
            COUNT(DISTINCT customer_id) 							AS customers_count,
            ROUND(AVG(amount_clean), 2) 							AS avg_check_clean
		FROM v_orders_joined
        WHERE is_successful = 1
        GROUP BY CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE)
),
mounthly_with_lag AS (
		SELECT 
			month,
            revenue_raw,
            revenue_clean,
            orders_count,
            customers_count,
            avg_check_clean,
            LAG(revenue_clean) OVER(ORDER BY month)                 AS prev_revenue_clean
		FROM mounthly
)
SELECT
	month,
    revenue_raw,
	revenue_clean,
	orders_count,
	customers_count,
	avg_check_clean,
    SUM(revenue_clean) OVER(ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 			AS revenue_cleaning_running,
    ROUND(AVG(revenue_clean) OVER(ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) 			AS revenue_clean_MA3,
    prev_revenue_clean,
    ROUND((revenue_clean - prev_revenue_clean) / NULLIF(prev_revenue_clean, 0) * 100, 2) 				AS MoM_growth_procent
FROM mounthly_with_lag
ORDER BY month
    

            


-- 4. ЭФФЕКТИВНОСТЬ КАНАЛОВ ПРИВЛЕЧЕНИЯ
CREATE OR REPLACE VIEW v_channel_performance AS
SELECT signup_channel,
		COUNT(DISTINCT customer_id) 					AS customers_with_orders,
        COUNT(DISTINCT order_id) 						AS successful_orders, 
        SUM(amount_raw) 								AS revenue_raw,
        SUM(amount_clean) 								AS revenue_clean,
        ROUND(AVG(amount_clean),2) 						AS avg_check_clean,
        ROUND(SUM(amount_raw) / NULLIF(COUNT(DISTINCT customer_id), 0),2) 	AS revenue_per_customer,
        ROUND(AVG(delivery_days),2) 					AS avg_delivery_days,
        ROUND(AVG(rating),2)							AS avg_rating
FROM v_orders_joined
WHERE is_successful = 1
GROUP BY signup_channel
ORDER BY signup_channel





-- 5. ПРОДАЖИ ПО КАНАЛАМ И МЕСЯЦАМ
CREATE OR REPLACE VIEW v_channel_month AS
SELECT 
		CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE) 	AS month,
        signup_channel,
        COUNT(DISTINCT order_id) 							AS orders_count,
        COUNT(DISTINCT customer_id)						    AS customers_count,
        SUM(amount_raw)										AS revenue_raw,
        SUM(amount_clean)									AS revenue_clean
FROM v_orders_joined
WHERE is_successful = 1
GROUP BY
		CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE),
        signup_channel
ORDER BY month, signup_channel




-- 6. РЕЙТИНГ КАТЕГОРИЙ ПО МЕСЯЦАМ
-- - RANK() для места категории внутри месяца
-- - SUM() OVER для доли категории
CREATE OR REPLACE VIEW v_caterogy_month_rank AS
WITH month_rank AS (
		SELECT
				CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE) AS month,
                category,
                SUM(amount_clean)								  AS revenue_clean
		FROM v_orders_joined
        WHERE is_successful = 1
        GROUP BY CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE), 
					category
)
SELECT 
		month,
        category,
        RANK() OVER(PARTITION BY month ORDER BY revenue_clean DESC) 				AS category_rank,
        revenue_clean / NULLIF(SUM(revenue_clean) OVER(PARTITION BY month),0) 		AS revenue_share
FROM month_rank
ORDER BY month, category_rank




-- 7. КАЧЕСТВО ДОСТАВКИ
CREATE OR REPLACE VIEW v_delivery_quality AS
SELECT
    CASE
        WHEN delivery_days <= 3 THEN '1-3 days'
        WHEN delivery_days <= 7 THEN '4-7 days'
        WHEN delivery_days <= 14 THEN '8-14 days'
        ELSE '15+ days'
    END 										AS delivery_bucket,
    COUNT(*) 									AS orders_count,
	ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(SUM(CASE
            WHEN rating <= 3 THEN 1
            ELSE 0
        END) / NULLIF(COUNT(rating), 0), 2) 	AS bad_rating_rate,
    ROUND(AVG(amount_clean),2) AS avg_check_clean
FROM v_orders_joined
WHERE is_successful = 1
    AND delivery_days IS NOT NULL
    AND rating IS NOT NULL
GROUP BY delivery_bucket;





-- 8. ПРОСТОЙ RFM ПО КЛИЕНТАМ
-- R - Recency - как давно клиент покупал
-- F - Frequency - как часто покупал
-- M - Monetary - сколько денег принес

CREATE OR REPLACE VIEW v_customer_rfm AS
WITH max_date AS (
	SELECT
			MAX(order_date) 		        AS universal_max_date
	FROM v_orders_joined
),
aggreg AS (
	SELECT 
			customer_id,
            MAX(order_date) 				AS last_order_date,
            SUM(amount_raw) 				AS monetary_raw,
            COUNT(DISTINCT order_id) 		AS frequenly
	FROM v_orders_joined
    WHERE is_successful = 1
    GROUP BY customer_id
    ORDER BY customer_id
)
SELECT 
		customer_id,
        last_order_date,
        DATEDIFF(universal_max_date, last_order_date) 	    AS recent_day,
        frequenly,
        monetary_raw,
        NTILE(5) OVER(ORDER BY last_order_date DESC)        AS r_score,
        NTILE(5) OVER(ORDER BY frequenly DESC)              AS f_score,
        NTILE(5) OVER(ORDER BY  monetary_raw DESC)          AS m_score
FROM aggreg, max_date
ORDER BY customer_id
            