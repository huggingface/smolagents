
WITH latest_question AS (
    SELECT question
    FROM {context_table}
    WHERE session_id = %s
      AND is_expired = FALSE
    ORDER BY id DESC
    LIMIT 1
),

undisplayed AS (
    SELECT rdd.*
    FROM {context_table} rdd
    JOIN latest_question lq ON rdd.question = lq.question
    WHERE rdd.session_id = %s
      AND rdd.is_displayed = FALSE
      AND rdd.is_expired = FALSE
    ORDER BY rdd.context_score DESC
    LIMIT %s
),

displayed AS (
    SELECT rdd.*, TRUE AS already_displayed
    FROM {context_table} rdd
    JOIN latest_question lq ON rdd.question = lq.question
    WHERE rdd.session_id = %s
      AND rdd.is_displayed = TRUE
      AND rdd.is_expired = FALSE
),

total AS (
    SELECT COUNT(*) AS total_count
    FROM {context_table} rdd
    JOIN latest_question lq ON rdd.question = lq.question
    WHERE rdd.session_id = %s
      AND rdd.is_expired = FALSE
)

SELECT combined.*, total.total_count
FROM (
        SELECT *
        FROM displayed
        UNION ALL
        SELECT u.*, FALSE AS already_displayed
        FROM undisplayed u
    ) combined
    CROSS JOIN total
ORDER BY context_score DESC
LIMIT % s;