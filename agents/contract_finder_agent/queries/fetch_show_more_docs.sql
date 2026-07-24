
WITH undisplayed AS (
    SELECT *
    FROM {context_table}
    WHERE session_id = %s
    AND question = %s
    AND is_displayed = FALSE
    AND is_expired = FALSE
    ORDER BY context_score DESC
    LIMIT %s
)
SELECT combined.*, total.total_count
FROM (
    SELECT *, TRUE AS already_displayed
    FROM {context_table}
    WHERE session_id = %s
    AND question = %s
    AND is_displayed = TRUE
    AND is_expired = FALSE

    UNION ALL

    SELECT *, FALSE AS already_displayed
    FROM undisplayed
) combined
CROSS JOIN (
    SELECT COUNT(*) AS total_count
    FROM {context_table}
    WHERE session_id = %s
    AND question = %s
    AND is_expired = FALSE
) total
ORDER BY context_score DESC
LIMIT %s;