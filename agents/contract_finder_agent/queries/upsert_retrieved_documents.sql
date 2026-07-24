UPDATE {context_table}
SET is_expired = TRUE
WHERE session_id = %s
AND question = %s
AND is_expired = FALSE;