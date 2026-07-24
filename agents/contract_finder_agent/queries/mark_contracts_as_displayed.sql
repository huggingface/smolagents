UPDATE {context_table} AS context
SET 
    is_displayed = TRUE,
    display_count = COALESCE(context.display_count, 0) + COALESCE(doc_counts.doc_count, 1)
FROM (
    VALUES
        {values_clause}
) AS doc_counts(doc_id, doc_count)
WHERE context.doc_id = doc_counts.doc_id
  AND context.session_id = %s
  AND context.question = %s;