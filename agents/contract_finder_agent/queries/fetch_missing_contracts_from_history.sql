SELECT doc_id, doc_context
FROM {context_table}
WHERE session_id = %s
AND doc_id = ANY(%s);