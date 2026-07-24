INSERT INTO {context_table} (session_id, doc_id, question, query_embeddings, doc_context, is_displayed, is_expired, created_date, context_score)
VALUES {values_clause}
RETURNING id, doc_id;