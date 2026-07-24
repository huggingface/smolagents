CREATE TABLE IF NOT EXISTS {context_table} (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    doc_id TEXT,
    question TEXT,
    query_embeddings VECTOR({embedding_dim}),
    doc_context JSONB,
    is_displayed BOOLEAN DEFAULT FALSE,
    is_expired BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT NOW(),
    context_score FLOAT DEFAULT 0.0,
    display_count INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_session_question ON {context_table}(session_id, question);

CREATE INDEX IF NOT EXISTS idx_session_displayed ON {context_table}(session_id, is_displayed, is_expired);

CREATE INDEX IF NOT EXISTS idx_context_score ON {context_table}(context_score DESC);