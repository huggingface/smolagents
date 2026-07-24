WITH ranked AS (
    SELECT
        {select_column_str},
        {return_column},
        ({search_column} <=> '{embedding_str}'::vector) AS similarity_score
    FROM "{table_name}"
    {isbn_exists_clause}
    ORDER BY similarity_score
    LIMIT {ids_per_filter}
),
assoc_books AS (
    SELECT DISTINCT
        jsonb_array_elements_text(r."{return_column}")::text AS book_id
    FROM ranked r
    WHERE (1 - (r.similarity_score / 2)) > {similarity_threshold}
)
SELECT
    r.*,
    1 - (r.similarity_score / 2) AS similarity_score,
    b.book_id AS book_ids,
    {book_cols},
    {metadata_cols}
FROM assoc_books ab
JOIN book_summary_backup_2 bs 
    ON bs."ISBN" = ab.book_id
    {isbn_join_clause}
JOIN book_publisher_backup_2 publisher
    ON publisher."ISBN" = ab.book_id
JOIN book_metadata_backup_2 metadata
    ON metadata."ISBN" = ab.book_id
JOIN ranked r
    ON (1 - (r.similarity_score / 2)) > {similarity_threshold}
JOIN LATERAL jsonb_array_elements_text(r.{return_column}) AS b(book_id)
    ON b.book_id = ab.book_id
ORDER BY r.similarity_score ASC
LIMIT {ids_per_filter}