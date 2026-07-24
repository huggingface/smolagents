SELECT
    MATCH.*,
    1 - (MATCH.similarity_score / 2) AS similarity_score,
    {book_cols},
    {metadata_cols}
FROM (
    SELECT
        {select_column_str},
        {book_id_alias},
        ({search_column} <=> '{embedding_str}'::vector) AS similarity_score
    FROM 
        "{table_name}"
        {isbn_clause}
    ORDER BY 
        similarity_score
    LIMIT 
        {ids_per_filter}
) AS MATCH
JOIN book_summary_backup_2 bs
    ON bs."ISBN" = MATCH.book_ids
JOIN book_publisher_backup_2 publisher
    ON publisher."ISBN" = MATCH.book_ids
JOIN book_metadata_backup_2 metadata
    ON metadata."ISBN" = MATCH.book_ids
WHERE (1 - (MATCH.similarity_score / 2)) > {similarity_threshold}
ORDER BY MATCH.similarity_score ASC