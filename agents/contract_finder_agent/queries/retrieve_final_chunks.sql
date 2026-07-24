SELECT
    MATCH.*,
    1 - (MATCH.similarity_score / 2) AS similarity_score,
    {book_cols}
FROM (
    SELECT
        {select_column_str},
        ({search_column} <=> {embedding_str}::vector) AS similarity_score
    FROM
        {table_name}
    WHERE
        "{return_column}" IN ({book_ids_str})
    ORDER BY
        similarity_score
    LIMIT
        {max_chunks_to_use}
) AS MATCH
LEFT JOIN book_summary_backup_2 bs
    ON bs."ISBN" = MATCH."ISBN"
LEFT JOIN book_publisher_backup_2 publisher
    ON publisher."ISBN" = MATCH."ISBN"
WHERE (1 - (MATCH.similarity_score / 2)) > {similarity_threshold}
ORDER BY MATCH.similarity_score ASC;