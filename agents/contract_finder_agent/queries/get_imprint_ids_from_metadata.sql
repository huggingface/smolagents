SELECT DISTINCT
    "{return_column}" AS imprint_id,
    "{match_column}"  AS imprint_name
FROM "{table_name}"
WHERE {where_clause};