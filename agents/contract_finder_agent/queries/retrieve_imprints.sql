SELECT {imprint_id_alias}, 
{select_column_str}
FROM "{table_name}"
ORDER BY ({search_column} <=> %s::vector) ASC
LIMIT 1;