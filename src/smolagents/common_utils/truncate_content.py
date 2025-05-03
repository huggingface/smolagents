# Utility function to truncate content to a specified maximum length

def truncate_content(content: str, max_length: int = 20000) -> str:
    """Truncates the content to the specified maximum length."""
    if len(content) > max_length:
        return content[:max_length] + "\n... [Content truncated]"
    return content