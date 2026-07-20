"""Search public X posts through the Xquik API.

Xquik is a closed-source hosted service. Xquik is an independent third-party
service. Not affiliated with X Corp. "Twitter" and "X" are trademarks of X Corp.
"""

import os

import requests

from smolagents import CodeAgent, InferenceClientModel, Tool


class XquikSearchPostsTool(Tool):
    name = "xquik_search_posts"
    description = "Search public X posts with the Xquik API."
    inputs = {
        "query": {
            "type": "string",
            "description": "Keyword query, username query, post ID, or X status URL.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of posts to return.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, limit: int | None = 5) -> str:
        api_key = os.environ.get("XQUIK_API_KEY")
        if not api_key:
            return "Set XQUIK_API_KEY before calling this tool."

        max_results = 5 if limit is None else max(1, min(limit, 20))
        response = requests.get(
            f"{os.environ.get('XQUIK_BASE_URL', 'https://xquik.com')}/api/v1/x/tweets/search",
            headers={"x-api-key": api_key},
            params={"q": query, "limit": max_results},
            timeout=30,
        )
        if response.status_code == 402:
            return "Xquik returned payment required. Check your Xquik account access."
        if response.status_code == 401:
            return "Xquik rejected the API key. Check XQUIK_API_KEY."
        response.raise_for_status()

        data = response.json()
        posts = data.get("tweets", data.get("items", []))
        if not posts:
            return "No matching posts found."

        return "\n\n".join(format_post(post) for post in posts[:max_results])


def format_post(post: dict) -> str:
    author = post.get("author") or {}
    username = author.get("username") or post.get("username") or "unknown"
    text = post.get("text") or post.get("fullText") or ""
    metrics = []
    for key, label in [
        ("likeCount", "likes"),
        ("replyCount", "replies"),
        ("retweetCount", "reposts"),
        ("quoteCount", "quotes"),
    ]:
        if key in post:
            metrics.append(f"{label}: {post[key]}")

    metrics_text = f" ({', '.join(metrics)})" if metrics else ""
    post_url = post.get("url") or post.get("tweetUrl")
    url_text = f"\nURL: {post_url}" if post_url else ""
    return f"@{username}{metrics_text}\n{text}{url_text}"


if __name__ == "__main__":
    agent = CodeAgent(
        tools=[XquikSearchPostsTool()],
        model=InferenceClientModel(),
        stream_outputs=True,
    )

    agent.run("Find recent public posts about open source AI agents.")
