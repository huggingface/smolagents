"""Thin HTTP client for the Memanto REST API."""

from __future__ import annotations

import os
from typing import Any

import httpx


class MemantoError(Exception):
    """Raised when a Memanto API request fails."""


class MemantoClient:
    """Client for Memanto's REST API (requires `memanto serve` to be running)."""

    def __init__(
        self,
        base_url: str | None = None,
        agent_id: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.getenv("MEMANTO_URL", "http://localhost:8000")).rstrip("/")
        self.agent_id = agent_id or os.getenv("MEMANTO_AGENT_ID", "smolagents-demo")
        self.session_token: str | None = None
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def activate(self) -> str:
        """Start or refresh the agent session and return the session token."""
        try:
            response = self._client.post(f"/api/v2/agents/{self.agent_id}/activate")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MemantoError(
                f"Could not activate Memanto session at {self.base_url}. "
                "Is `memanto serve` running? Did you create the agent with "
                f"`memanto agent create {self.agent_id}`?"
            ) from exc

        data = response.json()
        self.session_token = data["session_token"]
        return self.session_token

    def _headers(self) -> dict[str, str]:
        if not self.session_token:
            self.activate()
        return {
            "X-Session-Token": self.session_token,
            "Content-Type": "application/json",
        }

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Semantic search over stored memories."""
        try:
            response = self._client.post(
                f"/api/v2/agents/{self.agent_id}/recall",
                headers=self._headers(),
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MemantoError(f"Memanto recall failed: {exc}") from exc
        return response.json().get("memories", [])

    def recall_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recently stored memories, newest first."""
        try:
            response = self._client.post(
                f"/api/v2/agents/{self.agent_id}/recall/recent",
                headers=self._headers(),
                json={"limit": limit},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MemantoError(f"Memanto recall_recent failed: {exc}") from exc
        return response.json().get("memories", [])

    def has_memories(self) -> bool:
        """Return True if the agent namespace already has stored memories."""
        return len(self.recall_recent(limit=1)) > 0

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        title: str | None = None,
        confidence: float = 0.85,
        tags: list[str] | None = None,
    ) -> str:
        """Store a single typed memory and return its id."""
        payload: dict[str, Any] = {
            "content": content,
            "type": memory_type,
            "confidence": confidence,
        }
        if title is not None:
            payload["title"] = title
        if tags is not None:
            payload["tags"] = tags

        try:
            response = self._client.post(
                f"/api/v2/agents/{self.agent_id}/remember",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MemantoError(f"Memanto remember failed: {exc}") from exc
        return response.json()["memory_id"]

    def answer(self, question: str, limit: int | None = None) -> str:
        """Return an LLM answer grounded in stored memories."""
        payload: dict[str, Any] = {"question": question}
        if limit is not None:
            payload["limit"] = limit

        try:
            response = self._client.post(
                f"/api/v2/agents/{self.agent_id}/answer",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MemantoError(f"Memanto answer failed: {exc}") from exc
        return response.json().get("answer", "")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MemantoClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
