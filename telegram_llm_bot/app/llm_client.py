import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.getenv("LM_STUDIO_BASE", "http://192.168.0.102:1234")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60)

    async def list_models(self) -> List[str]:
        try:
            resp = await self._client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            return models
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch models: %s", exc)
            return []

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}).get("content", "")
        return message

    async def aclose(self) -> None:
        await self._client.aclose()

