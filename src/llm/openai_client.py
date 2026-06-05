# src/llm/openai_client.py
import asyncio
from typing import Dict, Any, Optional, List
import aiohttp
import os
import json

from config.settings import get_settings

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: Optional[int] = None):
        s = get_settings()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or getattr(s, "openai", None) and getattr(s.openai, "api_key", None)
        self.model = model or (getattr(s, "openai", None) and getattr(s.openai, "model", "gpt-4o-mini")) or "gpt-4o-mini"
        self.timeout = timeout or (getattr(s, "openai", None) and getattr(s.openai, "timeout", 30)) or 30

        # Allow overriding base URL (Azure/OpenAI proxy). Default to OpenAI.
        self.base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_session()
        url = f"{self.base}{path}"
        try:
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as r:
                text = await r.text()
                # Try to parse JSON either way
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"_raw": text}

                if r.status != 200:
                    # Standard OpenAI-style error
                    err = data.get("error", {})
                    msg = err.get("message") or data.get("_raw") or f"HTTP {r.status}"
                    raise RuntimeError(f"OpenAI API error {r.status}: {msg}")

                # If API returns an error field even with 200 (rare), surface it
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"OpenAI error: {data['error']}")

                return data
        except asyncio.TimeoutError:
            raise RuntimeError(f"OpenAI request timed out after {self.timeout}s")

    # Add this as a method of OpenAIClient
    def _get_token_param(self, model: str) -> str:
        """Newer OpenAI models use max_completion_tokens instead of max_tokens"""
        new_param_models = ("gpt-5", "o1", "o3", "o4")
        return "max_completion_tokens" if any(model.startswith(p) for p in new_param_models) else "max_tokens"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False
    ) -> str:
        s = get_settings()
        m = model or self.model
        t = temperature if temperature is not None else (getattr(s, "openai", None) and getattr(s.openai, "temperature", 0.2)) or 0.2
        mt = max_tokens or (getattr(s, "openai", None) and getattr(s.openai, "max_tokens", 2048)) or 2048

        payload = {
            "model": m,
            "messages": (
                ([{"role": "system", "content": system}] if system else []) +
                [{"role": "user", "content": prompt}]
            ),
            "temperature": t,
            self._get_token_param(m): mt,
            "stream": False
        }

        data = await self._post_json("/chat/completions", payload)

        # Defensive: explain what we got if structure is unexpected
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected response schema from OpenAI: {data}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        s = get_settings()
        m = model or self.model
        t = temperature if temperature is not None else (getattr(s, "openai", None) and getattr(s.openai, "temperature", 0.2)) or 0.2
        mt = max_tokens or (getattr(s, "openai", None) and getattr(s.openai, "max_tokens", 2048)) or 2048

        payload = {
            "model": m,
            "messages": messages,
            "temperature": t,
            self._get_token_param(m): mt,
            "stream": False
        }
        data = await self._post_json("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected response schema from OpenAI: {data}")

    async def health_check(self) -> bool:
        # Probe the models endpoint to verify auth + connectivity
        try:
            await self._ensure_session()
            async with self.session.get(
                f"{self.base}/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as r:
                return r.status == 200
        except Exception:
            return False

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
