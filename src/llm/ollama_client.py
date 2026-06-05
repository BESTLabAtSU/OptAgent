"""
Updated Ollama client with accurate token counting
"""
import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API with token metrics"""
    text: str
    model: str
    # Token counts from Ollama API
    prompt_eval_count: int = 0      # Input/prompt tokens
    eval_count: int = 0             # Output/completion tokens
    # Timing info (nanoseconds)
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion)"""
        return self.prompt_eval_count + self.eval_count
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens/second for generation"""
        if self.eval_duration and self.eval_count:
            # eval_duration is in nanoseconds
            seconds = self.eval_duration / 1e9
            return self.eval_count / seconds if seconds > 0 else None
        return None


class OllamaClient:
    """
    Ollama client with accurate token counting.
    
    Token counts come from two sources:
    1. Ollama API response (most accurate, when available)
    2. Fallback tokenizer estimation (when API doesn't return counts)
    """

    def __init__(
            self,
            host: Optional[str] = None,
            model: Optional[str] = None,
            timeout: Optional[int] = None,
            use_tokenizer_fallback: bool = True
    ):
        settings = get_settings()
        self.host = host or getattr(settings, 'ollama', None) and settings.ollama.host or "http://localhost:11434"
        self.default_model = model or getattr(settings, 'ollama', None) and settings.ollama.model or "llama3.2"
        self.timeout = timeout or getattr(settings, 'ollama', None) and settings.ollama.timeout or 120
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Token counter for fallback estimation
        self._token_counter = None
        self._use_tokenizer_fallback = use_tokenizer_fallback
        if use_tokenizer_fallback:
            self._init_token_counter()
    
    def _init_token_counter(self):
        """Initialize token counter for fallback estimation"""
        try:
            from .token_counter import TokenCounterFactory
            self._token_counter = TokenCounterFactory.get_counter("ollama", self.default_model)
            logger.info(f"Initialized token counter for {self.default_model}")
        except Exception as e:
            logger.warning(f"Could not initialize token counter: {e}")
            self._token_counter = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens using fallback counter"""
        if self._token_counter and text:
            return self._token_counter.count(text)
        # Last resort: ~4 chars per token
        return len(text) // 4 if text else 0

    def _estimate_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate tokens for messages"""
        if self._token_counter:
            return self._token_counter.count_messages(messages)
        # Fallback estimation
        total = 0
        for msg in messages:
            total += 4  # Overhead per message
            total += self._estimate_tokens(msg.get("content", ""))
        return total

    async def generate(
            self,
            prompt: str,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            system: Optional[str] = None,
            context: Optional[List[int]] = None,
            stream: bool = False
    ) -> OllamaResponse:
        """
        Generate text using Ollama.
        Returns OllamaResponse with token counts.
        """
        await self._ensure_session()

        model = model or self.default_model
        
        settings = get_settings()
        temperature = temperature if temperature is not None else (
            getattr(settings, 'ollama', None) and settings.ollama.temperature or 0.7
        )
        max_tokens = max_tokens or (
            getattr(settings, 'ollama', None) and settings.ollama.max_tokens or 2048
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Need full response for token counts
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system
        if context:
            payload["context"] = context

        url = f"{self.host}/api/generate"

        try:
            async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract token counts from response
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)
                    
                    # Fallback estimation if API didn't return counts
                    if prompt_tokens == 0 and self._use_tokenizer_fallback:
                        prompt_tokens = self._estimate_tokens(prompt)
                        if system:
                            prompt_tokens += self._estimate_tokens(system)
                    
                    if completion_tokens == 0 and self._use_tokenizer_fallback:
                        completion_tokens = self._estimate_tokens(data.get("response", ""))
                    
                    return OllamaResponse(
                        text=data.get("response", ""),
                        model=data.get("model", model),
                        prompt_eval_count=prompt_tokens,
                        eval_count=completion_tokens,
                        total_duration=data.get("total_duration"),
                        load_duration=data.get("load_duration"),
                        prompt_eval_duration=data.get("prompt_eval_duration"),
                        eval_duration=data.get("eval_duration")
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")

        except asyncio.TimeoutError:
            raise Exception(f"Ollama request timeout after {self.timeout} seconds")

    async def chat(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> OllamaResponse:
        """
        Chat completion using Ollama.
        Returns OllamaResponse with token counts.
        """
        await self._ensure_session()

        model = model or self.default_model
        
        settings = get_settings()
        temperature = temperature if temperature is not None else (
            getattr(settings, 'ollama', None) and settings.ollama.temperature or 0.7
        )
        max_tokens = max_tokens or (
            getattr(settings, 'ollama', None) and settings.ollama.max_tokens or 2048
        )

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        url = f"{self.host}/api/chat"

        try:
            async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract token counts
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)
                    
                    # Fallback estimation
                    if prompt_tokens == 0 and self._use_tokenizer_fallback:
                        prompt_tokens = self._estimate_message_tokens(messages)
                    
                    response_text = data.get("message", {}).get("content", "")
                    if completion_tokens == 0 and self._use_tokenizer_fallback:
                        completion_tokens = self._estimate_tokens(response_text)
                    
                    return OllamaResponse(
                        text=response_text,
                        model=data.get("model", model),
                        prompt_eval_count=prompt_tokens,
                        eval_count=completion_tokens,
                        total_duration=data.get("total_duration"),
                        load_duration=data.get("load_duration"),
                        prompt_eval_duration=data.get("prompt_eval_duration"),
                        eval_duration=data.get("eval_duration")
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama chat error: {response.status} - {error_text}")

        except asyncio.TimeoutError:
            raise Exception(f"Ollama request timeout after {self.timeout} seconds")

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        await self._ensure_session()
        try:
            async with self.session.get(
                    self.host,
                    timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except:
            return False

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None