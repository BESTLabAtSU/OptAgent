"""
LLM Interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    model: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion from a prompt"""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """Chat completion with message history"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available"""
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider type"""
        pass


class OpenAIAdapter(BaseLLMClient):
    """Adapter for OpenAI API (works with AsyncOpenAI client)"""

    def __init__(self, client, default_model: str = "gpt-4o-mini"):
        self._client = client
        self.default_model = default_model

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI

    def _get_token_param(self, model: str) -> str:
        new_param_models = ("gpt-5", "o1", "o3", "o4")
        return "max_completion_tokens" if any(model.startswith(p) for p in new_param_models) else "max_tokens"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        kwargs = {
            "model": model or self.default_model,
            "messages": messages
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs[self._get_token_param(kwargs["model"])] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            total_tokens=getattr(response.usage, 'total_tokens', 0) if response.usage else 0,
            prompt_tokens=getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
            completion_tokens=getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
            raw_response=response
        )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self):
        if hasattr(self._client, 'close'):
            await self._client.close()


class OllamaAdapter(BaseLLMClient):
    """Adapter for Ollama local models with accurate token counting"""

    def __init__(self, ollama_client, default_model: str = "llama3.2"):
        self._client = ollama_client
        self.default_model = default_model

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OLLAMA

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks"""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find raw JSON object/array
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            return json_match.group(1).strip()
        
        return text

    def _ensure_json_response(self, content: str, json_mode: bool) -> str:
        """Ensure response is valid JSON when json_mode is True"""
        if not json_mode:
            return content
        
        extracted = self._extract_json(content)
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            # Return a fallback JSON structure
            return json.dumps({"error": "Failed to parse JSON", "raw": content[:500]})

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> LLMResponse:
        # OllamaClient now returns OllamaResponse with token counts
        ollama_response = await self._client.generate(
            prompt=prompt,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system
        )

        return LLMResponse(
            content=ollama_response.text,
            model=ollama_response.model,
            prompt_tokens=ollama_response.prompt_eval_count,
            completion_tokens=ollama_response.eval_count,
            total_tokens=ollama_response.total_tokens,
            raw_response=ollama_response
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        # For JSON mode, add instruction to the last user message
        if json_mode:
            messages = [m.copy() for m in messages]  # Deep copy
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i] = {
                        "role": "user",
                        "content": messages[i]["content"] + "\n\nRespond ONLY with valid JSON, no other text."
                    }
                    break

        # OllamaClient now returns OllamaResponse with token counts
        ollama_response = await self._client.chat(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Process JSON mode response
        content = self._ensure_json_response(ollama_response.text, json_mode)

        return LLMResponse(
            content=content,
            model=ollama_response.model,
            prompt_tokens=ollama_response.prompt_eval_count,
            completion_tokens=ollama_response.eval_count,
            total_tokens=ollama_response.total_tokens,
            raw_response=ollama_response
        )

    async def health_check(self) -> bool:
        return await self._client.health_check()

    async def close(self):
        await self._client.close()
    
    def get_performance_metrics(self, response: LLMResponse) -> Dict[str, Any]:
        """Extract detailed performance metrics from Ollama response"""
        if not response.raw_response:
            return {}
        
        ollama_resp = response.raw_response
        metrics = {
            "prompt_tokens": ollama_resp.prompt_eval_count,
            "completion_tokens": ollama_resp.eval_count,
            "total_tokens": ollama_resp.total_tokens,
        }
        
        # Add timing metrics if available
        if ollama_resp.total_duration:
            metrics["total_duration_ms"] = ollama_resp.total_duration / 1e6
        if ollama_resp.eval_duration:
            metrics["generation_duration_ms"] = ollama_resp.eval_duration / 1e6
        if ollama_resp.tokens_per_second:
            metrics["tokens_per_second"] = ollama_resp.tokens_per_second
        
        return metrics


def create_llm_client(
    provider: str = "openai",
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client
    
    Args:
        provider: "openai" or "ollama"
        **kwargs: Provider-specific arguments
            OpenAI: api_key, model, organization
            Ollama: host, model, timeout
    
    Returns:
        BaseLLMClient instance
    """
    if provider.lower() == "openai":
        from openai import AsyncOpenAI
        api_key = kwargs.get("api_key")
        model = kwargs.get("model", "gpt-4o-mini")
        
        client = AsyncOpenAI(api_key=api_key)
        return OpenAIAdapter(client, default_model=model)
    
    elif provider.lower() == "ollama":
        from .ollama_client import OllamaClient
        
        host = kwargs.get("host", "http://localhost:11434")
        model = kwargs.get("model", "llama3.2")
        timeout = kwargs.get("timeout", 120)
        
        client = OllamaClient(host=host, model=model, timeout=timeout)
        return OllamaAdapter(client, default_model=model)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, ollama")