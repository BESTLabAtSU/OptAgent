# src/llm/ollama_client.py
"""
Ollama client for local LLM integration
"""
import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
import time

from config.settings import get_settings


@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    text: str
    model: str
    total_duration: Optional[float] = None
    load_duration: Optional[float] = None
    eval_duration: Optional[float] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama API for local LLM inference
    """

    def __init__(
            self,
            host: Optional[str] = None,
            model: Optional[str] = None,
            timeout: Optional[int] = None
    ):
        settings = get_settings()
        self.host = host or settings.ollama.host
        self.default_model = model or settings.ollama.model
        self.timeout = timeout or settings.ollama.timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()

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
        """
        Generate text using Ollama

        Args:
            prompt: The prompt to send to the model
            model: Model to use (defaults to configured model)
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
            system: System prompt
            context: Context from previous conversation
            stream: Whether to stream the response

        Returns:
            Generated text response
        """
        await self._ensure_session()

        settings = get_settings()
        model = model or self.default_model
        temperature = temperature if temperature is not None else settings.ollama.temperature
        max_tokens = max_tokens or settings.ollama.max_tokens

        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        if stream:
            return await self._generate_stream(payload)
        else:
            return await self._generate_complete(payload)

    async def _generate_complete(self, payload: Dict[str, Any]) -> str:
        """Generate complete response (non-streaming)"""
        url = f"{self.host}/api/generate"
        payload["stream"] = False

        try:
            async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")

        except asyncio.TimeoutError:
            raise Exception(f"Ollama request timeout after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")

    async def _generate_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        url = f"{self.host}/api/generate"
        payload["stream"] = True

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Ollama streaming failed: {str(e)}")

    async def chat(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat completion using Ollama

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Assistant's response
        """
        await self._ensure_session()

        settings = get_settings()
        model = model or self.default_model
        temperature = temperature if temperature is not None else settings.ollama.temperature
        max_tokens = max_tokens or settings.ollama.max_tokens

        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
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
                    return data.get("message", {}).get("content", "")
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama chat error: {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Ollama chat failed: {str(e)}")

    async def embeddings(
            self,
            text: str,
            model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embeddings for text

        Args:
            text: Text to embed
            model: Model to use for embeddings

        Returns:
            Embedding vector
        """
        await self._ensure_session()

        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": text
        }

        url = f"{self.host}/api/embeddings"

        try:
            async with self.session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama embeddings error: {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Ollama embeddings failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama

        Returns:
            List of available models
        """
        await self._ensure_session()

        url = f"{self.host}/api/tags"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama list models error: {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Failed to list Ollama models: {str(e)}")

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful
        """
        await self._ensure_session()

        url = f"{self.host}/api/pull"
        payload = {"name": model_name}

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                print(f"Pull status: {status}")
                            except:
                                continue
                    return True
                else:
                    return False

        except Exception as e:
            print(f"Failed to pull model: {str(e)}")
            return False

    async def health_check(self) -> bool:
        """
        Check if Ollama service is healthy

        Returns:
            True if service is healthy
        """
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