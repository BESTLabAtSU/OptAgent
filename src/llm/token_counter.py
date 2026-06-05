"""
Token Counter
"""
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TokenCounter(ABC):
    """Abstract base class for token counting"""
    
    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in chat messages"""
        pass


class TiktokenCounter(TokenCounter):
    """Token counter using tiktoken (OpenAI's tokenizer)"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import tiktoken
            self._tiktoken = tiktoken
            # Map model to encoding
            self._encoding = self._get_encoding(model)
        except ImportError:
            raise ImportError("tiktoken required: pip install tiktoken")
    
    def _get_encoding(self, model: str):
        """Get appropriate encoding for model"""
        try:
            return self._tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
            return self._tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))
    
    def count_messages(self, messages: List[Dict[str, str]], model: str = None) -> int:
        """
        Count tokens in messages, including message overhead.
        Based on OpenAI's token counting guidelines.
        """
        # Tokens per message overhead varies by model
        tokens_per_message = 3  # Default for gpt-3.5-turbo and gpt-4
        tokens_per_name = 1
        
        total = 0
        for msg in messages:
            total += tokens_per_message
            for key, value in msg.items():
                total += self.count(str(value))
                if key == "name":
                    total += tokens_per_name
        
        total += 3  # Every reply is primed with <|start|>assistant<|message|>
        return total


class HuggingFaceTokenCounter(TokenCounter):
    """
    Token counter using HuggingFace tokenizers.
    Most accurate for specific models like Llama, Mistral, etc.
    """
    
    # Model name to HuggingFace tokenizer mapping
    MODEL_TOKENIZERS = {
        # Llama family
        "llama3.2": "meta-llama/Llama-3.2-3B",
        "llama3.2:1b": "meta-llama/Llama-3.2-1B",
        "llama3.2:3b": "meta-llama/Llama-3.2-3B",
        "llama3.1": "meta-llama/Llama-3.1-8B",
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "llama2": "meta-llama/Llama-2-7b-hf",
        
        # Mistral family
        "mistral": "mistralai/Mistral-7B-v0.1",
        "mistral:7b": "mistralai/Mistral-7B-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-v0.1",
        
        # Qwen family
        "qwen2.5": "Qwen/Qwen2.5-7B",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B",
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
        "qwen2": "Qwen/Qwen2-7B",
        
        # Phi family
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
        
        # Gemma family
        "gemma2": "google/gemma-2-9b",
        "gemma2:9b": "google/gemma-2-9b",
        "gemma2:2b": "google/gemma-2-2b",
        "gemma": "google/gemma-7b",
        
        # CodeLlama
        "codellama": "codellama/CodeLlama-7b-hf",
        
        # DeepSeek
        "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-base",
    }
    
    def __init__(self, model: str = "llama3.2"):
        try:
            from transformers import AutoTokenizer
            self._AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")
        
        self.model = model
        self._tokenizer = None
        self._tokenizer_name = self._resolve_tokenizer(model)
    
    def _resolve_tokenizer(self, model: str) -> str:
        """Resolve Ollama model name to HuggingFace tokenizer"""
        # Normalize model name (remove tags like :latest)
        base_model = model.split(":")[0].lower() if ":" in model else model.lower()
        
        # Check exact match first
        if model.lower() in self.MODEL_TOKENIZERS:
            return self.MODEL_TOKENIZERS[model.lower()]
        
        # Check base model
        if base_model in self.MODEL_TOKENIZERS:
            return self.MODEL_TOKENIZERS[base_model]
        
        # Try to find partial match
        for key, value in self.MODEL_TOKENIZERS.items():
            if base_model in key or key in base_model:
                return value
        
        # Default fallback to Llama 3 tokenizer
        logger.warning(f"No tokenizer mapping for {model}, using Llama-3 tokenizer")
        return "meta-llama/Llama-3.2-3B"
    
    def _load_tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            try:
                self._tokenizer = self._AutoTokenizer.from_pretrained(
                    self._tokenizer_name,
                    trust_remote_code=True
                )
                logger.info(f"Loaded tokenizer: {self._tokenizer_name}")
            except Exception as e:
                logger.warning(f"Failed to load {self._tokenizer_name}: {e}")
                # Fallback to GPT-2 tokenizer (widely available)
                self._tokenizer = self._AutoTokenizer.from_pretrained("gpt2")
    
    def count(self, text: str) -> int:
        if not text:
            return 0
        self._load_tokenizer()
        return len(self._tokenizer.encode(text))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in chat messages with chat template overhead"""
        self._load_tokenizer()
        
        # Try to use chat template if available
        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                tokens = self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True,
                    add_generation_prompt=True
                )
                return len(tokens)
            except Exception:
                pass
        
        # Fallback: count each message with estimated overhead
        total = 0
        for msg in messages:
            # ~4 tokens overhead per message for role tags
            total += 4
            total += self.count(msg.get("content", ""))
        return total


class SimpleEstimator(TokenCounter):
    """
    Simple token estimator based on character/word counts.
    Use as last resort when no tokenizer is available.
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        # Average English text is ~4 chars per token
        # Code tends to be ~3-3.5 chars per token
        self.chars_per_token = chars_per_token
    
    def count(self, text: str) -> int:
        if not text:
            return 0
        return int(len(text) / self.chars_per_token)
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # Overhead per message
            total += self.count(msg.get("content", ""))
        return total


class TokenCounterFactory:
    """Factory for creating appropriate token counter"""
    
    _counters: Dict[str, TokenCounter] = {}
    
    @classmethod
    def get_counter(cls, provider: str, model: str) -> TokenCounter:
        """Get or create a token counter for the given provider/model"""
        cache_key = f"{provider}:{model}"
        
        if cache_key not in cls._counters:
            cls._counters[cache_key] = cls._create_counter(provider, model)
        
        return cls._counters[cache_key]
    
    @classmethod
    def _create_counter(cls, provider: str, model: str) -> TokenCounter:
        """Create appropriate token counter"""
        if provider.lower() == "openai":
            try:
                return TiktokenCounter(model)
            except ImportError:
                logger.warning("tiktoken not installed, using simple estimator")
                return SimpleEstimator()
        
        elif provider.lower() == "ollama":
            # Try HuggingFace tokenizer first (most accurate)
            try:
                return HuggingFaceTokenCounter(model)
            except ImportError:
                logger.warning("transformers not installed, trying tiktoken")
                # Tiktoken as fallback (less accurate but good enough)
                try:
                    return TiktokenCounter("gpt-4")
                except ImportError:
                    logger.warning("No tokenizer available, using simple estimator")
                    return SimpleEstimator()
        
        else:
            return SimpleEstimator()


# Convenience function
def count_tokens(
    text: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini"
) -> int:
    """Quick token count for a single text"""
    counter = TokenCounterFactory.get_counter(provider, model)
    return counter.count(text)


def count_message_tokens(
    messages: List[Dict[str, str]],
    provider: str = "openai",
    model: str = "gpt-4o-mini"
) -> int:
    """Quick token count for chat messages"""
    counter = TokenCounterFactory.get_counter(provider, model)
    return counter.count_messages(messages)