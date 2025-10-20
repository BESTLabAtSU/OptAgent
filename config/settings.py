# config/settings.py
"""
Central configuration management for the Multi-Agent DER Framework
"""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


class OllamaConfig(BaseSettings):
    """Ollama LLM configuration"""
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    host: str = Field(default="http://localhost:11434")
    model: str = Field(default="qwen3:1.7b")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    timeout: int = Field(default=30)


class MCPConfig(BaseSettings):
    """MCP Server configuration"""
    model_config = SettingsConfigDict(env_prefix="MCP_")

    host: str = Field(default="localhost")
    port: int = Field(default=8765)
    protocol_version: str = Field(default="1.0")
    enable_tls: bool = Field(default=False)


class MessageBusConfig(BaseSettings):
    """Message Bus configuration"""
    model_config = SettingsConfigDict(env_prefix="BUS_")

    type: str = Field(default="memory")  # memory, redis, kafka
    redis_url: Optional[str] = Field(default="redis://localhost:6379")
    kafka_brokers: Optional[str] = Field(default="localhost:9092")


class DERSystemConfig(BaseSettings):
    """DER System configuration"""
    model_config = SettingsConfigDict(env_prefix="DER_")

    config_path: str = Field(default="./der_configs")
    runtime_api_url: Optional[str] = Field(default=None)
    enable_simulation: bool = Field(default=True)


class Settings(BaseSettings):
    """Main application settings"""
    # Global .env handling + case rules
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Multi-Agent DER Framework"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Paths
    base_path: Path = Path(__file__).resolve().parent.parent
    agents_path: Path = base_path / "src" / "agents"
    tools_path: Path = base_path / "src" / "tools"

    # Sub-configurations (each reads its own env vars via its env_prefix)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    message_bus: MessageBusConfig = Field(default_factory=MessageBusConfig)
    der_system: DERSystemConfig = Field(default_factory=DERSystemConfig)

    # Agent Configuration
    max_concurrent_agents: int = Field(default=5)
    agent_timeout: int = Field(default=60)

    # Security
    enable_auth: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """Get the settings instance"""
    return settings


def update_settings(**kwargs) -> None:
    """Update settings dynamically"""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
