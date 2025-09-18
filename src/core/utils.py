"""Utility functions for OpenAI Agents SDK Demo"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """Setup logging configuration"""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("agent.log")
        ]
    )


def load_config(config_path: Optional[str] = None, env_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from environment variables and optional config file"""
    # Load environment variables
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()  # Load from .env file if it exists
    
    config = {
        # OpenAI Configuration
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_org_id": os.getenv("OPENAI_ORG_ID"),
        
        # Agent Configuration
        "default_model": os.getenv("DEFAULT_MODEL", "gpt-4"),
        "max_tokens": int(os.getenv("MAX_TOKENS", "2000")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        
        # Application Settings
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        
        # Server Configuration
        "host": os.getenv("HOST", "localhost"),
        "port": int(os.getenv("PORT", "8000")),
        "streamlit_port": int(os.getenv("STREAMLIT_PORT", "8501")),
    }
    
    # Load additional config from JSON file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    required_keys = ["openai_api_key"]
    
    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Missing required configuration: {key}")
    
    # Validate numeric values
    if config.get("max_tokens", 0) <= 0:
        raise ValueError("max_tokens must be positive")
    
    if not 0 <= config.get("temperature", 0.7) <= 2:
        raise ValueError("temperature must be between 0 and 2")
    
    return True


def create_project_structure(base_path: str) -> None:
    """Create the basic project directory structure"""
    directories = [
        "src/core",
        "src/examples",
        "src/tools",
        "src/agents",
        "tests",
        "docs",
        "data",
        "logs",
        "config"
    ]
    
    base = Path(base_path)
    for directory in directories:
        (base / directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/"):
            init_file = base / directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""Package: {directory}"""\n')


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "requirements.txt").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string with fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def format_conversation_for_display(messages: list) -> str:
    """Format conversation messages for display"""
    formatted = []
    
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        
        if msg.get("tool_calls"):
            tool_info = f" [Tool calls: {len(msg['tool_calls'])}]"
            content += tool_info
        
        formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def ensure_directory_exists(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_available_models() -> list:
    """Get list of available OpenAI models"""
    return [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (4 characters ≈ 1 token)"""
    return len(text) // 4


def format_error_message(error: Exception) -> str:
    """Format error message for user display"""
    error_type = type(error).__name__
    error_msg = str(error)
    return f"{error_type}: {error_msg}"


class ConfigManager:
    """Configuration manager with validation and persistence"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/agent_config.json"
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file and environment"""
        # Load from environment first
        self.config = load_config()
        
        # Override with file config if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            ensure_directory_exists(os.path.dirname(self.config_file))
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config to {self.config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config.update(updates)
    
    def validate(self) -> bool:
        """Validate current configuration"""
        return validate_config(self.config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()