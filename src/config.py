"""
Configuration management for the paraphrase generation system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "tuner007/pegasus_paraphrase"
    max_length: int = 100
    num_return_sequences: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    device: str = "auto"


@dataclass
class UIConfig:
    """UI configuration parameters."""
    title: str = "Paraphrase Generation System"
    description: str = "Generate high-quality paraphrases using state-of-the-art NLP models"
    max_input_length: int = 500
    default_num_paraphrases: int = 3
    show_similarity_scores: bool = True
    show_evaluation_metrics: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    ui: UIConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            ui=UIConfig(**config_dict.get('ui', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'model': asdict(self.model),
            'ui': asdict(self.ui),
            'logging': asdict(self.logging)
        }


class ConfigManager:
    """Configuration manager for loading and saving configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/config.yaml")
        self.config_path.parent.mkdir(exist_ok=True)
    
    def load_config(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig object
        """
        if not self.config_path.exists():
            logger.info("Config file not found, creating default configuration")
            config = self._get_default_config()
            self.save_config(config)
            return config
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    config_dict = yaml.safe_load(f)
            
            return AppConfig.from_dict(config_dict)
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def save_config(self, config: AppConfig) -> None:
        """
        Save configuration to file.
        
        Args:
            config: AppConfig object to save
        """
        try:
            config_dict = config.to_dict()
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _get_default_config(self) -> AppConfig:
        """Get default configuration."""
        return AppConfig(
            model=ModelConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )


# Import logger here to avoid circular imports
import logging
logger = logging.getLogger(__name__)
