#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - Configuration Management
Academic Implementation with Real API Integration
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfiguration:
    """Real API configuration for external services"""
    
    # Weather API Configuration
    WEATHER_API_KEY: str = "1308a047eb770ec1b04ccc653a76a4b9"
    WEATHER_API_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    
    # Flight API Configuration  
    FLIGHT_API_KEY: str = "10fa3e346f7606412b86a9cbde4b00a4"
    AVIATIONSTACK_BASE_URL: str = "https://api.aviationstack.com/v1"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = "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"
    OPENAI_MODEL: str = "gpt-4"
    
    # DeepSeek Configuration
    DEEPSEEK_API_KEY: str = "sk-488fe6d732734e3097631f0ba4eafd29"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.ai/v1"
    
    # OpenSky Network Configuration
    OPENSKY_BASE_URL: str = "https://opensky-network.org/api"
    
    # Safety Data Configuration
    SAFETY_DATA_SOURCES: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize safety data sources"""
        self.SAFETY_DATA_SOURCES = {
            "icao": "https://www.icao.int/safety/iStars/Pages/API-Data-Service.aspx",
            "faa": "https://www.faa.gov/data_research/accident_incident/",
            "easa": "https://www.easa.europa.eu/domains/safety-management"
        }

# Global configuration instance
API_CONFIG = APIConfiguration()

class SystemConfiguration:
    """System-wide configuration management"""
    
    def __init__(self):
        self.config = {
            # Core System Configuration
            "system": {
                "name": "MAMA Flight Selection Assistant",
                "version": "1.0.0",
                "environment": "production",
                "debug": False,
                "log_level": "INFO"
            },
            
            # API Configuration
            "apis": {
                "weather": {
                    "provider": "openweathermap",
                    "api_key": API_CONFIG.WEATHER_API_KEY,
                    "base_url": API_CONFIG.WEATHER_API_BASE_URL,
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "flight": {
                    "provider": "aviationstack",
                    "api_key": API_CONFIG.FLIGHT_API_KEY,
                    "base_url": API_CONFIG.AVIATIONSTACK_BASE_URL,
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "openai": {
                    "api_key": API_CONFIG.OPENAI_API_KEY,
                    "model": API_CONFIG.OPENAI_MODEL,
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "deepseek": {
                    "api_key": API_CONFIG.DEEPSEEK_API_KEY,
                    "base_url": API_CONFIG.DEEPSEEK_BASE_URL,
                    "model": "deepseek-chat"
                },
                "opensky": {
                    "base_url": API_CONFIG.OPENSKY_BASE_URL,
                    "timeout": 30
                }
            },
            
            # MAMA System Configuration
            "mama": {
                "max_concurrent_agents": 5,
                "timeout_seconds": 30.0,
                "trust_threshold": 0.3,
                "confidence_threshold": 0.8,
                "consensus_threshold": 0.75,
                "similarity_threshold": 0.6,
                "ranking_depth": 10,
                "feature_dimension": 128,
                "learning_rate": 0.001,
                "discount_factor": 0.95,
                "trust_weight": 0.4
            },
            
            # SBERT Configuration
            "sbert": {
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "similarity_threshold": 0.6,
                "cache_embeddings": True
            },
            
            # MARL Configuration
            "marl": {
                "state_dimension": 128,
                "action_dimension": 64,
                "hidden_dimension": 256,
                "learning_rate": 0.001,
                "discount_factor": 0.95,
                "exploration_rate": 0.1,
                "batch_size": 32,
                "memory_size": 10000
            },
            
            # LTR Configuration
            "ltr": {
                "feature_dimension": 128,
                "ranking_depth": 10,
                "learning_rate": 0.01,
                "regularization": 0.001,
                "iterations": 100
            },
            
            # Trust System Configuration
            "trust": {
                "initial_trust_score": 0.5,
                "trust_decay_factor": 0.95,
                "trust_update_rate": 0.1,
                "min_trust_threshold": 0.1,
                "max_trust_threshold": 0.9,
                "trust_dimensions": [
                    "reliability",
                    "competence", 
                    "fairness",
                    "security",
                    "transparency"
                ]
            },
            
            # Database Configuration
            "database": {
                "type": "sqlite",
                "path": "data/mama_system.db",
                "backup_interval": 3600,
                "max_connections": 10
            },
            
            # Logging Configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/mama_system.log",
                "max_size": 10485760,  # 10MB
                "backup_count": 5
            },
            
            # Web Interface Configuration
            "web": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False,
                "secret_key": "mama_flight_assistant_secure_key_2024",
                "session_timeout": 3600,
                "max_content_length": 16777216  # 16MB
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value by dot-separated path"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that all required API keys are present"""
        validation_results = {}
        
        # Check Weather API
        weather_key = self.get("apis.weather.api_key")
        validation_results["weather_api"] = bool(weather_key and len(weather_key) > 10)
        
        # Check Flight API
        flight_key = self.get("apis.flight.api_key")
        validation_results["flight_api"] = bool(flight_key and len(flight_key) > 10)
        
        # Check OpenAI API
        openai_key = self.get("apis.openai.api_key")
        validation_results["openai_api"] = bool(openai_key and openai_key.startswith("sk-"))
        
        # Check DeepSeek API
        deepseek_key = self.get("apis.deepseek.api_key")
        validation_results["deepseek_api"] = bool(deepseek_key and deepseek_key.startswith("sk-"))
        
        return validation_results
    
    def get_api_config(self, service: str) -> Optional[Dict[str, Any]]:
        """Get API configuration for specific service"""
        return self.get(f"apis.{service}")

# Global configuration instance
CONFIG = SystemConfiguration()

# LLM Configuration for backward compatibility
LLM_CONFIG = {
    "openai": {
        "api_key": API_CONFIG.OPENAI_API_KEY,
        "model": API_CONFIG.OPENAI_MODEL,
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "deepseek": {
        "api_key": API_CONFIG.DEEPSEEK_API_KEY,
        "base_url": API_CONFIG.DEEPSEEK_BASE_URL,
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 2000
    }
}

# Milestone Configuration - Real Data Space Connection
MILESTONE_URL = "http://localhost:1026/ngsi-ld/v1/entities"
MILESTONE_REALTIME_URL = "http://localhost:9090/ngsi-ld/v1/entities"
PROTECTED_URL = "http://localhost:6003/ngsi-ld/v1/entities"
CONTEXT_URL = "http://localhost:8080/ngsi-ld.jsonld"
JWT_TOKEN = "PASTE_YOUR_JWT_TOKEN_HERE"  # Must be configured with real token

# System Configuration for Proxies
PROXIES = {'http': None, 'https': None}

# OpenAI API Key for backward compatibility
OPENAI_API_KEY = API_CONFIG.OPENAI_API_KEY

# API Keys Configuration
WEATHER_API_KEY = "498ae38fb9831291de1d0432ea2fdf07"
FLIGHT_API_KEY = "10fa3e346f7606412b86a9cbde4b00a4"
OPENAI_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-488fe6d732734e3097631f0ba4eafd29"

# Environment variables setup
os.environ['WEATHER_API_KEY'] = WEATHER_API_KEY
os.environ['FLIGHT_API_KEY'] = FLIGHT_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY

def initialize_configuration() -> SystemConfiguration:
    """Initialize and validate system configuration"""
    logger.info("🔧 Initializing MAMA system configuration...")
    
    # Validate API keys
    validation_results = CONFIG.validate_api_keys()
    
    for service, is_valid in validation_results.items():
        if is_valid:
            logger.info(f"✅ {service.upper()} API key validated")
        else:
            logger.error(f"❌ {service.upper()} API key invalid or missing")
    
    # Check if all critical APIs are configured
    critical_apis = ["weather_api", "flight_api"]
    all_critical_valid = all(validation_results.get(api, False) for api in critical_apis)
    
    if all_critical_valid:
        logger.info("✅ All critical API configurations validated")
    else:
        logger.warning("⚠️ Some critical API configurations are invalid")
    
    return CONFIG

def get_config() -> SystemConfiguration:
    """Get global configuration instance"""
    return CONFIG

__all__ = [
    'CONFIG',
    'API_CONFIG', 
    'SystemConfiguration',
    'APIConfiguration',
    'initialize_configuration',
    'get_config'
]
