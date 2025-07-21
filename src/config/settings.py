"""
MAMA Flight Assistant - Configuration Settings

Central configuration management for the entire system.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

@dataclass
class APIConfiguration:
    """API Configuration settings"""
    # Flight data APIs
    amadeus_api_key: str = field(default_factory=lambda: os.getenv('AMADEUS_API_KEY', ''))
    amadeus_api_secret: str = field(default_factory=lambda: os.getenv('AMADEUS_API_SECRET', ''))
    amadeus_base_url: str = "https://test.api.amadeus.com"
    
    skyscanner_api_key: str = field(default_factory=lambda: os.getenv('SKYSCANNER_API_KEY', ''))
    skyscanner_base_url: str = "https://rapidapi.com/skyscanner/api/skyscanner-flight-search"
    
    # Rate limiting
    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    
    # Timeout settings
    connection_timeout: int = 10
    read_timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_backoff_factor: float = 0.3

@dataclass
class CacheConfiguration:
    """Cache Configuration settings"""
    # Cache directories
    cache_dir: Path = CACHE_DIR
    
    # Cache TTL (Time To Live) in seconds
    flight_data_ttl: int = 3600  # 1 hour
    api_response_ttl: int = 1800  # 30 minutes
    processed_data_ttl: int = 7200  # 2 hours
    
    # Cache size limits
    max_cache_size_mb: int = 100
    max_cached_items: int = 1000
    
    # Cache cleanup
    cleanup_interval_hours: int = 24
    auto_cleanup_enabled: bool = True

@dataclass
class LoggingConfiguration:
    """Logging Configuration settings"""
    # Log directories
    log_dir: Path = LOGS_DIR
    
    # Log levels
    root_log_level: str = "INFO"
    api_log_level: str = "DEBUG"
    agent_log_level: str = "INFO"
    
    # Log file settings
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Log formats
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Console logging
    console_logging_enabled: bool = True
    console_log_level: str = "INFO"

@dataclass
class AgentConfiguration:
    """Agent system configuration"""
    # Agent behavior
    max_response_length: int = 2000
    response_timeout_seconds: int = 30
    
    # Conversation settings
    max_conversation_history: int = 50
    context_window_size: int = 10
    
    # Flight search defaults
    default_search_limit: int = 20
    max_search_results: int = 100
    
    # AI model settings
    temperature: float = 0.7
    max_tokens: int = 1500
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: ['zh-CN', 'en-US'])
    default_language: str = 'zh-CN'

@dataclass
class SecurityConfiguration:
    """Security Configuration settings"""
    # API key encryption
    encrypt_api_keys: bool = True
    encryption_key: str = field(default_factory=lambda: os.getenv('ENCRYPTION_KEY', ''))
    
    # Rate limiting
    rate_limit_enabled: bool = True
    max_requests_per_ip_per_hour: int = 100
    
    # Input validation
    max_input_length: int = 1000
    allowed_characters_pattern: str = r'^[a-zA-Z0-9\s\-_.,!?()]+$'
    
    # Data protection
    mask_sensitive_data: bool = True
    log_sensitive_data: bool = False

@dataclass
class DatabaseConfiguration:
    """Database Configuration settings (for future use)"""
    # SQLite settings (for local storage)
    sqlite_path: Path = DATA_DIR / "mama_assistant.db"
    
    # Connection settings
    connection_pool_size: int = 5
    connection_timeout: int = 30
    
    # Backup settings
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7

@dataclass
class SystemConfiguration:
    """Main system configuration that combines all settings"""
    api: APIConfiguration = field(default_factory=APIConfiguration)
    cache: CacheConfiguration = field(default_factory=CacheConfiguration)
    logging: LoggingConfiguration = field(default_factory=LoggingConfiguration)
    agent: AgentConfiguration = field(default_factory=AgentConfiguration)
    security: SecurityConfiguration = field(default_factory=SecurityConfiguration)
    database: DatabaseConfiguration = field(default_factory=DatabaseConfiguration)
    
    # System settings
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG', 'False').lower() == 'true')
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.cache.cache_dir,
            self.logging.log_dir,
            DATA_DIR,
            self.database.sqlite_path.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'api': {
                'amadeus_base_url': self.api.amadeus_base_url,
                'skyscanner_base_url': self.api.skyscanner_base_url,
                'requests_per_minute': self.api.requests_per_minute,
                'requests_per_hour': self.api.requests_per_hour,
                'connection_timeout': self.api.connection_timeout,
                'read_timeout': self.api.read_timeout,
                'max_retries': self.api.max_retries,
                'retry_backoff_factor': self.api.retry_backoff_factor
            },
            'cache': {
                'cache_dir': str(self.cache.cache_dir),
                'flight_data_ttl': self.cache.flight_data_ttl,
                'api_response_ttl': self.cache.api_response_ttl,
                'processed_data_ttl': self.cache.processed_data_ttl,
                'max_cache_size_mb': self.cache.max_cache_size_mb,
                'max_cached_items': self.cache.max_cached_items,
                'cleanup_interval_hours': self.cache.cleanup_interval_hours,
                'auto_cleanup_enabled': self.cache.auto_cleanup_enabled
            },
            'logging': {
                'log_dir': str(self.logging.log_dir),
                'root_log_level': self.logging.root_log_level,
                'api_log_level': self.logging.api_log_level,
                'agent_log_level': self.logging.agent_log_level,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count,
                'console_logging_enabled': self.logging.console_logging_enabled,
                'console_log_level': self.logging.console_log_level
            },
            'agent': {
                'max_response_length': self.agent.max_response_length,
                'response_timeout_seconds': self.agent.response_timeout_seconds,
                'max_conversation_history': self.agent.max_conversation_history,
                'context_window_size': self.agent.context_window_size,
                'default_search_limit': self.agent.default_search_limit,
                'max_search_results': self.agent.max_search_results,
                'temperature': self.agent.temperature,
                'max_tokens': self.agent.max_tokens,
                'supported_languages': self.agent.supported_languages,
                'default_language': self.agent.default_language
            },
            'security': {
                'encrypt_api_keys': self.security.encrypt_api_keys,
                'rate_limit_enabled': self.security.rate_limit_enabled,
                'max_requests_per_ip_per_hour': self.security.max_requests_per_ip_per_hour,
                'max_input_length': self.security.max_input_length,
                'mask_sensitive_data': self.security.mask_sensitive_data,
                'log_sensitive_data': self.security.log_sensitive_data
            },
            'database': {
                'sqlite_path': str(self.database.sqlite_path),
                'connection_pool_size': self.database.connection_pool_size,
                'connection_timeout': self.database.connection_timeout,
                'auto_backup_enabled': self.database.auto_backup_enabled,
                'backup_interval_hours': self.database.backup_interval_hours,
                'max_backups': self.database.max_backups
            },
            'system': {
                'debug_mode': self.debug_mode,
                'environment': self.environment,
                'version': self.version
            }
        }
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate configuration settings"""
        errors = {}
        
        # Validate API configuration
        if not self.api.amadeus_api_key and not self.api.skyscanner_api_key:
            errors.setdefault('api', []).append("At least one API key must be configured")
        
        if self.api.requests_per_minute <= 0:
            errors.setdefault('api', []).append("requests_per_minute must be positive")
        
        if self.api.requests_per_hour <= 0:
            errors.setdefault('api', []).append("requests_per_hour must be positive")
        
        if self.api.connection_timeout <= 0:
            errors.setdefault('api', []).append("connection_timeout must be positive")
        
        # Validate cache configuration
        if self.cache.flight_data_ttl <= 0:
            errors.setdefault('cache', []).append("flight_data_ttl must be positive")
        
        if self.cache.max_cache_size_mb <= 0:
            errors.setdefault('cache', []).append("max_cache_size_mb must be positive")
        
        # Validate agent configuration
        if self.agent.max_response_length <= 0:
            errors.setdefault('agent', []).append("max_response_length must be positive")
        
        if self.agent.temperature < 0 or self.agent.temperature > 2:
            errors.setdefault('agent', []).append("temperature must be between 0 and 2")
        
        if self.agent.default_language not in self.agent.supported_languages:
            errors.setdefault('agent', []).append("default_language must be in supported_languages")
        
        # Validate security configuration
        if self.security.max_input_length <= 0:
            errors.setdefault('security', []).append("max_input_length must be positive")
        
        return errors

# Global configuration instance
settings = SystemConfiguration()

# Configuration constants
FLIGHT_SEARCH_ENDPOINTS = {
    'amadeus': {
        'flight_offers': '/v2/shopping/flight-offers',
        'flight_dates': '/v1/shopping/flight-dates',
        'flight_destinations': '/v1/shopping/flight-destinations'
    },
    'skyscanner': {
        'flight_search': '/v1.0/flights/search',
        'flight_details': '/v1.0/flights/details'
    }
}

SUPPORTED_CITIES = {
    'Beijing': {'code': 'BJS', 'iata': 'PEK', 'country': 'CN'},
    'Shanghai': {'code': 'SHA', 'iata': 'PVG', 'country': 'CN'},
    'Guangzhou': {'code': 'CAN', 'iata': 'CAN', 'country': 'CN'},
    'Shenzhen': {'code': 'SZX', 'iata': 'SZX', 'country': 'CN'},
    'Chengdu': {'code': 'CTU', 'iata': 'CTU', 'country': 'CN'},
    'Hangzhou': {'code': 'HGH', 'iata': 'HGH', 'country': 'CN'},
    'Nanjing': {'code': 'NKG', 'iata': 'NKG', 'country': 'CN'},
    'Xi\'an': {'code': 'XIY', 'iata': 'XIY', 'country': 'CN'},
    'Chongqing': {'code': 'CKG', 'iata': 'CKG', 'country': 'CN'},
    'Tianjin': {'code': 'TSN', 'iata': 'TSN', 'country': 'CN'},
    'New York': {'code': 'NYC', 'iata': 'JFK', 'country': 'US'},
    'Los Angeles': {'code': 'LAX', 'iata': 'LAX', 'country': 'US'},
    'London': {'code': 'LON', 'iata': 'LHR', 'country': 'GB'},
    'Paris': {'code': 'PAR', 'iata': 'CDG', 'country': 'FR'},
    'Tokyo': {'code': 'TYO', 'iata': 'NRT', 'country': 'JP'},
    'Seoul': {'code': 'SEL', 'iata': 'ICN', 'country': 'KR'},
    'Singapore': {'code': 'SIN', 'iata': 'SIN', 'country': 'SG'},
    'Hong Kong': {'code': 'HKG', 'iata': 'HKG', 'country': 'HK'},
    'Taipei': {'code': 'TPE', 'iata': 'TPE', 'country': 'TW'},
    'Bangkok': {'code': 'BKK', 'iata': 'BKK', 'country': 'TH'}
}

AIRLINE_CODES = {
    'CA': 'Air China',
    'MU': 'China Eastern Airlines',
    'CZ': 'China Southern Airlines',
    'HU': 'Hainan Airlines',
    '9C': 'Spring Airlines',
    'UO': 'Juneyao Airlines',
    'FM': 'Shanghai Airlines',
    'SC': 'Shandong Airlines',
    'JD': 'Capital Airlines',
    'G5': 'China Express Airlines',
    'AA': 'American Airlines',
    'UA': 'United Airlines',
    'DL': 'Delta Air Lines',
    'BA': 'British Airways',
    'AF': 'Air France',
    'LH': 'Lufthansa',
    'NH': 'All Nippon Airways',
    'JL': 'Japan Airlines',
    'KE': 'Korean Air',
    'OZ': 'Asiana Airlines',
    'SQ': 'Singapore Airlines',
    'CX': 'Cathay Pacific',
    'TG': 'Thai Airways'
}

# Error messages in English
ERROR_MESSAGES = {
    'api_unavailable': 'API service temporarily unavailable, please try again later',
    'invalid_city': 'Invalid city name, please check your input',
    'invalid_date': 'Invalid date format, please use YYYY-MM-DD format',
    'no_flights_found': 'No flights found matching the criteria',
    'rate_limit_exceeded': 'Request rate limit exceeded, please try again later',
    'internal_error': 'System internal error, please contact technical support',
    'timeout_error': 'Request timed out, please check your network connection',
    'validation_error': 'Input validation failed, please check parameters',
    'authentication_error': 'API authentication failed, please check configuration'
}

# Success messages in English
SUCCESS_MESSAGES = {
    'flights_found': 'Successfully found flight information',
    'data_cached': 'Data has been cached',
    'request_processed': 'Request processed successfully',
    'service_healthy': 'Service is running normally'
} 