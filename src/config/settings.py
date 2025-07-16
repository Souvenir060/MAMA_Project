#!/usr/bin/env python3
"""
MAMA Framework Configuration Settings
System-wide configuration parameters for the MAMA multi-agent system
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

@dataclass
class SystemConfiguration:
    """Core system configuration"""
    
    debug_mode: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    request_timeout: float = 30.0
    
    data_directory: str = "data"
    results_directory: str = "results"
    figures_directory: str = "figures"
    logs_directory: str = "logs"
    
    enable_caching: bool = True
    cache_expiry_hours: int = 24
    max_cache_size_mb: int = 100
    
    random_seed: int = 42
    
    def __post_init__(self):
        for directory in [self.data_directory, self.results_directory, 
                         self.figures_directory, self.logs_directory]:
            Path(directory).mkdir(exist_ok=True)

@dataclass
class ExperimentConfiguration:
    """Experiment-specific configuration"""
    
    max_iterations: int = 150
    batch_size: int = 32
    learning_rate: float = 0.001
    
    trust_threshold: float = 0.5
    similarity_threshold: float = 0.7
    
    enable_marl: bool = True
    enable_trust_system: bool = True
    enable_sbert: bool = True
    enable_ltr: bool = True
    
    save_intermediate_results: bool = True
    generate_plots: bool = True
    
    performance_metrics: List[str] = field(default_factory=lambda: [
        'mrr', 'ndcg_5', 'ndcg_10', 'precision_5', 'recall_5'
    ])

@dataclass
class ModelConfiguration:
    """Model-specific configuration"""
    
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    marl_state_size: int = 10
    marl_action_size: int = 5
    marl_epsilon: float = 0.1
    marl_gamma: float = 0.95
    
    trust_dimensions: List[str] = field(default_factory=lambda: [
        'reliability', 'competence', 'fairness', 'security', 'transparency'
    ])
    
    trust_weights: Dict[str, float] = field(default_factory=lambda: {
        'reliability': 0.25,
        'competence': 0.25,
        'fairness': 0.20,
        'security': 0.15,
        'transparency': 0.15
    })

@dataclass
class AgentConfiguration:
    """Agent system configuration"""
    
    max_response_length: int = 2000
    response_timeout_seconds: int = 30
    
    max_conversation_history: int = 50
    context_window_size: int = 10
    
    default_search_limit: int = 20
    max_search_results: int = 100
    
    temperature: float = 0.7
    max_tokens: int = 1500
    
    supported_languages: List[str] = field(default_factory=lambda: ['en-US'])
    default_language: str = 'en-US'

@dataclass
class APIConfiguration:
    """External API configuration"""
    
    amadeus_api_key: Optional[str] = None
    amadeus_api_secret: Optional[str] = None
    amadeus_base_url: str = "https://api.amadeus.com"
    
    opensky_username: Optional[str] = None
    opensky_password: Optional[str] = None
    opensky_base_url: str = "https://opensky-network.org/api"
    
    weather_api_key: Optional[str] = None
    weather_base_url: str = "https://api.openweathermap.org/data/2.5"
    
    safety_api_key: Optional[str] = None
    safety_base_url: str = "https://api.aviation-safety.net"
    
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    enable_ssl_verification: bool = True
    request_timeout_seconds: float = 30.0

@dataclass
class SecurityConfiguration:
    """Security and privacy configuration"""
    
    enable_encryption: bool = True
    encryption_key_length: int = 256
    
    enable_authentication: bool = False
    jwt_secret_key: Optional[str] = None
    jwt_expiry_hours: int = 24
    
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    enable_logging: bool = True
    log_sensitive_data: bool = False
    
    allowed_origins: List[str] = field(default_factory=lambda: ['*'])
    
    data_retention_days: int = 30
    anonymize_user_data: bool = True

@dataclass
class DatabaseConfiguration:
    """Database configuration"""
    
    database_type: str = "sqlite"
    database_path: str = "data/mama_system.db"
    
    connection_pool_size: int = 10
    connection_timeout_seconds: float = 30.0
    
    enable_migrations: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    query_timeout_seconds: float = 30.0
    max_query_results: int = 1000

@dataclass
class UIConfiguration:
    """User interface configuration"""
    
    theme: str = "light"
    language: str = "en"
    
    enable_dark_mode: bool = False
    enable_animations: bool = True
    
    results_per_page: int = 20
    max_search_history: int = 100
    
    enable_auto_complete: bool = True
    enable_spell_check: bool = True
    
    map_provider: str = "openstreetmap"
    default_map_zoom: int = 10

@dataclass
class PerformanceConfiguration:
    """Performance optimization configuration"""
    
    enable_caching: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    
    enable_compression: bool = True
    compression_level: int = 6
    
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 4
    
    enable_gpu_acceleration: bool = False
    gpu_memory_limit_mb: int = 2048
    
    enable_profiling: bool = False
    profiling_output_directory: str = "profiling"

@dataclass
class TestConfiguration:
    """Testing configuration"""
    
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = False
    
    test_data_directory: str = "tests/data"
    test_results_directory: str = "tests/results"
    
    mock_external_apis: bool = True
    test_timeout_seconds: float = 60.0
    
    generate_test_reports: bool = True
    test_coverage_threshold: float = 0.8

settings = SystemConfiguration()
experiment_config = ExperimentConfiguration()
model_config = ModelConfiguration()
agent_config = AgentConfiguration()
api_config = APIConfiguration()
security_config = SecurityConfiguration()
database_config = DatabaseConfiguration()
ui_config = UIConfiguration()
performance_config = PerformanceConfiguration()
test_config = TestConfiguration()

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
    'Xian': {'code': 'XIY', 'iata': 'XIY', 'country': 'CN'},
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

ERROR_MESSAGES = {
    'api_unavailable': 'API service temporarily unavailable, please try again later',
    'invalid_city': 'Invalid city name, please check input',
    'invalid_date': 'Invalid date format, please use YYYY-MM-DD format',
    'no_flights_found': 'No flights found matching the criteria',
    'rate_limit_exceeded': 'Too many requests, please try again later',
    'internal_error': 'Internal system error, please contact technical support',
    'timeout_error': 'Request timeout, please check network connection',
    'validation_error': 'Input validation failed, please check parameters',
    'authentication_error': 'API authentication failed, please check configuration'
}

SUCCESS_MESSAGES = {
    'flights_found': 'Successfully found flight information',
    'data_cached': 'Data has been cached',
    'request_processed': 'Request processed successfully',
    'service_healthy': 'Service is running normally'
}

def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Configuration file not found: {config_file}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        return {}

def save_config_to_file(config: Dict[str, Any], config_file: str) -> bool:
    """Save configuration to JSON file"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}")
        return False

def get_config_value(key: str, default: Any = None) -> Any:
    """Get configuration value with fallback"""
    try:
        return getattr(settings, key, default)
    except AttributeError:
        return default

def validate_configuration() -> List[str]:
    """Validate system configuration"""
    errors = []
    
    if not os.path.exists(settings.data_directory):
        errors.append(f"Data directory does not exist: {settings.data_directory}")
    
    if not os.path.exists(settings.results_directory):
        errors.append(f"Results directory does not exist: {settings.results_directory}")
    
    if experiment_config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if experiment_config.trust_threshold < 0 or experiment_config.trust_threshold > 1:
        errors.append("Trust threshold must be between 0 and 1")
    
    return errors

def setup_logging():
    """Setup system logging"""
    log_level = getattr(logging, settings.log_level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{settings.logs_directory}/mama_system.log"),
            logging.StreamHandler()
        ]
    )
    
    if settings.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)

def initialize_system():
    """Initialize the MAMA system"""
    setup_logging()
    
    errors = validate_configuration()
    if errors:
        for error in errors:
            logging.error(error)
        raise ValueError(f"Configuration validation failed: {errors}")
    
    logging.info("MAMA system configuration loaded successfully")
    return True 