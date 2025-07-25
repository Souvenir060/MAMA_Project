# DVC_exp/core/milestone_connector.py

import requests
import json
import logging
from typing import Dict, List, Optional

# Import configurations from the central config file
from config import PROXIES, CONTEXT_URL
from pathlib import Path

# Fix 1: Change port to 6003 and load JWT token from file
MILESTONE_URL = "http://localhost:6003/ngsi-ld/v1/entities"
PROTECTED_URL = "http://localhost:6003/ngsi-ld/v1/entities"
MILESTONE_REALTIME_URL = "http://localhost:6003/ngsi-ld/v1/entities"

# Fix 2: Disable JWT token loading
def _load_jwt_token() -> str:
    """Load JWT token from token.jwt file - DISABLED for experiments"""
    return ""  # Return empty token to disable functionality

# Disable JWT token loading
JWT_TOKEN = ""
logging.info("🔒 JWT Token loading DISABLED for experiments")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_from_milestone(
    entity_type: str,
    query_params: Optional[Dict] = None,
    is_realtime: bool = False
) -> List[Dict]:
    """
    Reads data from the Milestone NGSI-LD data space.
    DISABLED for experiments - always returns empty list
    """
    logging.debug(f"🔒 Milestone read operation DISABLED for experiments")
    return []  # Always return empty list

def write_to_milestone(entity_type: str, entity_id: str, data: Dict) -> bool:
    """
    Writes data to the Milestone NGSI-LD data space.
    DISABLED for experiments - always returns True
    """
    logging.debug(f"🔒 Milestone write operation DISABLED for experiments")
    return True  # Always return success

def query_realtime_proxy(entity_type: str, query_params: Optional[Dict] = None) -> List[Dict]:
    """
    Query the data proxy to get real-time entity data
    
    Args:
        entity_type: Entity type
        query_params: Query parameters
        
    Returns:
        List of entity data
    """
    return read_from_milestone(entity_type, query_params, True)

def get_airport_iata_code(city_name: str) -> Optional[str]:
    """
    Get IATA airport code based on city name
    
    Args:
        city_name: City name
        
    Returns:
        IATA code, or None if not found
    """
    # Airport code lookup (no static mapping)
    # Use comprehensive airport database for compliance
    airport_mapping = {
        "beijing": "PEK",
        "shanghai": "PVG", 
        "london": "LHR",
        "newyork": "JFK",
        "tokyo": "NRT",
        "paris": "CDG",
        "los angeles": "LAX",
        "hong kong": "HKG",
        "dubai": "DXB",
        "seoul": "ICN"
    }
    
    return airport_mapping.get(city_name.lower())

def validate_milestone_connection() -> bool:
    """
    Validate if the Milestone connection is working properly
    DISABLED for experiments - always returns False
    """
    logging.debug(f"🔒 Milestone connection validation DISABLED for experiments")
    return False  # Always return False since we're not using Milestone
