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

# Fix 2: Load JWT token from file
def _load_jwt_token() -> str:
    """Load JWT token from token.jwt file"""
    token_file = Path('token.jwt')
    if not token_file.exists():
        raise FileNotFoundError("token.jwt not found in MAMA project directory! Please copy it from the Milestone project.")
    with open(token_file, 'r') as f:
        return f.read().strip()

try:
    JWT_TOKEN = _load_jwt_token()
except FileNotFoundError as e:
    logging.error(f"JWT Token loading failed: {e}")
    JWT_TOKEN = ""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_from_milestone(
    entity_type: str,
    query_params: Optional[Dict] = None,
    is_realtime: bool = False
) -> List[Dict]:
    """
    Reads data from the Milestone NGSI-LD data space.

    Args:
        entity_type (str): The type of entity to query (e.g., 'Flight', 'Weather', 'Airline').
        query_params (dict, optional): A dictionary of query parameters (e.g., 'q', 'attrs').
        is_realtime (bool): If True, queries the real-time data proxy. Otherwise, queries the main broker.

    Returns:
        list: A list of entity dictionaries if successful, otherwise an empty list.
    """
    base_url = MILESTONE_REALTIME_URL if is_realtime else MILESTONE_URL
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JWT_TOKEN}",  # Fix 3: Also include authentication header in read requests
        "Link": f'<{CONTEXT_URL}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }

    # Standardize query parameters
    params = {'type': entity_type}
    if query_params:
        params.update(query_params)

    try:
        logging.info(f"Reading from Milestone: URL={base_url}, Params={params}")
        response = requests.get(base_url, headers=headers, params=params, proxies=PROXIES, timeout=30)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            logging.warning(f"Unexpected data format from Milestone: {data}")
            return []

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to read from Milestone. Error: {e}")
        return []

def write_to_milestone(entity_type: str, entity_id: str, data: Dict) -> bool:
    """
    Writes or updates an entity in the Milestone NGSI-LD data space using the protected endpoint.

    Args:
        entity_type (str): The type of the entity (e.g., 'Recommendation').
        entity_id (str): A unique ID for the entity (e.g., 'urn:ngsi-ld:Recommendation:1234').
        data (dict): The NGSI-LD formatted entity data to write.

    Returns:
        bool: True if the write operation was successful, False otherwise.
    """
    url = f"{PROTECTED_URL}/{entity_id}/attrs"
    
    headers = {
        "Content-Type": "application/ld+json",  # Fix 4: Use correct Content-Type
        "Authorization": f"Bearer {JWT_TOKEN}",
        # Fix 6: When using application/ld+json, do not use Link header simultaneously
        # "Link": f'<{CONTEXT_URL}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }

    # Ensure data conforms to NGSI-LD structure
    if 'id' not in data or 'type' not in data:
        logging.error("Data for writing must contain 'id' and 'type' keys.")
        return False
        
    try:
        logging.info(f"Writing to Milestone: URL={PROTECTED_URL}, EntityID={entity_id}")
        # First, try to create the entity. If it exists, a 409 will be returned.
        create_response = requests.post(PROTECTED_URL, headers=headers, data=json.dumps(data), proxies=PROXIES, timeout=30)

        if create_response.status_code == 201:
            logging.info(f"Successfully created entity {entity_id}")
            return True
        elif create_response.status_code == 409: # Conflict, entity already exists
            logging.warning(f"Entity {entity_id} already exists. Attempting to update.")
            # We need to remove id and type for patching attributes
            patch_data = {k: v for k, v in data.items() if k not in ['id', 'type']}
            update_response = requests.patch(url, headers=headers, data=json.dumps(patch_data), proxies=PROXIES, timeout=30)
            update_response.raise_for_status()
            logging.info(f"Successfully updated entity {entity_id}")
            return True
        else:
            create_response.raise_for_status()
            return False

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to write to Milestone for entity {entity_id}. Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response Status Code: {e.response.status_code}")
            logging.error(f"Response Headers: {e.response.headers}")
            logging.error(f"Response Body: {e.response.text}")
        return False
def query_realtime_proxy(entity_type: str, query_params: Optional[Dict] = None) -> List[Dict]:
    """
    Query the real-time data proxy to get real-time entity data
    
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
    # This function would normally query the Milestone database
    # For academic simplicity, we use a static mapping
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
    
    Returns:
        Whether the connection is working
    """
    try:
        # Fix 5: Use correct port and authentication header to validate connection
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Just try to access the types endpoint which should always exist
        response = requests.get(f"{MILESTONE_URL}/types", 
                              headers=headers, 
                              proxies=PROXIES, 
                              timeout=10)
        
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Failed to validate Milestone connection: {e}")
        return False
