#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - Flight Search API Integration
Uses OpenSky Network for free flight data
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import os
from config import CONFIG

logger = logging.getLogger(__name__)

def search_flights_amadeus(origin: str, destination: str, date: str, passenger_count: int = 1, cabin_class: str = "economy") -> Dict[str, Any]:
    """
    Search flights using OpenSky Network data
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        date: Flight date (YYYY-MM-DD)
        passenger_count: Number of passengers
        cabin_class: Cabin class preference
        
    Returns:
        Dict with status and flight list
    """
    try:
        # Convert airport codes to coordinates (simplified)
        origin_coords = _get_airport_coordinates(origin)
        dest_coords = _get_airport_coordinates(destination)
        
        if not origin_coords or not dest_coords:
            logger.error("❌ Could not find airport coordinates")
            return {
                'status': 'error',
                'error': 'Could not find airport coordinates',
                'flights': [],
                'count': 0,
                'source': 'amadeus'
            }
        
        # Get live flights from OpenSky
        url = f"{CONFIG.apis['opensky']['base_url']}/states/all"
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"❌ OpenSky API request failed: {response.status_code}")
            return {
                'status': 'error',
                'error': f'API request failed: {response.status_code}',
                'flights': [],
                'count': 0,
                'source': 'amadeus'
            }
            
        data = response.json()
        flights = []
        
        # Filter and transform flight data
        for state in data['states']:
            if state[5] and state[6]:  # Has coordinates
                flight = {
                    'flight_number': state[1].strip() if state[1] else 'Unknown',
                    'airline_code': state[1][:3] if state[1] else 'Unknown',
                    'aircraft_type': state[8] if state[8] else 'Unknown',
                    'departure': origin,
                    'destination': destination,
                    'departure_time': datetime.fromtimestamp(state[3]).isoformat() if state[3] else None,
                    'arrival_time': None,  # Not available in free data
                    'price': None,  # Not available in free data
                    'seats_available': None,  # Not available in free data
                    'flight_duration': None,  # Not available in free data
                    'data_source': 'amadeus'
                }
                flights.append(flight)
        
        logger.info(f"✅ Found {len(flights)} flights from {origin} to {destination}")
        return {
            'status': 'success',
            'flights': flights,
            'count': len(flights),
            'source': 'amadeus'
        }
        
    except Exception as e:
        logger.error(f"❌ Flight search failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'flights': [],
            'count': 0,
            'source': 'amadeus'
        }

def _get_airport_coordinates(airport_code: str) -> Optional[Dict[str, float]]:
    """Get airport coordinates (simplified implementation)"""
    # Simplified airport database
    airports = {
        'JFK': {'lat': 40.6413, 'lon': -73.7781},
        'LAX': {'lat': 33.9416, 'lon': -118.4085},
        'ORD': {'lat': 41.9742, 'lon': -87.9073},
        'LHR': {'lat': 51.4700, 'lon': -0.4543},
        'CDG': {'lat': 49.0097, 'lon': 2.5479},
        'HKG': {'lat': 22.3080, 'lon': 113.9185},
        'SIN': {'lat': 1.3644, 'lon': 103.9915},
        'DXB': {'lat': 25.2532, 'lon': 55.3657},
        'PEK': {'lat': 40.0799, 'lon': 116.6031},
        'NRT': {'lat': 35.7720, 'lon': 140.3929}
    }
    return airports.get(airport_code.upper())

# Export main functions
__all__ = ['search_flights_amadeus'] 