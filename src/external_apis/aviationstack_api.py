#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAMA Flight Selection Assistant - AviationStack API Integration
Real AviationStack API Implementation for Flight Data
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from config import CONFIG
import time

logger = logging.getLogger(__name__)

class AviationStackAPI:
    """AviationStack Flight Data API with rate limiting and retry logic"""
    
    def __init__(self):
        self.api_key = CONFIG.get("apis.flight.api_key")
        self.base_url = CONFIG.get("apis.flight.base_url")
        self.timeout = CONFIG.get("apis.flight.timeout", 30)
        self.retry_attempts = CONFIG.get("apis.flight.retry_attempts", 3)
        self.last_request_time = 0
        self.min_interval = 1.0  # Minimum 1 second between requests
        
        if not self.api_key:
            raise ValueError("Flight API key not configured")
        
        logger.info("✈️ AviationStack API initialized")
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        
    def _make_api_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        self._wait_for_rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded, waiting 5 seconds...")
                time.sleep(5)
                # Try once more after waiting
                self._wait_for_rate_limit()
                response = requests.get(url, params=params, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ AviationStack API request failed: {e}")
            return None
    
    def search_flights(self, departure: str, destination: str, date: str = None) -> List[Dict[str, Any]]:
        """Search for flights using AviationStack API"""
        try:
            # Use flights endpoint for real-time flight data
            url = f"{self.base_url}/flights"
            
            params = {
                'access_key': self.api_key,
                'dep_iata': self._get_airport_code(departure),
                'arr_iata': self._get_airport_code(destination),
                'limit': 50
            }
            
            # Add date filter if provided
            if date:
                try:
                    flight_date = datetime.strptime(date, '%Y-%m-%d')
                    params['flight_date'] = flight_date.strftime('%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid date format: {date}, searching without date filter")
            
            for attempt in range(self.retry_attempts):
                try:
                    data = self._make_api_request('flights', params)
                    
                    if 'data' not in data:
                        logger.warning(f"No flight data returned from API")
                        return []
                    
                    flights = []
                    for flight_data in data['data']:
                        flight = self._parse_flight_data(flight_data)
                        if flight:
                            flights.append(flight)
                    
                    logger.info(f"✅ Retrieved {len(flights)} flights from {departure} to {destination}")
                    return flights
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️ Flight API attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_attempts - 1:
                        raise
                        
        except Exception as e:
            logger.error(f"❌ Failed to search flights: {e}")
            raise
    
    def get_flight_status(self, flight_number: str, airline: str = None) -> Dict[str, Any]:
        """Get real-time flight status"""
        try:
            url = f"{self.base_url}/flights"
            
            params = {
                'access_key': self.api_key,
                'flight_iata': flight_number,
                'limit': 10
            }
            
            if airline:
                params['airline_iata'] = airline
            
            response = self._make_api_request('flights', params)
            
            if 'data' not in response or not response['data']:
                logger.warning(f"No status data for flight {flight_number}")
                return {}
            
            # Return the most recent flight data
            flight_data = response['data'][0]
            status = self._parse_flight_status(flight_data)
            
            logger.info(f"✅ Retrieved status for flight {flight_number}")
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get flight status for {flight_number}: {e}")
            raise
    
    def get_airline_info(self, airline_code: str) -> Dict[str, Any]:
        """Get airline information"""
        try:
            url = f"{self.base_url}/airlines"
            
            params = {
                'access_key': self.api_key,
                'airline_iata': airline_code
            }
            
            response = self._make_api_request('airlines', params)
            
            if 'data' not in response or not response['data']:
                logger.warning(f"No airline data for {airline_code}")
                return {}
            
            airline_data = response['data'][0]
            airline_info = {
                'airline_name': airline_data.get('airline_name', ''),
                'iata_code': airline_data.get('iata_code', ''),
                'icao_code': airline_data.get('icao_code', ''),
                'country_name': airline_data.get('country_name', ''),
                'country_iso2': airline_data.get('country_iso2', ''),
                'fleet_size': airline_data.get('fleet_size', 0),
                'hub_code': airline_data.get('hub_code', ''),
                'status': airline_data.get('status', 'active')
            }
            
            logger.info(f"✅ Retrieved airline info for {airline_code}")
            return airline_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get airline info for {airline_code}: {e}")
            raise
    
    def get_airport_info(self, airport_code: str) -> Dict[str, Any]:
        """Get airport information"""
        try:
            url = f"{self.base_url}/airports"
            
            params = {
                'access_key': self.api_key,
                'airport_iata': airport_code
            }
            
            response = self._make_api_request('airports', params)
            
            if 'data' not in response or not response['data']:
                logger.warning(f"No airport data for {airport_code}")
                return {}
            
            airport_data = response['data'][0]
            airport_info = {
                'airport_name': airport_data.get('airport_name', ''),
                'iata_code': airport_data.get('iata_code', ''),
                'icao_code': airport_data.get('icao_code', ''),
                'country_name': airport_data.get('country_name', ''),
                'country_iso2': airport_data.get('country_iso2', ''),
                'city_iata_code': airport_data.get('city_iata_code', ''),
                'latitude': airport_data.get('latitude', 0),
                'longitude': airport_data.get('longitude', 0),
                'timezone': airport_data.get('timezone', ''),
                'gmt_offset': airport_data.get('gmt_offset', 0)
            }
            
            logger.info(f"✅ Retrieved airport info for {airport_code}")
            return airport_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get airport info for {airport_code}: {e}")
            raise
    
    def _parse_flight_data(self, flight_data: Dict) -> Optional[Dict[str, Any]]:
        """Parse raw flight data from AviationStack API"""
        try:
            departure = flight_data.get('departure', {})
            arrival = flight_data.get('arrival', {})
            flight = flight_data.get('flight', {})
            aircraft = flight_data.get('aircraft', {})
            airline = flight_data.get('airline', {})
            
            # Calculate duration if both times are available
            duration_minutes = None
            if departure.get('scheduled') and arrival.get('scheduled'):
                try:
                    dep_time = datetime.fromisoformat(departure['scheduled'].replace('Z', '+00:00'))
                    arr_time = datetime.fromisoformat(arrival['scheduled'].replace('Z', '+00:00'))
                    duration_minutes = int((arr_time - dep_time).total_seconds() / 60)
                except:
                    pass
            
            # Estimate price based on flight characteristics
            estimated_price = self._estimate_flight_price(
                departure.get('iata', ''),
                arrival.get('iata', ''),
                duration_minutes,
                airline.get('name', ''),
                aircraft.get('iata', '')
            )
            
            parsed_flight = {
                'flight_id': f"{flight.get('iata', '')}-{flight.get('number', '')}-{datetime.now().strftime('%Y%m%d')}",
                'flight_number': flight.get('iata', ''),
                'airline': {
                    'name': airline.get('name', ''),
                    'iata': airline.get('iata', ''),
                    'icao': airline.get('icao', '')
                },
                'aircraft': {
                    'type': aircraft.get('iata', ''),
                    'registration': aircraft.get('registration', '')
                },
                'departure': {
                    'airport': departure.get('airport', ''),
                    'iata': departure.get('iata', ''),
                    'icao': departure.get('icao', ''),
                    'terminal': departure.get('terminal', ''),
                    'gate': departure.get('gate', ''),
                    'scheduled_time': departure.get('scheduled', ''),
                    'estimated_time': departure.get('estimated', ''),
                    'actual_time': departure.get('actual', ''),
                    'delay': departure.get('delay', 0),
                    'timezone': departure.get('timezone', '')
                },
                'arrival': {
                    'airport': arrival.get('airport', ''),
                    'iata': arrival.get('iata', ''),
                    'icao': arrival.get('icao', ''),
                    'terminal': arrival.get('terminal', ''),
                    'gate': arrival.get('gate', ''),
                    'scheduled_time': arrival.get('scheduled', ''),
                    'estimated_time': arrival.get('estimated', ''),
                    'actual_time': arrival.get('actual', ''),
                    'delay': arrival.get('delay', 0),
                    'timezone': arrival.get('timezone', '')
                },
                'status': flight_data.get('flight_status', 'scheduled'),
                'duration_minutes': duration_minutes,
                'estimated_price': estimated_price,
                'data_source': 'aviationstack_api',
                'last_updated': datetime.now().isoformat()
            }
            
            return parsed_flight
            
        except Exception as e:
            logger.error(f"Failed to parse flight data: {e}")
            return None
    
    def _parse_flight_status(self, flight_data: Dict) -> Dict[str, Any]:
        """Parse flight status data"""
        try:
            departure = flight_data.get('departure', {})
            arrival = flight_data.get('arrival', {})
            flight = flight_data.get('flight', {})
            
            status = {
                'flight_number': flight.get('iata', ''),
                'status': flight_data.get('flight_status', 'unknown'),
                'departure': {
                    'airport': departure.get('airport', ''),
                    'scheduled': departure.get('scheduled', ''),
                    'estimated': departure.get('estimated', ''),
                    'actual': departure.get('actual', ''),
                    'delay': departure.get('delay', 0)
                },
                'arrival': {
                    'airport': arrival.get('airport', ''),
                    'scheduled': arrival.get('scheduled', ''),
                    'estimated': arrival.get('estimated', ''),
                    'actual': arrival.get('actual', ''),
                    'delay': arrival.get('delay', 0)
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to parse flight status: {e}")
            return {}
    
    def _estimate_flight_price(self, dep_iata: str, arr_iata: str, duration_minutes: int, 
                              airline: str, aircraft: str) -> Dict[str, float]:
        """Estimate flight prices based on route and characteristics"""
        try:
            # Base price calculation
            base_price = 100  # Base price in USD
            
            # Duration-based pricing
            if duration_minutes:
                base_price += (duration_minutes / 60) * 50  # $50 per hour
            
            # Route-based adjustments
            route_multiplier = self._get_route_multiplier(dep_iata, arr_iata)
            base_price *= route_multiplier
            
            # Airline-based adjustments
            airline_multiplier = self._get_airline_multiplier(airline)
            base_price *= airline_multiplier
            
            # Aircraft-based adjustments
            aircraft_multiplier = self._get_aircraft_multiplier(aircraft)
            base_price *= aircraft_multiplier
            
            # Generate price range
            economy_price = base_price
            business_price = economy_price * 3.5
            first_price = economy_price * 6.0
            
            return {
                'economy': round(economy_price, 2),
                'business': round(business_price, 2),
                'first': round(first_price, 2),
                'currency': 'USD'
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate flight price: {e}")
            return {'economy': 200.0, 'business': 700.0, 'first': 1200.0, 'currency': 'USD'}
    
    def _get_route_multiplier(self, dep_iata: str, arr_iata: str) -> float:
        """Get price multiplier based on route popularity and distance"""
        # Major international routes
        major_routes = ['JFK', 'LAX', 'LHR', 'CDG', 'NRT', 'PEK', 'PVG', 'HKG', 'SIN', 'DXB']
        
        if dep_iata in major_routes or arr_iata in major_routes:
            return 1.3
        
        # Regional routes
        return 1.0
    
    def _get_airline_multiplier(self, airline: str) -> float:
        """Get price multiplier based on airline type"""
        if not airline:
            return 1.0
            
        airline_lower = airline.lower()
        
        # Premium airlines
        premium_airlines = ['emirates', 'singapore', 'cathay', 'lufthansa', 'british airways']
        if any(premium in airline_lower for premium in premium_airlines):
            return 1.4
        
        # Budget airlines
        budget_airlines = ['ryanair', 'easyjet', 'southwest', 'jetblue', 'spirit']
        if any(budget in airline_lower for budget in budget_airlines):
            return 0.7
        
        return 1.0
    
    def _get_aircraft_multiplier(self, aircraft: str) -> float:
        """Get price multiplier based on aircraft type"""
        if not aircraft:
            return 1.0
        
        aircraft_upper = aircraft.upper()
        
        # Wide-body aircraft (typically more expensive)
        widebody_aircraft = ['A330', 'A340', 'A350', 'A380', 'B747', 'B767', 'B777', 'B787']
        if any(wb in aircraft_upper for wb in widebody_aircraft):
            return 1.2
        
        # Regional aircraft (typically cheaper)
        regional_aircraft = ['CRJ', 'ERJ', 'ATR', 'DH8']
        if any(reg in aircraft_upper for reg in regional_aircraft):
            return 0.8
        
        return 1.0
    
    def _get_airport_code(self, location: str) -> str:
        """Convert city name or airport name to IATA code"""
        # Common city to airport code mappings
        city_to_iata = {
            'beijing': 'PEK',
            'shanghai': 'PVG',
            'guangzhou': 'CAN',
            'shenzhen': 'SZX',
            'hong kong': 'HKG',
            'tokyo': 'NRT',
            'osaka': 'KIX',
            'seoul': 'ICN',
            'singapore': 'SIN',
            'bangkok': 'BKK',
            'kuala lumpur': 'KUL',
            'jakarta': 'CGK',
            'manila': 'MNL',
            'new york': 'JFK',
            'los angeles': 'LAX',
            'chicago': 'ORD',
            'london': 'LHR',
            'paris': 'CDG',
            'frankfurt': 'FRA',
            'amsterdam': 'AMS',
            'dubai': 'DXB',
            'doha': 'DOH',
            'istanbul': 'IST',
            'moscow': 'SVO',
            'sydney': 'SYD',
            'melbourne': 'MEL'
        }
        
        location_lower = location.lower().strip()
        
        # Check if it's already an IATA code (3 letters)
        if len(location) == 3 and location.isalpha():
            return location.upper()
        
        # Look up city name
        return city_to_iata.get(location_lower, location_lower[:3].upper())

# Global AviationStack API instance
aviation_api = AviationStackAPI()

def search_flights_aviationstack(departure: str, destination: str, date: str = None) -> Dict[str, Any]:
    """Search for flights using AviationStack API"""
    try:
        flights = aviation_api.search_flights(departure, destination, date)
        return {
            'status': 'success',
            'flights': flights,
            'count': len(flights),
            'source': 'aviationstack'
        }
    except Exception as e:
        logger.error(f"Failed to search flights: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'flights': [],
            'count': 0,
            'source': 'aviationstack'
        }

def get_flight_status(flight_number: str, airline: str = None) -> Dict[str, Any]:
    """Get real-time flight status"""
    try:
        return aviation_api.get_flight_status(flight_number, airline)
    except Exception as e:
        logger.error(f"Failed to get flight status: {e}")
        raise

def get_airline_information(airline_code: str) -> Dict[str, Any]:
    """Get airline information"""
    try:
        return aviation_api.get_airline_info(airline_code)
    except Exception as e:
        logger.error(f"Failed to get airline information: {e}")
        raise

def get_airport_information(airport_code: str) -> Dict[str, Any]:
    """Get airport information"""
    try:
        return aviation_api.get_airport_info(airport_code)
    except Exception as e:
        logger.error(f"Failed to get airport information: {e}")
        raise

if __name__ == "__main__":
    # Test the API
    import sys
    
    if len(sys.argv) >= 3:
        dep = sys.argv[1]
        dest = sys.argv[2]
        date_param = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"🧪 Testing AviationStack API: {dep} → {dest}")
        result = search_flights_aviationstack(dep, dest, date_param)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python aviationstack_api.py <departure> <destination> [date]")
        print("Example: python aviationstack_api.py Beijing Shanghai 2025-01-15") 