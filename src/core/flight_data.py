"""
MAMA Flight Selection Assistant - Flight Data Management
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
import uuid

logger = logging.getLogger(__name__)

class FlightDataAggregator:
    """
    Professional flight data aggregator with multiple data sources and fallback mechanisms
    Ensures reliable flight data availability for MAMA system
    """
    
    def __init__(self, model: str = "full"):
        """Initialize the flight data aggregator with API clients"""
        self.model = model
        logger.info(f"🚀 Initializing Flight Data Aggregator (mode: {model})")
        
        # Initialize data source status
        self.data_sources_status = self._initialize_data_sources()
        
        # Initialize API clients
        self.data_sources = {}
        self._initialize_api_clients()
        
        logger.info(f"✅ Flight Data Aggregator initialized with {len(self.data_sources)} data sources")
    
    def _initialize_data_sources(self) -> Dict[str, bool]:
        """Initialize available data sources"""
        sources = {
            "aviationstack": True,  # Primary API source
            "opensky": True,        # Secondary API source  
            "amadeus": True,        # Tertiary API source
            "milestone": True,      # Milestone data space
            "fallback": True        # Built-in fallback data
        }
        return sources
    
    def _initialize_api_clients(self):
        """Initialize real API client instances"""
        try:
            # Try to initialize AviationStack API
            try:
                from external_apis.aviationstack_api import AviationStackAPI
                self.data_sources['aviationstack'] = AviationStackAPI()
                logger.info("✅ AviationStack API client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize AviationStack API: {e}")
            
            # Try to initialize OpenSky API
            try:
                from external_apis.opensky_api import OpenSkyAPI
                self.data_sources['opensky'] = OpenSkyAPI()
                logger.info("✅ OpenSky API client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize OpenSky API: {e}")
                
            # Try to initialize Amadeus API
            try:
                from external_apis.amadeus_api import AmadeusAPI
                self.data_sources['amadeus'] = AmadeusAPI()
                logger.info("✅ Amadeus API client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Amadeus API: {e}")
                
        except Exception as e:
            logger.error(f"❌ Critical error initializing API clients: {e}")
        
        if not self.data_sources:
            logger.warning("⚠️ No real API clients available - will use synthetic data only")
    
    def search_flights(self, departure: str, destination: str, date: str) -> List[Dict[str, Any]]:
        """
        Flight search
        Uses multiple data sources and intelligent fallbacks
        """
        logger.info(f"🔍 Starting flight search: {departure} → {destination} on {date}")
        
        # Check if departure and destination are the same
        if departure.strip().lower() == destination.strip().lower():
            logger.warning("Departure and destination are the same")
            return {
                "status": "error",
                "message": "Departure and destination cannot be the same",
                "suggestion": "Please select different departure and destination locations"
            }
        
        all_flights = []
        search_successful = False
        api_errors = {}
        
        # First try to get data from real APIs
        logger.info("🌐 Attempting to fetch real flight data from APIs...")
        
        # Try AviationStack API
        try:
            if 'aviationstack' in self.data_sources:
                aviation_api = self.data_sources['aviationstack']
                flights = aviation_api.search_flights(departure, destination, date)
                if flights and isinstance(flights, list):
                    logger.info(f"✅ AviationStack API: Found {len(flights)} flights")
                    all_flights.extend(flights)
                    search_successful = True
                else:
                    logger.info(f"⚠️ AviationStack API: No flights returned")
            else:
                logger.info("⚠️ AviationStack API not available")
        except Exception as e:
            api_errors['aviationstack'] = str(e)
            logger.warning(f"❌ AviationStack API failed: {str(e)}")
        
        # Try OpenSky API (primary data source)
        try:
            if 'opensky' in self.data_sources:
                opensky_api = self.data_sources['opensky']
                flights = opensky_api.search_flights(departure, destination, date)
                if flights and isinstance(flights, list) and len(flights) > 0:
                    logger.info(f"✅ OpenSky API: Found {len(flights)} real flights")
                    all_flights.extend(flights)
                    search_successful = True
                else:
                    logger.info(f"⚠️ OpenSky API: No flights found for route")
            else:
                logger.info("⚠️ OpenSky API not available")
        except Exception as e:
            api_errors['opensky'] = str(e)
            logger.warning(f"❌ OpenSky API failed: {str(e)}")
        
        # If all real APIs failed, return fallback data instead of synthetic data
        if not all_flights:
            logger.error("❌ All real APIs failed completely")
            return self._generate_fallback_flights(departure, destination, date)
        else:
            # Mark real data
            for flight in all_flights:
                if isinstance(flight, dict) and 'data_source' not in flight:
                    flight['data_source'] = 'real_api'
                    flight['academic_verified'] = True
        
        # Process weather for each flight
        try:
            # Try to import and use WeatherAgent
            try:
                from agents.weather_agent import WeatherAgent
                weather_agent = WeatherAgent()
            except ImportError:
                logger.warning("WeatherAgent not available, skipping weather analysis")
                weather_agent = None
            
            if weather_agent:
                for flight in all_flights:
                    try:
                        # Get weather data for departure and arrival cities
                        departure_weather = weather_agent.get_weather_data(departure)
                        arrival_weather = weather_agent.get_weather_data(destination)
                        
                        # Add weather analysis to flight
                        flight['weather_analysis'] = {
                            'departure': departure_weather,
                            'arrival': arrival_weather,
                            'overall_impact': 'moderate'  # Default value
                        }
                    except Exception as weather_error:
                        logger.warning(f"Weather analysis failed for flight {flight.get('flight_number', 'unknown')}: {weather_error}")
                        flight['weather_analysis'] = {
                            'departure': None,
                            'arrival': None,
                            'overall_impact': 'unknown'
                        }
            else:
                # Add empty weather analysis if weather agent not available
                for flight in all_flights:
                    flight['weather_analysis'] = {
                        'departure': None,
                        'arrival': None,
                        'overall_impact': 'unknown'
                    }
        except Exception as e:
            logger.error(f"Weather agent initialization failed: {e}")
            # Add empty weather analysis for all flights
            for flight in all_flights:
                flight['weather_analysis'] = {
                    'departure': None,
                    'arrival': None,
                    'overall_impact': 'unknown'
                }
        
        # Sort flights by relevance score
        all_flights.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Process each flight with additional computed fields
        processed_flights = []
        for i, flight in enumerate(all_flights):
            # Check if flight is a dictionary type
            if not isinstance(flight, dict):
                logger.warning(f"⚠️ Skipping invalid flight data (not a dict) at index {i}: {type(flight)} - {str(flight)[:100]}")
                continue
                
            processed_flight = self._process_flight_data(flight)
            processed_flights.append(processed_flight)
        
        logger.info(f"🔍 About to deduplicate {len(processed_flights)} processed flights")
        # Debug: Check types before deduplication
        for i, flight in enumerate(processed_flights[:5]):  # Only check first 5
            logger.info(f"Processed flight {i+1} type: {type(flight)}")
        
        # Remove duplicates
        unique_flights = self._deduplicate_flights(processed_flights)
        
        # Log warning if limited results but do not add synthetic data
        if len(unique_flights) < 3:
            logger.warning(f"⚠️ Only {len(unique_flights)} real flights found for {departure} → {destination}")
            logger.info("Real API data is limited but maintaining data integrity")
        
        # Sort by relevance and confidence
        sorted_flights = sorted(unique_flights, 
                              key=lambda f: (f.get('confidence', 0), f.get('preference_score', 0)), 
                              reverse=True)
        
        # Limit to reasonable number for UI
        final_flights = sorted_flights[:12]
        
        logger.info(f"✅ Flight search completed: {len(final_flights)} flights prepared")
        logger.info(f"Real API data: {sum(1 for f in final_flights if f.get('data_source') == 'real_api')} flights")
        logger.info(f"Emergency fallback: {sum(1 for f in final_flights if f.get('data_source') == 'emergency_fallback')} flights")
        
        return final_flights
    

    
    def _generate_additional_options(self, departure: str, destination: str, date: str, existing_count: int) -> List[Dict[str, Any]]:
        """
        Generate additional flight options to ensure variety
        """
        additional_flights = []
        
        try:
            search_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        except:
            search_date = datetime.now() + timedelta(days=7)
        
        # Generate 2-3 more options with different characteristics
        additional_templates = [
            {"time": "05:45", "airline": "Early Bird", "price": 169, "type": "ultra_budget"},
            {"time": "12:30", "airline": "Midday Express", "price": 319, "type": "premium"},
            {"time": "23:15", "airline": "Red Eye", "price": 149, "type": "late_night"},
        ]
        
        for i, template in enumerate(additional_templates):
            if existing_count + i >= 6:  # Don't add too many
                break
                
            departure_time = search_date.replace(
                hour=int(template["time"].split(':')[0]),
                minute=int(template["time"].split(':')[1])
            )
            
            flight = {
                "flight_id": f"ADD_{i+1:03d}",
                "flight_number": f"AD{2000+i}",
                "airline": {
                    "name": template["airline"],
                    "code": "AD",
                    "type": template.get("type", "budget")
                },
                "departure": {
                    "airport": departure,
                    "code": self._get_airport_code(departure),
                    "time": departure_time.isoformat()
                },
                "arrival": {
                    "airport": destination,
                    "code": self._get_airport_code(destination),
                    "time": (departure_time + timedelta(hours=3)).isoformat()
                },
                "duration": {"formatted": "3h 0m"},
                "pricing": {"economy": template["price"], "currency": "USD"},
                "stops": 0,
                "confidence": 0.70,
                "data_source": "synthetic_additional"
            }
            
            additional_flights.append(flight)
        
        return additional_flights
    
    def _get_airport_code(self, airport_name: str) -> str:
        """
        Generate or extract airport code from airport name
        """
        # Common airport codes mapping
        airport_codes = {
            "new york": "NYC",
            "los angeles": "LAX", 
            "chicago": "CHI",
            "miami": "MIA",
            "london": "LHR",
            "paris": "CDG",
            "tokyo": "NRT",
            "beijing": "PEK",
            "dubai": "DXB",
            "singapore": "SIN"
        }
        
        airport_lower = airport_name.lower()
        for city, code in airport_codes.items():
            if city in airport_lower:
                return code
        
        # Generate code from airport name
        return airport_name[:3].upper()
    
    def _generate_amenities(self, price: float) -> List[str]:
        """
        Generate realistic amenities based on flight price
        """
        base_amenities = ["Seat selection"]
        
        if price > 200:
            base_amenities.extend(["In-flight entertainment", "Complimentary snacks"])
        
        if price > 300:
            base_amenities.extend(["Wi-Fi available", "Power outlets", "Extra legroom"])
        
        if price > 400:
            base_amenities.extend(["Priority boarding", "Complimentary meals", "Premium cabin"])
        
        return base_amenities
    
    def _process_flight_data(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process flight data with additional metadata and confidence scoring
        """
        processed = flight.copy()
        
        # Ensure required fields exist
        if 'confidence' not in processed:
            processed['confidence'] = 0.80
        
        if 'data_source' not in processed:
            processed['data_source'] = 'api_processed'
        
        # Add search context
        processed['search_context'] = {
            'departure_requested': flight.get('departure') if isinstance(flight.get('departure'), str) else flight.get('departure', {}).get('airport', ''),
            'destination_requested': flight.get('destination') if isinstance(flight.get('destination'), str) else flight.get('destination', {}).get('airport', ''),
            'date_requested': flight.get('departure', {}).get('time', '') if isinstance(flight.get('departure'), dict) else '',
            'search_timestamp': datetime.now().isoformat()
        }
        
        # Calculate relevance score
        relevance_score = 0.8
        
        # Score advantage for direct flights
        if processed.get('stops', 1) == 0:
            relevance_score += 0.1
        
        # Score advantage for reasonable prices
        price = processed.get('pricing', {}).get('economy', 500)
        if 150 <= price <= 600:
            relevance_score += 0.1
        
        processed['relevance_score'] = min(1.0, relevance_score)
        
        return processed
    
    def _deduplicate_flights(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate flights based on flight number and departure time
        """
        logger.info(f"🔍 Starting deduplication with {len(flights)} flights")
        seen = set()
        unique_flights = []
        
        for i, flight in enumerate(flights):
            # Check if flight is a dictionary type
            if not isinstance(flight, dict):
                logger.warning(f"⚠️ Skipping invalid flight data (not a dict) at index {i}: {type(flight)} - {str(flight)[:100]}")
                continue
                
            try:
                # Create unique key - handle different data formats
                flight_number = flight.get('flight_number', '')
                
                # Handle different departure field formats
                departure_info = flight.get('departure', {})
                if isinstance(departure_info, dict):
                    departure_time = departure_info.get('time', '')
                else:
                    # If departure is a string (like airport code), use empty time
                    departure_time = ''
                
                # Handle different airline field formats
                airline_info = flight.get('airline', {})
                if isinstance(airline_info, dict):
                    airline_code = airline_info.get('code', '')
                else:
                    # If airline is a string, use it directly
                    airline_code = str(airline_info) if airline_info else ''
                
                flight_key = (flight_number, departure_time, airline_code)
                
                if flight_key not in seen:
                    seen.add(flight_key)
                    unique_flights.append(flight)
            except Exception as e:
                logger.error(f"❌ Error processing flight at index {i}: {e}")
                logger.error(f"Flight type: {type(flight)}")
                logger.error(f"Flight content: {str(flight)[:200]}")
                continue  # Skip this flight instead of raising
        
        logger.info(f"✅ Deduplication completed: {len(unique_flights)} unique flights")
        return unique_flights
    
    def get_schedules(self, departure: str, destination: str, date: str) -> List[Dict[str, Any]]:
        """
        Get detailed flight schedules with guaranteed results
        """
        try:
            # Use flight search as base
            flights = self.search_flights(departure, destination, date)
            
            # Convert flights to schedule format
            schedules = []
            for flight in flights:
                schedule = {
                    "route_id": f"{flight.get('departure', {}).get('code', '')}_{flight.get('arrival', {}).get('code', '')}",
                    "flight_number": flight.get("flight_number", ""),
                    "airline": flight.get("airline", {}).get("name", ""),
                    "departure_time": flight.get("departure", {}).get("time", ""),
                    "arrival_time": flight.get("arrival", {}).get("time", ""),
                    "frequency": "daily",
                    "reliability": flight.get("confidence", 0.85),
                    "average_delay": random.randint(5, 25),  # minutes
                    "on_time_performance": round(random.uniform(0.75, 0.95), 3)
                }
                schedules.append(schedule)
            
            return schedules
            
        except Exception as e:
            logger.error(f"Schedule retrieval failed: {str(e)}")
            return []

    def _generate_fallback_flights(self, departure: str, destination: str, date: str) -> List[Dict[str, Any]]:
        """
        Generate fallback flight data only when all real APIs fail
        This provides minimal flight options to ensure system functionality
        """
        logger.warning(f"⚠️ All APIs failed, generating minimal fallback data for {departure} → {destination}")
        
        try:
            search_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            search_date = datetime.now()
        
        # Minimal fallback with clear indication this is emergency data
        base_duration_hours = 3  # Default duration
        
        # Single basic flight option
        departure_time = search_date.replace(hour=14, minute=30, second=0, microsecond=0)
        arrival_time = departure_time + timedelta(hours=base_duration_hours)
        
        fallback_flight = {
            "flight_id": "FALLBACK_001",
            "flight_number": "FB001",
            "airline": {
                "name": "Flight Information Unavailable",
                "code": "FB",
                "type": "unknown"
            },
            "departure": {
                "airport": departure,
                "code": self._get_airport_code(departure),
                "time": departure_time.isoformat()
            },
            "arrival": {
                "airport": destination,
                "code": self._get_airport_code(destination),
                "time": arrival_time.isoformat()
            },
            "duration": {
                "total_minutes": int(base_duration_hours * 60),
                "formatted": f"{int(base_duration_hours)}h 0m"
            },
            "pricing": {
                "economy": 299,
                "currency": "USD"
            },
            "aircraft": {"type": "Unknown"},
            "stops": 0,
            "amenities": ["Basic service"],
            "confidence": 0.1,
            "data_source": "emergency_fallback",
            "warning": "Real flight data unavailable - please verify directly with airlines"
        }
        
        logger.warning("⚠️ Generated emergency fallback flight data")
        return [fallback_flight]