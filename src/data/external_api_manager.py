"""
MAMA Flight Assistant - External API Manager

Central coordinator for all external data services including flight data,
weather information, and other third-party API integrations.
"""

import logging
import requests
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import FLIGHT_API_KEY, WEATHER_API_KEY, PROXIES
except ImportError:
    # Fallback configuration
    FLIGHT_API_KEY = "10fa3e346f7606412b86a9cbde4b00a4"
    WEATHER_API_KEY = "498ae38fb9831291de1d0432ea2fdf07"
    PROXIES = {'http': None, 'https': None}

logger = logging.getLogger(__name__)

class ExternalAPIManager:
    """
    Central manager for all external API communications
    
    This class coordinates data retrieval from multiple external sources
    and provides unified interfaces for the agent system.
    """
    
    def __init__(self):
        """Initialize the External API Manager"""
        self.flight_api_key = FLIGHT_API_KEY
        self.weather_api_key = WEATHER_API_KEY
        self.proxies = PROXIES
        
        # API base URLs
        self.aviationstack_base_url = "http://api.aviationstack.com/v1"
        self.amap_weather_base_url = "https://restapi.amap.com/v3/weather"
        
        # Rate limiting and caching
        self.request_cache = {}
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        logger.info("External API Manager initialized with live data sources")
    
    def _make_request(self, url: str, params: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with rate limiting and error handling
        
        Args:
            url: Request URL
            params: Request parameters
            timeout: Request timeout in seconds
            
        Returns:
            Response data as dictionary or None if failed
        """
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            
            # Make request
            logger.debug(f"Making API request to: {url}")
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                proxies=self.proxies
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def get_flight_data(self, departure: str, destination: str, date: str) -> Dict[str, Any]:
        """
        Get flight data from Aviationstack API
        
        Args:
            departure: Departure city/airport
            destination: Destination city/airport
            date: Flight date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing flight data and metadata
        """
        try:
            # Convert city names to airport codes
            departure_code = self._get_airport_code(departure)
            destination_code = self._get_airport_code(destination)
            
            # Prepare API parameters
            params = {
                'access_key': self.flight_api_key,
                'dep_iata': departure_code,
                'arr_iata': destination_code,
                'flight_date': date,
                'limit': 50
            }
            
            # Check cache first
            cache_key = f"flights_{departure_code}_{destination_code}_{date}"
            if cache_key in self.request_cache:
                cache_time, cached_data = self.request_cache[cache_key]
                if time.time() - cache_time < 3600:  # 1 hour cache
                    logger.info("Returning cached flight data")
                    return cached_data
            
            # Make API request
            url = f"{self.aviationstack_base_url}/flights"
            response_data = self._make_request(url, params)
            
            if response_data and 'data' in response_data:
                processed_data = self._process_flight_data(response_data['data'])
                
                # Cache the result
                self.request_cache[cache_key] = (time.time(), processed_data)
                
                logger.info(f"Retrieved {len(processed_data.get('flights', []))} flights from Aviationstack API")
                return processed_data
            else:
                logger.warning("No flight data received from API")
                return self._get_fallback_flight_data(departure, destination, date)
                
        except Exception as e:
            logger.error(f"Error getting flight data: {e}")
            return self._get_fallback_flight_data(departure, destination, date)
    
    def get_weather_data(self, city: str, date: str) -> Dict[str, Any]:
        """
        Get weather data from Amap Weather API
        
        Args:
            city: City name
            date: Date for weather forecast (YYYY-MM-DD)
            
        Returns:
            Dictionary containing weather data
        """
        try:
            # Get city code
            city_code = self._get_city_code(city)
            
            # Prepare API parameters for current weather
            current_params = {
                'key': self.weather_api_key,
                'city': city_code,
                'extensions': 'base'
            }
            
            # Prepare API parameters for forecast
            forecast_params = {
                'key': self.weather_api_key,
                'city': city_code,
                'extensions': 'all'
            }
            
            # Check cache
            cache_key = f"weather_{city_code}_{date}"
            if cache_key in self.request_cache:
                cache_time, cached_data = self.request_cache[cache_key]
                if time.time() - cache_time < 1800:  # 30 minutes cache
                    logger.info("Returning cached weather data")
                    return cached_data
            
            # Get current weather
            current_url = f"{self.amap_weather_base_url}/weatherInfo"
            current_data = self._make_request(current_url, current_params)
            
            # Get forecast
            forecast_url = f"{self.amap_weather_base_url}/weatherInfo"
            forecast_data = self._make_request(forecast_url, forecast_params)
            
            # Process weather data
            processed_data = self._process_weather_data(current_data, forecast_data, city, date)
            
            # Cache the result
            self.request_cache[cache_key] = (time.time(), processed_data)
            
            logger.info(f"Retrieved weather data for {city}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return self._get_fallback_weather_data(city, date)
    
    def _get_airport_code(self, city: str) -> str:
        """Convert city name to IATA airport code"""
        airport_mapping = {
            'Beijing': 'PEK',
            'Shanghai': 'PVG',
            'Guangzhou': 'CAN',
            'Shenzhen': 'SZX',
            'Chengdu': 'CTU',
            'Hangzhou': 'HGH',
            'Nanjing': 'NKG',
            'Wuhan': 'WUH',
            'Xian': 'XIY',
            'Chongqing': 'CKG',
            'Tianjin': 'TSN',
            'Qingdao': 'TAO',
            'New York': 'JFK',
            'London': 'LHR',
            'Tokyo': 'NRT',
            'Paris': 'CDG',
            'Frankfurt': 'FRA',
            'Amsterdam': 'AMS',
            'Dubai': 'DXB',
            'Singapore': 'SIN',
            'Hong Kong': 'HKG',
            'Seoul': 'ICN',
            'Bangkok': 'BKK',
            'Kuala Lumpur': 'KUL',
            'Sydney': 'SYD',
            'Melbourne': 'MEL',
            'Los Angeles': 'LAX',
            'San Francisco': 'SFO',
            'Toronto': 'YYZ',
            'Vancouver': 'YVR'
        }
        
        return airport_mapping.get(city, city[:3].upper())
    
    def _get_city_code(self, city: str) -> str:
        """Convert city name to Amap city code"""
        city_mapping = {
            'Beijing': '110000',
            'Shanghai': '310000',
            'Guangzhou': '440100',
            'Shenzhen': '440300',
            'Chengdu': '510100',
            'Hangzhou': '330100',
            'Nanjing': '320100',
            'Wuhan': '420100',
            'Xian': '610100',
            'Chongqing': '500000',
            'Tianjin': '120000',
            'Qingdao': '370200'
        }
        
        return city_mapping.get(city, '110000')  # Default to Beijing
    
    def _process_flight_data(self, flight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw flight data from API"""
        processed_flights = []
        
        for flight in flight_data:
            try:
                processed_flight = {
                    'flight_id': f"flight_{flight.get('flight', {}).get('number', 'unknown')}_{flight.get('flight_date', '')}",
                    'flight_number': flight.get('flight', {}).get('number', 'N/A'),
                    'airline': flight.get('airline', {}).get('name', 'Unknown'),
                    'airline_iata': flight.get('airline', {}).get('iata', 'N/A'),
                    'aircraft_type': flight.get('aircraft', {}).get('registration', 'N/A'),
                    'departure_airport': flight.get('departure', {}).get('airport', 'N/A'),
                    'departure_iata': flight.get('departure', {}).get('iata', 'N/A'),
                    'departure_time': flight.get('departure', {}).get('scheduled', 'N/A'),
                    'departure_delay': flight.get('departure', {}).get('delay', 0),
                    'arrival_airport': flight.get('arrival', {}).get('airport', 'N/A'),
                    'arrival_iata': flight.get('arrival', {}).get('iata', 'N/A'),
                    'arrival_time': flight.get('arrival', {}).get('scheduled', 'N/A'),
                    'arrival_delay': flight.get('arrival', {}).get('delay', 0),
                    'flight_status': flight.get('flight_status', 'scheduled'),
                    'flight_date': flight.get('flight_date', ''),
                    'duration': self._calculate_duration(
                        flight.get('departure', {}).get('scheduled'),
                        flight.get('arrival', {}).get('scheduled')
                    ),
                    'price': self._estimate_price(flight),
                    'total_cost': self._estimate_total_cost(flight)
                }
                
                processed_flights.append(processed_flight)
                
            except Exception as e:
                logger.error(f"Error processing flight data: {e}")
                continue
        
        return {
            'status': 'success',
            'flights': processed_flights,
            'total_count': len(processed_flights),
            'source': 'aviationstack_api',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_weather_data(self, current_data: Dict[str, Any], forecast_data: Dict[str, Any], 
                            city: str, date: str) -> Dict[str, Any]:
        """Process weather data from API"""
        try:
            processed_weather = {
                'city': city,
                'date': date,
                'source': 'amap_weather_api',
                'timestamp': datetime.now().isoformat()
            }
            
            # Process current weather
            if current_data and 'lives' in current_data:
                current_info = current_data['lives'][0] if current_data['lives'] else {}
                processed_weather['current'] = {
                    'temperature': current_info.get('temperature', 'N/A'),
                    'weather': current_info.get('weather', 'N/A'),
                    'wind_direction': current_info.get('winddirection', 'N/A'),
                    'wind_power': current_info.get('windpower', 'N/A'),
                    'humidity': current_info.get('humidity', 'N/A'),
                    'report_time': current_info.get('reporttime', 'N/A')
                }
            
            # Process forecast data
            if forecast_data and 'forecasts' in forecast_data:
                forecasts = forecast_data['forecasts'][0] if forecast_data['forecasts'] else {}
                casts = forecasts.get('casts', [])
                
                processed_weather['forecast'] = []
                for cast in casts:
                    forecast_item = {
                        'date': cast.get('date', ''),
                        'week': cast.get('week', ''),
                        'dayweather': cast.get('dayweather', ''),
                        'nightweather': cast.get('nightweather', ''),
                        'daytemp': cast.get('daytemp', ''),
                        'nighttemp': cast.get('nighttemp', ''),
                        'daywind': cast.get('daywind', ''),
                        'nightwind': cast.get('nightwind', ''),
                        'daypower': cast.get('daypower', ''),
                        'nightpower': cast.get('nightpower', '')
                    }
                    processed_weather['forecast'].append(forecast_item)
            
            # Generate weather assessment
            processed_weather['assessment'] = self._assess_weather_conditions(processed_weather)
            
            return {
                'status': 'success',
                'weather_data': processed_weather
            }
            
        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
            return self._get_fallback_weather_data(city, date)
    
    def _calculate_duration(self, departure_time: str, arrival_time: str) -> str:
        """Calculate flight duration"""
        try:
            if not departure_time or not arrival_time:
                return "N/A"
            
            # Parse times (assuming same date)
            dep_time = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
            arr_time = datetime.fromisoformat(arrival_time.replace('Z', '+00:00'))
            
            duration = arr_time - dep_time
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            return f"{hours}h {minutes}m"
            
        except Exception:
            return "N/A"
    
    def _estimate_price(self, flight_data: Dict[str, Any]) -> float:
        """Estimate flight price based on flight data"""
        try:
            # Base price estimation logic
            base_price = 500.0
            
            # Adjust by airline
            airline = flight_data.get('airline', {}).get('name', '').lower()
            if 'china' in airline or 'air china' in airline:
                base_price *= 1.1
            elif 'spring' in airline or 'lucky' in airline:
                base_price *= 0.8
            
            # Adjust by time
            departure_time = flight_data.get('departure', {}).get('scheduled', '')
            if departure_time:
                hour = int(departure_time.split('T')[1].split(':')[0]) if 'T' in departure_time else 12
                if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
                    base_price *= 1.2
                elif hour < 6 or hour > 22:  # Early/late hours
                    base_price *= 0.9
            
            return round(base_price, 2)
            
        except Exception:
            return 500.0
    
    def _estimate_total_cost(self, flight_data: Dict[str, Any]) -> float:
        """Estimate total cost including additional fees"""
        base_price = self._estimate_price(flight_data)
        
        # Add estimated fees
        airport_fee = 50.0
        fuel_surcharge = 30.0
        
        return round(base_price + airport_fee + fuel_surcharge, 2)
    
    def _assess_weather_conditions(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess weather conditions for flight safety"""
        try:
            current = weather_data.get('current', {})
            weather_condition = current.get('weather', '').lower()
            wind_power = current.get('wind_power', '1')
            
            # Safety assessment
            safety_score = 1.0
            risk_factors = []
            
            # Weather condition assessment
            if any(condition in weather_condition for condition in ['rain', 'storm', 'thunder']):
                safety_score -= 0.3
                risk_factors.append("precipitation")
            
            if any(condition in weather_condition for condition in ['fog', 'haze']):
                safety_score -= 0.2
                risk_factors.append("visibility")
            
            # Wind assessment
            try:
                wind_level = int(wind_power.split('-')[0]) if '-' in wind_power else int(wind_power)
                if wind_level > 6:
                    safety_score -= 0.2
                    risk_factors.append("high_wind")
            except (ValueError, AttributeError):
                pass
            
            safety_score = max(0.0, min(1.0, safety_score))
            
            return {
                'safety_score': safety_score,
                'risk_factors': risk_factors,
                'recommendation': 'safe' if safety_score > 0.8 else 'caution' if safety_score > 0.6 else 'consider_delay',
                'assessment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing weather: {e}")
            return {
                'safety_score': 0.8,
                'risk_factors': [],
                'recommendation': 'safe',
                'assessment_time': datetime.now().isoformat()
            }
    
    def _get_fallback_flight_data(self, departure: str, destination: str, date: str) -> Dict[str, Any]:
        """Provide fallback flight data when API fails"""
        logger.warning("Using fallback flight data")
        
        fallback_flights = [
            {
                'flight_id': f'fallback_001_{date}',
                'flight_number': 'FB001',
                'airline': 'Fallback Airlines',
                'airline_iata': 'FB',
                'aircraft_type': 'Boeing 737',
                'departure_airport': f'{departure} Airport',
                'departure_iata': self._get_airport_code(departure),
                'departure_time': f'{date}T08:00:00Z',
                'departure_delay': 0,
                'arrival_airport': f'{destination} Airport',
                'arrival_iata': self._get_airport_code(destination),
                'arrival_time': f'{date}T12:00:00Z',
                'arrival_delay': 0,
                'flight_status': 'scheduled',
                'flight_date': date,
                'duration': '4h 0m',
                'price': 450.0,
                'total_cost': 530.0
            },
            {
                'flight_id': f'fallback_002_{date}',
                'flight_number': 'FB002',
                'airline': 'Fallback Express',
                'airline_iata': 'FE',
                'aircraft_type': 'Airbus A320',
                'departure_airport': f'{departure} Airport',
                'departure_iata': self._get_airport_code(departure),
                'departure_time': f'{date}T14:00:00Z',
                'departure_delay': 0,
                'arrival_airport': f'{destination} Airport',
                'arrival_iata': self._get_airport_code(destination),
                'arrival_time': f'{date}T18:30:00Z',
                'arrival_delay': 0,
                'flight_status': 'scheduled',
                'flight_date': date,
                'duration': '4h 30m',
                'price': 380.0,
                'total_cost': 460.0
            }
        ]
        
        return {
            'status': 'fallback',
            'flights': fallback_flights,
            'total_count': len(fallback_flights),
            'source': 'fallback_data',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_fallback_weather_data(self, city: str, date: str) -> Dict[str, Any]:
        """Provide fallback weather data when API fails"""
        logger.warning("Using fallback weather data")
        
        return {
            'status': 'fallback',
            'weather_data': {
                'city': city,
                'date': date,
                'current': {
                    'temperature': '20',
                    'weather': 'Clear',
                    'wind_direction': 'East',
                    'wind_power': '3',
                    'humidity': '60',
                    'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'forecast': [
                    {
                        'date': date,
                        'week': '1',
                        'dayweather': 'Clear',
                        'nightweather': 'Clear',
                        'daytemp': '25',
                        'nighttemp': '15',
                        'daywind': 'East',
                        'nightwind': 'East',
                        'daypower': '3',
                        'nightpower': '2'
                    }
                ],
                'assessment': {
                    'safety_score': 0.9,
                    'risk_factors': [],
                    'recommendation': 'safe',
                    'assessment_time': datetime.now().isoformat()
                },
                'source': 'fallback_data',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of external API connections"""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Check Aviationstack API
        try:
            test_params = {
                'access_key': self.flight_api_key,
                'limit': 1
            }
            response = self._make_request(f"{self.aviationstack_base_url}/flights", test_params)
            health_status['services']['aviationstack'] = 'healthy' if response else 'degraded'
        except Exception:
            health_status['services']['aviationstack'] = 'down'
        
        # Check Amap Weather API
        try:
            test_params = {
                'key': self.weather_api_key,
                'city': '110000',
                'extensions': 'base'
            }
            response = self._make_request(f"{self.amap_weather_base_url}/weatherInfo", test_params)
            health_status['services']['amap_weather'] = 'healthy' if response else 'degraded'
        except Exception:
            health_status['services']['amap_weather'] = 'down'
        
        # Determine overall status
        service_statuses = list(health_status['services'].values())
        if all(status == 'healthy' for status in service_statuses):
            health_status['overall_status'] = 'healthy'
        elif any(status == 'healthy' for status in service_statuses):
            health_status['overall_status'] = 'degraded'
        else:
            health_status['overall_status'] = 'down'
        
        return health_status 