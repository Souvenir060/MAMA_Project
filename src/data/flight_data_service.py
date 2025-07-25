"""
MAMA Flight Assistant - Flight Data Service

Specialized service for flight data retrieval, processing, and caching.
This module encapsulates all flight-related data operations.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .external_api_manager import ExternalAPIManager

logger = logging.getLogger(__name__)

class FlightDataService:
    """
    Specialized service for flight data operations
    
    This service handles all flight-related data retrieval, processing,
    and provides standardized interfaces for the agent system.
    """
    
    def __init__(self):
        """Initialize the Flight Data Service"""
        self.api_manager = ExternalAPIManager()
        self.flight_cache = {}
        self.processed_flights_cache = {}
        
        logger.info("Flight Data Service initialized")
    
    def search_flights(self, departure: str, destination: str, date: str, 
                      filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for flights with optional filters
        
        Args:
            departure: Departure city
            destination: Destination city
            date: Flight date (YYYY-MM-DD)
            filters: Optional filters (airline, price_range, time_range, etc.)
            
        Returns:
            Dictionary containing flight search results
        """
        try:
            logger.info(f"Searching flights: {departure} -> {destination} on {date}")
            
            # Get raw flight data from API
            raw_data = self.api_manager.get_flight_data(departure, destination, date)
            
            if raw_data.get('status') in ['success', 'fallback']:
                flights = raw_data.get('flights', [])
                
                # Apply filters if provided
                if filters:
                    flights = self._apply_filters(flights, filters)
                
                # Process flight data with additional computed fields
                processed_flights = self._process_flight_data(flights, departure, destination, date)
                
                return {
                    'status': 'success',
                    'flights': processed_flights,
                    'total_count': len(processed_flights),
                    'search_params': {
                        'departure': departure,
                        'destination': destination,
                        'date': date,
                        'filters': filters
                    },
                    'data_source': raw_data.get('source', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error("Failed to retrieve flight data")
                return {
                    'status': 'error',
                    'message': 'Failed to retrieve flight data',
                    'flights': [],
                    'total_count': 0
                }
                
        except Exception as e:
            logger.error(f"Error in flight search: {e}")
            return {
                'status': 'error',
                'message': f'Flight search error: {str(e)}',
                'flights': [],
                'total_count': 0
            }
    
    def get_flight_details(self, flight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific flight
        
        Args:
            flight_id: Unique flight identifier
            
        Returns:
            Detailed flight information or None if not found
        """
        try:
            # Check cache first
            if flight_id in self.processed_flights_cache:
                logger.info(f"Returning cached flight details for {flight_id}")
                return self.processed_flights_cache[flight_id]
            
            # If not in cache, would need to search or query API
            logger.warning(f"Flight details not found for {flight_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting flight details: {e}")
            return None
    
    def _apply_filters(self, flights: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to flight list"""
        filtered_flights = flights
        
        try:
            # Airline filter
            if 'airlines' in filters and filters['airlines']:
                airline_list = filters['airlines'] if isinstance(filters['airlines'], list) else [filters['airlines']]
                filtered_flights = [f for f in filtered_flights 
                                 if f.get('airline', '').lower() in [a.lower() for a in airline_list]]
            
            # Price range filter
            if 'price_min' in filters and filters['price_min'] is not None:
                filtered_flights = [f for f in filtered_flights 
                                 if f.get('price', 0) >= filters['price_min']]
            
            if 'price_max' in filters and filters['price_max'] is not None:
                filtered_flights = [f for f in filtered_flights 
                                 if f.get('price', float('inf')) <= filters['price_max']]
            
            # Time range filter
            if 'departure_time_min' in filters and filters['departure_time_min']:
                min_time = filters['departure_time_min']
                filtered_flights = [f for f in filtered_flights 
                                 if self._compare_time(f.get('departure_time', ''), min_time, '>=')]
            
            if 'departure_time_max' in filters and filters['departure_time_max']:
                max_time = filters['departure_time_max']
                filtered_flights = [f for f in filtered_flights 
                                 if self._compare_time(f.get('departure_time', ''), max_time, '<=')]
            
            # Direct flights only
            if filters.get('direct_only', False):
                filtered_flights = [f for f in filtered_flights if f.get('stops', 1) == 0]
            
            # Aircraft type filter
            if 'aircraft_types' in filters and filters['aircraft_types']:
                aircraft_list = filters['aircraft_types'] if isinstance(filters['aircraft_types'], list) else [filters['aircraft_types']]
                filtered_flights = [f for f in filtered_flights 
                                 if any(aircraft.lower() in f.get('aircraft_type', '').lower() 
                                       for aircraft in aircraft_list)]
            
            logger.info(f"Applied filters: {len(flights)} -> {len(filtered_flights)} flights")
            return filtered_flights
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return flights
    
    def _process_flight_data(self, flights: List[Dict[str, Any]], departure: str, 
                           destination: str, date: str) -> List[Dict[str, Any]]:
        """Process flight data with additional computed fields"""
        processed_flights = []
        
        for flight in flights:
            try:
                processed_flight = flight.copy()
                
                # Add computed fields
                processed_flight['route'] = f"{departure} -> {destination}"
                processed_flight['search_date'] = date
                
                # Calculate stops (default to direct flight if not specified)
                if 'stops' not in processed_flight:
                    processed_flight['stops'] = 0
                
                # Add flight type classification
                processed_flight['flight_type'] = self._classify_flight_type(processed_flight)
                
                # Add time-based classification
                processed_flight['time_category'] = self._classify_time_category(processed_flight.get('departure_time', ''))
                
                # Add price category
                processed_flight['price_category'] = self._classify_price_category(processed_flight.get('price', 0))
                
                # Add reliability score based on airline and flight status
                processed_flight['reliability_score'] = self._calculate_reliability_score(processed_flight)
                
                # Add convenience score
                processed_flight['convenience_score'] = self._calculate_convenience_score(processed_flight)
                
                # Store in cache for quick retrieval
                flight_id = processed_flight.get('flight_id')
                if flight_id:
                    self.processed_flights_cache[flight_id] = processed_flight
                
                processed_flights.append(processed_flight)
                
            except Exception as e:
                logger.error(f"Error processing flight data: {e}")
                processed_flights.append(flight)  # Add original if processing fails
        
        return processed_flights
    
    def _compare_time(self, flight_time: str, comparison_time: str, operator: str) -> bool:
        """Compare flight times"""
        try:
            # Extract time from datetime string
            if 'T' in flight_time:
                flight_time_part = flight_time.split('T')[1].split(':')
            else:
                flight_time_part = flight_time.split(':')
            
            comp_time_part = comparison_time.split(':')
            
            flight_minutes = int(flight_time_part[0]) * 60 + int(flight_time_part[1])
            comp_minutes = int(comp_time_part[0]) * 60 + int(comp_time_part[1])
            
            if operator == '>=':
                return flight_minutes >= comp_minutes
            elif operator == '<=':
                return flight_minutes <= comp_minutes
            elif operator == '==':
                return flight_minutes == comp_minutes
            
        except (ValueError, IndexError):
            return True  # If parsing fails, include the flight
        
        return True
    
    def _classify_flight_type(self, flight: Dict[str, Any]) -> str:
        """Classify flight type based on stops and duration"""
        stops = flight.get('stops', 0)
        duration = flight.get('duration', '')
        
        if stops == 0:
            return 'direct'
        elif stops == 1:
            return 'one_stop'
        else:
            return 'multi_stop'
    
    def _classify_time_category(self, departure_time: str) -> str:
        """Classify flight by departure time"""
        try:
            if 'T' in departure_time:
                time_part = departure_time.split('T')[1]
            else:
                time_part = departure_time
            
            hour = int(time_part.split(':')[0])
            
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'
                
        except (ValueError, IndexError):
            return 'unknown'
    
    def _classify_price_category(self, price: float) -> str:
        """Classify flight by price range"""
        if price <= 300:
            return 'budget'
        elif price <= 600:
            return 'economy'
        elif price <= 1000:
            return 'business'
        else:
            return 'premium'
    
    def _calculate_reliability_score(self, flight: Dict[str, Any]) -> float:
        """Calculate reliability score based on various factors"""
        score = 0.8  # Base score
        
        try:
            # Airline reputation adjustment
            airline = flight.get('airline', '').lower()
            if any(premium in airline for premium in ['air china', 'china eastern', 'china southern']):
                score += 0.1
            elif any(budget in airline for budget in ['spring', 'lucky', 'west air']):
                score -= 0.1
            
            # Flight status adjustment
            status = flight.get('flight_status', '').lower()
            if status == 'scheduled':
                score += 0.05
            elif status in ['delayed', 'cancelled']:
                score -= 0.2
            
            # Delay history adjustment
            departure_delay = flight.get('departure_delay', 0)
            if departure_delay > 30:
                score -= 0.1
            elif departure_delay > 60:
                score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.8
    
    def _calculate_convenience_score(self, flight: Dict[str, Any]) -> float:
        """Calculate convenience score based on various factors"""
        score = 0.7  # Base score
        
        try:
            # Direct flight bonus
            if flight.get('stops', 1) == 0:
                score += 0.2
            
            # Departure time convenience
            time_category = flight.get('time_category', '')
            if time_category in ['morning', 'afternoon']:
                score += 0.1
            elif time_category == 'night':
                score -= 0.1
            
            # Duration convenience
            duration = flight.get('duration', '')
            if 'h' in duration:
                try:
                    hours = int(duration.split('h')[0])
                    if hours <= 2:
                        score += 0.1
                    elif hours >= 6:
                        score -= 0.1
                except (ValueError, IndexError):
                    pass
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.7
    
    def get_flight_statistics(self, flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics for a list of flights"""
        if not flights:
            return {'message': 'No flights to analyze'}
        
        try:
            stats = {
                'total_flights': len(flights),
                'airlines': {},
                'price_stats': {},
                'time_distribution': {},
                'aircraft_types': {},
                'average_scores': {}
            }
            
            # Collect data
            prices = []
            reliability_scores = []
            convenience_scores = []
            
            for flight in flights:
                # Airline distribution
                airline = flight.get('airline', 'Unknown')
                stats['airlines'][airline] = stats['airlines'].get(airline, 0) + 1
                
                # Price data
                price = flight.get('price', 0)
                if price > 0:
                    prices.append(price)
                
                # Time distribution
                time_cat = flight.get('time_category', 'unknown')
                stats['time_distribution'][time_cat] = stats['time_distribution'].get(time_cat, 0) + 1
                
                # Aircraft types
                aircraft = flight.get('aircraft_type', 'Unknown')
                stats['aircraft_types'][aircraft] = stats['aircraft_types'].get(aircraft, 0) + 1
                
                # Scores
                reliability_scores.append(flight.get('reliability_score', 0.8))
                convenience_scores.append(flight.get('convenience_score', 0.7))
            
            # Calculate price statistics
            if prices:
                stats['price_stats'] = {
                    'min': min(prices),
                    'max': max(prices),
                    'average': round(sum(prices) / len(prices), 2),
                    'range': max(prices) - min(prices)
                }
            
            # Calculate average scores
            stats['average_scores'] = {
                'reliability': round(sum(reliability_scores) / len(reliability_scores), 3),
                'convenience': round(sum(convenience_scores) / len(convenience_scores), 3)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating flight statistics: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the flight data service"""
        try:
            # Check API manager health
            api_health = self.api_manager.health_check()
            
            return {
                'service': 'FlightDataService',
                'status': 'healthy',
                'api_manager_status': api_health.get('overall_status', 'unknown'),
                'cached_flights': len(self.processed_flights_cache),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'service': 'FlightDataService',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 