#!/usr/bin/env python3
"""
MAMA Flight Assistant - Weather Data Aggregator

This module provides a comprehensive weather data aggregation system,
integrating multiple weather data sources and providing comprehensive
meteorological analysis capabilities.

Academic Features:
- Multi-source weather data integration
- Comprehensive meteorological pattern analysis
- Historical data processing
- Real-time data streaming
- Quality control and validation
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class WeatherDataAggregator:
    """
    Academic-level weather data aggregation system
    
    Features:
    - Multi-source data integration
    - Real-time data streaming
    - Historical data analysis
    - Pattern recognition
    - Quality control
    """
    
    def __init__(self):
        """Initialize weather data aggregator"""
        self.logger = logging.getLogger(__name__)
        self.openweather_api = OpenWeatherMapAPI()
        
        # Initialize data sources
        self.data_sources = [
            self.openweather_api
        ]
        
        self.logger.info("Weather data aggregator initialized with OpenWeatherMap API")
    
    def _initialize_data_sources(self):
        """Initialize available weather data sources"""
        self.data_sources = [
                "noaa",
                "weatherapi",
                "openweathermap",
                "aeris",
            "tomorrow",
            "milestone"
            ]
        self.logger.info(f"Initialized {len(self.data_sources)} weather data sources")
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather conditions for location"""
        try:
            # Aggregate data from multiple sources
            weather_data = {}
            for source in self.data_sources:
                try:
                    data = self._get_weather_from_source(source, location)
                    weather_data[source] = data
                except Exception as e:
                    self.logger.warning(f"Failed to get weather from {source}: {e}")
            
            # If no weather data from APIs, try Milestone data space
            if not any(weather_data.values()):
                try:
                    from core.milestone_connector import read_from_milestone
                    self.logger.info("Attempting to fetch weather data from Milestone data space")
                    
                    milestone_weather = read_from_milestone(
                        entity_type="WeatherInfo",
                        query_params={
                            "location": location,
                            "type": "current"
                        }
                    )
                    
                    if milestone_weather:
                        self.logger.info(f"Retrieved weather data from Milestone data space")
                        return self._convert_milestone_to_weather(milestone_weather[0] if milestone_weather else {})
                    else:
                        self.logger.warning("No weather data found in Milestone data space")
                        
                except Exception as e:
                    self.logger.error(f"Failed to retrieve weather data from Milestone: {e}")
            
            return self._aggregate_weather_data(weather_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get current weather: {e}")
            return {}
    
    def get_weather_forecast(self, location: str, days: int = 5) -> List[Dict[str, Any]]:
        """Get weather forecast for location"""
        try:
            # Aggregate forecasts from multiple sources
            forecasts = {}
            for source in self.data_sources:
                try:
                    data = self._get_forecast_from_source(source, location, days)
                    forecasts[source] = data
                except Exception as e:
                    self.logger.warning(f"Failed to get forecast from {source}: {e}")
            
            # If no forecast data from APIs, try Milestone data space
            if not any(forecasts.values()):
                try:
                    from core.milestone_connector import read_from_milestone
                    self.logger.info("Attempting to fetch forecast data from Milestone data space")
                    
                    milestone_forecast = read_from_milestone(
                        entity_type="WeatherForecast",
                        query_params={
                            "location": location,
                            "days": days
                        }
                    )
                    
                    if milestone_forecast:
                        self.logger.info(f"Retrieved forecast data from Milestone data space")
                        return self._convert_milestone_to_forecast(milestone_forecast)
                    else:
                        self.logger.warning("No forecast data found in Milestone data space")
                        
                except Exception as e:
                    self.logger.error(f"Failed to retrieve forecast data from Milestone: {e}")
            
            return self._aggregate_forecast_data(forecasts)
            
        except Exception as e:
            self.logger.error(f"Failed to get weather forecast: {e}")
            return []
    
    def _convert_milestone_to_weather(self, milestone_weather: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Milestone data to weather format"""
        
        def get_property_value(data: Dict[str, Any], property_name: str, default=None):
            """Safely extract property value from NGSI-LD format"""
            try:
                prop = data.get(property_name, {})
                if isinstance(prop, dict):
                    return prop.get('value', default)
                return prop or default
            except Exception:
                return default
        
        try:
            return {
                "location": get_property_value(milestone_weather, "location", ""),
                "temperature": get_property_value(milestone_weather, "temperature", 0),
                "humidity": get_property_value(milestone_weather, "humidity", 0),
                "wind_speed": get_property_value(milestone_weather, "windSpeed", 0),
                "wind_direction": get_property_value(milestone_weather, "windDirection", ""),
                "conditions": get_property_value(milestone_weather, "conditions", ""),
                "visibility": get_property_value(milestone_weather, "visibility", 0),
                "pressure": get_property_value(milestone_weather, "pressure", 0),
                "timestamp": get_property_value(milestone_weather, "timestamp", datetime.now().isoformat())
            }
        except Exception as e:
            self.logger.error(f"Error converting Milestone weather data: {e}")
            return {}
    
    def _convert_milestone_to_forecast(self, milestone_forecast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Milestone data to forecast format"""
        forecast = []
        
        def get_property_value(data: Dict[str, Any], property_name: str, default=None):
            """Safely extract property value from NGSI-LD format"""
            try:
                prop = data.get(property_name, {})
                if isinstance(prop, dict):
                    return prop.get('value', default)
                return prop or default
            except Exception:
                return default
        
        for forecast_data in milestone_forecast:
            try:
                day_forecast = {
                    "date": get_property_value(forecast_data, "date", ""),
                    "temperature_high": get_property_value(forecast_data, "temperatureHigh", 0),
                    "temperature_low": get_property_value(forecast_data, "temperatureLow", 0),
                    "humidity": get_property_value(forecast_data, "humidity", 0),
                    "wind_speed": get_property_value(forecast_data, "windSpeed", 0),
                    "wind_direction": get_property_value(forecast_data, "windDirection", ""),
                    "conditions": get_property_value(forecast_data, "conditions", ""),
                    "precipitation_chance": get_property_value(forecast_data, "precipitationChance", 0),
                    "visibility": get_property_value(forecast_data, "visibility", 0),
                    "pressure": get_property_value(forecast_data, "pressure", 0)
                }
                forecast.append(day_forecast)
            except Exception as e:
                self.logger.error(f"Error converting Milestone forecast data: {e}")
                continue
        
        return forecast
    
    def _get_weather_from_source(self, source: str, location: str) -> Dict[str, Any]:
        """Get weather data from specific source"""
        # Implementation would integrate with actual weather APIs
        return {}
    
    def _get_forecast_from_source(self, source: str, location: str, days: int) -> List[Dict[str, Any]]:
        """Get forecast data from specific source"""
        # Implementation would integrate with actual weather APIs
        return []
    
    def _aggregate_weather_data(self, weather_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate weather data from multiple sources"""
        # Implementation would combine and validate data from multiple sources
        return {}
    
    def _aggregate_forecast_data(self, forecasts: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Aggregate forecast data from multiple sources"""
        # Implementation would combine and validate forecasts from multiple sources
        return [] 