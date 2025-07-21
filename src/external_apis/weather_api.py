#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather API Module
"""

import os
import sys
import logging
import requests
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

class OpenWeatherMapAPI:
    """
    OpenWeatherMap API client for weather data retrieval
    Academic implementation with proper error handling
    """
    
    def __init__(self, api_key: str = "498ae38fb9831291de1d0432ea2fdf07"):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.timeout = 10
        self.logger = logging.getLogger(__name__)
        
        # Common city coordinates for quick lookup
        self.city_coordinates = {
            "Beijing": {"lat": 39.9042, "lon": 116.4074},
            "Shanghai": {"lat": 31.2304, "lon": 121.4737},
            "Guangzhou": {"lat": 23.1291, "lon": 113.2644},
            "Shenzhen": {"lat": 22.5431, "lon": 114.0579},
            "Hangzhou": {"lat": 30.2741, "lon": 120.1551},
            "Chengdu": {"lat": 30.5728, "lon": 104.0668},
            "Xi'an": {"lat": 34.3416, "lon": 108.9398},
            "Nanjing": {"lat": 32.0603, "lon": 118.7969},
            "Tianjin": {"lat": 39.3434, "lon": 117.3616},
            "Chongqing": {"lat": 29.4316, "lon": 106.9123},
            "Singapore": {"lat": 1.3521, "lon": 103.8198},
            "New York": {"lat": 40.7128, "lon": -74.0060},
            "London": {"lat": 51.5074, "lon": -0.1278},
            "Paris": {"lat": 48.8566, "lon": 2.3522},
            "Tokyo": {"lat": 35.6762, "lon": 139.6503},
            "Sydney": {"lat": -33.8688, "lon": 151.2093},
        }
    
    def get_weather_assessment(self, location: str, date: str = None) -> Dict[str, Any]:
        """Get comprehensive weather assessment for aviation safety"""
        try:
            # Get current weather data
            current_weather = self._get_current_weather(location)
            
            if not current_weather:
                self.logger.warning(f"No current weather data for {location}")
                return self._generate_fallback_weather_assessment(location)
            
            # Calculate safety scores
            safety_scores = self._calculate_weather_safety_scores(current_weather)
            
            # Generate assessment
            assessment = {
                "location": location,
                "weather_data": current_weather,
                "safety_scores": safety_scores,
                "overall_score": safety_scores.get("overall", 0.8),
                "safety_rating": self._get_safety_rating(safety_scores.get("overall", 0.8)),
                "conditions_description": self._generate_conditions_description(current_weather),
                "recommendations": self._generate_weather_recommendations(safety_scores),
                "timestamp": datetime.now().isoformat(),
                "data_source": "openweathermap_api"
            }
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Weather assessment failed for {location}: {e}")
            return self._generate_fallback_weather_assessment(location)
    
    def get_route_weather_analysis(self, departure: str, destination: str, date: str = None) -> Dict[str, Any]:
        """Get weather analysis for entire flight route"""
        try:
            # Get weather for both locations
            dep_weather = self.get_weather_assessment(departure, date)
            dest_weather = self.get_weather_assessment(destination, date)
            
            # Combine assessments
            route_analysis = {
                "departure_weather": dep_weather,
                "destination_weather": dest_weather,
                "overall_route_score": (dep_weather.get("overall_score", 0.8) + dest_weather.get("overall_score", 0.8)) / 2,
                "route_recommendations": self._generate_route_recommendations(dep_weather, dest_weather),
                "optimal_flight_time": self._suggest_optimal_flight_time(dep_weather, dest_weather),
                "weather_risks": self._identify_weather_risks(dep_weather, dest_weather),
                "timestamp": datetime.now().isoformat()
            }
            
            return route_analysis
            
        except Exception as e:
            self.logger.error(f"Route weather analysis failed: {e}")
            return {
                "departure_weather": self._generate_fallback_weather_assessment(departure),
                "destination_weather": self._generate_fallback_weather_assessment(destination),
                "overall_route_score": 0.8,
                "route_recommendations": ["Weather data unavailable - verify conditions directly"],
                "optimal_flight_time": "Data unavailable",
                "weather_risks": ["Unable to assess weather risks"],
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather from OpenWeatherMap API"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'location': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'conditions': data['weather'][0]['description'],
                'cloud_coverage': data['clouds']['all'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current weather: {e}")
            return {}
    
    def _calculate_weather_safety_scores(self, weather: Dict[str, Any]) -> Dict[str, float]:
        """Calculate safety scores based on real weather data"""
        # Visibility score (most critical for aviation)
        visibility = weather.get('visibility', 10)
        if visibility >= 10:
            visibility_score = 1.0
        elif visibility >= 5:
            visibility_score = 0.8
        elif visibility >= 2:
            visibility_score = 0.6
        else:
            visibility_score = 0.3
        
        # Wind score
        wind_speed = weather.get('wind_speed', 0)
        if wind_speed <= 10:
            wind_score = 1.0
        elif wind_speed <= 20:
            wind_score = 0.8
        elif wind_speed <= 30:
            wind_score = 0.6
        else:
            wind_score = 0.3
        
        # Temperature score (extreme temperatures affect operations)
        temperature = weather.get('temperature', 20)
        if -10 <= temperature <= 35:
            temp_score = 1.0
        elif -20 <= temperature <= 40:
            temp_score = 0.8
        else:
            temp_score = 0.6
        
        # Cloud coverage score
        cloud_coverage = weather.get('cloud_coverage', 0)
        if cloud_coverage <= 25:
            cloud_score = 1.0
        elif cloud_coverage <= 50:
            cloud_score = 0.8
        elif cloud_coverage <= 75:
            cloud_score = 0.7
        else:
            cloud_score = 0.6
        
        # Calculate overall score with weights
        overall_score = (
            visibility_score * 0.35 +
            wind_score * 0.25 +
            temp_score * 0.20 +
            cloud_score * 0.20
        )
        
        return {
            "visibility": visibility_score,
            "wind": wind_score,
            "temperature": temp_score,
            "cloud_coverage": cloud_score,
            "overall": overall_score
        }
    
    def _get_safety_rating(self, overall_score: float) -> str:
        """Convert safety score to rating"""
        if overall_score >= 0.9:
            return "Excellent"
        elif overall_score >= 0.8:
            return "Good"
        elif overall_score >= 0.7:
            return "Fair"
        elif overall_score >= 0.6:
            return "Poor"
        else:
            return "Dangerous"
    
    def _generate_conditions_description(self, weather: Dict[str, Any]) -> str:
        """Generate human-readable conditions description"""
        temp = weather.get('temperature', 20)
        wind = weather.get('wind_speed', 0)
        visibility = weather.get('visibility', 10)
        conditions = weather.get('conditions', 'clear')
        
        description = f"Temperature {temp}°C, {conditions}"
        if wind > 15:
            description += f", windy ({wind} m/s)"
        if visibility < 5:
            description += f", limited visibility ({visibility} km)"
        
        return description
    
    def _generate_weather_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate weather-based recommendations"""
        recommendations = []
        
        if scores.get("visibility", 1.0) < 0.7:
            recommendations.append("Limited visibility - consider flight delays")
        
        if scores.get("wind", 1.0) < 0.7:
            recommendations.append("Strong winds - expect turbulence")
        
        if scores.get("temperature", 1.0) < 0.7:
            recommendations.append("Extreme temperatures - verify aircraft performance")
    
        if scores.get("overall", 1.0) >= 0.9:
            recommendations.append("Excellent weather conditions for flying")
        
        return recommendations or ["Weather conditions acceptable for aviation"]
    
    def _generate_route_recommendations(self, dep_weather: Dict[str, Any], dest_weather: Dict[str, Any]) -> List[str]:
        """Generate route-specific recommendations"""
        recommendations = []
    
        dep_score = dep_weather.get("overall_score", 0.8)
        dest_score = dest_weather.get("overall_score", 0.8)
        
        if dep_score < 0.7:
            recommendations.append("Consider delayed departure due to weather")
        
        if dest_score < 0.7:
            recommendations.append("Monitor destination weather closely")
        
        if abs(dep_score - dest_score) > 0.3:
            recommendations.append("Significant weather differences between airports")
        
        return recommendations or ["Route weather conditions acceptable"]
    
    def _suggest_optimal_flight_time(self, dep_weather: Dict[str, Any], dest_weather: Dict[str, Any]) -> str:
        """Suggest optimal flight timing based on weather"""
        dep_score = dep_weather.get("overall_score", 0.8)
        dest_score = dest_weather.get("overall_score", 0.8)
        
        if dep_score >= 0.8 and dest_score >= 0.8:
            return "Current conditions favorable for immediate departure"
        elif dep_score < 0.6 or dest_score < 0.6:
            return "Consider delaying flight 2-4 hours for improved conditions"
        else:
            return "Acceptable for flight with standard precautions"
    
    def _identify_weather_risks(self, dep_weather: Dict[str, Any], dest_weather: Dict[str, Any]) -> List[str]:
        """Identify potential weather risks for the route"""
        risks = []
        
        dep_data = dep_weather.get("weather_data", {})
        dest_data = dest_weather.get("weather_data", {})
    
        if dep_data.get("wind_speed", 0) > 20:
            risks.append("Strong winds at departure")
        
        if dest_data.get("wind_speed", 0) > 20:
            risks.append("Strong winds at destination")
        
        if dep_data.get("visibility", 10) < 3:
            risks.append("Poor visibility at departure")
        
        if dest_data.get("visibility", 10) < 3:
            risks.append("Poor visibility at destination")
        
        return risks or ["No significant weather risks identified"]
    
    def _generate_fallback_weather_assessment(self, location: str) -> Dict[str, Any]:
        """Generate fallback weather assessment when API fails"""
        return {
            "location": location,
            "weather_data": {
                "temperature": 20,
                "humidity": 60,
                "wind_speed": 8,
                "visibility": 10,
                "conditions": "data unavailable"
            },
            "safety_scores": {
                "visibility": 0.8,
                "wind": 0.8,
                "temperature": 0.8,
                "cloud_coverage": 0.8,
                "overall": 0.8
            },
            "overall_score": 0.8,
            "safety_rating": "Good",
            "conditions_description": "Data unavailable - using fallback assessment",
            "recommendations": ["Weather data unavailable - proceed with caution"],
            "data_source": "fallback"
        }


def get_weather_data(location: str, date: str = None) -> Dict[str, Any]:
    """Convenience function to get weather data"""
    api = OpenWeatherMapAPI()
    return api.get_weather_assessment(location, date)


def get_route_weather(departure: str, destination: str, date: str = None) -> Dict[str, Any]:
    """Convenience function to get route weather analysis"""
    api = OpenWeatherMapAPI()
    return api.get_route_weather_analysis(departure, destination, date) 