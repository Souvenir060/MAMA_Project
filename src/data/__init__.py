"""
MAMA Flight Assistant - Data Service Module

This module provides modularized data services for external API integration.
All flight, weather, and other external data queries are centralized here.
"""

from .flight_data_service import FlightDataService
from .external_api_manager import ExternalAPIManager

__all__ = [
    'FlightDataService',
    'ExternalAPIManager'
] 