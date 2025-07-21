#!/usr/bin/env python3
"""
Single Agent System - Single Agent Baseline Model
True implementation of a single agent serially processing all tasks
Completely removes artificial penalties to let performance differences emerge naturally
"""

import numpy as np
import random
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class SingleAgentSystem(BaseModel):
    """
    Single Agent System - True Serial Processing Implementation
    
    This class simulates a real, serially-working single agent system.
    It must personally complete all subtasks without specialized division of labor.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize single agent system"""
        super().__init__(config)
        self.model_description = "Single Agent System (true serial implementation)"
    
    def _initialize_model(self):
        """Initialize model specific components"""
        # No need for specialized components in single agent system
        self.agent_id = "single_agent"
        logger.info("✅ Single agent system initialized")
    
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Always selects only the single agent
        
        Args:
            query_data: Query data
            
        Returns:
            Single-element list with agent ID and selection score
        """
        # Single agent system always selects itself with perfect score
        return [("single_agent", 1.0)]
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                          selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Process query with single agent by serially executing all subtasks
        
        Args:
            query_data: Query data
            selected_agents: Selected agents (ignored, always uses single agent)
            
        Returns:
            Single agent processing results
        """
        # Serial execution: the single agent must process each flight one by one
        logger.info("🔄 Single agent processing all flights serially...")
        
        # Get flight options
        flight_options = query_data.get('flight_options', [])
        if not flight_options:
            # Generate dummy flight IDs if not provided
            flight_options = [f"flight_{i:04d}" for i in range(10)]
        
        # Process each flight one by one (true serial implementation)
        flight_results = {}
        overall_recommendations = []
        
        for i, flight in enumerate(flight_options):
            # --- Subtask 1: Get weather information ---
            weather_info = self._get_weather_for_flight(flight)
            
            # --- Subtask 2: Get safety information ---
            safety_info = self._get_safety_for_flight(flight)
            
            # --- Subtask 3: Get economic information ---
            economic_info = self._get_economy_for_flight(flight)
            
            # --- Subtask 4: Compile flight details ---
            flight_details = self._compile_flight_details(flight)
            
            # --- Integrate information for this flight ---
            flight_score, confidence, reasoning = self._integrate_flight_information(
                flight, weather_info, safety_info, economic_info, flight_details
            )
            
            # Store result for this flight
            flight_results[flight] = {
                'weather_info': weather_info,
                'safety_info': safety_info,
                'economic_info': economic_info,
                'flight_details': flight_details,
                'score': flight_score,
                'confidence': confidence
            }
            
            # Add to overall recommendations
            overall_recommendations.append({
                'flight_id': flight,
                'score': flight_score,
                'agent_confidence': confidence,
                'reasoning': reasoning
            })
        
        # Sort recommendations by score
        overall_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'agent_id': self.agent_id,
            'success': True,
            'recommendations': overall_recommendations,
            'flight_details': flight_results,
            'agent_type': 'single_agent',
            'processing_time': len(flight_options) * 2.0  # Each flight takes time
        }
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate single agent results (simple passthrough)
        
        Args:
            agent_results: Agent processing results
            query_data: Query data
            
        Returns:
            Integrated results
        """
        # For single agent, we just need the recommendations
        single_agent_result = agent_results.get('single_agent', {})
        
        if not single_agent_result or not single_agent_result.get('success', False):
            # If processing failed, return empty results
            return {
                'query_id': query_data.get('query_id', 'unknown'),
                'success': False,
                'error': 'Single agent processing failed',
                'ranking': [],
                'recommendations': [],
                'model_name': self.model_name
            }
        
        # Get recommendations from single agent
        recommendations = single_agent_result.get('recommendations', [])
        
        # Create ranking from recommendations
        ranking = [rec['flight_id'] for rec in recommendations]
        
        return {
            'query_id': query_data.get('query_id', 'unknown'),
            'success': True,
            'ranking': ranking,
            'recommendations': recommendations,
            'system_confidence': 0.8,
            'processing_summary': {
                'selected_agents': [self.agent_id],
                'successful_agents': [self.agent_id] if single_agent_result.get('success', False) else [],
                'total_processing_time': single_agent_result.get('processing_time', 0.0)
            },
            'model_name': self.model_name
        }
    
    def _get_weather_for_flight(self, flight_id: str) -> Dict[str, Any]:
        """Get weather information for flight (simulates dedicated agent)"""
        # Simulate processing time
        time.sleep(0.1)
        
        # Simulate random weather conditions
        weather_score = random.uniform(0.6, 1.0)
        conditions = random.choice(['clear', 'partly cloudy', 'cloudy', 'rain', 'snow'])
        
        return {
            'weather_score': weather_score,
            'conditions': conditions,
            'temperature': random.randint(-10, 35),
            'wind_speed': random.randint(0, 50),
            'precipitation': random.uniform(0, 100),
            'confidence': random.uniform(0.7, 0.9)
        }
    
    def _get_safety_for_flight(self, flight_id: str) -> Dict[str, Any]:
        """Get safety information for flight (simulates dedicated agent)"""
        # Simulate processing time
        time.sleep(0.15)
        
        # Simulate random safety data
        safety_score = random.uniform(0.75, 0.98)
        
        return {
            'safety_score': safety_score,
            'airline_rating': random.uniform(3.0, 5.0),
            'maintenance_record': random.uniform(0.8, 1.0),
            'incident_history': random.uniform(0.0, 0.2),
            'confidence': random.uniform(0.8, 0.95)
        }
    
    def _get_economy_for_flight(self, flight_id: str) -> Dict[str, Any]:
        """Get economic information for flight (simulates dedicated agent)"""
        # Simulate processing time
        time.sleep(0.12)
        
        # Simulate random economic data
        base_price = 100 + random.randint(50, 500)
        
        return {
            'economic_score': random.uniform(0.6, 0.95),
            'price': base_price,
            'value_rating': random.uniform(0.5, 1.0),
            'price_trend': random.uniform(-0.2, 0.3),
            'confidence': random.uniform(0.75, 0.9)
        }
    
    def _compile_flight_details(self, flight_id: str) -> Dict[str, Any]:
        """Compile general flight details (simulates dedicated agent)"""
        # Simulate processing time
        time.sleep(0.08)
        
        # Parse flight ID to get airline code
        airline_code = flight_id[:2] if len(flight_id) > 2 else "AA"
        
        # Simulate flight details
        return {
            'flight_number': flight_id,
            'airline': airline_code,
            'aircraft_type': random.choice(['Boeing 737', 'Airbus A320', 'Boeing 777', 'Airbus A380']),
            'departure_time': f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            'arrival_time': f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            'duration': random.randint(60, 600),
            'confidence': random.uniform(0.85, 0.95)
        }
    
    def _integrate_flight_information(self, flight_id: str, weather: Dict[str, Any],
                                    safety: Dict[str, Any], economy: Dict[str, Any],
                                    details: Dict[str, Any]) -> Tuple[float, float, str]:
        """Integrate information from all subtasks for a single flight"""
        # Simulate processing time for integration
        time.sleep(0.05)
        
        # Calculate integrated score
        weather_factor = weather.get('weather_score', 0.8) * 0.2
        safety_factor = safety.get('safety_score', 0.9) * 0.4
        economy_factor = economy.get('economic_score', 0.7) * 0.3
        details_factor = 0.1
        
        # Calculate integrated score
        score = weather_factor + safety_factor + economy_factor + details_factor
        score = min(0.95, score)  # Cap at 0.95 to allow for randomness
        
        # Add small random factor
        score += random.uniform(-0.05, 0.05)
        score = max(0.5, min(0.98, score))
        
        # Calculate confidence
        confidence = (
            weather.get('confidence', 0.8) * 0.2 +
            safety.get('confidence', 0.9) * 0.4 +
            economy.get('confidence', 0.7) * 0.3 +
            details.get('confidence', 0.85) * 0.1
        )
        
        # Generate reasoning
        reasoning = (
            f"Flight {flight_id} scored {score:.2f} based on "
            f"weather ({weather.get('conditions', 'unknown')}), "
            f"safety rating ({safety.get('airline_rating', 0):.1f}/5.0), "
            f"and price (${economy.get('price', 0):.2f})."
        )
        
        return score, confidence, reasoning 