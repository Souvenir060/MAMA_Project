"""
Trust Management Simulation System

This module generates realistic test data and scenarios for evaluating
the trust management system with authentic algorithms and metrics.
"""

import random
import time
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json

@dataclass
class TaskSimulation:
    """Simulates realistic task execution with varying outcomes"""
    task_type: str
    difficulty: float  # 0-1 scale
    expected_duration: float  # seconds
    success_probability: float  # base probability
    
class AgentPersonality:
    """Models agent behavioral characteristics"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.base_competence = self._get_base_competence(agent_id)
        self.learning_rate = random.uniform(0.01, 0.05)
        self.fatigue_factor = random.uniform(0.8, 1.0)
        self.bias_tendencies = self._generate_bias_tendencies()
        self.security_awareness = random.uniform(0.6, 0.95)
        
    def _get_base_competence(self, agent_id: str) -> float:
        """Set realistic base competence levels for different agents"""
        competence_profiles = {
            'WeatherAgent': random.uniform(0.75, 0.9),
            'FlightInfoAgent': random.uniform(0.7, 0.85),
            'SafetyAgent': random.uniform(0.8, 0.95),
            'EconomicAgent': random.uniform(0.65, 0.8),
            'IntegrationAgent': random.uniform(0.7, 0.85)
        }
        return competence_profiles.get(agent_id, random.uniform(0.6, 0.8))
        
    def _generate_bias_tendencies(self) -> Dict[str, float]:
        """Generate realistic bias tendencies"""
        return {
            'airline_preference': random.uniform(-0.2, 0.2),  # -0.2 to 0.2
            'price_bias': random.uniform(-0.15, 0.15),
            'time_preference': random.uniform(-0.1, 0.1),
            'route_bias': random.uniform(-0.1, 0.1)
        }

class WeatherDataSimulator:
    """Simulates realistic weather data and prediction accuracy"""
    
    def __init__(self):
        self.weather_patterns = {
            'clear': {'probability': 0.4, 'prediction_accuracy': 0.9},
            'cloudy': {'probability': 0.3, 'prediction_accuracy': 0.85},
            'rainy': {'probability': 0.2, 'prediction_accuracy': 0.75},
            'stormy': {'probability': 0.1, 'prediction_accuracy': 0.65}
        }
        
    def generate_weather_task(self, difficulty: float) -> Dict[str, Any]:
        """Generate realistic weather prediction task"""
        # Select weather type based on probability
        rand_val = random.random()
        cumulative_prob = 0
        selected_weather = 'clear'
        
        for weather_type, data in self.weather_patterns.items():
            cumulative_prob += data['probability']
            if rand_val <= cumulative_prob:
                selected_weather = weather_type
                break
        
        base_accuracy = self.weather_patterns[selected_weather]['prediction_accuracy']
        
        # Adjust accuracy based on task difficulty
        actual_accuracy = base_accuracy * (1 - difficulty * 0.3)
        
        # Add some randomness
        actual_accuracy += random.uniform(-0.1, 0.1)
        actual_accuracy = max(0.3, min(0.98, actual_accuracy))
        
        return {
            'task_type': 'weather_prediction',
            'weather_type': selected_weather,
            'difficulty': difficulty,
            'prediction_accuracy': actual_accuracy,
            'response_time': random.uniform(1.0, 5.0),
            'data_quality': random.uniform(0.7, 0.95)
        }

class FlightSearchSimulator:
    """Simulates flight search operations with realistic data"""
    
    def __init__(self):
        self.airlines = ['American', 'Delta', 'United', 'Southwest', 'JetBlue', 'Alaska']
        self.routes = [
            ('NYC', 'LAX'), ('NYC', 'SFO'), ('CHI', 'MIA'),
            ('BOS', 'SEA'), ('DEN', 'ATL'), ('LAS', 'NYC')
        ]
        
    def generate_flight_search_task(self, agent_personality: AgentPersonality) -> Dict[str, Any]:
        """Generate realistic flight search task with bias simulation"""
        origin, destination = random.choice(self.routes)
        
        # Generate flight options
        num_flights = random.randint(3, 8)
        flights = []
        
        for i in range(num_flights):
            airline = random.choice(self.airlines)
            base_price = random.uniform(200, 800)
            
            # Apply agent bias
            if airline in ['American', 'Delta']:  # Premium airlines
                price_bias = agent_personality.bias_tendencies['airline_preference']
                base_price *= (1 + price_bias)
            
            flights.append({
                'airline': airline,
                'price': round(base_price, 2),
                'duration': random.randint(180, 480),  # minutes
                'stops': random.choice([0, 1, 2]),
                'departure_time': self._generate_departure_time()
            })
        
        # Simulate search accuracy
        search_accuracy = agent_personality.base_competence + random.uniform(-0.1, 0.1)
        search_accuracy = max(0.5, min(0.98, search_accuracy))
        
        return {
            'task_type': 'flight_search',
            'origin': origin,
            'destination': destination,
            'flights_found': flights,
            'search_accuracy': search_accuracy,
            'response_time': random.uniform(2.0, 8.0),
            'api_success_rate': random.uniform(0.85, 0.98)
        }
    
    def _generate_departure_time(self) -> str:
        """Generate realistic departure time"""
        base_time = datetime.now() + timedelta(days=random.randint(1, 30))
        hour = random.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        minute = random.choice([0, 15, 30, 45])
        
        departure = base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return departure.isoformat()

class SafetyAssessmentSimulator:
    """Simulates safety risk assessment with realistic scenarios"""
    
    def __init__(self):
        self.risk_factors = {
            'weather_risk': {'weight': 0.3, 'base_probability': 0.15},
            'aircraft_age': {'weight': 0.2, 'base_probability': 0.1},
            'pilot_fatigue': {'weight': 0.2, 'base_probability': 0.05},
            'route_congestion': {'weight': 0.15, 'base_probability': 0.2},
            'maintenance_status': {'weight': 0.15, 'base_probability': 0.08}
        }
        
    def generate_safety_assessment_task(self, agent_personality: AgentPersonality) -> Dict[str, Any]:
        """Generate realistic safety assessment task"""
        # Calculate individual risk components
        risk_scores = {}
        overall_risk = 0
        
        for factor, data in self.risk_factors.items():
            # Simulate factor assessment accuracy
            assessment_noise = random.uniform(-0.1, 0.1)
            agent_competence_factor = agent_personality.base_competence
            
            actual_risk = data['base_probability'] + random.uniform(-0.05, 0.05)
            assessed_risk = actual_risk + assessment_noise * (1 - agent_competence_factor)
            assessed_risk = max(0, min(1, assessed_risk))
            
            risk_scores[factor] = {
                'actual_risk': actual_risk,
                'assessed_risk': assessed_risk,
                'weight': data['weight']
            }
            
            overall_risk += assessed_risk * data['weight']
        
        # Calculate assessment accuracy
        actual_overall = sum(scores['actual_risk'] * scores['weight'] 
                           for scores in risk_scores.values())
        assessment_accuracy = 1 - abs(overall_risk - actual_overall)
        assessment_accuracy = max(0.3, min(0.98, assessment_accuracy))
        
        return {
            'task_type': 'safety_assessment',
            'risk_factors': risk_scores,
            'overall_risk_score': overall_risk,
            'assessment_accuracy': assessment_accuracy,
            'response_time': random.uniform(3.0, 10.0),
            'confidence_level': agent_personality.base_competence + random.uniform(-0.1, 0.1)
        }

class EconomicAnalysisSimulator:
    """Simulates economic analysis with market dynamics"""
    
    def __init__(self):
        self.market_conditions = {
            'fuel_price': random.uniform(80, 120),  # per barrel
            'demand_level': random.uniform(0.6, 1.2),
            'seasonal_factor': random.uniform(0.8, 1.3),
            'competition_index': random.uniform(0.7, 1.1)
        }
        
    def generate_economic_analysis_task(self, agent_personality: AgentPersonality) -> Dict[str, Any]:
        """Generate realistic economic analysis task"""
        base_cost = random.uniform(300, 600)
        
        # Apply market factors
        fuel_impact = (self.market_conditions['fuel_price'] - 100) * 0.02
        demand_impact = (self.market_conditions['demand_level'] - 1) * 0.15
        seasonal_impact = (self.market_conditions['seasonal_factor'] - 1) * 0.1
        competition_impact = (1 - self.market_conditions['competition_index']) * 0.2
        
        actual_cost = base_cost * (1 + fuel_impact + demand_impact + seasonal_impact + competition_impact)
        
        # Agent prediction with competence factor
        prediction_error = random.uniform(-0.2, 0.2) * (1 - agent_personality.base_competence)
        predicted_cost = actual_cost * (1 + prediction_error)
        
        # Calculate accuracy
        cost_accuracy = 1 - abs(predicted_cost - actual_cost) / actual_cost
        cost_accuracy = max(0.4, min(0.95, cost_accuracy))
        
        # Hidden costs detection
        hidden_costs = {
            'baggage_fees': random.uniform(25, 50),
            'seat_selection': random.uniform(10, 30),
            'change_fees': random.uniform(50, 200)
        }
        
        detection_rate = agent_personality.base_competence + random.uniform(-0.15, 0.15)
        detection_rate = max(0.3, min(0.9, detection_rate))
        
        return {
            'task_type': 'economic_analysis',
            'base_cost': base_cost,
            'actual_total_cost': actual_cost,
            'predicted_cost': predicted_cost,
            'cost_accuracy': cost_accuracy,
            'hidden_costs': hidden_costs,
            'hidden_cost_detection': detection_rate,
            'market_conditions': self.market_conditions.copy(),
            'response_time': random.uniform(5.0, 15.0)
        }

class SecurityAttackSimulator:
    """Simulates various types of security attacks for testing agent resistance"""
    
    def __init__(self):
        self.attack_types = {
            'data_injection': {
                'severity': 'high',
                'detection_difficulty': 0.7,
                'impact_potential': 0.8
            },
            'api_manipulation': {
                'severity': 'medium',
                'detection_difficulty': 0.5,
                'impact_potential': 0.6
            },
            'social_engineering': {
                'severity': 'high',
                'detection_difficulty': 0.8,
                'impact_potential': 0.9
            },
            'ddos_simulation': {
                'severity': 'medium',
                'detection_difficulty': 0.3,
                'impact_potential': 0.5
            },
            'privilege_escalation': {
                'severity': 'critical',
                'detection_difficulty': 0.9,
                'impact_potential': 0.95
            }
        }
        
    def simulate_security_attacks(self, agent_personality: AgentPersonality, 
                                num_attacks: int = 10) -> Dict[str, Any]:
        """Simulate series of security attacks"""
        results = {
            'total_attacks': num_attacks,
            'successful_defenses': 0,
            'attack_details': []
        }
        
        for i in range(num_attacks):
            attack_type = random.choice(list(self.attack_types.keys()))
            attack_data = self.attack_types[attack_type]
            
            # Calculate defense success probability
            agent_defense_capability = agent_personality.security_awareness
            detection_difficulty = attack_data['detection_difficulty']
            
            defense_probability = agent_defense_capability * (1 - detection_difficulty * 0.7)
            defense_success = random.random() < defense_probability
            
            if defense_success:
                results['successful_defenses'] += 1
            
            attack_detail = {
                'attack_type': attack_type,
                'severity': attack_data['severity'],
                'defense_successful': defense_success,
                'detection_time': random.uniform(0.5, 10.0) if defense_success else None,
                'impact_mitigated': random.uniform(0.7, 0.95) if defense_success else 0
            }
            
            results['attack_details'].append(attack_detail)
        
        return results

class DecisionBiasSimulator:
    """Simulates decision-making scenarios to test for bias"""
    
    def __init__(self):
        self.decision_scenarios = [
            'airline_selection',
            'price_range_preference',
            'time_slot_selection',
            'route_preference'
        ]
        
    def generate_decision_history(self, agent_personality: AgentPersonality, 
                                num_decisions: int = 20) -> List[Dict[str, Any]]:
        """Generate decision history with realistic bias patterns"""
        decisions = []
        
        # Airline distribution with bias
        airlines = ['American', 'Delta', 'United', 'Southwest', 'JetBlue', 'Alaska']
        airline_weights = [1.0] * len(airlines)
        
        # Apply airline preference bias
        bias_strength = abs(agent_personality.bias_tendencies['airline_preference'])
        if agent_personality.bias_tendencies['airline_preference'] > 0:
            # Bias toward premium airlines
            airline_weights[0] *= (1 + bias_strength * 2)  # American
            airline_weights[1] *= (1 + bias_strength * 2)  # Delta
        else:
            # Bias toward budget airlines
            airline_weights[3] *= (1 + bias_strength * 2)  # Southwest
            airline_weights[4] *= (1 + bias_strength * 1.5)  # JetBlue
        
        for i in range(num_decisions):
            # Select airline based on weighted probabilities
            selected_airline = random.choices(airlines, weights=airline_weights)[0]
            
            # Generate price with bias
            base_price = random.uniform(200, 800)
            price_bias = agent_personality.bias_tendencies['price_bias']
            
            if price_bias > 0:  # Bias toward higher prices
                price = base_price * (1 + price_bias * random.uniform(0, 0.5))
            else:  # Bias toward lower prices  
                price = base_price * (1 + price_bias * random.uniform(-0.5, 0))
            
            decision = {
                'decision_id': f'decision_{i+1}',
                'timestamp': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                'recommended_flight': {
                    'airline': selected_airline,
                    'price': round(price, 2),
                    'duration': random.randint(120, 480),
                    'stops': random.choice([0, 1, 2])
                },
                'alternatives_considered': random.randint(3, 8),
                'decision_confidence': random.uniform(0.6, 0.95),
                'user_acceptance': random.random() < 0.7  # 70% acceptance rate
            }
            
            decisions.append(decision)
        
        return decisions

class ExplanationGenerator:
    """Generates realistic explanation logs with varying quality"""
    
    def __init__(self):
        self.explanation_templates = {
            'weather': [
                "Weather prediction based on {data_sources} with {confidence}% confidence",
                "Analysis shows {weather_type} conditions due to {factors}",
                "Current atmospheric patterns indicate {prediction} with {accuracy} accuracy"
            ],
            'flight': [
                "Flight selection based on price ({price}), duration ({duration}), and reliability score ({score})",
                "Recommended {airline} due to best value proposition considering all factors",
                "Selected option provides optimal balance of cost and convenience"
            ],
            'safety': [
                "Safety assessment considers {factors} with overall risk score of {risk_score}",
                "Risk analysis based on historical data and current conditions",
                "Safety evaluation incorporates multiple risk factors and mitigation strategies"
            ],
            'economic': [
                "Cost analysis includes base price, market conditions, and hidden fees totaling ${total_cost}",
                "Economic evaluation considers fuel prices, demand, and seasonal factors",
                "Price prediction based on market trends and historical patterns"
            ]
        }
        
    def generate_explanations(self, agent_id: str, task_results: List[Dict[str, Any]]) -> List[str]:
        """Generate explanations for task results"""
        explanations = []
        
        for task in task_results:
            task_type = task.get('task_type', 'general')
            
            if 'weather' in task_type:
                template = random.choice(self.explanation_templates['weather'])
                explanation = template.format(
                    data_sources=random.choice(['satellite data', 'weather stations', 'meteorological models']),
                    confidence=round(task.get('prediction_accuracy', 0.8) * 100, 1),
                    weather_type=task.get('weather_type', 'variable'),
                    factors=random.choice(['pressure systems', 'temperature gradients', 'humidity levels']),
                    prediction=task.get('weather_type', 'mixed'),
                    accuracy=round(task.get('prediction_accuracy', 0.8) * 100, 1)
                )
            elif 'flight' in task_type:
                template = random.choice(self.explanation_templates['flight'])
                flights = task.get('flights_found', [{}])
                selected_flight = flights[0] if flights else {}
                explanation = template.format(
                    price=selected_flight.get('price', 'N/A'),
                    duration=selected_flight.get('duration', 'N/A'),
                    score=round(task.get('search_accuracy', 0.8) * 100, 1),
                    airline=selected_flight.get('airline', 'N/A')
                )
            elif 'safety' in task_type:
                template = random.choice(self.explanation_templates['safety'])
                explanation = template.format(
                    factors=', '.join(task.get('risk_factors', {}).keys()),
                    risk_score=round(task.get('overall_risk_score', 0.1), 3)
                )
            elif 'economic' in task_type:
                template = random.choice(self.explanation_templates['economic'])
                explanation = template.format(
                    total_cost=round(task.get('actual_total_cost', 500), 2)
                )
            else:
                explanation = f"Task completed with {round(task.get('accuracy', 0.8) * 100, 1)}% accuracy"
            
            explanations.append(explanation)
        
        return explanations

class TrustSimulationEngine:
    """Main simulation engine that coordinates all simulation components"""
    
    def __init__(self):
        self.weather_sim = WeatherDataSimulator()
        self.flight_sim = FlightSearchSimulator()
        self.safety_sim = SafetyAssessmentSimulator()
        self.economic_sim = EconomicAnalysisSimulator()
        self.security_sim = SecurityAttackSimulator()
        self.bias_sim = DecisionBiasSimulator()
        self.explanation_gen = ExplanationGenerator()
        
        # Initialize agent personalities
        self.agent_personalities = {
            'WeatherAgent': AgentPersonality('WeatherAgent'),
            'FlightInfoAgent': AgentPersonality('FlightInfoAgent'),
            'SafetyAgent': AgentPersonality('SafetyAgent'),
            'EconomicAgent': AgentPersonality('EconomicAgent'),
            'IntegrationAgent': AgentPersonality('IntegrationAgent')
        }
        
    def simulate_agent_performance(self, agent_id: str, num_tasks: int = 10, 
                                 days_period: int = 30) -> Dict[str, Any]:
        """Simulate comprehensive agent performance over time"""
        if agent_id not in self.agent_personalities:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        personality = self.agent_personalities[agent_id]
        task_results = []
        
        # Generate tasks based on agent type
        for i in range(num_tasks):
            if agent_id == 'WeatherAgent':
                task_data = self.weather_sim.generate_weather_task(
                    difficulty=random.uniform(0.2, 0.8)
                )
            elif agent_id == 'FlightInfoAgent':
                task_data = self.flight_sim.generate_flight_search_task(personality)
            elif agent_id == 'SafetyAgent':
                task_data = self.safety_sim.generate_safety_assessment_task(personality)
            elif agent_id == 'EconomicAgent':
                task_data = self.economic_sim.generate_economic_analysis_task(personality)
            else:  # IntegrationAgent
                # Mix of different task types
                task_type = random.choice(['weather', 'flight', 'safety', 'economic'])
                if task_type == 'weather':
                    task_data = self.weather_sim.generate_weather_task(random.uniform(0.2, 0.8))
                elif task_type == 'flight':
                    task_data = self.flight_sim.generate_flight_search_task(personality)
                elif task_type == 'safety':
                    task_data = self.safety_sim.generate_safety_assessment_task(personality)
                else:
                    task_data = self.economic_sim.generate_economic_analysis_task(personality)
            
            # Add success/failure status based on accuracy
            accuracy = task_data.get('prediction_accuracy') or task_data.get('search_accuracy') or \
                      task_data.get('assessment_accuracy') or task_data.get('cost_accuracy', 0.8)
            
            task_data['status'] = 'success' if accuracy > 0.6 else 'failure'
            task_data['timestamp'] = (datetime.now() - timedelta(
                days=random.randint(0, days_period)
            )).isoformat()
            
            task_results.append(task_data)
        
        # Generate decision history
        decision_history = self.bias_sim.generate_decision_history(personality, num_tasks)
        
        # Simulate security attacks
        attack_results = self.security_sim.simulate_security_attacks(personality, num_attacks=5)
        
        # Generate explanations
        explanations = self.explanation_gen.generate_explanations(agent_id, task_results)
        
        # Calculate performance metrics
        performance_data = self._calculate_performance_metrics(agent_id, task_results)
        
        return {
            'agent_id': agent_id,
            'task_results': task_results,
            'performance_data': performance_data,
            'decision_history': decision_history,
            'attack_simulation': attack_results,
            'explanations': explanations,
            'simulation_period_days': days_period,
            'total_tasks': num_tasks
        }
    
    def _calculate_performance_metrics(self, agent_id: str, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate realistic performance metrics"""
        if not task_results:
            return {}
        
        successful_tasks = [t for t in task_results if t.get('status') == 'success']
        success_rate = len(successful_tasks) / len(task_results)
        
        # Agent-specific metrics
        if agent_id == 'WeatherAgent':
            accuracies = [t.get('prediction_accuracy', 0.8) for t in task_results]
            response_times = [t.get('response_time', 3.0) for t in task_results]
            
            return {
                'accuracy': np.mean(accuracies),
                'weather_accuracy': np.mean(accuracies),
                'avg_response_time': np.mean(response_times),
                'success_rate': success_rate,
                'efficiency': 1.0 / (np.mean(response_times) / 3.0)  # normalized to 3s baseline
            }
            
        elif agent_id == 'SafetyAgent':
            accuracies = [t.get('assessment_accuracy', 0.8) for t in task_results]
            response_times = [t.get('response_time', 5.0) for t in task_results]
            
            # Simulate correlation with actual events (would be real in production)
            actual_correlation = np.mean(accuracies) + random.uniform(-0.1, 0.1)
            actual_correlation = max(0.4, min(0.95, actual_correlation))
            
            return {
                'accuracy': np.mean(accuracies),
                'risk_accuracy': np.mean(accuracies),
                'actual_correlation': actual_correlation,
                'avg_response_time': np.mean(response_times),
                'success_rate': success_rate,
                'efficiency': 1.0 / (np.mean(response_times) / 5.0)
            }
            
        elif agent_id == 'EconomicAgent':
            cost_accuracies = [t.get('cost_accuracy', 0.7) for t in task_results]
            detection_rates = [t.get('hidden_cost_detection', 0.6) for t in task_results]
            response_times = [t.get('response_time', 8.0) for t in task_results]
            
            return {
                'accuracy': np.mean(cost_accuracies),
                'cost_accuracy': np.mean(cost_accuracies),
                'hidden_cost_detection': np.mean(detection_rates),
                'avg_response_time': np.mean(response_times),
                'success_rate': success_rate,
                'efficiency': 1.0 / (np.mean(response_times) / 8.0)
            }
            
        else:  # FlightInfoAgent or IntegrationAgent
            accuracies = [t.get('search_accuracy', 0.8) for t in task_results if 'search_accuracy' in t]
            if not accuracies:
                accuracies = [0.8]  # fallback
            response_times = [t.get('response_time', 4.0) for t in task_results]
            
            return {
                'accuracy': np.mean(accuracies),
                'search_accuracy': np.mean(accuracies),
                'avg_response_time': np.mean(response_times),
                'success_rate': success_rate,
                'efficiency': 1.0 / (np.mean(response_times) / 4.0)
            }
    
    def run_comprehensive_simulation(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive simulation for all agents"""
        results = {}
        
        for agent_id in self.agent_personalities.keys():
            print(f"Simulating {agent_id}...")
            num_tasks = random.randint(15, 25)  # Realistic task volume
            
            agent_results = self.simulate_agent_performance(
                agent_id=agent_id,
                num_tasks=num_tasks,
                days_period=days
            )
            
            results[agent_id] = agent_results
        
        return results
    
    def simulate_trust_evolution(self, agent_id: str, weeks: int = 12) -> List[Dict[str, Any]]:
        """Simulate trust score evolution over time"""
        evolution_data = []
        
        for week in range(weeks):
            # Simulate performance variation over time
            base_performance = self.agent_personalities[agent_id].base_competence
            
            # Add realistic performance variations
            weekly_variation = random.uniform(-0.1, 0.1)
            learning_effect = week * self.agent_personalities[agent_id].learning_rate * 0.5
            fatigue_effect = math.sin(week * 0.5) * 0.05  # Cyclical fatigue
            
            adjusted_performance = base_performance + weekly_variation + learning_effect + fatigue_effect
            adjusted_performance = max(0.3, min(0.95, adjusted_performance))
            
            # Generate week's data
            week_data = self.simulate_agent_performance(
                agent_id=agent_id,
                num_tasks=random.randint(8, 15),
                days_period=7
            )
            
            # Adjust performance based on calculated factors
            for task in week_data['task_results']:
                if 'prediction_accuracy' in task:
                    task['prediction_accuracy'] = max(0.3, min(0.95, 
                        task['prediction_accuracy'] * adjusted_performance / base_performance))
                if 'search_accuracy' in task:
                    task['search_accuracy'] = max(0.3, min(0.95,
                        task['search_accuracy'] * adjusted_performance / base_performance))
                if 'assessment_accuracy' in task:
                    task['assessment_accuracy'] = max(0.3, min(0.95,
                        task['assessment_accuracy'] * adjusted_performance / base_performance))
                if 'cost_accuracy' in task:
                    task['cost_accuracy'] = max(0.3, min(0.95,
                        task['cost_accuracy'] * adjusted_performance / base_performance))
            
            week_data['week'] = week + 1
            week_data['adjusted_performance'] = adjusted_performance
            evolution_data.append(week_data)
        
        return evolution_data

# Global simulation engine instance
simulation_engine = TrustSimulationEngine()

def generate_test_data(agent_id: str = None, weeks: int = 4) -> Dict[str, Any]:
    """Generate comprehensive test data for trust system evaluation"""
    if agent_id:
        return simulation_engine.simulate_agent_performance(agent_id, num_tasks=20, days_period=weeks * 7)
    else:
        return simulation_engine.run_comprehensive_simulation(days=weeks * 7)

def simulate_trust_scenarios() -> Dict[str, Any]:
    """Generate specific trust scenarios for testing"""
    scenarios = {
        'high_trust_scenario': {
            'agent_id': 'WeatherAgent',
            'performance_modifier': 1.2,  # Boost performance
            'description': 'Agent consistently performs above expectations'
        },
        'declining_trust_scenario': {
            'agent_id': 'FlightInfoAgent', 
            'performance_modifier': 0.7,  # Reduce performance
            'description': 'Agent showing declining performance over time'
        },
        'bias_detected_scenario': {
            'agent_id': 'EconomicAgent',
            'bias_amplifier': 2.0,  # Amplify existing biases
            'description': 'Agent showing significant decision bias'
        },
        'security_incident_scenario': {
            'agent_id': 'SafetyAgent',
            'attack_frequency': 3.0,  # More frequent attacks
            'description': 'Agent under increased security pressure'
        }
    }
    
    results = {}
    for scenario_name, config in scenarios.items():
        agent_id = config['agent_id']
        
        # Modify agent personality for scenario
        original_personality = simulation_engine.agent_personalities[agent_id]
        
        if 'performance_modifier' in config:
            original_personality.base_competence *= config['performance_modifier']
            original_personality.base_competence = max(0.1, min(0.98, original_personality.base_competence))
        
        if 'bias_amplifier' in config:
            for bias_type in original_personality.bias_tendencies:
                original_personality.bias_tendencies[bias_type] *= config['bias_amplifier']
        
        if 'attack_frequency' in config:
            original_personality.security_awareness *= (1 / config['attack_frequency'])
            original_personality.security_awareness = max(0.2, min(0.95, original_personality.security_awareness))
        
        # Generate scenario data
        scenario_data = simulation_engine.simulate_agent_performance(
            agent_id=agent_id,
            num_tasks=15,
            days_period=14
        )
        
        scenario_data['scenario_description'] = config['description']
        results[scenario_name] = scenario_data
        
        # Reset personality (in real system, would use copies)
        simulation_engine.agent_personalities[agent_id] = AgentPersonality(agent_id)
    
    return results 