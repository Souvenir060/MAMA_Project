#!/usr/bin/env python3
"""
MAMA System Base Model Class
Provides unified interface for all models (including complete MAMA and baseline models)
"""

import abc
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1
    
    trust_weights: Dict[str, float] = None
    
    max_agents: int = 3
    trust_threshold: float = 0.5
    response_timeout: float = 30.0
    
    random_seed: int = 42
    
    def __post_init__(self):
        if self.trust_weights is None:
            self.trust_weights = {
                'reliability': 0.25,
                'accuracy': 0.25,
                'consistency': 0.20,
                'transparency': 0.15,
                'robustness': 0.15
            }

class BaseModel(abc.ABC):
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model_name = self.__class__.__name__
        self.model_description = "Base Model"
        self.performance_history = []
        self.initialization_time = time.time()
        
        self.agents = self._initialize_agents()
        self._initialize_model()
        
        logger.info(f"✅ {self.model_name} initialized successfully")
    
    def _initialize_agents(self) -> Dict[str, Dict[str, Any]]:
        agents = {
            'safety_assessment_agent': {
                'name': 'Safety Assessment Agent',
                'expertise_keywords': ['safety', 'risk', 'incident', 'accident', 'security'],
                'trust_history': [0.8, 0.82, 0.85, 0.83, 0.87],
                'performance_history': [0.75, 0.78, 0.82, 0.80, 0.85],
                'response_time_history': [2.1, 1.9, 2.3, 2.0, 1.8],
                'specialization': 'aviation_safety'
            },
            'economic_agent': {
                'name': 'Economic Agent',
                'expertise_keywords': ['cost', 'price', 'budget', 'economic', 'financial'],
                'trust_history': [0.75, 0.78, 0.80, 0.82, 0.84],
                'performance_history': [0.72, 0.75, 0.78, 0.80, 0.82],
                'response_time_history': [1.8, 1.7, 1.9, 1.6, 1.5],
                'specialization': 'cost_analysis'
            },
            'weather_agent': {
                'name': 'Weather Agent',
                'expertise_keywords': ['weather', 'climate', 'meteorology', 'forecast', 'conditions'],
                'trust_history': [0.85, 0.87, 0.84, 0.89, 0.91],
                'performance_history': [0.82, 0.85, 0.83, 0.87, 0.89],
                'response_time_history': [1.5, 1.4, 1.6, 1.3, 1.2],
                'specialization': 'weather_analysis'
            },
            'flight_info_agent': {
                'name': 'Flight Information Agent',
                'expertise_keywords': ['flight', 'schedule', 'route', 'airline', 'aircraft'],
                'trust_history': [0.78, 0.80, 0.82, 0.84, 0.86],
                'performance_history': [0.76, 0.79, 0.81, 0.83, 0.85],
                'response_time_history': [2.0, 1.9, 2.1, 1.8, 1.7],
                'specialization': 'flight_information'
            },
            'integration_agent': {
                'name': 'Integration Agent',
                'expertise_keywords': ['integration', 'synthesis', 'ranking', 'decision', 'recommendation'],
                'trust_history': [0.82, 0.84, 0.86, 0.88, 0.90],
                'performance_history': [0.80, 0.82, 0.84, 0.86, 0.88],
                'response_time_history': [1.2, 1.1, 1.3, 1.0, 0.9],
                'specialization': 'result_integration'
            }
        }
        
        return agents
    
    @abc.abstractmethod
    def _initialize_model(self):
        pass
    
    def _calculate_semantic_similarity(self, query: str, agent_id: str) -> float:
        query_words = set(query.lower().split())
        agent_keywords = set(self.agents[agent_id]['expertise_keywords'])
        
        if not query_words or not agent_keywords:
            return 0.5
        
        intersection = query_words.intersection(agent_keywords)
        union = query_words.union(agent_keywords)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        base_similarity = 0.3 + 0.7 * jaccard_similarity
        
        noise = np.random.normal(0, 0.05)
        return np.clip(base_similarity + noise, 0.0, 1.0)
    
    def _calculate_trust_score(self, agent_id: str) -> float:
        trust_history = self.agents[agent_id]['trust_history']
        if not trust_history:
            return 0.5
        
        recent_scores = trust_history[-5:]
        base_trust = np.mean(recent_scores)
        
        trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores) if len(recent_scores) > 1 else 0
        
        trust_score = base_trust + 0.1 * trend
        return np.clip(trust_score, 0.0, 1.0)
    
    def _calculate_historical_performance(self, agent_id: str) -> float:
        performance_history = self.agents[agent_id]['performance_history']
        if not performance_history:
            return 0.5
        
        recent_performance = performance_history[-3:]
        return np.mean(recent_performance)
    
    def _simulate_agent_execution(self, agent_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_info = self.agents[agent_id]
        
        execution_time = np.random.uniform(1.0, 3.0)
        time.sleep(0.1)
        
        success_probability = np.mean(agent_info['trust_history'])
        success = np.random.random() < success_probability
        
        if success:
            num_recommendations = np.random.randint(3, 8)
            recommendations = []
            
            for i in range(num_recommendations):
                recommendations.append({
                    'flight_id': f"FL{np.random.randint(1000, 9999)}",
                    'score': np.random.uniform(0.6, 0.95),
                    'agent_confidence': np.random.uniform(0.7, 0.9),
                    'reasoning': f"Recommendation from {agent_info['name']}"
                })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
        else:
            recommendations = []
        
        return {
            'agent_id': agent_id,
            'agent_name': agent_info['name'],
            'success': success,
            'execution_time': execution_time,
            'recommendations': recommendations,
            'agent_confidence': np.random.uniform(0.6, 0.9) if success else 0.0,
            'error_message': None if success else "Agent execution failed"
        }
    
    def _create_final_ranking(self, agent_results: Dict[str, Any]) -> List[str]:
        flight_scores = {}
        
        for agent_id, result in agent_results.items():
            if not result.get('success', False):
                continue
            
            trust_weight = self._calculate_trust_score(agent_id)
            
            for rec in result.get('recommendations', []):
                flight_id = rec['flight_id']
                weighted_score = rec['score'] * trust_weight
                
                if flight_id not in flight_scores:
                    flight_scores[flight_id] = []
                flight_scores[flight_id].append(weighted_score)
        
        final_scores = {}
        for flight_id, scores in flight_scores.items():
            final_scores[flight_id] = np.mean(scores)
        
        ranked_flights = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [flight_id for flight_id, _ in ranked_flights]
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            selected_agents = self._select_agents(query_data)
            agent_results = self._process_with_agents(query_data, selected_agents)
            final_result = self._integrate_results(agent_results, query_data)
            
            final_result['processing_time'] = time.time() - start_time
            
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'query_id': query_data.get('query_id', 'unknown'),
                'success': final_result.get('success', False),
                'processing_time': final_result['processing_time'],
                'system_confidence': final_result.get('system_confidence', 0.0)
            })
            
            return final_result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'query_id': query_data.get('query_id', 'unknown'),
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'model_name': self.model_name
            }
    
    @abc.abstractmethod
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        pass
    
    @abc.abstractmethod
    def _process_with_agents(self, query_data: Dict[str, Any], selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        pass
    
    @abc.abstractmethod
    def _integrate_results(self, agent_results: Dict[str, Any], query_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {
                'total_queries': 0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'avg_confidence': 0.0
            }
        
        successful_queries = [q for q in self.performance_history if q['success']]
        
        return {
            'total_queries': len(self.performance_history),
            'success_rate': len(successful_queries) / len(self.performance_history),
            'avg_processing_time': np.mean([q['processing_time'] for q in self.performance_history]),
            'avg_confidence': np.mean([q.get('system_confidence', 0.0) for q in successful_queries]) if successful_queries else 0.0,
            'model_name': self.model_name,
            'uptime': time.time() - self.initialization_time
        }
    
    def reset_performance_history(self):
        self.performance_history = []
        logger.info(f"Performance history reset for {self.model_name}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        return {
            'agents': {
                agent_id: {
                    'name': info['name'],
                    'specialization': info['specialization'],
                    'avg_trust': np.mean(info['trust_history']),
                    'avg_performance': np.mean(info['performance_history']),
                    'avg_response_time': np.mean(info['response_time_history'])
                }
                for agent_id, info in self.agents.items()
            },
            'total_agents': len(self.agents)
        } 