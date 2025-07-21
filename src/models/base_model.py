#!/usr/bin/env python3
"""
MAMA System Base Model Class
Provides unified interface for all models (including full MAMA and baseline models)
"""

import os
import json
import logging
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

# Import MAMA system core components
from ..core.sbert_similarity import SBERTSimilarity
from ..agents.manager import AgentManager

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model Configuration"""
    # Selection score weights
    alpha: float = 0.7  # SBERT similarity weight
    beta: float = 0.2   # Trust score weight
    gamma: float = 0.1  # Historical performance weight
    
    # Trust score thresholds
    trust_threshold: float = 0.5  # Minimum trust score for selection
    
    # System parameters
    max_agents: int = 3  # Maximum number of agents to select
    max_interactions: int = 50  # Maximum number of interactions to track
    confidence_threshold: float = 0.7  # Minimum confidence for valid response
    
    # Random seed
    random_seed: int = 42  # Fixed seed for reproducible experiments

class BaseModel(ABC):
    """Base Model Abstract Class"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize base model
        
        Args:
            config: Model configuration
        """
        # Set random seed
        self.config = config if config is not None else ModelConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Model information
        self.model_name = self.__class__.__name__
        self.model_description = "Base Model"
        
        # Initialize SBERT similarity engine
        self.sbert = SBERTSimilarity()
        self.sbert_enabled = False
        
        # Agent management
        self.agent_manager = AgentManager()
        self.agents = self.agent_manager.get_all_agents()
        
        # Performance tracking
        self.performance_history = []
        
        # Initialize model-specific components
        self._initialize_model()
        logger.info(f"✅ {self.model_name} initialized")
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize model specific components"""
        pass
    
    def _calculate_semantic_similarity(self, query: str, agent_id: str) -> float:
        """
        Calculate semantic similarity between query and agent expertise
        
        Args:
            query: User query text
            agent_id: Agent identifier
            
        Returns:
            Similarity score [0.0-1.0]
        """
        if not self.sbert_enabled:
            # Return random similarity if SBERT is disabled
            return random.uniform(0.5, 1.0)
        
        # Get agent expertise description
        agent = self.agents.get(agent_id)
        if not agent:
            return 0.0
            
        expertise_desc = agent.get('expertise', '')
        
        # Calculate similarity using SBERT
        similarity = self.sbert.calculate_similarity(query, expertise_desc)
        
        return similarity
    
    def _calculate_trust_score(self, agent_id: str) -> float:
        """
        Calculate agent trust score
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Trust score [0.0-1.0]
        """
        # For base model, return default trust
        return 0.75
    
    def _calculate_historical_performance(self, agent_id: str) -> float:
        """
        Calculate agent historical performance
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Performance score [0.0-1.0]
        """
        # For base model, return default performance
        return 0.7
    
    def _simulate_agent_execution(self, agent_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate agent execution
        
        Args:
            agent_id: Agent identifier
            query_data: Query data
            
        Returns:
            Agent execution results
        """
        # Get agent
        agent = self.agents.get(agent_id)
        if not agent:
            return {
                'success': False,
                'error': f"Agent {agent_id} not found"
            }
        
        # Simulate success probability based on agent quality
        success_prob = agent.get('quality', 0.9)
        success = random.random() < success_prob
        
        if success:
            # Generate simulated recommendations
            recommendations = self._generate_recommendations(agent_id, query_data)
            
            return {
                'success': True,
                'agent_id': agent_id,
                'agent_type': agent.get('type', 'unknown'),
                'recommendations': recommendations,
                'agent_confidence': random.uniform(0.7, 0.95)
            }
        else:
            return {
            'success': False,
                'agent_id': agent_id,
                'error': "Simulated execution failure"
        }
    
    def _generate_recommendations(self, agent_id: str, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate simulated recommendations from an agent
        
        Args:
            agent_id: Agent identifier
            query_data: Query data
            
        Returns:
            List of recommendations
        """
        recs = []
        flight_options = query_data.get('flight_options', [])
        
        if not flight_options:
            # Generate dummy flight IDs
            flight_options = [f"flight_{i:04d}" for i in range(10)]
        
        for flight_id in flight_options:
            score = random.uniform(0.5, 1.0)
            rec = {
                'flight_id': flight_id,
                'score': score,
                'agent_confidence': random.uniform(0.7, 0.95),
                'reasoning': f"Agent {agent_id} recommendation"
            }
            recs.append(rec)
        
        # Sort by score
        recs.sort(key=lambda x: x['score'], reverse=True)
        
        return recs
    
    def _create_final_ranking(self, agent_results: Dict[str, Any]) -> List[str]:
        """
        Create final flight ranking from agent results
        
        Args:
            agent_results: Agent execution results
            
        Returns:
            Ordered list of flight IDs
        """
        # Collect all flight scores
        flight_scores = {}
        
        for agent_id, result in agent_results.items():
            if not result.get('success', False):
                continue
                
            recommendations = result.get('recommendations', [])
            
            for rec in recommendations:
                flight_id = rec.get('flight_id')
                score = rec.get('score', 0.5)
                
                if flight_id not in flight_scores:
                    flight_scores[flight_id] = []
                    
                flight_scores[flight_id].append(score)
        
        # Calculate average score for each flight
        final_scores = {}
        for flight_id, scores in flight_scores.items():
            final_scores[flight_id] = sum(scores) / len(scores)
            
        # Sort flights by score
        ranked_flights = sorted(final_scores.keys(), 
                               key=lambda fid: final_scores[fid],
                               reverse=True)
                               
        return ranked_flights
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query using the model
        
        Args:
            query_data: Query data including text and parameters
            
        Returns:
            Processing results including rankings and recommendations
        """
        # Track the start of processing time
        start_time = 0
        
        try:
            # 1. Select agents based on model-specific strategy
            selected_agents = self._select_agents(query_data)
            
            # 2. Process query with selected agents
            agent_results = self._process_with_agents(query_data, selected_agents)
            
            # 3. Integrate results
            integrated_results = self._integrate_results(agent_results, query_data)
            
            return integrated_results
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': self.model_name
            }
    
    @abstractmethod
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Select appropriate agents based on query
        
        Args:
            query_data: Query data
            
        Returns:
            List of selected agent IDs with selection scores
        """
        pass
    
    @abstractmethod
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Process query with selected agents
        
        Args:
            query_data: Query data
            selected_agents: Selected agents
            
        Returns:
            Agent processing results
        """
        pass
    
    @abstractmethod
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate results from multiple agents
        
        Args:
            agent_results: Results from each agent
            query_data: Original query data
            
        Returns:
            Integrated results
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'description': self.model_description,
            'configuration': {
                'alpha': self.config.alpha,
                'beta': self.config.beta, 
                'gamma': self.config.gamma,
                'max_agents': self.config.max_agents
            }
        } 