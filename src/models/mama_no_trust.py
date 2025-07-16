#!/usr/bin/env python3
"""
MAMA No Trust Model - Ablation Study Baseline Model
Removes trust mechanism, retains only SBERT semantic similarity and historical performance
Demonstrates the importance of trust mechanism
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base_model import BaseModel
from core.sbert_similarity import SBERTSimilarity

logger = logging.getLogger(__name__)

class MAMANoTrustModel(BaseModel):
    """MAMA model without trust mechanism (ablation study)"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MAMA model without trust
        
        Args:
            config: Model configuration
        """
        # Force set beta (trust weight) to 0
        config = config.copy()
        config['beta'] = 0.0  # Remove trust mechanism
        super().__init__(config)
        
        self.model_description = "Ablation Study Model - SBERT + Historical (No Trust Mechanism)"
    
    def _initialize_model(self):
        """Initialize MAMA model without trust"""
        # Initialize SBERT simulator
        self.sbert_similarity = SBERTSimilarity()
        
        # Initialize agent historical performance tracking
        self.agent_history = {
            'safety_assessment_agent': [],
            'economic_agent': [],
            'weather_agent': [],
            'flight_info_agent': [],
            'integration_agent': []
        }
        
        logger.info("MAMA No Trust model initialized (ablation study)")
    
    def calculate_agent_score(self, agent_id: str, query: str, context: Dict[str, Any] = None) -> float:
        """
        Calculate agent selection score without trust mechanism
        
        Args:
            agent_id: Agent identifier
            query: Query string
            context: Additional context
            
        Returns:
            Selection score (SBERT similarity + historical performance only)
        """
        # Calculate SBERT semantic similarity
        similarity_score = self.sbert_similarity.calculate_similarity(
            query, 
            agent_id,
            context
        )
        
        # Calculate historical performance (simple average)
        historical_score = self._get_historical_performance(agent_id)
        
        # Selection score without trust (beta = 0)
        # SelectionScore = α × SBERT_similarity + γ × Historical_performance
        selection_score = (
            self.config['alpha'] * similarity_score +
            self.config['gamma'] * historical_score
        )
        
        return selection_score
    
    def _get_historical_performance(self, agent_id: str) -> float:
        """Get historical performance score for agent"""
        if agent_id not in self.agent_history:
            return 0.5  # Default neutral score
        
        history = self.agent_history[agent_id]
        if not history:
            return 0.5
        
        # Simple average of historical performance
        return np.mean(history)
    
    def update_agent_performance(self, agent_id: str, performance_score: float):
        """Update agent historical performance"""
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = []
        
        self.agent_history[agent_id].append(performance_score)
        
        # Keep only recent history (sliding window)
        max_history = 100
        if len(self.agent_history[agent_id]) > max_history:
            self.agent_history[agent_id] = self.agent_history[agent_id][-max_history:]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'MAMA_NoTrust',
            'description': self.model_description,
            'components': ['SBERT', 'Historical_Performance'],
            'trust_mechanism': False,
            'alpha': self.config['alpha'],
            'beta': 0.0,  # Trust weight forced to 0
            'gamma': self.config['gamma']
        } 