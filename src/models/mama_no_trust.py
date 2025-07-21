#!/usr/bin/env python3
"""
MAMA No Trust Model - Ablation Study Baseline Model
Removes trust mechanism, keeping only SBERT semantic similarity and historical performance
Demonstrates the importance of trust mechanism
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy
from .base_model import BaseModel, ModelConfig
from .mama_full import MAMAFull

logger = logging.getLogger(__name__)

class MAMANoTrust(MAMAFull):
    """MAMA model without trust mechanism (ablation study)"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize MAMA model without trust
        
        Args:
            config: Model configuration
        """
        # Force beta (trust weight) to be 0
        modified_config = deepcopy(config) if config else ModelConfig()
        
        modified_config.beta = 0.0  # Remove trust mechanism
        
        super().__init__(modified_config)
        self.model_description = "Ablation Study Model - SBERT + Historical (No Trust Mechanism)"
    
    def _initialize_model(self):
        """Initialize MAMA model without trust"""
        # Initialize SBERT simulator
        self.sbert_enabled = True
        
        # Disable trust system
        self.trust_enabled = False
        
        # Initialize historical performance system
        self.historical_enabled = True
        
        # Initialize MARL system
        self.marl_enabled = True
        
        # MARL Q-tables (simplified implementation)
        self.q_tables = {}
        for agent_id in self.agents.keys():
            self.q_tables[agent_id] = np.random.uniform(0.5, 1.0, size=(10, 5))  # 10 states x 5 actions
        
        logger.info("✅ MAMA No Trust model initialized - Trust mechanism disabled")
    
    def _calculate_trust_score(self, agent_id: str) -> float:
        """
        Calculate trust score (always returns 0 in this model)
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Always 0.0 since trust is disabled
        """
        # Trust is disabled in this model, always returns 0
        return 0.0
    
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Select agents without using trust scores
        
        Implementation formula: SelectionScore = α * SBERT_similarity + γ * Historical_performance
        
        Args:
            query_data: Query data
            
        Returns:
            List of selected agents (agent_id, selection_score)
        """
        query_text = query_data.get('query_text', '')
        agent_scores = []
        
        for agent_id in self.agents.keys():
            # 1. Calculate SBERT semantic similarity
            sbert_similarity = self._calculate_semantic_similarity(query_text, agent_id)
            
            # 2. Calculate historical performance
            historical_performance = self._calculate_historical_performance(agent_id)
            
            # 3. MARL Q-value
            marl_q_value = self._get_marl_q_value(agent_id, query_data)
            
            # 4. Calculate selection score (without trust component)
            selection_score = (
                self.config.alpha * sbert_similarity +
                self.config.gamma * historical_performance +
                0.1 * marl_q_value
            )
            
            agent_scores.append((agent_id, selection_score))
            
            logger.debug(f"Agent {agent_id}: SBERT={sbert_similarity:.3f}, "
                        f"Historical={historical_performance:.3f}, MARL={marl_q_value:.3f}, "
                        f"Final={selection_score:.3f}")
        
        # Sort by score and select top N
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agents = agent_scores[:self.config.max_agents]
        
        logger.info(f"🎯 Selected agents (no trust): {[f'{agent_id}({score:.3f})' for agent_id, score in selected_agents]}")
        
        return selected_agents
    
    def _create_decision_trace(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create decision tracing information (modified for no-trust model)"""
        trace = super()._create_decision_trace(agent_results, query_data)
        
        # Update selection algorithm
        trace['agent_selection']['selection_algorithm'] = 'MAMA_NoTrust'
        trace['agent_selection']['selection_criteria'] = 'SBERT + Historical (No Trust)'
        
        # Remove trust evaluation
        del trace['trust_evaluation']
        trace['ablation_note'] = 'Trust mechanism removed for ablation study'
        
        return trace
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (modified for no-trust model)"""
        info = super().get_model_info()
        info['model_type'] = 'MAMA_NoTrust'
        info['description'] = self.model_description
        info['features']['trust_mechanism'] = False
        info['ablation_purpose'] = 'Evaluate impact of trust mechanism'
        
        return info 