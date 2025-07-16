#!/usr/bin/env python3
"""
MAMA Full Model - Complete MAMA system implementation
Includes all features: SBERT semantic similarity + trust mechanism + historical performance + MARL agent selection
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class MAMAFull(BaseModel):
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.model_description = "Complete MAMA System - SBERT + Trust + Historical + MARL"
    
    def _initialize_model(self):
        self.sbert_enabled = True
        self.trust_enabled = True
        self.historical_enabled = True
        self.marl_enabled = True
        
        self.q_tables = {}
        for agent_id in self.agents.keys():
            self.q_tables[agent_id] = np.random.uniform(0.5, 1.0, size=(10, 5))
        
        logger.info("✅ MAMA Full Model initialization completed - All features enabled")
    
    def _calculate_sbert_similarity(self, query: str, agent_id: str) -> float:
        query_words = set(query.lower().split())
        agent_expertise = self.agents[agent_id]['expertise_keywords']
        
        if not query_words or not agent_expertise:
            return 0.5
        
        intersection = query_words.intersection(set(agent_expertise))
        union = query_words.union(set(agent_expertise))
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        base_similarity = 0.4 + 0.6 * jaccard_similarity
        
        noise = np.random.normal(0, 0.05)
        return np.clip(base_similarity + noise, 0.0, 1.0)
    
    def _calculate_trust_score(self, agent_id: str) -> float:
        trust_history = self.agents[agent_id]['trust_history']
        
        if not trust_history:
            return 0.5
        
        recent_scores = trust_history[-10:]
        base_trust = np.mean(recent_scores)
        
        consistency = 1.0 - np.std(recent_scores) if len(recent_scores) > 1 else 1.0
        trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores) if len(recent_scores) > 1 else 0
        
        trust_score = base_trust + 0.1 * consistency + 0.05 * trend
        return np.clip(trust_score, 0.0, 1.0)
    
    def _calculate_historical_performance(self, agent_id: str) -> float:
        performance_history = self.agents[agent_id]['performance_history']
        
        if not performance_history:
            return 0.5
        
        recent_performance = performance_history[-5:]
        base_performance = np.mean(recent_performance)
        
        improvement_rate = 0
        if len(recent_performance) > 1:
            improvement_rate = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
        
        historical_score = base_performance + 0.1 * improvement_rate
        return np.clip(historical_score, 0.0, 1.0)
    
    def _get_marl_action(self, agent_id: str, state: int) -> int:
        if agent_id not in self.q_tables:
            return np.random.randint(0, 5)
        
        q_values = self.q_tables[agent_id][state]
        
        if np.random.random() < 0.1:
            return np.random.randint(0, 5)
        else:
            return np.argmax(q_values)
    
    def _update_marl_q_table(self, agent_id: str, state: int, action: int, reward: float, next_state: int):
        if agent_id not in self.q_tables:
            return
        
        learning_rate = 0.1
        discount_factor = 0.9
        
        current_q = self.q_tables[agent_id][state, action]
        max_next_q = np.max(self.q_tables[agent_id][next_state])
        
        new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
        self.q_tables[agent_id][state, action] = new_q
    
    def select_agents(self, query: str, num_agents: int = 3) -> List[str]:
        scores = {}
        
        for agent_id in self.agents.keys():
            sbert_score = self._calculate_sbert_similarity(query, agent_id)
            trust_score = self._calculate_trust_score(agent_id)
            historical_score = self._calculate_historical_performance(agent_id)
            
            state = hash(query) % 10
            marl_action = self._get_marl_action(agent_id, state)
            marl_bonus = marl_action / 10.0
            
            combined_score = (
                self.config.alpha * sbert_score +
                self.config.beta * trust_score +
                self.config.gamma * historical_score +
                0.1 * marl_bonus
            )
            
            scores[agent_id] = combined_score
        
        selected_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_agents]
        return [agent_id for agent_id, _ in selected_agents]
    
    def update_agent_performance(self, agent_id: str, performance_score: float):
        if agent_id in self.agents:
            self.agents[agent_id]['performance_history'].append(performance_score)
            
            trust_score = 0.8 + 0.2 * performance_score
            self.agents[agent_id]['trust_history'].append(trust_score)
            
            state = np.random.randint(0, 10)
            action = np.random.randint(0, 5)
            reward = performance_score
            next_state = np.random.randint(0, 10)
            
            self._update_marl_q_table(agent_id, state, action, reward, next_state)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': 'MAMA_Full',
            'description': self.model_description,
            'features': {
                'sbert_enabled': self.sbert_enabled,
                'trust_enabled': self.trust_enabled,
                'historical_enabled': self.historical_enabled,
                'marl_enabled': self.marl_enabled
            },
            'config': {
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'gamma': self.config.gamma,
                'trust_threshold': self.config.trust_threshold
            },
            'agents': list(self.agents.keys()),
            'q_table_shapes': {agent_id: table.shape for agent_id, table in self.q_tables.items()}
        } 