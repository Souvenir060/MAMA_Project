#!/usr/bin/env python3
"""
MAMA Full Model - Complete MAMA System Implementation
Including all features: SBERT Semantic Similarity + Trust Mechanism + Historical Performance + MARL Agent Selection
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class MAMAFull(BaseModel):
    """Complete MAMA model implementation"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize complete MAMA model
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_description = "Complete MAMA System - SBERT + Trust + Historical + MARL"
    
    def _initialize_model(self):
        """Initialize MAMA model specific components"""
        # Initialize SBERT simulator (in real implementation, this loads a pre-trained model)
        self.sbert_enabled = True
        
        # Initialize trust system
        self.trust_enabled = True
        
        # Initialize historical performance system
        self.historical_enabled = True
        
        # Initialize MARL system
        self.marl_enabled = True
        
        # MARL Q-tables (simplified implementation)
        self.q_tables = {}
        for agent_id in self.agents.keys():
            self.q_tables[agent_id] = np.random.uniform(0.5, 1.0, size=(10, 5))  # 10 states x 5 actions
        
        logger.info("✅ MAMA complete model initialized - all features enabled")
    
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Use complete MAMA algorithm to select agents
        
        Implementation formula: SelectionScore = α * SBERT_similarity + β * TrustScore + γ * Historical_performance
        
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
            
            # 2. Calculate trust score
            trust_score = self._calculate_trust_score(agent_id)
            
            # 3. Calculate historical performance
            historical_performance = self._calculate_historical_performance(agent_id)
            
            # 4. MARL Q-value
            marl_q_value = self._get_marl_q_value(agent_id, query_data)
            
            # 5. Calculate selection score (complete MAMA formula)
            selection_score = (
                self.config.alpha * sbert_similarity +
                self.config.beta * trust_score +
                self.config.gamma * historical_performance +
                0.1 * marl_q_value  # MARL bonus
            )
            
            agent_scores.append((agent_id, selection_score))
            
            logger.debug(f"Agent {agent_id}: SBERT={sbert_similarity:.3f}, Trust={trust_score:.3f}, "
                        f"Historical={historical_performance:.3f}, MARL={marl_q_value:.3f}, "
                        f"Final={selection_score:.3f}")
        
        # Sort by score and select top N
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agents = agent_scores[:self.config.max_agents]
        
        logger.info(f"🎯 Selected agents: {[f'{agent_id}({score:.3f})' for agent_id, score in selected_agents]}")
        
        return selected_agents
    
    def _get_marl_q_value(self, agent_id: str, query_data: Dict[str, Any]) -> float:
        """Get MARL Q-value"""
        # Simplified state representation
        query_complexity = query_data.get('metadata', {}).get('query_complexity', 0.5)
        state_index = min(int(query_complexity * 10), 9)
        
        # Random action (in real implementation would use policy)
        action_index = np.random.randint(0, 5)
        
        # Get Q-value
        q_value = self.q_tables[agent_id][state_index, action_index]
        
        return q_value
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Process query using selected agents
        
        Args:
            query_data: Query data
            selected_agents: Selected agents
            
        Returns:
            Agent processing results
        """
        agent_results = {}
        
        # Parallel agent execution (simulated)
        for agent_id, selection_score in selected_agents:
            start_time = time.time()
            
            # Simulate agent execution
            result = self._simulate_agent_execution(agent_id, query_data)
            
            # Add selection score information
            result['selection_score'] = selection_score
            result['processing_time'] = time.time() - start_time
            
            agent_results[agent_id] = result
            
            logger.debug(f"✅ {agent_id} execution completed: success={result['success']}, "
                        f"time={result['processing_time']:.2f}s")
        
        return agent_results
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate agent results using trust-weighted method
        
        Args:
            agent_results: Agent processing results
            query_data: Query data
            
        Returns:
            Integrated results
        """
        # Trust-weighted decision integration
        final_ranking = self._create_final_ranking(agent_results)
        
        # Generate comprehensive recommendations
        recommendations = self._create_comprehensive_recommendations(agent_results, final_ranking)
        
        # Calculate system confidence
        system_confidence = self._calculate_system_confidence(agent_results)
        
        # Record decision process
        decision_trace = self._create_decision_trace(agent_results, query_data)
        
        return {
            'query_id': query_data.get('query_id', 'unknown'),
            'success': len(agent_results) > 0,
            'ranking': final_ranking,
            'recommendations': recommendations,
            'system_confidence': system_confidence,
            'decision_trace': decision_trace,
            'agent_results': agent_results,
            'model_name': self.model_name,
            'processing_summary': {
                'selected_agents': list(agent_results.keys()),
                'successful_agents': [aid for aid, result in agent_results.items() 
                                    if result.get('success', False)],
                'total_processing_time': sum(result.get('processing_time', 0) 
                                           for result in agent_results.values())
            }
        }
    
    def _create_comprehensive_recommendations(self, agent_results: Dict[str, Any], 
                                            ranking: List[str]) -> List[Dict[str, Any]]:
        """Create comprehensive recommendations"""
        recommendations = []
        
        for i, flight_id in enumerate(ranking[:5]):  # Top 5 recommendations
            # Collect opinions from all agents for this flight
            agent_opinions = []
            total_score = 0.0
            total_confidence = 0.0
            
            for agent_id, result in agent_results.items():
                if result.get('success', False):
                    agent_recs = result.get('recommendations', [])
                    trust_score = self._calculate_trust_score(agent_id)
                    
                    for rec in agent_recs:
                        if rec['flight_id'] == flight_id:
                            weighted_score = rec['score'] * trust_score
                            total_score += weighted_score
                            total_confidence += rec.get('agent_confidence', 0.0) * trust_score
                            
                            agent_opinions.append({
                                'agent_id': agent_id,
                                'score': rec['score'],
                                'confidence': rec.get('agent_confidence', 0.0),
                                'reasoning': rec.get('reasoning', ''),
                                'trust_weight': trust_score
                            })
            
            if agent_opinions:
                avg_score = total_score / len(agent_opinions)
                avg_confidence = total_confidence / len(agent_opinions)
            else:
                avg_score = 0.5
                avg_confidence = 0.5
            
            recommendations.append({
                'rank': i + 1,
                'flight_id': flight_id,
                'overall_score': avg_score,
                'confidence': avg_confidence,
                'agent_consensus': len(agent_opinions),
                'agent_opinions': agent_opinions,
                'recommendation_strength': 'strong' if avg_score > 0.8 else 'moderate' if avg_score > 0.6 else 'weak'
            })
        
        return recommendations
    
    def _calculate_system_confidence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate overall system confidence"""
        if not agent_results:
            return 0.0
        
        # Based on number of successful agents
        successful_agents = [aid for aid, result in agent_results.items() 
                           if result.get('success', False)]
        success_ratio = len(successful_agents) / len(agent_results)
        
        # Based on agent trust scores
        total_trust = sum(self._calculate_trust_score(aid) for aid in successful_agents)
        avg_trust = total_trust / len(successful_agents) if successful_agents else 0.0
        
        # Based on agent consistency
        consistency_score = self._calculate_agent_consistency(agent_results)
        
        # Combined confidence
        system_confidence = 0.4 * success_ratio + 0.4 * avg_trust + 0.2 * consistency_score
        
        return min(system_confidence, 1.0)
    
    def _calculate_agent_consistency(self, agent_results: Dict[str, Any]) -> float:
        """Calculate consistency between agents"""
        successful_results = [result for result in agent_results.values() 
                            if result.get('success', False)]
        
        if len(successful_results) < 2:
            return 1.0  # With a single agent, consistency is perfect
        
        # Compare overlap of top 3 recommendations
        top3_lists = []
        for result in successful_results:
            recommendations = result.get('recommendations', [])
            top3 = [rec['flight_id'] for rec in recommendations[:3]]
            top3_lists.append(set(top3))
        
        # Calculate average overlap
        total_overlap = 0.0
        comparisons = 0
        
        for i in range(len(top3_lists)):
            for j in range(i + 1, len(top3_lists)):
                overlap = len(top3_lists[i].intersection(top3_lists[j]))
                total_overlap += overlap / 3.0  # Maximum overlap is 3
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        return total_overlap / comparisons
    
    def _create_decision_trace(self, agent_results: Dict[str, Any], 
                             query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create decision tracing information"""
        return {
            'query_analysis': {
                'query_text': query_data.get('query_text', ''),
                'query_complexity': query_data.get('metadata', {}).get('query_complexity', 0.0),
                'route_popularity': query_data.get('metadata', {}).get('route_popularity', 0.0)
            },
            'agent_selection': {
                'selection_algorithm': 'MAMA_Full',
                'selection_criteria': 'SBERT + Trust + Historical + MARL',
                'weights': {
                    'alpha': self.config.alpha,
                    'beta': self.config.beta,
                    'gamma': self.config.gamma
                }
            },
            'trust_evaluation': {
                agent_id: {
                    'trust_score': self._calculate_trust_score(agent_id),
                    'historical_performance': self._calculate_historical_performance(agent_id),
                    'selection_score': agent_results.get(agent_id, {}).get('selection_score', 0.0)
                }
                for agent_id in agent_results.keys()
            },
            'integration_method': 'trust_weighted_voting',
            'quality_indicators': {
                'agent_consensus': self._calculate_agent_consistency(agent_results),
                'system_confidence': self._calculate_system_confidence(agent_results),
                'successful_executions': len([r for r in agent_results.values() 
                                            if r.get('success', False)])
            }
        }
    
    def update_performance(self, query_data: Dict[str, Any], 
                         result: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        """
        Update model performance (MARL learning)
        
        Args:
            query_data: Query data
            result: Processing result
            feedback: User feedback
        """
        if not self.marl_enabled:
            return
        
        # Calculate reward
        reward = self._calculate_reward(result, feedback)
        
        # Update Q-tables
        for agent_id in result.get('agent_results', {}).keys():
            self._update_q_table(agent_id, query_data, reward)
        
        logger.debug(f"📚 MARL learning update: reward={reward:.3f}")
    
    def _calculate_reward(self, result: Dict[str, Any], 
                         feedback: Optional[Dict[str, Any]] = None) -> float:
        """Calculate MARL reward"""
        base_reward = 0.5
        
        # Reward based on system confidence
        confidence_bonus = result.get('system_confidence', 0.0) * 0.3
        
        # Penalty based on processing time
        processing_time = result.get('processing_summary', {}).get('total_processing_time', 5.0)
        time_penalty = max(0.0, (processing_time - 2.0) * 0.1)
        
        # Reward based on user feedback
        feedback_bonus = 0.0
        if feedback:
            user_rating = feedback.get('rating', 0.5)
            feedback_bonus = (user_rating - 0.5) * 0.4
        
        reward = base_reward + confidence_bonus - time_penalty + feedback_bonus
        return np.clip(reward, 0.0, 1.0)
    
    def _update_q_table(self, agent_id: str, query_data: Dict[str, Any], reward: float):
        """Update agent's Q-table"""
        # Simplified Q-learning update
        query_complexity = query_data.get('metadata', {}).get('query_complexity', 0.5)
        state_index = min(int(query_complexity * 10), 9)
        action_index = np.random.randint(0, 5)
        
        learning_rate = 0.1
        discount_factor = 0.95
        
        # Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        current_q = self.q_tables[agent_id][state_index, action_index]
        max_future_q = np.max(self.q_tables[agent_id][state_index, :])
        
        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        self.q_tables[agent_id][state_index, action_index] = new_q
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': 'MAMA_Full',
            'description': self.model_description,
            'features': {
                'sbert_similarity': self.sbert_enabled,
                'trust_mechanism': self.trust_enabled,
                'historical_performance': self.historical_enabled,
                'marl_learning': self.marl_enabled
            },
            'configuration': {
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'gamma': self.config.gamma,
                'max_agents': self.config.max_agents,
                'trust_threshold': self.config.trust_threshold
            },
            'agents': list(self.agents.keys()),
            'performance_history_size': len(self.performance_history)
        } 