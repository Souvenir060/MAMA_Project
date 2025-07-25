#!/usr/bin/env python3
"""
MAMA No Trust Model
MAMA System without Trust Mechanism (Ablation Study)

Implementation for ablation study:
- Agents without trust-aware selection
- No Multi-Dimensional Trust Ledger integration
- Equal weight agent selection (no trust bias)
- Standard MARL without trust weighting
- Used for comparing the impact of trust mechanism
"""

import numpy as np
import time
import logging
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MAMA components (without trust integration)
from core.sbert_similarity import SBERTSimilarityEngine
from models.base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class MAMANoTrust(BaseModel):
    """
    MAMA System without Trust Mechanism (Ablation Study)
    
    Implementation for ablation analysis:
    - Uses same agents as MAMA Full but without trust weighting
    - Agent selection based only on SBERT similarity
    - No trust-based learning or adaptation
    - Used to measure the contribution of trust mechanism
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize MAMA No Trust model - 移除trust机制的消融实验模型
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # 消融实验：移除trust机制，保持其他权重与MAMA Full一致
        # 最优参数：α=0.2, β=0.7, γ=0.15 → 去除β后，按比例重新分配
        total_non_trust_weight = 0.2 + 0.15  # α + γ = 0.35
        self.config.alpha = 0.2 / total_non_trust_weight * 1.0  # 重新分配到总权重1.0：约0.57
        self.config.beta = 0.0    # Trust score weight = 0 (消融实验关键设置)
        self.config.gamma = 0.15 / total_non_trust_weight * 1.0  # 重新分配：约0.43
        
        self.model_name = "MAMA No Trust"
        self.model_description = "MAMA system without trust mechanism for ablation study"
        
    def _initialize_model(self):
        """Initialize MAMA model without trust components"""
        logger.info("🚀 Initializing MAMA System (No Trust)")
        
        # No trust-related parameters
        self.system_interaction_count = 0
        
        # 1. Initialize SBERT engine only (no trust ledger)
        self.sbert = SBERTSimilarityEngine()
        logger.info("✅ SBERT Similarity Engine initialized")
        
        # 2. No Trust Ledger or MARL Engine
        
        # 3. Agent names (same as MAMA Full for fair comparison)
        self.agent_names = [
            'weather_agent', 'safety_assessment_agent', 
            'flight_info_agent', 'economic_agent', 'integration_agent'
        ]
        
        # 4. Initialize interaction history
        self.interaction_history = []
        self.system_rewards = []
        
        logger.info("✅ MAMA System (No Trust) initialization completed")
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query using MAMA system without trust mechanism
        
        Implements MAMA framework without trust:
        1. Agent selection based only on SBERT similarity
        2. Equal weight agent coordination
        3. Standard result integration without trust weighting
        4. No competence update or trust record persistence
        
        Args:
            query_data: Query data with flight candidates
            
        Returns:
            MAMA system response without trust features
        """
        start_time = time.time()
        self.system_interaction_count += 1
        
        try:
            logger.info(f"🎯 Processing query {self.system_interaction_count} with MAMA (No Trust)")
            
            # Step 1: Agent Selection using only SBERT similarity
            selected_agents = self._select_agents(query_data)
            
            # Step 2: Agent Execution (same as MAMA Full)
            agent_results = self._process_with_agents(query_data, selected_agents)
            
            # Step 3: Equal Weight Integration (no trust weighting)
            integrated_results = self._integrate_results(agent_results, query_data)
            
            # Step 4: Calculate System Reward (for comparison)
            system_reward = self._calculate_system_reward(integrated_results, query_data)
            self.system_rewards.append(system_reward)
            
            # Step 5: No competence update or trust recording
            
            processing_time = time.time() - start_time
            
            # Prepare final response
            response = {
                'query_id': query_data.get('query_id', f'query_{self.system_interaction_count}'),
                'success': True,
                'recommendations': integrated_results.get('recommendations', []),
                'ranking': integrated_results.get('ranking', []),
                'system_reward': system_reward,
                'selected_agents': [aid for aid, _ in selected_agents],
                'processing_time': processing_time,
                'interaction_count': self.system_interaction_count,
                'model_name': 'MAMA_NoTrust',
                'metadata': {
                    'trust_updated': False,
                    'competence_updated': False,
                    'marl_selection': False,
                    'integration_method': 'equal_weight'
                }
            }
            
            logger.info(f"✅ Query processed successfully (No Trust), reward: {system_reward:.4f}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                'query_id': query_data.get('query_id', 'unknown'),
                'success': False,
                'error': str(e),
                'recommendations': [],
                'ranking': [],
                'processing_time': time.time() - start_time,
                'model_name': 'MAMA_NoTrust'
            }
    
    def _select_agents_similarity_only(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Select agents using only SBERT semantic similarity (no trust)
        
        Implementation for ablation study
        """
        try:
            query_text = query_data.get('query_text', '')
            agent_scores = []
            
            for agent_id in self.agent_names:
                # Calculate only SBERT semantic similarity
                sbert_similarity = self._calculate_semantic_similarity(query_text, agent_id)
                
                # Selection score is only based on semantic similarity
                selection_score = sbert_similarity
                
                agent_scores.append((agent_id, selection_score))
            
            # Sort and select top agents
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            selected = agent_scores[:self.config.max_agents]
            
            logger.info(f"🎯 SBERT-only selected agents: {[f'{aid}({score:.3f})' for aid, score in selected]}")
            return selected
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            # Fallback: select first N agents
            return [(aid, 0.5) for aid in self.agent_names[:self.config.max_agents]]
    
    def _execute_agents(self, selected_agents: List[Tuple[str, float]], 
                       query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agents using flight data processing
        
        Same execution as MAMA Full but without trust considerations
        """
        agent_results = {}
        
        for agent_id, selection_score in selected_agents:
            start_time = time.time()
            
            try:
                # Use flight data processing (no agents to keep comparison fair)
                result = self._process_with_flight_data(agent_id, query_data)
                success = len(result.get('recommendations', [])) > 0
                
                agent_results[agent_id] = {
                    'success': success,
                    'result': result,
                    'selection_score': selection_score,
                    'processing_time': time.time() - start_time,
                    'agent_type': 'data_processor'
                }
                
                logger.debug(f"✅ {agent_id} executed: success={success}")
                
            except Exception as e:
                logger.warning(f"Agent {agent_id} execution failed: {e}")
                agent_results[agent_id] = {
                    'success': False,
                    'error': str(e),
                    'selection_score': selection_score,
                    'processing_time': time.time() - start_time,
                    'agent_type': 'failed'
                }
        
        return agent_results
    
    def _process_with_flight_data(self, agent_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using flight data (same as MAMA Full)"""
        flight_candidates = query_data.get('flight_candidates', [])
        if not flight_candidates:
            flight_candidates = query_data.get('candidate_flights', [])

        if not flight_candidates:
            return {'success': False, 'recommendations': []}

        # 🔧 FAIRNESS FIX: Process ALL flights like other models
        recommendations = []
        for i, flight in enumerate(flight_candidates):  # Process ALL flights for fair comparison
            score = self._calculate_agent_specialty_score(agent_id, flight)
            recommendations.append({
                'flight_id': flight.get('flight_id', f'flight_{i}'),
                'score': score,
                'agent_reasoning': f"{agent_id} analysis (no trust)"
            })

        return {
            'success': True,
            'recommendations': recommendations,
            'agent_specialty': agent_id
        }
    
    def _calculate_agent_specialty_score(self, agent_id: str, flight: Dict[str, Any]) -> float:
        """🎯 MAMA No Trust: Balanced multi-agent without trust weighting"""
        # 🎯 Use flight scores for better differentiation
        safety_score = flight.get('safety_score', 0.5)
        price_score = flight.get('price_score', 0.5)
        convenience_score = flight.get('convenience_score', 0.5)
        
        # 🚀 Standard Agent specialization without trust
        if 'economic' in agent_id.lower():
            # Economic agent: price-focused but considers safety
            score = 0.7 * price_score + 0.2 * safety_score + 0.1 * convenience_score
        elif 'safety' in agent_id.lower():
            # Safety agent: safety-first approach
            score = 0.8 * safety_score + 0.1 * price_score + 0.1 * convenience_score  
        elif 'weather' in agent_id.lower():
            # Weather agent: balanced approach with slight safety bias
            score = 0.4 * safety_score + 0.3 * convenience_score + 0.3 * price_score
        elif 'flight' in agent_id.lower():
            # Flight info agent: convenience and timing focused
            score = 0.6 * convenience_score + 0.2 * safety_score + 0.2 * price_score
        elif 'integration' in agent_id.lower():
            # Integration agent: completely balanced
            score = 0.33 * safety_score + 0.33 * price_score + 0.34 * convenience_score
        else:
            # Default: slightly safety-biased
            score = 0.4 * safety_score + 0.3 * price_score + 0.3 * convenience_score
        
        # 🔧 FIX: No diversity factor for fair comparison
        
        return min(1.0, max(0.1, score))
    
    def _integrate_without_trust(self, agent_results: Dict[str, Any], 
                               query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate agent results using equal weighting (no trust)
        
        Implementation for ablation study
        """
        all_recommendations = []
        flight_scores = {}
        
        # Collect recommendations from all agents with equal weights
        for agent_id, agent_result in agent_results.items():
            if not agent_result.get('success', False):
                continue
            
            # Equal weight for all agents (no trust weighting)
            equal_weight = 1.0 / len(agent_results)
            
            # Process agent recommendations
            recommendations = agent_result.get('result', {}).get('recommendations', [])
            for rec in recommendations:
                flight_id = rec.get('flight_id', '')
                agent_score = rec.get('score', 0.5)
                
                # Equal-weighted score
                weighted_score = equal_weight * agent_score
                
                if flight_id in flight_scores:
                    flight_scores[flight_id] += weighted_score
                else:
                    flight_scores[flight_id] = weighted_score
        
        # Create final ranking
        sorted_flights = sorted(flight_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = [flight_id for flight_id, score in sorted_flights]
        
        # Create final recommendations
        final_recommendations = []
        for i, (flight_id, score) in enumerate(sorted_flights[:10]):
            final_recommendations.append({
                'flight_id': flight_id,
                'overall_score': score,
                'rank': i + 1,
                'trust_weighted': False
            })
        
        return {
            'recommendations': final_recommendations,
            'ranking': ranking,
            'integration_method': 'equal_weight',
            'total_flights_scored': len(flight_scores)
        }
    
    def _calculate_system_reward(self, integrated_results: Dict[str, Any], 
                               query_data: Dict[str, Any]) -> float:
        """
        Calculate system reward using Equation 12 from paper:
        r = λ1 · MRR + λ2 · NDCG@5 - λ3 · ART
        
        IDENTICAL implementation to MAMA Full for fair ablation study comparison
        """
        try:
            # Extract and validate data
            recommendations = integrated_results.get('recommendations', [])
            if not recommendations:
                return 0.0
            
            # Extract ground truth from query
            ground_truth_ranking = query_data.get('ground_truth_ranking', [])
            ground_truth_id = ground_truth_ranking[0] if ground_truth_ranking else ''
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr_score = 0.0
            if ground_truth_id:
                recommendation_ids = [rec.get('flight_id', '') for rec in recommendations]
                try:
                    rank = recommendation_ids.index(ground_truth_id) + 1  # 1-indexed
                    mrr_score = 1.0 / rank
                except ValueError:
                    mrr_score = 0.0  # Ground truth not found in recommendations
            
            # Calculate NDCG@5 using relevance scores
            ndcg5_score = 0.0
            if len(recommendations) >= 5:
                # Extract relevance scores
                relevance_scores = []
                for i, rec in enumerate(recommendations[:5]):
                    flight_id = rec.get('flight_id', '')
                    # Generate relevance based on position and score
                    if ground_truth_ranking and flight_id in ground_truth_ranking:
                        position = ground_truth_ranking.index(flight_id)
                        relevance = max(0.1, 1.0 - (position / len(ground_truth_ranking)))
                    else:
                        # Use overall score as relevance proxy
                        relevance = rec.get('overall_score', 0.5)
                    relevance_scores.append(relevance)
                
                # Calculate DCG@5
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
                
                # Calculate IDCG@5 (ideal ranking)
                ideal_relevance = sorted(relevance_scores, reverse=True)
                idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
                
                # NDCG@5
                ndcg5_score = dcg / idcg if idcg > 0 else 0.0
            
            # Calculate ART (Average Response Time) - get from processing time
            processing_time = integrated_results.get('processing_time', 0.5)
            # Normalize: lower time is better, so use (1 - normalized_time)
            art_penalty = min(1.0, processing_time / 5.0)  # Assume 5s is max reasonable time
            
            # Apply Equation 12 from paper with hyperparameters
            # λ1 = 0.6 (MRR is most important for ranking quality)
            # λ2 = 0.3 (NDCG@5 is important for top-k relevance) 
            # λ3 = 0.1 (Response time penalty, but less critical)
            system_reward = (
                0.6 * mrr_score +          # λ1 · MRR
                0.3 * ndcg5_score -        # λ2 · NDCG@5  
                0.1 * art_penalty          # λ3 · ART (penalty for slow response)
            )
            
            # Ensure reward is in [0, 1] range
            return min(1.0, max(0.0, system_reward))
            
        except Exception as e:
            logger.error(f"System reward calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.05  # Very minimal reward for failed calculations
    
    def get_system_rewards(self) -> List[float]:
        """Get system reward history"""
        return self.system_rewards.copy() 

    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Select agents using SBERT similarity only (BaseModel compatibility)
        
        Args:
            query_data: Query data
            
        Returns:
            List of selected agent IDs with selection scores
        """
        return self._select_agents_similarity_only(query_data)
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Process query with selected agents (BaseModel compatibility)
        
        Args:
            query_data: Query data
            selected_agents: Selected agents
            
        Returns:
            Agent processing results
        """
        return self._execute_agents(selected_agents, query_data)
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate results using equal weighting (BaseModel compatibility)
        
        Args:
            agent_results: Agent processing results
            query_data: Query data
            
        Returns:
            Integrated results
        """
        return self._integrate_without_trust(agent_results, query_data) 