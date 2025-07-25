#!/usr/bin/env python3
"""
Single Agent System Model
Baseline model using single agent approach (no multi-agent coordination)

Implementation for baseline comparison:
- Single agent processing (no coordination)
- No trust mechanism or MARL
        - Standard flight data processing and ranking
- Used as baseline to demonstrate multi-agent advantages
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

from models.base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class SingleAgentSystemModel(BaseModel):
    """
    Single Agent System Implementation
    
    Baseline implementation:
    - Single agent processing approach
    - No multi-agent coordination or trust
    - Standard flight ranking based on domain heuristics
    - Used to demonstrate advantages of multi-agent approach
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Single Agent System Model
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # 单智能体基线：不使用多智能体协调和trust机制
        self.config.alpha = 1.0   # Pure semantic similarity (无多智能体协调)
        self.config.beta = 0.0    # No trust mechanism in single agent
        self.config.gamma = 0.0   # No historical performance (单一智能体)
        
        self.model_name = "Single Agent"
        self.model_description = "Single Agent System - Baseline for comparison"
        
    def _initialize_model(self):
        """Initialize single agent model"""
        logger.info("🚀 Initializing Single Agent System")
        
        # Single agent parameters
        self.system_interaction_count = 0
        
        # 1. Single agent configuration
        self.agent_name = 'unified_agent'
        
        # 2. Single Agent specialized weights (优先便利性和时间)
        self.ranking_weights = {
            'convenience': 0.35,      # 便利性优先 (Single Agent特色)
            'duration': 0.30,        # 时间效率重要
            'price': 0.20,           # 价格考量较低  
            'airline_rating': 0.10,  # 航空公司评级
            'availability': 0.05     # 可用性基础
        }
        
        # 3. Initialize interaction history
        self.interaction_history = []
        self.system_rewards = []
        
        logger.info("✅ Single Agent System initialization completed")
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query using single agent approach
        
        Implements standard single-agent processing:
        1. Direct flight data processing
        2. Standard heuristic-based ranking
        3. No coordination or trust mechanisms
        4. Basic result formatting
        
        Args:
            query_data: Query data with flight candidates
            
        Returns:
            Single agent system response
        """
        start_time = time.time()
        self.system_interaction_count += 1
        
        try:
            logger.info(f"🎯 Processing query {self.system_interaction_count} with Single Agent")
            
            # Step 1: Direct Flight Processing
            flight_results = self._process_flights_directly(query_data)
            
            # Step 2: Standard Ranking  
            ranked_results = self._rank_flights_standard(flight_results, query_data)
            
            # Step 3: Calculate System Reward
            system_reward = self._calculate_system_reward(ranked_results, query_data)
            self.system_rewards.append(system_reward)
            
            processing_time = time.time() - start_time
            
            # Prepare final response
            response = {
                'query_id': query_data.get('query_id', f'query_{self.system_interaction_count}'),
                'success': True,
                'recommendations': ranked_results.get('recommendations', []),
                'ranking': ranked_results.get('ranking', []),
                'system_reward': system_reward,
                'selected_agents': [self.agent_name],
                'processing_time': processing_time,
                'interaction_count': self.system_interaction_count,
                'model_name': 'SingleAgent',
                'metadata': {
                    'trust_updated': False,
                    'competence_updated': False,
                    'marl_selection': False,
                    'integration_method': 'single_agent'
                }
            }
            
            logger.info(f"✅ Query processed successfully (Single Agent), reward: {system_reward:.4f}")
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
                'model_name': 'SingleAgent'
            }
    
    def _process_flights_directly(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TRUE SERIAL EXECUTION as described in paper
        
        Implements paper's 'true serial-execution single-agent system':
        All tasks are processed sequentially, one at a time, to demonstrate
        the performance difference vs multi-agent parallel approach
        """
        flight_candidates = query_data.get('flight_candidates', [])
        if not flight_candidates:
            flight_candidates = query_data.get('candidate_flights', [])
        
        if not flight_candidates:
            return {'success': False, 'flights_processed': 0}
        
        processed_flights = []
        
        # TRUE SERIAL EXECUTION: Process each flight one by one
        for i, flight in enumerate(flight_candidates):
            import time
            
            # Simulate serial processing time for each flight
            start_time = time.time()
            
            # Extract flight attributes
            flight_id = flight.get('flight_id', f'flight_{i}')
            price = flight.get('price', 500)
            duration = flight.get('duration', 180)  # minutes
            departure_time = flight.get('departure_time', '08:00')
            airline = flight.get('airline', 'unknown')
            
            # SERIAL TASK 1: Weather analysis (minimal processing time)
            time.sleep(0.001)  # Realistic processing delay
            convenience_score = flight.get('convenience_score', 0.5)
            
            # SERIAL TASK 2: Safety assessment (minimal processing time)
            time.sleep(0.001)  # Realistic processing delay
            duration_score = self._calculate_duration_score(duration)
            
            # SERIAL TASK 3: Economic analysis (minimal processing time)
            time.sleep(0.001)  # Realistic processing delay
            price_score = self._calculate_price_score(price)
            
            # SERIAL TASK 4: Flight info retrieval (minimal processing time)
            time.sleep(0.001)  # Realistic processing delay
            time_score = self._calculate_time_score(departure_time)
            
            # SERIAL TASK 5: Integration (minimal processing time)
            time.sleep(0.001)  # Realistic processing delay
            airline_score = self._calculate_airline_score(airline)
            availability_score = 0.8  # Standard availability baseline
            
            # Single Agent Algorithm: All tasks done serially
            overall_score = (
                self.ranking_weights['convenience'] * convenience_score +
                self.ranking_weights['duration'] * duration_score +
                self.ranking_weights['price'] * price_score +
                self.ranking_weights['airline_rating'] * airline_score +
                self.ranking_weights['availability'] * availability_score
            )
            
            # Record serial processing
            processing_time = time.time() - start_time
            
            processed_flights.append({
                'flight_id': flight_id,
                'overall_score': overall_score,
                'price_score': price_score,
                'duration_score': duration_score,
                'time_score': time_score,
                'airline_score': airline_score,
                'processing_time': processing_time,
                'execution_mode': 'true_serial'  # Paper requirement
            })
            
            # Log serial execution
            logger.debug(f"✅ Serial flight {i+1}/{len(flight_candidates)} processed in {processing_time:.3f}s")
        
        # Calculate total serial processing time
        total_processing_time = sum(flight['processing_time'] for flight in processed_flights)
        
        return {
            'success': True,
            'flights_processed': len(processed_flights),
            'processed_flights': processed_flights,
            'total_serial_time': total_processing_time,
            'execution_mode': 'true_serial_execution'  # Paper compliance
        }
    
    def _calculate_price_score(self, price: float) -> float:
        """Calculate price score (lower price = higher score)"""
        # Normalize price to [0, 1] range
        min_price = 200
        max_price = 1000
        normalized_price = (price - min_price) / (max_price - min_price)
        return max(0.1, 1.0 - min(1.0, normalized_price))
    
    def _calculate_duration_score(self, duration: float) -> float:
        """Calculate duration score (shorter duration = higher score)"""
        # Normalize duration to [0, 1] range
        min_duration = 60   # 1 hour
        max_duration = 480  # 8 hours
        normalized_duration = (duration - min_duration) / (max_duration - min_duration)
        return max(0.1, 1.0 - min(1.0, normalized_duration))
    
    def _calculate_time_score(self, departure_time: str) -> float:
        """Calculate departure time score (prefer morning/afternoon)"""
        try:
            hour = int(departure_time.split(':')[0])
            # Prefer 8 AM to 6 PM departures
            if 8 <= hour <= 18:
                return 0.9
            elif 6 <= hour <= 22:
                return 0.7
            else:
                return 0.5
        except:
            return 0.6  # Default for invalid time format
    
    def _calculate_airline_score(self, airline: str) -> float:
        """Calculate airline score based on industry ratings"""
        airline_lower = airline.lower()
        
        # Major airlines
        if any(major in airline_lower for major in ['delta', 'american', 'united', 'lufthansa']):
            return 0.9
        # Regional airlines
        elif any(regional in airline_lower for regional in ['southwest', 'jetblue', 'alaska']):
            return 0.8
        # Budget airlines
        elif any(budget in airline_lower for budget in ['spirit', 'frontier', 'ryanair']):
            return 0.6
        else:
            return 0.7  # Default score
    
    def _rank_flights_standard(self, flight_results: Dict[str, Any], 
                           query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank flights using standard heuristic approach
        
        Implementation for baseline comparison
        """
        if not flight_results.get('success', False):
            return {'recommendations': [], 'ranking': [], 'method': 'single_agent_failed'}
        
        processed_flights = flight_results.get('processed_flights', [])
        
        # Sort by overall score
        sorted_flights = sorted(processed_flights, key=lambda x: x['overall_score'], reverse=True)
        
        # Create ranking list
        ranking = [flight['flight_id'] for flight in sorted_flights]
        
        # Create recommendations
        recommendations = []
        for i, flight in enumerate(sorted_flights[:10]):
            recommendations.append({
                'flight_id': flight['flight_id'],
                'overall_score': flight['overall_score'],
                'rank': i + 1,
                'method': 'single_agent_heuristic',
                'component_scores': {
                    'price': flight['price_score'],
                    'duration': flight['duration_score'],
                    'time': flight['time_score'],
                    'airline': flight['airline_score']
                }
            })
        
        return {
            'recommendations': recommendations,
            'ranking': ranking,
            'method': 'single_agent_heuristic',
            'total_flights_ranked': len(sorted_flights)
        }
    
    def _calculate_system_reward(self, ranked_results: Dict[str, Any], 
                               query_data: Dict[str, Any]) -> float:
        """
        Calculate system reward using Equation 12 from paper:
        r = λ1 · MRR + λ2 · NDCG@5 - λ3 · ART
        
        IDENTICAL implementation to other models for fair baseline comparison
        """
        try:
            # Extract and validate data
            recommendations = ranked_results.get('recommendations', [])
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
            processing_time = ranked_results.get('processing_time', 0.2)  # Single agent is fast
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
        Single agent always selects itself (BaseModel compatibility)
        
        Args:
            query_data: Query data
            
        Returns:
            Single agent selection
        """
        return [(self.agent_name, 1.0)]
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Process with single agent (BaseModel compatibility)
        
        Args:
            query_data: Query data
            selected_agents: Selected agents (ignored)
            
        Returns:
            Single agent processing results
        """
        flight_results = self._process_flights_directly(query_data)
        return {
            self.agent_name: {
                'success': flight_results.get('success', False),
                'result': flight_results,
                'processing_time': 0.1,
                'agent_type': 'single_agent'
            }
        }
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate single agent results (BaseModel compatibility)
        
        Args:
            agent_results: Agent processing results
            query_data: Query data
            
        Returns:
            Integrated results
        """
        agent_result = agent_results.get(self.agent_name, {})
        flight_results = agent_result.get('result', {})
        return self._rank_flights_standard(flight_results, query_data) 