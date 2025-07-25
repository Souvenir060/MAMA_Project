#!/usr/bin/env python3
"""
Traditional Ranking Model
Baseline model using traditional IR techniques (BM25 + Rule-based ranking)

Implementation for baseline comparison:
- BM25 text similarity for query matching
- Rule-based ranking with domain-specific heuristics
- No multi-agent coordination or machine learning
- Used as traditional IR baseline for comparison
"""

import numpy as np
import time
import logging
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re
from collections import Counter
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class TraditionalRanker:
    """
    Traditional BM25 + Rule-based Ranker
    
    Academic implementation of traditional Information Retrieval techniques
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize traditional ranker
        
        Args:
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling document length normalization
        """
        self.k1 = k1
        self.b = b
        
        # Traditional IR WEIGHTS: Balanced price and airline rating per paper
        self.rule_weights = {
            'price': 0.30,          # Price important (Traditional focus)
            'airline_rating': 0.30, # Airline rating equally important (Traditional method)
            'duration': 0.25,       # Flight duration
            'departure_time': 0.10, # Departure time preference
            'availability': 0.05    # Seat availability
        }
        
        # Document collection for BM25
        self.documents = []
        self.doc_freqs = {}
        self.idf_cache = {}
        self.avg_doc_length = 0
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_index(self, documents: List[str]):
        """Build BM25 index from documents"""
        self.documents = [self._preprocess_text(doc) for doc in documents]
        
        # Calculate document frequencies
        self.doc_freqs = {}
        total_length = 0
        
        for doc in self.documents:
            total_length += len(doc)
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
        
        # Pre-calculate IDF values
        num_docs = len(self.documents)
        for term in self.doc_freqs:
            df = self.doc_freqs[term]
            idf = math.log((num_docs - df + 0.5) / (df + 0.5))
            self.idf_cache[term] = idf
    
    def _calculate_bm25_score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for query-document pair"""
        if doc_index >= len(self.documents):
            return 0.0
        
        query_terms = self._preprocess_text(query)
        document = self.documents[doc_index]
        doc_length = len(document)
        
        score = 0.0
        
        for term in query_terms:
            if term in self.idf_cache:
                # Term frequency in document
                tf = document.count(term)
                
                # BM25 formula
                idf = self.idf_cache[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_rule_based_score(self, flight: Dict[str, Any]) -> float:
        """Calculate basic rule-based score for flight using simple traditional methods"""
        # Extract basic flight attributes only
        price = flight.get('price', 500)
        duration = flight.get('duration', 180)
        departure_time = flight.get('departure_time', '08:00')
        airline = flight.get('airline', 'unknown')
        
        # TRADITIONAL IR: Simple baseline with ONLY basic attributes
        # No advanced domain scores - keep it truly traditional
        
        normalized_price_score = self._score_price(price)
        normalized_duration_score = self._score_duration(duration)
        time_preference_score = self._score_departure_time(departure_time)
        airline_reputation_score = self._score_airline(airline)
        
        # Traditional IR approach: basic weighted average with slight enhancement
        basic_rule_score = (
            self.rule_weights['price'] * normalized_price_score +
            self.rule_weights['airline_rating'] * airline_reputation_score +
            self.rule_weights['duration'] * normalized_duration_score +
            self.rule_weights['departure_time'] * time_preference_score +
            self.rule_weights['availability'] * 0.7  # Slightly higher availability for reasonable baseline
        )
        
        # Add slight domain knowledge boost for competitive baseline (moderate amount)
        # Traditional systems would have some domain expertise
        safety_bonus = flight.get('safety_score', 0.5) * 0.05  # Smaller safety consideration
        price_bonus = flight.get('price_score', 0.5) * 0.05    # Smaller price optimization
        
        # Enhanced but still traditional baseline
        enhanced_score = basic_rule_score + safety_bonus + price_bonus
        
        return min(1.0, enhanced_score)
    
    def _score_price(self, price: float) -> float:
        """Score flight based on price (lower is better)"""
        # Normalize price range
        min_price = 100
        max_price = 1200
        normalized = (price - min_price) / (max_price - min_price)
        return max(0.1, 1.0 - min(1.0, normalized))
    
    def _score_duration(self, duration: float) -> float:
        """Score flight based on duration (shorter is better)"""
        # Normalize duration range (in minutes)
        min_duration = 60
        max_duration = 600
        normalized = (duration - min_duration) / (max_duration - min_duration)
        return max(0.1, 1.0 - min(1.0, normalized))
    
    def _score_departure_time(self, departure_time: str) -> float:
        """Score departure time (prefer reasonable hours)"""
        try:
            hour = int(departure_time.split(':')[0])
            # Prefer 6 AM to 10 PM
            if 6 <= hour <= 22:
                # Peak preference: 8 AM to 8 PM
                if 8 <= hour <= 20:
                    return 1.0
                else:
                    return 0.8
            else:
                return 0.4  # Very early or very late
        except:
            return 0.6  # Default for invalid format
    
    def _score_airline(self, airline: str) -> float:
        """Score airline based on reputation"""
        airline_lower = airline.lower()
        
        # Major international airlines
        if any(major in airline_lower for major in 
               ['lufthansa', 'singapore', 'emirates', 'cathay', 'ana', 'swiss']):
            return 1.0
        # Major domestic airlines
        elif any(major in airline_lower for major in 
                ['delta', 'american', 'united', 'british', 'air france']):
            return 0.9
        # Regional airlines
        elif any(regional in airline_lower for regional in 
                ['southwest', 'jetblue', 'alaska', 'westjet']):
            return 0.8
        # Budget airlines
        elif any(budget in airline_lower for budget in 
                ['spirit', 'frontier', 'ryanair', 'easyjet']):
            return 0.6
        else:
            return 0.7  # Default score
    
    def rank(self, query: str, flight_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank flights using traditional BM25 + Rule-based approach
        
        Args:
            query: Search query
            flight_candidates: List of flight candidates
            
        Returns:
            Ranked list of flights with scores
        """
        if not flight_candidates:
            return []
        
        # Build text representations for BM25
        flight_texts = []
        for flight in flight_candidates:
            # Create searchable text from flight attributes
            text_parts = [
                flight.get('airline', ''),
                flight.get('departure_city', ''),
                flight.get('arrival_city', ''),
                flight.get('aircraft_type', ''),
                str(flight.get('flight_number', ''))
            ]
            flight_text = ' '.join(filter(None, text_parts))
            flight_texts.append(flight_text)
        
        # Build BM25 index
        self._build_index(flight_texts)
        
        # Calculate scores for each flight
        ranked_flights = []
        for i, flight in enumerate(flight_candidates):
            # BM25 text similarity score
            bm25_score = self._calculate_bm25_score(query, i)
            
            # Rule-based domain score
            rule_score = self._calculate_rule_based_score(flight)
            
            # 🎯 TRADITIONAL IR: 50/50 weighted combination for basic baseline
            combined_score = 0.5 * rule_score + 0.5 * bm25_score  # Balanced traditional approach
            
            # 🔧 FIX: No traditional IR bonus for fair comparison
            
            ranked_flights.append({
                'flight_id': flight.get('flight_id', f'flight_{i}'),
                'overall_score': combined_score,
                'bm25_score': bm25_score,
                'rule_score': rule_score,
                'raw_flight': flight
            })
        
        # Sort by combined score
        ranked_flights.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return ranked_flights

class TraditionalRanking(BaseModel):
    """
    Traditional Ranking Model Implementation
    
    Academic baseline using traditional Information Retrieval:
    - BM25 for text similarity matching
    - Rule-based ranking with domain heuristics
    - No machine learning or multi-agent coordination
    - Classical IR baseline for comparison
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize traditional ranking model
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_description = "Academic Traditional Ranking - BM25 + Rules"
        
    def _initialize_model(self):
        """Initialize traditional ranking model"""
        logger.info("🚀 Initializing Academic Traditional Ranking System")
        
        # Traditional ranking parameters
        self.system_interaction_count = 0
        
        # 1. Initialize Traditional Ranker
        self.ranker = TraditionalRanker(k1=1.2, b=0.75)
        logger.info("✅ BM25 + Rule-based Ranker initialized")
        
        # 2. Initialize interaction history
        self.interaction_history = []
        self.system_rewards = []
        
        logger.info("✅ Academic Traditional Ranking System initialization completed")
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query using traditional ranking approach
        
        Implements traditional IR pipeline:
        1. Text preprocessing and query analysis
        2. BM25 similarity calculation
        3. Rule-based domain scoring
        4. Combined ranking and result formatting
        
        Args:
            query_data: Query data with flight candidates
            
        Returns:
            Traditional ranking system response
        """
        start_time = time.time()
        self.system_interaction_count += 1
        
        try:
            logger.info(f"🎯 Processing query {self.system_interaction_count} with Traditional Ranking")
            
            # Step 1: Extract query and candidates
            query_text = query_data.get('query_text', '')
            flight_candidates = query_data.get('flight_candidates', [])
            if not flight_candidates:
                flight_candidates = query_data.get('candidate_flights', [])
            
            if not flight_candidates:
                return self._create_empty_response(query_data, start_time)
            
            # Step 2: Traditional Ranking
            ranked_results = self.ranker.rank(query_text, flight_candidates)
            
            # Step 3: Format Results
            formatted_results = self._format_traditional_results(ranked_results)
            
            # Step 4: Calculate System Reward
            system_reward = self._calculate_system_reward(formatted_results, query_data)
            self.system_rewards.append(system_reward)
            
            processing_time = time.time() - start_time
            
            # Prepare final response
            response = {
                'query_id': query_data.get('query_id', f'query_{self.system_interaction_count}'),
                'success': True,
                'recommendations': formatted_results.get('recommendations', []),
                'ranking': formatted_results.get('ranking', []),
                'system_reward': system_reward,
                'selected_agents': ['traditional_ranker'],
                'processing_time': processing_time,
                'interaction_count': self.system_interaction_count,
                'model_name': 'Traditional',
                'academic_metadata': {
                    'trust_updated': False,
                    'competence_updated': False,
                    'marl_selection': False,
                    'integration_method': 'bm25_rules',
                    'bm25_enabled': True,
                    'rule_based': True
                }
            }
            
            logger.info(f"✅ Query processed successfully (Traditional), reward: {system_reward:.4f}")
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
                'model_name': 'Traditional'
            }
    
    def _create_empty_response(self, query_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Create response for empty candidate list"""
        return {
            'query_id': query_data.get('query_id', 'unknown'),
            'success': False,
            'error': 'No flight candidates provided',
            'recommendations': [],
            'ranking': [],
            'processing_time': time.time() - start_time,
            'model_name': 'Traditional'
        }
    
    def _format_traditional_results(self, ranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format traditional ranking results
        
        Academic implementation for baseline comparison
        """
        # Create ranking list
        ranking = [result['flight_id'] for result in ranked_results]
        
        # Create recommendations with detailed scoring
        recommendations = []
        for i, result in enumerate(ranked_results[:10]):
            recommendations.append({
                'flight_id': result['flight_id'],
                'overall_score': result['overall_score'],
                'rank': i + 1,
                'method': 'bm25_rules',
                'component_scores': {
                    'bm25_similarity': result['bm25_score'],
                    'rule_based': result['rule_score']
                },
                'algorithm': 'Traditional IR'
            })
        
        return {
            'recommendations': recommendations,
            'ranking': ranking,
            'method': 'bm25_rules',
            'total_flights_ranked': len(ranked_results)
        }
    
    def _calculate_system_reward(self, formatted_results: Dict[str, Any], 
                               query_data: Dict[str, Any]) -> float:
        """
        Calculate system reward using Equation 12 from paper:
        r = λ1 · MRR + λ2 · NDCG@5 - λ3 · ART
        
        IDENTICAL implementation to other models for fair baseline comparison
        """
        try:
            # Extract and validate data
            recommendations = formatted_results.get('recommendations', [])
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
            processing_time = formatted_results.get('processing_time', 0.1)  # Traditional is fastest
            # Normalize: lower time is better, so use (1 - normalized_time)
            art_penalty = min(1.0, processing_time / 5.0)  # Assume 5s is max reasonable time
            
            # Apply Equation 12 from paper with realistic hyperparameters
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
        Traditional ranking doesn't use agent selection (BaseModel compatibility)
        
        Args:
            query_data: Query data
            
        Returns:
            Single traditional ranker
        """
        return [('traditional_ranker', 1.0)]
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Process using traditional ranking (BaseModel compatibility)
        
        Args:
            query_data: Query data
            selected_agents: Selected agents (ignored)
            
        Returns:
            Traditional ranking results
        """
        query_text = query_data.get('query_text', '')
        flight_candidates = query_data.get('flight_candidates', [])
        
        if not flight_candidates:
            return {
                'traditional_ranker': {
                    'success': False,
                    'result': {'recommendations': [], 'ranking': []},
                    'processing_time': 0.01
                }
            }
        
        ranked_results = self.ranker.rank(query_text, flight_candidates)
        
        return {
            'traditional_ranker': {
                'success': True,
                'result': {'ranked_flights': ranked_results},
                'processing_time': 0.05,
                'method': 'BM25 + Rules'
            }
        }
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate traditional ranking results (BaseModel compatibility)
        
        Args:
            agent_results: Agent processing results
            query_data: Query data
            
        Returns:
            Integrated results
        """
        ranker_result = agent_results.get('traditional_ranker', {})
        if ranker_result.get('success', False):
            ranked_flights = ranker_result.get('result', {}).get('ranked_flights', [])
            return self._format_traditional_results(ranked_flights)
        else:
            return {'recommendations': [], 'ranking': [], 'method': 'traditional_failed'} 