#!/usr/bin/env python3
"""
Traditional Ranking System - Baseline model for traditional ranking systems
Using traditional information retrieval methods (BM25, TF-IDF, rule-based ranking)
"""

import numpy as np
import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter
from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class TraditionalRanker:
    """
    Traditional Ranker - BM25 and rule-based flight ranking system
    
    This class implements a traditional information retrieval system using the BM25 algorithm for text matching,
    combined with rule-based multi-factor weighted ranking. Does not use any modern AI technology.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize traditional ranker
        
        Args:
            k1: BM25 parameter controlling term frequency saturation (default 1.2)
            b: BM25 parameter controlling document length normalization (default 0.75)
        """
        # BM25 parameters
        self.bm25_k1 = k1
        self.bm25_b = b
        
        # Ranking rule weights
        self.ranking_weights = {
            'price': 0.30,           # Price factor weight
            'duration': 0.20,        # Flight duration weight
            'departure_time': 0.15,  # Departure time weight
            'airline_rating': 0.20,  # Airline rating weight
            'availability': 0.15     # Seat availability weight
        }
        
        # Initialize flight database and index
        self.flight_database = self._initialize_flight_database()
        self.inverted_index = self._build_inverted_index()
        self.document_frequencies = self._calculate_document_frequencies()
        
        logger.info(f"✅ Traditional ranker initialized - BM25(k1={k1}, b={b})")
    
    def rank(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Rank flights for user query, return sorted flight list
        
        Args:
            user_query: User query string
            
        Returns:
            Sorted flight list, each element containing flight_id and score
        """
        start_time = time.time()
        
        # Step 1: BM25 text similarity calculation
        bm25_scores = self._calculate_bm25_scores(user_query)
        
        # Step 2: Rule-based multi-factor scoring
        rule_scores = self._calculate_rule_based_scores()
        
        # Step 3: Combined scoring (50% BM25 + 50% rule-based)
        final_scores = []
        for flight_id in self.flight_database.keys():
            bm25_score = bm25_scores.get(flight_id, 0.0)
            rule_score = rule_scores.get(flight_id, 0.0)
            
            # Combined score
            final_score = 0.5 * bm25_score + 0.5 * rule_score
            
            final_scores.append({
                'flight_id': flight_id,
                'score': final_score,
                'bm25_component': bm25_score,
                'rule_component': rule_score
            })
        
        # Step 4: Sort by score
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        
        processing_time = time.time() - start_time
        logger.debug(f"Traditional ranking completed, processing time: {processing_time:.3f}s")
        
        return final_scores
    
    def _initialize_flight_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize simulated flight database
        
        Returns:
            Flight database dictionary
        """
        flight_db = {}
        
        for i in range(1, 11):  # 10 flights
            flight_id = f"flight_{i:03d}"
            flight_db[flight_id] = {
                'flight_id': flight_id,
                'price': np.random.uniform(300, 1200),
                'duration': np.random.uniform(2.0, 8.0),  # hours
                'departure_time': np.random.choice(['morning', 'afternoon', 'evening']),
                'airline_rating': np.random.uniform(0.6, 0.95),
                'availability': np.random.choice([True, False], p=[0.8, 0.2]),
                'description': f"Flight {flight_id} from Beijing to Shanghai via {np.random.choice(['direct', 'one-stop', 'two-stop'])} route"
            }
        
        return flight_db
    
    def _build_inverted_index(self) -> Dict[str, Set[str]]:
        """
        Build inverted index
        
        Returns:
            Inverted index dictionary {term: {flight_ids}}
        """
        index = {}
        
        for flight_id, flight_data in self.flight_database.items():
            # Extract document terms
            doc_text = flight_data['description'].lower()
            terms = doc_text.split()
            
            for term in terms:
                if term not in index:
                    index[term] = set()
                index[term].add(flight_id)
        
        return index
    
    def _calculate_document_frequencies(self) -> Dict[str, int]:
        """
        Calculate document frequencies
        
        Returns:
            Document frequency dictionary {term: frequency}
        """
        doc_freq = {}
        total_docs = len(self.flight_database)
        
        for term, doc_set in self.inverted_index.items():
            doc_freq[term] = len(doc_set)
        
        return doc_freq
    
    def _calculate_bm25_scores(self, query: str) -> Dict[str, float]:
        """
        Calculate BM25 similarity scores
        
        Args:
            query: Query string
            
        Returns:
            BM25 score dictionary {flight_id: score}
        """
        query_terms = query.lower().split()
        scores = {}
        
        # Calculate average document length
        total_length = sum(len(flight['description'].split()) 
                          for flight in self.flight_database.values())
        avg_doc_length = total_length / len(self.flight_database)
        
        for flight_id, flight_data in self.flight_database.items():
            doc_terms = flight_data['description'].lower().split()
            doc_length = len(doc_terms)
            term_freq = Counter(doc_terms)
            
            bm25_score = 0.0
            
            for term in query_terms:
                if term in term_freq:
                    # TF component
                    tf = term_freq[term]
                    tf_component = (tf * (self.bm25_k1 + 1)) / (tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (doc_length / avg_doc_length)))
                    
                    # IDF component
                    df = self.document_frequencies.get(term, 0)
                    if df > 0:
                        idf = math.log((len(self.flight_database) - df + 0.5) / (df + 0.5))
                        bm25_score += idf * tf_component
            
            scores[flight_id] = bm25_score
        
        # Normalize to [0,1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def _calculate_rule_based_scores(self) -> Dict[str, float]:
        """
        Calculate rule-based scores
        
        Returns:
            Rule-based score dictionary {flight_id: score}
        """
        scores = {}
        
        # Get numerical ranges for normalization
        prices = [flight['price'] for flight in self.flight_database.values()]
        durations = [flight['duration'] for flight in self.flight_database.values()]
        ratings = [flight['airline_rating'] for flight in self.flight_database.values()]
        
        price_range = (min(prices), max(prices))
        duration_range = (min(durations), max(durations))
        
        for flight_id, flight_data in self.flight_database.items():
            # Price score (lower is better, normalized to [0,1])
            price_score = 1.0 - (flight_data['price'] - price_range[0]) / (price_range[1] - price_range[0])
            
            # Duration score (shorter is better, normalized to [0,1])
            duration_score = 1.0 - (flight_data['duration'] - duration_range[0]) / (duration_range[1] - duration_range[0])
            
            # Departure time score (preference for morning flights)
            time_scores = {'morning': 1.0, 'afternoon': 0.7, 'evening': 0.5}
            departure_score = time_scores.get(flight_data['departure_time'], 0.5)
            
            # Airline rating score (already in [0,1] range)
            airline_score = flight_data['airline_rating']
            
            # Availability score
            availability_score = 1.0 if flight_data['availability'] else 0.0
            
            # Weighted combined score
            final_score = (
                self.ranking_weights['price'] * price_score +
                self.ranking_weights['duration'] * duration_score +
                self.ranking_weights['departure_time'] * departure_score +
                self.ranking_weights['airline_rating'] * airline_score +
                self.ranking_weights['availability'] * availability_score
            )
            
            scores[flight_id] = final_score
        
        return scores

def generate_decision_tree_ground_truth(flight_options: List[Dict[str, Any]], 
                                      user_preferences: Dict[str, str]) -> List[str]:
    """
    Generate Ground Truth ranking based on strict decision tree logic
    
    This function uses strict decision tree logic, completely different from the weighted model of the MAMA system,
    ensuring that the Ground Truth generation is completely decoupled from the model decision logic.
    
    Args:
        flight_options: List of 10 candidate flight objects
        user_preferences: User preference dictionary, e.g., {'priority': 'safety', 'budget': 'medium'}
        
    Returns:
        Sorted list of flight IDs as Ground Truth
    """
    
    # Step 1: Hard filtering (Deal-breaker Filters)
    filtered_flights = []
    
    for flight in flight_options:
        # Safety score must be > 0.4
        safety_score = flight.get('safety_score', np.random.uniform(0.3, 0.95))
        if safety_score <= 0.4:
            continue
        
        # Seat availability must be True
        if not flight.get('availability', True):
            continue
        
        # Budget constraint
        price = flight.get('price', np.random.uniform(300, 1200))
        budget = user_preferences.get('budget', 'medium')
        
        if budget == 'low' and price >= 500:
            continue
        elif budget == 'medium' and price >= 1000:
            continue
        # high budget has no price limit
        
        # Flights passing filters
        filtered_flights.append({
            'flight_id': flight.get('flight_id', f"flight_{len(filtered_flights)+1:03d}"),
            'safety_score': safety_score,
            'price': price,
            'duration': flight.get('duration', np.random.uniform(2.0, 8.0)),
            'original_data': flight
        })
    
    # If too few flights after filtering, relax conditions
    if len(filtered_flights) < 3:
        logger.warning("Too few flights after hard filtering, relaxing safety score requirement")
        filtered_flights = []
        for flight in flight_options:
            safety_score = flight.get('safety_score', np.random.uniform(0.3, 0.95))
            if safety_score > 0.3 and flight.get('availability', True):
                filtered_flights.append({
                    'flight_id': flight.get('flight_id', f"flight_{len(filtered_flights)+1:03d}"),
                    'safety_score': safety_score,
                    'price': flight.get('price', np.random.uniform(300, 1200)),
                    'duration': flight.get('duration', np.random.uniform(2.0, 8.0)),
                    'original_data': flight
                })
    
    # Step 2: Primary sorting (Primary Sorting)
    priority = user_preferences.get('priority', 'safety')
    
    if priority == 'safety':
        # Sort by safety score from high to low
        filtered_flights.sort(key=lambda x: x['safety_score'], reverse=True)
    elif priority == 'cost':
        # Sort by price from low to high
        filtered_flights.sort(key=lambda x: x['price'], reverse=False)
    elif priority == 'time':
        # Sort by total flight duration from low to high
        filtered_flights.sort(key=lambda x: x['duration'], reverse=False)
    else:
        # Default sort by safety score
        filtered_flights.sort(key=lambda x: x['safety_score'], reverse=True)
    
    # Step 3.1: Handle ties (Tie-Breaking)
    # For flights with the same priority metric, use price as the secondary sorting criterion
    if priority == 'safety':
        # Group by safety score, sort within groups by price
        filtered_flights.sort(key=lambda x: (-x['safety_score'], x['price']))
    elif priority == 'cost':
        # Group by price, sort within groups by safety score
        filtered_flights.sort(key=lambda x: (x['price'], -x['safety_score']))
    elif priority == 'time':
        # Group by duration, sort within groups by price
        filtered_flights.sort(key=lambda x: (x['duration'], x['price']))
    
    # Step 3.2: Handle final ties (Final Tie-Breaking)
    # Finally, use flight duration as the decisive criterion
    if priority == 'safety':
        filtered_flights.sort(key=lambda x: (-x['safety_score'], x['price'], x['duration']))
    elif priority == 'cost':
        filtered_flights.sort(key=lambda x: (x['price'], -x['safety_score'], x['duration']))
    elif priority == 'time':
        filtered_flights.sort(key=lambda x: (x['duration'], x['price']))
    
    # Step 4: Generate final ranking
    ground_truth_ranking = [flight['flight_id'] for flight in filtered_flights]
    
    # If the ranking is less than 10, fill with remaining flights
    all_flight_ids = [f.get('flight_id', f"flight_{i:03d}") for i, f in enumerate(flight_options)]
    for flight_id in all_flight_ids:
        if flight_id not in ground_truth_ranking:
            ground_truth_ranking.append(flight_id)
    
    logger.debug(f"Decision tree Ground Truth generated: priority={priority}, filtered={len(filtered_flights)} flights")
    
    return ground_truth_ranking[:10]  # Return top 10

class TraditionalRanking(BaseModel):
    """Traditional ranking system baseline model"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize traditional ranking system
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_description = "Traditional ranking system - BM25 + TF-IDF + rule-based ranking"
        
        # Initialize traditional ranker
        self.ranker = TraditionalRanker()
    
    def _initialize_model(self):
        """Initialize traditional ranking system"""
        # Disable all modern AI features
        self.sbert_enabled = False
        self.trust_enabled = False
        self.historical_enabled = False
        self.marl_enabled = False
        
        logger.info("✅ Traditional ranking system initialized - modern AI features disabled")
    
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Traditional system does not use intelligent agent selection, returns empty list"""
        return []
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Process query using traditional ranker"""
        query_text = query_data.get('query_text', '')
        
        # Use traditional ranker
        ranking_results = self.ranker.rank(query_text)
        
        return {
            'traditional_ranker': {
                'success': True,
                'recommendations': ranking_results,
                'processing_time': 0.1,  # Traditional methods are usually faster
                'method': 'BM25 + Rule-based ranking'
            }
        }
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate traditional ranking results"""
        ranker_result = agent_results['traditional_ranker']
        recommendations = ranker_result['recommendations']
        
        # Extract ranking
        final_ranking = [rec['flight_id'] for rec in recommendations]
        
        return {
            'query_id': query_data.get('query_id', 'unknown'),
            'success': ranker_result['success'],
            'ranking': final_ranking,
            'recommendations': recommendations,
            'system_confidence': 0.8,  # Fixed confidence for traditional methods
            'model_name': self.model_name,
            'processing_summary': {
                'total_time': ranker_result['processing_time'],
                'method': ranker_result['method'],
                'architecture_type': 'traditional_ir'
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': 'Traditional_Ranking_Baseline',
            'description': self.model_description,
            'methodology': {
                'text_retrieval': 'BM25 (Best Matching 25)',
                'ranking_algorithm': 'Rule-based scoring',
                'feature_engineering': 'Manual feature definition',
                'learning_method': 'none'
            },
            'features': {
                'modern_ai': False,
                'machine_learning': False,
                'deep_learning': False,
                'semantic_understanding': False,
                'trust_mechanism': False,
                'agent_collaboration': False
            },
            'algorithms': {
                'bm25': {
                    'k1': self.ranker.bm25_k1,
                    'b': self.ranker.bm25_b,
                    'purpose': 'Text similarity calculation'
                },
                'rule_based_ranking': {
                    'weights': self.ranker.ranking_weights,
                    'purpose': 'Multi-factor ranking'
                }
            },
            'limitations': {
                'no_learning': 'Cannot improve from user feedback',
                'static_rules': 'Fixed ranking rules',
                'no_semantic_understanding': 'Limited to keyword matching',
                'no_personalization': 'Same ranking for all users'
            },
            'academic_purpose': 'Baseline representing traditional IR methods',
            'expected_performance': 'Moderate but consistent performance'
        }
    
    def get_flight_by_id(self, flight_id: str) -> Optional[Dict[str, Any]]:
        """Get flight information by ID"""
        return self.ranker.flight_database.get(flight_id)
    
    def get_ranking_explanation(self, flight_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get ranking explanation"""
        flight_info = self.get_flight_by_id(flight_id)
        if not flight_info:
            return {}
        
        return {
            'flight_id': flight_id,
            'ranking_method': 'Traditional BM25 + Rule-based',
            'factors': {
                'text_similarity': 'BM25 score based on query terms',
                'price': f"Price: {flight_info['price']:.2f}",
                'duration': f"Duration: {flight_info['duration']:.2f} hours",
                'departure_time': f"Departure: {flight_info['departure_time']}",
                'airline_rating': f"Rating: {flight_info['airline_rating']:.2f}",
                'availability': f"Seats: {'Available' if flight_info['availability'] else 'Not Available'}"
            },
            'algorithm_details': {
                'bm25_parameters': {'k1': self.ranker.bm25_k1, 'b': self.ranker.bm25_b},
                'combination_method': 'Weighted sum of BM25 and rule-based scores',
                'personalization': 'none'
            }
        } 