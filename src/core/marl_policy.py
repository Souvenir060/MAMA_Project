"""
MARL Policy for MAMA Flight Assistant System

Academic implementation of trust-aware multi-agent reinforcement learning policy
for dynamic agent selection based on research paper formulas.

Key Mathematical Formulations:
1. Agent Selection Score: SelectionScore = α × similarity + β × popularity
2. Trust-Aware Q-Value: Q'(s,a) = Q(s,a) + trust_weight × trust_score
3. Agent Popularity Update: popularity_t = popularity_{t-1} + η × (reward - popularity_{t-1})
4. Dynamic Selection Reward: r = α × MRR + β × NDCG@5 - γ × ART
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import pickle
import warnings

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """
    Agent capability representation for MAMA system
    
    Academic formulation for agent expertise and performance metrics
    """
    agent_id: str
    agent_type: str
    expertise_domains: List[str]
    performance_history: Dict[str, float]
    trust_score: float
    response_time_avg: float
    success_rate: float
    specialization_score: float
    
    # Academic metrics
    mrr_score: float = 0.0          # Mean Reciprocal Rank
    ndcg_score: float = 0.0         # NDCG@5
    precision_at_k: float = 0.0     # Precision@K
    recall_score: float = 0.0       # Recall
    f1_score: float = 0.0           # F1 Score
    
    # Dynamic parameters
    current_load: float = 0.0       # Current workload [0,1]
    availability: bool = True       # Agent availability
    last_interaction: datetime = field(default_factory=datetime.now)
    
    def compute_expertise_similarity(self, query_embedding: np.ndarray, 
                                   domain_embeddings: Dict[str, np.ndarray]) -> float:
        """
        Compute expertise similarity using SBERT embeddings
        
        Args:
            query_embedding: Query embedding vector
            domain_embeddings: Domain expertise embeddings
            
        Returns:
            Similarity score [0,1]
        """
        max_similarity = 0.0
        for domain in self.expertise_domains:
            if domain in domain_embeddings:
                domain_emb = domain_embeddings[domain]
                similarity = np.dot(query_embedding, domain_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(domain_emb)
                )
                max_similarity = max(max_similarity, similarity)
        
        return max(0.0, min(1.0, max_similarity))

@dataclass
class TrustMetrics:
    """
    Trust metrics for academic trust-aware MARL implementation
    
    Implements trust computation based on historical performance and reliability
    """
    historical_accuracy: float
    consistency_score: float
    reliability_index: float
    peer_trust_score: float
    domain_expertise: float
    
    # Trust update parameters
    alpha_trust: float = 0.7    # Historical weight
    beta_trust: float = 0.2     # Consistency weight  
    gamma_trust: float = 0.1    # Peer weight
    
    def compute_overall_trust(self) -> float:
        """
        Compute overall trust score using weighted combination
        
        Returns:
            Overall trust score [0,1]
        """
        trust_score = (self.alpha_trust * self.historical_accuracy +
                      self.beta_trust * self.consistency_score +
                      self.gamma_trust * self.peer_trust_score)
        
        # Apply reliability and expertise modulation
        trust_score *= (0.5 * self.reliability_index + 0.5 * self.domain_expertise)
        
        return max(0.0, min(1.0, trust_score))

class TrustAwareQNetwork(nn.Module):
    """
    Trust-aware Q-Network for agent selection
    
    Academic implementation integrating trust scores into Q-value computation:
    Q'(s,a) = Q(s,a) + trust_weight × trust_score
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize trust-aware Q-network
        
        Args:
            state_dim: State vector dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Main Q-network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.q_network = nn.Sequential(*layers)
        
        # Trust integration layer
        self.trust_weight = nn.Parameter(torch.tensor(0.3))  # Learnable trust weight
        
        # Value network for baseline
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 1)
        )
        
    def forward(self, state: torch.Tensor, trust_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with trust-aware Q-value computation
        
        Args:
            state: State tensor
            trust_scores: Trust scores for agents
            
        Returns:
            Trust-aware Q-values
        """
        # Compute base Q-values
        q_values = self.q_network(state)
        
        if trust_scores is not None:
            # Apply trust-aware modification: Q'(s,a) = Q(s,a) + trust_weight × trust_score
            trust_adjustment = self.trust_weight * trust_scores
            q_values = q_values + trust_adjustment
        
        return q_values
    
    def compute_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value for baseline"""
        return self.value_network(state)

class MARLPolicy:
    """
    Multi-Agent Reinforcement Learning Policy for MAMA System
    
    Academic implementation of trust-aware dynamic agent selection using
    deep reinforcement learning with mathematical formulations from research paper.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MARL policy
        
        Args:
            config: Policy configuration parameters
        """
        self.config = config or {}
        
        # Model paths and configuration
        self.model_path = self.config.get('model_path', 'models/marl_policy.pth')
        self.embeddings_path = self.config.get('embeddings_path', 'models/domain_embeddings.pkl')
        
        # Academic parameters from research paper
        self.alpha = self.config.get('alpha', 0.7)      # Similarity weight
        self.beta = self.config.get('beta', 0.3)        # Popularity weight
        self.gamma = self.config.get('gamma', 0.6)      # Accuracy weight in reward
        self.delta = self.config.get('delta', 0.4)      # Efficiency weight in reward
        self.eta = self.config.get('eta', 0.1)          # Popularity learning rate
        
        # Agent configuration (compatible with MAMA system)
        self.agent_types = [
            'weather_agent',
            'safety_assessment_agent', 
            'flight_info_agent',
            'economic_agent',
            'integration_agent'
        ]
        
        # Initialize agent capabilities
        self.agent_capabilities = self._initialize_agent_capabilities()
        
        # Initialize neural networks
        self.state_dim = self._calculate_state_dimension()
        self.action_dim = len(self.agent_types)
        
        self.q_network = TrustAwareQNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.get('hidden_dims', [256, 128, 64])
        )
        
        # Load pre-trained model if available
        self.model_loaded = self._load_model()
        
        # Initialize embeddings model (simplified without external dependencies)
        self.domain_embeddings = self._load_domain_embeddings()
        
        # Historical tracking for academic metrics
        self.agent_popularities = {agent_id: 0.5 for agent_id in self.agent_types}
        self.performance_history = []
        self.trust_metrics = self._initialize_trust_metrics()
        
        logger.info(f"MARL Policy initialized with {len(self.agent_types)} agents")
        if not self.model_loaded:
            logger.warning("No pre-trained model found. Using random initialization.")
    
    def _initialize_agent_capabilities(self) -> Dict[str, AgentCapability]:
        """Initialize agent capabilities for MAMA system"""
        capabilities = {}
        
        # Define agent expertise domains (based on MAMA system)
        agent_expertise = {
            'weather_agent': ['weather_analysis', 'meteorology', 'atmospheric_conditions'],
            'safety_assessment_agent': ['safety_evaluation', 'risk_analysis', 'hazard_assessment'],
            'flight_info_agent': ['flight_search', 'airline_data', 'route_optimization'],
            'economic_agent': ['cost_analysis', 'price_comparison', 'economic_evaluation'],
            'integration_agent': ['data_integration', 'result_synthesis', 'decision_making']
        }
        
        for agent_id in self.agent_types:
            capabilities[agent_id] = AgentCapability(
                agent_id=agent_id,
                agent_type=agent_id.replace('_agent', ''),
                expertise_domains=agent_expertise.get(agent_id, []),
                performance_history={},
                trust_score=0.8,  # Initial trust score
                response_time_avg=1.0,
                success_rate=0.7,
                specialization_score=0.8,
                mrr_score=0.5,
                ndcg_score=0.6,
                precision_at_k=0.7,
                recall_score=0.6,
                f1_score=0.65
            )
        
        return capabilities
    
    def _initialize_trust_metrics(self) -> Dict[str, TrustMetrics]:
        """Initialize trust metrics for each agent"""
        trust_metrics = {}
        
        for agent_id in self.agent_types:
            trust_metrics[agent_id] = TrustMetrics(
                historical_accuracy=0.8,
                consistency_score=0.7,
                reliability_index=0.85,
                peer_trust_score=0.75,
                domain_expertise=0.8
            )
        
        return trust_metrics
    
    def _calculate_state_dimension(self) -> int:
        """Calculate state vector dimension for neural network"""
        # Based on MARLState in environment.py
        base_features = 3  # query_complexity, urgency, type
        agent_features = len(self.agent_types) * 5  # per-agent features
        system_features = 3  # system_load, time_budget, quality_requirement
        academic_features = len(self.agent_types) * 3  # MRR, NDCG, ART per agent
        
        return base_features + agent_features + system_features + academic_features
    
    def _load_model(self) -> bool:
        """Load pre-trained MARL model"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.q_network.load_state_dict(checkpoint['model_state_dict'])
                
                # Load additional metadata if available
                if 'agent_popularities' in checkpoint:
                    self.agent_popularities = checkpoint['agent_popularities']
                if 'trust_metrics' in checkpoint:
                    self.trust_metrics = checkpoint['trust_metrics']
                
                logger.info(f"Successfully loaded MARL model from {self.model_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
        
        return False
    
    def _load_domain_embeddings(self) -> Dict[str, np.ndarray]:
        """Load pre-computed domain embeddings"""
        try:
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load domain embeddings: {e}")
        
        # Return default embeddings if file not found
        return self._compute_default_domain_embeddings()
    
    def _compute_default_domain_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute default domain embeddings"""
        domains = [
            'weather_analysis', 'meteorology', 'atmospheric_conditions',
            'safety_evaluation', 'risk_analysis', 'hazard_assessment',
            'flight_search', 'airline_data', 'route_optimization',
            'cost_analysis', 'price_comparison', 'economic_evaluation',
            'data_integration', 'result_synthesis', 'decision_making'
        ]
        
        embeddings = {}
        # Create simple embeddings based on domain keywords
        for i, domain in enumerate(domains):
            # Create a simple embedding vector
            embedding = np.random.normal(0, 1, 384)  # Standard embedding size
            embedding[i % 384] += 2.0  # Add domain-specific signal
            embeddings[domain] = embedding / np.linalg.norm(embedding)
        
        return embeddings
    
    def select_agents(self, query: str, chat_history: List[Dict[str, Any]], 
                     available_agents: List[str], max_agents: int = 3) -> List[str]:
        """
        Select optimal agents using academic MARL formulation
        
        Implements agent selection based on:
        SelectionScore = α × similarity + β × popularity
        
        Args:
            query: User query text
            chat_history: Conversation history
            available_agents: List of available agent IDs
            max_agents: Maximum number of agents to select
            
        Returns:
            List of selected agent IDs
        """
        try:
            # Compute query embedding
            query_embedding = self._compute_query_embedding(query)
            
            # Prepare state vector
            state_vector = self._create_state_vector(query, chat_history, available_agents)
            
            # Compute trust scores
            trust_scores = self._compute_trust_scores(available_agents)
            
            # Get Q-values from neural network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                trust_tensor = torch.FloatTensor([trust_scores.get(agent, 0.5) for agent in self.agent_types])
                q_values = self.q_network(state_tensor, trust_tensor).squeeze(0)
            
            # Compute academic selection scores
            selection_scores = {}
            for i, agent_id in enumerate(self.agent_types):
                if agent_id in available_agents:
                    # Compute similarity using domain embeddings
                    similarity = self._compute_agent_similarity(agent_id, query_embedding)
                    
                    # Get current popularity
                    popularity = self.agent_popularities.get(agent_id, 0.5)
                    
                    # Academic formula: SelectionScore = α × similarity + β × popularity
                    academic_score = self.alpha * similarity + self.beta * popularity
                    
                    # Combine with Q-network output
                    q_value = q_values[i].item()
                    combined_score = 0.6 * academic_score + 0.4 * q_value
                    
                    selection_scores[agent_id] = combined_score
            
            # Select top agents
            sorted_agents = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)
            selected_agents = [agent_id for agent_id, _ in sorted_agents[:max_agents]]
            
            logger.info(f"Selected agents: {selected_agents} with scores: {selection_scores}")
            return selected_agents
            
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            # Fallback to simple selection
            return available_agents[:max_agents]
    
    def _compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute query embedding using simple method"""
        # Simple query embedding based on keywords
        keywords = query.lower().split()
        embedding = np.random.normal(0, 1, 384)
        
        # Add keyword-specific signals
        for i, word in enumerate(keywords):
            if i < 384:
                embedding[i] += len(word) * 0.1
        
        return embedding / np.linalg.norm(embedding)
    
    def _compute_agent_similarity(self, agent_id: str, query_embedding: np.ndarray) -> float:
        """Compute agent-query similarity using domain expertise"""
        if agent_id in self.agent_capabilities:
            capability = self.agent_capabilities[agent_id]
            return capability.compute_expertise_similarity(query_embedding, self.domain_embeddings)
        
        return 0.5  # Default similarity
    
    def _compute_trust_scores(self, available_agents: List[str]) -> Dict[str, float]:
        """Compute trust scores for available agents"""
        trust_scores = {}
        
        for agent_id in available_agents:
            if agent_id in self.trust_metrics:
                trust_metrics = self.trust_metrics[agent_id]
                trust_scores[agent_id] = trust_metrics.compute_overall_trust()
            else:
                trust_scores[agent_id] = 0.5  # Default trust
        
        return trust_scores
    
    def _create_state_vector(self, query: str, chat_history: List[Dict[str, Any]], 
                           available_agents: List[str]) -> np.ndarray:
        """Create state vector for neural network input"""
        features = []
        
        # Query features (simplified)
        query_complexity = min(1.0, len(query.split()) / 20.0)  # Normalized by word count
        query_urgency = 0.5  # Default urgency
        query_type_hash = hash(query) % 1000 / 1000.0  # Normalized hash
        
        features.extend([query_complexity, query_urgency, query_type_hash])
        
        # Agent features
        for agent_id in self.agent_types:
            if agent_id in available_agents and agent_id in self.agent_capabilities:
                capability = self.agent_capabilities[agent_id]
                features.extend([
                    0.8,  # similarity (placeholder)
                    self.agent_popularities.get(agent_id, 0.5),
                    capability.trust_score,
                    capability.response_time_avg,
                    capability.current_load
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 1.0, 1.0])  # Unavailable agent
        
        # System features
        system_load = len([msg for msg in chat_history if 'content' in msg]) / 10.0
        time_budget = 1.0  # Full time budget initially
        quality_requirement = 0.8  # Default quality requirement
        
        features.extend([min(1.0, system_load), time_budget, quality_requirement])
        
        # Academic metrics
        for agent_id in self.agent_types:
            if agent_id in self.agent_capabilities:
                capability = self.agent_capabilities[agent_id]
                features.extend([
                    capability.mrr_score,
                    capability.ndcg_score,
                    capability.response_time_avg
                ])
            else:
                features.extend([0.0, 0.0, 1.0])
        
        return np.array(features, dtype=np.float32)
    
    def update_agent_performance(self, agent_id: str, performance_metrics: Dict[str, Any]):
        """
        Update agent performance and popularity using academic formulas
        
        Implements popularity update: popularity_t = popularity_{t-1} + η × (reward - popularity_{t-1})
        
        Args:
            agent_id: Agent identifier
            performance_metrics: Performance metrics including accuracy, efficiency, etc.
        """
        if agent_id not in self.agent_types:
            return
        
        # Extract performance metrics
        accuracy = performance_metrics.get('accuracy', 0.5)
        efficiency = performance_metrics.get('efficiency', 0.5)
        response_time = performance_metrics.get('response_time', 1.0)
        success = performance_metrics.get('success', False)
        
        # Compute reward using academic formula: r = γ × accuracy + δ × efficiency
        reward = self.gamma * accuracy + self.delta * efficiency
        
        # Update popularity using academic formula: popularity_t = popularity_{t-1} + η × (reward - popularity_{t-1})
        old_popularity = self.agent_popularities.get(agent_id, 0.5)
        new_popularity = old_popularity + self.eta * (reward - old_popularity)
        self.agent_popularities[agent_id] = max(0.0, min(1.0, new_popularity))
        
        # Update agent capabilities
        if agent_id in self.agent_capabilities:
            capability = self.agent_capabilities[agent_id]
            
            # Update performance history
            capability.performance_history[str(datetime.now())] = {
                'accuracy': accuracy,
                'efficiency': efficiency,
                'response_time': response_time,
                'reward': reward
            }
            
            # Update academic metrics
            if success:
                capability.mrr_score = min(1.0, capability.mrr_score + 0.05)
                capability.ndcg_score = min(1.0, capability.ndcg_score + 0.03)
                capability.success_rate = min(1.0, capability.success_rate + 0.02)
            else:
                capability.mrr_score = max(0.0, capability.mrr_score - 0.02)
                capability.ndcg_score = max(0.0, capability.ndcg_score - 0.01)
            
            # Update trust metrics
            if agent_id in self.trust_metrics:
                trust_metrics = self.trust_metrics[agent_id]
                trust_metrics.historical_accuracy = 0.9 * trust_metrics.historical_accuracy + 0.1 * accuracy
                trust_metrics.consistency_score = 0.95 * trust_metrics.consistency_score + 0.05 * (1.0 - abs(accuracy - trust_metrics.historical_accuracy))
        
        # Keep performance history manageable
        self.performance_history.append({
            'agent_id': agent_id,
            'timestamp': datetime.now(),
            'metrics': performance_metrics,
            'reward': reward
        })
        
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.info(f"Updated performance for {agent_id}: popularity={self.agent_popularities[agent_id]:.3f}, reward={reward:.3f}")
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained MARL model"""
        save_path = path or self.model_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.q_network.state_dict(),
                'agent_popularities': self.agent_popularities,
                'trust_metrics': self.trust_metrics,
                'config': self.config,
                'timestamp': datetime.now()
            }
            
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'agent_popularities': self.agent_popularities.copy(),
            'agent_trust_scores': {
                agent_id: metrics.compute_overall_trust()
                for agent_id, metrics in self.trust_metrics.items()
            },
            'agent_capabilities': {
                agent_id: {
                    'mrr_score': cap.mrr_score,
                    'ndcg_score': cap.ndcg_score,
                    'success_rate': cap.success_rate,
                    'trust_score': cap.trust_score
                }
                for agent_id, cap in self.agent_capabilities.items()
            },
            'recent_performance': self.performance_history[-10:] if self.performance_history else [],
            'total_interactions': len(self.performance_history)
        }
        
        return summary

def create_marl_policy(config: Optional[Dict[str, Any]] = None) -> MARLPolicy:
    """
    Factory function to create MARL policy
    
    Args:
        config: Policy configuration
        
    Returns:
        Configured MARLPolicy instance
    """
    return MARLPolicy(config) 