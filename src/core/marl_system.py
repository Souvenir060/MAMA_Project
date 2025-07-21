#!/usr/bin/env python3
"""
MARL (Multi-Agent Reinforcement Learning) System

Academic implementation of trust-aware multi-agent reinforcement learning for
dynamic agent selection and coordination in the MAMA system.

Key Formulas:
1. Q-value update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
2. Trust-weighted Q-value: Q_trust(s,a) = Σ w_i * Q_i(s,a), where w_i is trust weight
3. Agent selection probability: P(a_i) = exp(Q_trust(s,a_i)/τ) / Σ exp(Q_trust(s,a_j)/τ)
4. Trust-aware reward: R_trust = R_base + λ * T_score
"""

import numpy as np
import logging
import json
import pickle
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio
from enum import Enum
import threading
import uuid
from collections import Counter

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Action types for MARL system"""
    SELECT_AGENT = "select_agent"
    COORDINATE_AGENTS = "coordinate_agents"
    TRUST_UPDATE = "trust_update"
    PERFORMANCE_EVALUATION = "performance_evaluation"

class StateType(Enum):
    """State types for MARL system"""
    QUERY_STATE = "query_state"
    AGENT_STATE = "agent_state"
    SYSTEM_STATE = "system_state"
    TRUST_STATE = "trust_state"

@dataclass
class MARLState:
    """MARL system state representation"""
    state_id: str
    state_type: StateType
    query_features: np.ndarray
    agent_features: Dict[str, np.ndarray]
    trust_scores: Dict[str, float]
    system_metrics: Dict[str, float]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MARLAction:
    """MARL action representation"""
    action_id: str
    action_type: ActionType
    agent_id: str
    parameters: Dict[str, Any]
    expected_reward: float
    confidence: float
    timestamp: datetime

@dataclass
class MARLExperience:
    """Experience tuple for MARL learning"""
    state: MARLState
    action: MARLAction
    reward: float
    next_state: Optional[MARLState]
    done: bool
    trust_change: float
    performance_metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class AgentQTable:
    """Q-table for individual agent"""
    agent_id: str
    q_values: Dict[str, Dict[str, float]]  # state_id -> action_type -> q_value
    visit_counts: Dict[str, Dict[str, int]]
    last_updated: datetime
    learning_rate: float = 0.1
    discount_factor: float = 0.95

class TrustAwareMARLEngine:
    """
    Trust-aware Multi-Agent Reinforcement Learning Engine
    
    Implements Q-learning with trust-weighted coordination for dynamic
    agent selection and collaborative decision making.
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, trust_weight: float = 0.3, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Trust-Aware MARL Engine
        
        Args:
            learning_rate: Q-learning rate α
            discount_factor: Future reward discount γ  
            trust_weight: Trust weighting factor in coordination
            config: Configuration dictionary
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.trust_weight = trust_weight
        self.config = config or {}
        
        # Initialize agent management components
        self.agents = {}
        self.agent_trust_scores = {}
        self.coordination_history = []
        self.experience_replay_buffer = []
        self.max_replay_buffer_size = 1000
        
        # Add thread safety lock
        self.lock = threading.Lock()
        
        # Add agent performance tracking
        self.agent_performance = {}
        
        # Add missing attributes for MARL functionality
        self.trust_history = defaultdict(list)  # Store trust score history for each agent
        self.state_history = []  # Store all states
        self.action_history = []  # Store all actions
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy exploration
        self.episode_rewards = []  # Store episode rewards
        self.coordination_success_rate = 0.0  # Success rate of coordination
        self.temperature = 1.0  # Temperature for softmax selection
        self.current_state = None  # Current MARL state
        
        # Performance tracking
        self.total_interactions = 0
        self.successful_coordinations = 0
        self.average_rewards: Dict[str, float] = {}
        
        logger.info(f"MARL Engine initialized with lr={learning_rate}, gamma={discount_factor}, trust_weight={trust_weight}")
    
    def register_agent(self, agent_id: str, capabilities: List[str], 
                      initial_trust: float = 0.5) -> None:
        """
        Register new agent in MARL system
        
        Args:
            agent_id: Agent identifier
            capabilities: Agent capabilities
            initial_trust: Initial trust score
        """
        try:
            with self.lock:
                if agent_id not in self.agents:
                    self.agents[agent_id] = AgentQTable(
                        agent_id=agent_id,
                        q_values=defaultdict(lambda: defaultdict(float)),
                        visit_counts=defaultdict(lambda: defaultdict(int)),
                        last_updated=datetime.now()
                    )
                
                # Initialize trust
                self.agent_trust_scores[agent_id] = initial_trust
                
                # Initialize performance metrics
                self.agent_performance[agent_id] = {
                    'success_rate': 0.0,
                    'avg_reward': 0.0,
                    'response_time': 0.0,
                    'trust_trend': 0.0,
                    'collaboration_score': 0.0
                }
                
            logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    def create_state(self, query_text: str, available_agents: List[str],
                    system_context: Dict[str, Any]) -> MARLState:
        """
        Create MARL state from current system context
        
        Args:
            query_text: User query
            available_agents: Available agent IDs
            system_context: System context information
            
        Returns:
            MARLState object
        """
        try:
            # Extract query features
            query_features = self._extract_query_features(query_text)
            
            # Extract agent features
            agent_features = {}
            for agent_id in available_agents:
                if agent_id in self.agents:
                    agent_features[agent_id] = self._extract_agent_features(
                        agent_id, system_context
                    )
            
            # Get current trust scores
            current_trust = {aid: self.agent_trust_scores.get(aid, 0.5) 
                           for aid in available_agents}
            
            # Extract system metrics
            system_metrics = self._extract_system_metrics(system_context)
            
            # Create state
            state = MARLState(
                state_id=f"state_{uuid.uuid4().hex[:12]}",
                state_type=StateType.QUERY_STATE,
                query_features=query_features,
                agent_features=agent_features,
                trust_scores=current_trust,
                system_metrics=system_metrics,
                timestamp=datetime.now(),
                context=system_context
            )
            
            self.current_state = state
            self.state_history.append(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to create MARL state: {e}")
            raise
    
    def select_agents(self, state: MARLState, num_agents: int = 3,
                     selection_strategy: str = "trust_weighted") -> List[Tuple[str, float]]:
        """
        Select optimal agents using trust-aware MARL
        
        Implementation of: P(a_i) = exp(Q_trust(s,a_i)/τ) / Σ exp(Q_trust(s,a_j)/τ)
        
        Args:
            state: Current MARL state
            num_agents: Number of agents to select
            selection_strategy: Selection strategy
            
        Returns:
            List of (agent_id, selection_probability) tuples
        """
        try:
            available_agents = list(state.agent_features.keys())
            if not available_agents:
                return []
            
            # Compute trust-weighted Q-values for each agent
            agent_scores = {}
            for agent_id in available_agents:
                q_trust = self._compute_trust_weighted_q_value(state, agent_id)
                agent_scores[agent_id] = q_trust
            
            # Apply selection strategy
            if selection_strategy == "trust_weighted":
                selected_agents = self._softmax_selection(
                    agent_scores, num_agents, self.temperature
                )
            elif selection_strategy == "epsilon_greedy":
                selected_agents = self._epsilon_greedy_selection(
                    agent_scores, num_agents
                )
            elif selection_strategy == "thompson_sampling":
                selected_agents = self._thompson_sampling_selection(
                    state, agent_scores, num_agents
                )
            else:
                # Default to trust-weighted
                selected_agents = self._softmax_selection(
                    agent_scores, num_agents, self.temperature
                )
            
            # Create selection actions
            for agent_id, prob in selected_agents:
                action = MARLAction(
                    action_id=f"action_{uuid.uuid4().hex[:12]}",
                    action_type=ActionType.SELECT_AGENT,
                    agent_id=agent_id,
                    parameters={'selection_probability': prob, 'q_value': agent_scores[agent_id]},
                    expected_reward=agent_scores[agent_id],
                    confidence=prob,
                    timestamp=datetime.now()
                )
                self.action_history.append(action)
            
            logger.info(f"Selected {len(selected_agents)} agents using {selection_strategy}")
            return selected_agents
            
        except Exception as e:
            logger.error(f"Failed to select agents: {e}")
            return []
    
    def coordinate_agents(self, selected_agents: List[str], 
                         coordination_strategy: str = "hierarchical") -> Dict[str, Any]:
        """
        Coordinate selected agents for collaborative task execution
        
        Args:
            selected_agents: List of selected agent IDs
            coordination_strategy: Coordination strategy
            
        Returns:
            Coordination plan
        """
        try:
            if not selected_agents:
                return {}
            
            coordination_plan = {
                'strategy': coordination_strategy,
                'agents': selected_agents,
                'roles': {},
                'dependencies': {},
                'communication_plan': {},
                'success_probability': 0.0
            }
            
            if coordination_strategy == "hierarchical":
                coordination_plan = self._hierarchical_coordination(selected_agents)
            elif coordination_strategy == "collaborative":
                coordination_plan = self._collaborative_coordination(selected_agents)
            elif coordination_strategy == "competitive":
                coordination_plan = self._competitive_coordination(selected_agents)
            else:
                coordination_plan = self._default_coordination(selected_agents)
            
            # Create coordination action
            action = MARLAction(
                action_id=f"action_{uuid.uuid4().hex[:12]}",
                action_type=ActionType.COORDINATE_AGENTS,
                agent_id="system",
                parameters=coordination_plan,
                expected_reward=coordination_plan.get('success_probability', 0.0),
                confidence=coordination_plan.get('success_probability', 0.0),
                timestamp=datetime.now()
            )
            self.action_history.append(action)
            
            logger.info(f"Coordinated {len(selected_agents)} agents using {coordination_strategy}")
            return coordination_plan
            
        except Exception as e:
            logger.error(f"Failed to coordinate agents: {e}")
            return {}
    
    def update_q_values(self, experience: MARLExperience) -> None:
        """
        Update Q-values using trust-aware Q-learning
        
        Implementation of: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        With trust-aware reward: R_trust = R_base + λ * T_score
        
        Args:
            experience: MARL experience tuple
        """
        try:
            with self.lock:
                state = experience.state
                action = experience.action
                reward = experience.reward
                next_state = experience.next_state
                
                # Compute trust-aware reward
                trust_bonus = self.trust_weight * experience.trust_change
                trust_aware_reward = reward + trust_bonus
                
                # Get agent Q-table
                agent_id = action.agent_id
                if agent_id not in self.agents:
                    self.register_agent(agent_id, [])
                
                q_table = self.agents[agent_id]
                state_key = state.state_id
                action_key = action.action_type.value
                
                # Current Q-value
                current_q = q_table.q_values[state_key][action_key]
                
                # Compute max Q-value for next state
                max_next_q = 0.0
                if next_state is not None and not experience.done:
                    max_next_q = self._compute_max_q_value(next_state, agent_id)
                
                # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                td_target = trust_aware_reward + self.discount_factor * max_next_q
                td_error = td_target - current_q
                new_q = current_q + self.learning_rate * td_error
                
                # Update Q-table
                q_table.q_values[state_key][action_key] = new_q
                q_table.visit_counts[state_key][action_key] += 1
                q_table.last_updated = datetime.now()
                
                # Add to experience buffer
                self.experience_replay_buffer.append(experience)
                
                logger.debug(f"Updated Q-value for {agent_id}: {current_q:.3f} -> {new_q:.3f} (TD error: {td_error:.3f})")
                
        except Exception as e:
            logger.error(f"Failed to update Q-values: {e}")
    
    def update_trust_scores(self, agent_performance: Dict[str, Dict[str, float]]) -> None:
        """
        Update agent trust scores based on performance
        
        Args:
            agent_performance: Performance metrics for each agent
        """
        try:
            with self.lock:
                for agent_id, metrics in agent_performance.items():
                    if agent_id not in self.agent_trust_scores:
                        continue
                    
                    # Compute trust update based on performance
                    current_trust = self.agent_trust_scores[agent_id]
                    
                    # Performance factors
                    success_factor = metrics.get('success_rate', 0.5)
                    quality_factor = metrics.get('quality_score', 0.5)
                    timeliness_factor = metrics.get('timeliness_score', 0.5)
                    
                    # Weighted trust update
                    performance_score = (
                        0.4 * success_factor +
                        0.3 * quality_factor +
                        0.3 * timeliness_factor
                    )
                    
                    # Adaptive trust update
                    trust_change = 0.1 * (performance_score - current_trust)
                    new_trust = np.clip(current_trust + trust_change, 0.0, 1.0)
                    
                    # Update trust
                    self.agent_trust_scores[agent_id] = new_trust
                    
                    # Update agent performance metrics
                    self.agent_performance[agent_id].update(metrics)
                    
                    logger.debug(f"Updated trust for {agent_id}: {current_trust:.3f} -> {new_trust:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to update trust scores: {e}")
    
    def replay_experience(self, batch_size: Optional[int] = None) -> None:
        """
        Replay experiences for batch learning
        
        Args:
            batch_size: Size of experience batch
        """
        try:
            if len(self.experience_replay_buffer) < (batch_size or self.max_replay_buffer_size):
                return
            
            # Sample random experiences
            batch_size = batch_size or self.max_replay_buffer_size
            experiences = random.sample(self.experience_replay_buffer, batch_size)
            
            # Update Q-values for sampled experiences
            for experience in experiences:
                self.update_q_values(experience)
            
            logger.debug(f"Replayed {len(experiences)} experiences")
            
        except Exception as e:
            logger.error(f"Failed to replay experience: {e}")
    
    def _compute_trust_weighted_q_value(self, state: MARLState, agent_id: str) -> float:
        """
        Compute trust-weighted Q-value for agent selection
        
        Implementation of: Q_trust(s,a) = Σ w_i * Q_i(s,a)
        
        Args:
            state: Current state
            agent_id: Agent ID
            
        Returns:
            Trust-weighted Q-value
        """
        try:
            if agent_id not in self.agents:
                return 0.0
            
            q_table = self.agents[agent_id]
            state_key = state.state_id
            
            # Get Q-values for all action types
            base_q_values = []
            for action_type in ActionType:
                q_val = q_table.q_values[state_key][action_type.value]
                base_q_values.append(q_val)
            
            # Base Q-value (average across action types)
            base_q = np.mean(base_q_values) if base_q_values else 0.0
            
            # Trust weight
            trust_score = self.agent_trust_scores.get(agent_id, 0.5)
            
            # Agent-specific features
            agent_features = state.agent_features.get(agent_id, np.zeros(10))
            feature_score = np.mean(agent_features) if len(agent_features) > 0 else 0.0
            
            # Trust-weighted Q-value
            q_trust = base_q + self.trust_weight * trust_score + 0.1 * feature_score
            
            return q_trust
            
        except Exception as e:
            logger.error(f"Failed to compute trust-weighted Q-value: {e}")
            return 0.0
    
    def _softmax_selection(self, agent_scores: Dict[str, float], 
                          num_agents: int, temperature: float) -> List[Tuple[str, float]]:
        """
        Softmax-based agent selection
        
        Implementation of: P(a_i) = exp(Q_trust(s,a_i)/τ) / Σ exp(Q_trust(s,a_j)/τ)
        
        Args:
            agent_scores: Agent Q-values
            num_agents: Number of agents to select
            temperature: Softmax temperature
            
        Returns:
            Selected agents with probabilities
        """
        try:
            if not agent_scores:
                return []
            
            # Compute softmax probabilities
            scores = np.array(list(agent_scores.values()))
            exp_scores = np.exp(scores / temperature)
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Create probability distribution
            agent_ids = list(agent_scores.keys())
            agent_probs = list(zip(agent_ids, probabilities))
            
            # Sort by probability (descending)
            agent_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top agents
            selected = agent_probs[:num_agents]
            
            return selected
            
        except Exception as e:
            logger.error(f"Failed to perform softmax selection: {e}")
            return []
    
    def _epsilon_greedy_selection(self, agent_scores: Dict[str, float], 
                                 num_agents: int) -> List[Tuple[str, float]]:
        """
        Epsilon-greedy agent selection
        
        Args:
            agent_scores: Agent Q-values
            num_agents: Number of agents to select
            
        Returns:
            Selected agents with scores
        """
        try:
            if not agent_scores:
                return []
            
            selected = []
            available_agents = list(agent_scores.keys())
            
            for _ in range(min(num_agents, len(available_agents))):
                if random.random() < self.exploration_rate:
                    # Explore: random selection
                    agent_id = random.choice(available_agents)
                else:
                    # Exploit: greedy selection
                    agent_id = max(available_agents, key=lambda x: agent_scores[x])
                
                selected.append((agent_id, agent_scores[agent_id]))
                available_agents.remove(agent_id)
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
            
            return selected
            
        except Exception as e:
            logger.error(f"Failed to perform epsilon-greedy selection: {e}")
            return []
    
    def _thompson_sampling_selection(self, state: MARLState, 
                                   agent_scores: Dict[str, float],
                                   num_agents: int) -> List[Tuple[str, float]]:
        """
        Thompson sampling for agent selection
        
        Args:
            state: Current state
            agent_scores: Agent Q-values
            num_agents: Number of agents to select
            
        Returns:
            Selected agents with sampled values
        """
        try:
            if not agent_scores:
                return []
            
            selected = []
            
            for agent_id in agent_scores.keys():
                # Get visitation statistics
                if agent_id in self.agents:
                    q_table = self.agents[agent_id]
                    state_key = state.state_id
                    visit_count = sum(q_table.visit_counts[state_key].values())
                    
                    # Thompson sampling with Beta distribution
                    alpha = 1 + visit_count  # Success count
                    beta = 1 + max(0, 10 - visit_count)  # Failure count
                    
                    sampled_value = np.random.beta(alpha, beta)
                    selected.append((agent_id, sampled_value))
            
            # Sort by sampled values and select top agents
            selected.sort(key=lambda x: x[1], reverse=True)
            return selected[:num_agents]
            
        except Exception as e:
            logger.error(f"Failed to perform Thompson sampling: {e}")
            return []
    
    def _hierarchical_coordination(self, agents: List[str]) -> Dict[str, Any]:
        """Hierarchical coordination strategy"""
        if not agents:
            return {}
        
        # Assign roles based on trust scores
        sorted_agents = sorted(agents, key=lambda x: self.agent_trust_scores.get(x, 0.5), reverse=True)
        
        coordination_plan = {
            'strategy': 'hierarchical',
            'agents': agents,
            'roles': {
                'leader': sorted_agents[0] if sorted_agents else None,
                'coordinators': sorted_agents[1:3] if len(sorted_agents) > 1 else [],
                'workers': sorted_agents[3:] if len(sorted_agents) > 3 else []
            },
            'dependencies': {agent: [] for agent in agents},
            'communication_plan': {'type': 'hierarchical', 'levels': 3},
            'success_probability': np.mean([self.agent_trust_scores.get(a, 0.5) for a in agents])
        }
        
        return coordination_plan
    
    def _collaborative_coordination(self, agents: List[str]) -> Dict[str, Any]:
        """Collaborative coordination strategy"""
        if not agents:
            return {}
        
        coordination_plan = {
            'strategy': 'collaborative',
            'agents': agents,
            'roles': {agent: 'collaborator' for agent in agents},
            'dependencies': {agent: [a for a in agents if a != agent] for agent in agents},
            'communication_plan': {'type': 'all-to-all', 'rounds': 3},
            'success_probability': np.mean([self.agent_trust_scores.get(a, 0.5) for a in agents]) * 0.9
        }
        
        return coordination_plan
    
    def _competitive_coordination(self, agents: List[str]) -> Dict[str, Any]:
        """Competitive coordination strategy"""
        if not agents:
            return {}
        
        coordination_plan = {
            'strategy': 'competitive',
            'agents': agents,
            'roles': {agent: 'competitor' for agent in agents},
            'dependencies': {agent: [] for agent in agents},
            'communication_plan': {'type': 'minimal', 'rounds': 1},
            'success_probability': max([self.agent_trust_scores.get(a, 0.5) for a in agents])
        }
        
        return coordination_plan
    
    def _default_coordination(self, agents: List[str]) -> Dict[str, Any]:
        """Default coordination strategy"""
        if not agents:
            return {}
        
        coordination_plan = {
            'strategy': 'default',
            'agents': agents,
            'roles': {agent: 'agent' for agent in agents},
            'dependencies': {agent: [] for agent in agents},
            'communication_plan': {'type': 'sequential', 'rounds': 2},
            'success_probability': np.mean([self.agent_trust_scores.get(a, 0.5) for a in agents])
        }
        
        return coordination_plan
    
    def _extract_query_features(self, query_text: str) -> np.ndarray:
        """Extract features from query text for MARL state representation"""
        features = []
        
        # Basic text statistics
        words = query_text.split()
        features.extend([
            len(query_text),  # Text length
            len(words),       # Word count
            len(set(words)),  # Unique word count
            sum(c.isupper() for c in query_text) / max(len(query_text), 1),  # Uppercase ratio
            sum(c.isdigit() for c in query_text) / max(len(query_text), 1),  # Digit ratio
        ])
        
        # Character frequency analysis (normalized)
        char_freq = np.zeros(26)  # Only consider English letters
        total_chars = 0
        for c in query_text.lower():
            # Only process English letters
            if ord('a') <= ord(c) <= ord('z'):
                char_freq[ord(c) - ord('a')] += 1
                total_chars += 1
        
        # Normalize character frequency
        if total_chars > 0:
            char_freq = char_freq / total_chars
        features.extend(char_freq)
        
        # Add padding if needed
        feature_dim = self.config.get("feature_dimension", 128)
        current_dim = len(features)
        if current_dim < feature_dim:
            features.extend([0.0] * (feature_dim - current_dim))
        elif current_dim > feature_dim:
            features = features[:feature_dim]
            
        return np.array(features, dtype=np.float32)
    
    def _extract_agent_features(self, agent_id: str, context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features for agent"""
        try:
            # Agent performance metrics
            performance = self.agent_performance.get(agent_id, {})
            trust_score = self.agent_trust_scores.get(agent_id, 0.5)
            
            # Feature vector
            features = [
                trust_score,  # Trust score
                performance.get('success_rate', 0.0),  # Success rate
                performance.get('avg_reward', 0.0),  # Average reward
                performance.get('response_time', 1.0),  # Response time (normalized)
                performance.get('collaboration_score', 0.5),  # Collaboration score
                len(self.trust_history.get(agent_id, [])),  # Trust history length
                1 if agent_id in context.get('recent_agents', []) else 0,  # Recently used
                1 if agent_id in context.get('preferred_agents', []) else 0,  # User preference
                context.get('system_load', 0.5),  # System load
                context.get('time_pressure', 0.5),  # Time pressure
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract agent features: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _extract_system_metrics(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract system-level metrics"""
        try:
            metrics = {
                'system_load': context.get('system_load', 0.5),
                'response_time': context.get('avg_response_time', 1.0),
                'success_rate': context.get('system_success_rate', 0.8),
                'agent_count': len(self.agents),
                'query_complexity': context.get('query_complexity', 0.5),
                'time_pressure': context.get('time_pressure', 0.5),
                'resource_availability': context.get('resource_availability', 0.8),
                'coordination_overhead': context.get('coordination_overhead', 0.3)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to extract system metrics: {e}")
            return {}
    
    def _compute_max_q_value(self, state: MARLState, agent_id: str) -> float:
        """Compute maximum Q-value for next state"""
        try:
            if agent_id not in self.agents:
                return 0.0
            
            q_table = self.agents[agent_id]
            state_key = state.state_id
            
            # Get maximum Q-value across all actions
            q_values = [q_table.q_values[state_key][action_type.value] 
                       for action_type in ActionType]
            
            return max(q_values) if q_values else 0.0
            
        except Exception as e:
            logger.error(f"Failed to compute max Q-value: {e}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            metrics = {
                'learning_stats': {
                    'total_states': len(self.state_history),
                    'total_actions': len(self.action_history),
                    'experience_buffer_size': len(self.experience_replay_buffer),
                    'exploration_rate': self.exploration_rate,
                    'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0
                },
                'agent_stats': {
                    'total_agents': len(self.agents),
                    'avg_trust_score': np.mean(list(self.agent_trust_scores.values())) if self.agent_trust_scores else 0.0,
                    'trust_variance': np.var(list(self.agent_trust_scores.values())) if self.agent_trust_scores else 0.0
                },
                'coordination_stats': {
                    'success_rate': self.coordination_success_rate,
                    'avg_agents_per_task': np.mean([len(s.agent_features) for s in self.state_history]) if self.state_history else 0.0
                },
                'q_table_stats': {
                    'total_q_entries': sum(len(qt.q_values) for qt in self.agents.values()),
                    'avg_q_value': np.mean([q for qt in self.agents.values() 
                                          for state_q in qt.q_values.values() 
                                          for q in state_q.values()]) if self.agents else 0.0
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save MARL model to file"""
        try:
            save_data = {
                'agents': {
                    aid: {
                        'agent_id': qt.agent_id,
                        'q_values': dict(qt.q_values),
                        'visit_counts': dict(qt.visit_counts),
                        'last_updated': qt.last_updated.isoformat(),
                        'learning_rate': qt.learning_rate,
                        'discount_factor': qt.discount_factor
                    } for aid, qt in self.agents.items()
                },
                'agent_trust_scores': self.agent_trust_scores,
                'trust_history': {
                    aid: [(ts.isoformat(), score) for ts, score in history]
                    for aid, history in self.trust_history.items()
                },
                'agent_performance': dict(self.agent_performance),
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'trust_weight': self.trust_weight
                },
                'performance_metrics': self.get_performance_metrics()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Saved MARL model to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save MARL model: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load MARL model from file"""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore Q-tables
            for aid, qt_data in save_data.get('agents', {}).items():
                self.agents[aid] = AgentQTable(
                    agent_id=qt_data['agent_id'],
                    q_values=defaultdict(lambda: defaultdict(float), qt_data['q_values']),
                    visit_counts=defaultdict(lambda: defaultdict(int), qt_data['visit_counts']),
                    last_updated=datetime.fromisoformat(qt_data['last_updated']),
                    learning_rate=qt_data.get('learning_rate', self.learning_rate),
                    discount_factor=qt_data.get('discount_factor', self.discount_factor)
                )
            
            # Restore trust scores and history
            self.agent_trust_scores = save_data.get('agent_trust_scores', {})
            trust_history_data = save_data.get('trust_history', {})
            for aid, history in trust_history_data.items():
                self.trust_history[aid] = [(datetime.fromisoformat(ts), score) 
                                         for ts, score in history]
            
            # Restore agent performance
            self.agent_performance = defaultdict(lambda: defaultdict(float))
            for aid, perf in save_data.get('agent_performance', {}).items():
                self.agent_performance[aid].update(perf)
            
            # Restore hyperparameters
            hyperparams = save_data.get('hyperparameters', {})
            self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
            self.discount_factor = hyperparams.get('discount_factor', self.discount_factor)
            self.trust_weight = hyperparams.get('trust_weight', self.trust_weight)
            
            logger.info(f"Loaded MARL model from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load MARL model: {e}")


# Global MARL engine instance
marl_engine = None

def initialize_marl_engine(**kwargs) -> TrustAwareMARLEngine:
    """Initialize global MARL engine"""
    global marl_engine
    if marl_engine is None:
        default_kwargs = {
            'learning_rate': 0.001,
            'discount_factor': 0.95, 
            'trust_weight': 0.4
        }
        default_kwargs.update(kwargs)
        marl_engine = TrustAwareMARLEngine(**default_kwargs)
    return marl_engine

def select_optimal_agents(query_text: str, available_agents: List[str],
                        num_agents: int = 3) -> List[Tuple[str, float]]:
    """Global function to select optimal agents using MARL"""
    if marl_engine is None:
        initialize_marl_engine()
    
    # Create state
    state = marl_engine.create_state(query_text, available_agents, {})
    
    # Select agents
    return marl_engine.select_agents(state, num_agents)

def update_agent_performance(agent_performance: Dict[str, Dict[str, float]]) -> None:
    """Global function to update agent performance"""
    if marl_engine is None:
        initialize_marl_engine()
    
    marl_engine.update_trust_scores(agent_performance) 