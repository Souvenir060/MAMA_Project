# DVC_exp/marl/environment.py

"""
MARL Environment for MAMA Flight Assistant System

Academic implementation of reinforcement learning environment for dynamic agent
selection based on research paper formulas.

Key Mathematical Formulations:
1. Agent Selection Score: SelectionScore = α × similarity + β × popularity
2. Reward Function: r = γ × accuracy + δ × efficiency  
3. Popularity Update: popularity_t = popularity_{t-1} + η × (reward - popularity_{t-1})
4. Dynamic Agent Selection: r = α × MRR + β × NDCG@5 - γ × ART
"""

import gym
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Agent types in MAMA system"""
    WEATHER_AGENT = "weather_agent"
    SAFETY_ASSESSMENT_AGENT = "safety_assessment_agent"
    FLIGHT_INFO_AGENT = "flight_info_agent" 
    ECONOMIC_AGENT = "economic_agent"
    INTEGRATION_AGENT = "integration_agent"

class TaskType(Enum):
    """Task types for agent selection"""
    WEATHER_ANALYSIS = "weather_analysis"
    SAFETY_EVALUATION = "safety_evaluation"
    FLIGHT_SEARCH = "flight_search"
    ECONOMIC_ANALYSIS = "economic_analysis"
    DATA_INTEGRATION = "data_integration"

@dataclass
class MARLState:
    """
    MARL environment state representation
    
    Academic formulation based on paper specifications for agent selection
    and trust-aware coordination in multi-agent systems.
    """
    # Query characteristics
    query_complexity: float  # Normalized complexity score [0,1]
    query_urgency: float    # Time sensitivity [0,1]
    query_type: str         # Query category
    
    # Agent availability and performance
    agent_similarities: Dict[str, float]  # SBERT similarity scores
    agent_popularities: Dict[str, float]  # Historical success rates
    agent_trust_scores: Dict[str, float]  # Trust evaluation scores
    agent_response_times: Dict[str, float]  # Average response times
    agent_load_factors: Dict[str, float]   # Current workload [0,1]
    
    # System metrics
    system_load: float      # Overall system utilization [0,1]
    time_budget: float      # Remaining time budget [0,1]
    quality_requirement: float  # Required quality threshold [0,1]
    
    # Historical context
    recent_agent_performance: Dict[str, Dict[str, float]]
    coordination_history: List[Dict[str, Any]]
    
    # Academic metrics tracking
    mrr_scores: Dict[str, float]        # Mean Reciprocal Rank
    ndcg_scores: Dict[str, float]       # NDCG@5 scores
    response_times: Dict[str, float]    # Average Response Times
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        """
        Convert state to feature vector for neural network processing
        
        Returns:
            numpy array of normalized features
        """
        features = []
        
        # Query features
        features.extend([
            self.query_complexity,
            self.query_urgency,
            hash(self.query_type) % 1000 / 1000.0  # Normalized hash
        ])
        
        # Agent features (assuming fixed agent order)
        agent_ids = sorted(self.agent_similarities.keys())
        for agent_id in agent_ids:
            features.extend([
                self.agent_similarities.get(agent_id, 0.0),
                self.agent_popularities.get(agent_id, 0.0),
                self.agent_trust_scores.get(agent_id, 0.5),
                self.agent_response_times.get(agent_id, 1.0),
                self.agent_load_factors.get(agent_id, 0.0)
            ])
        
        # System features
        features.extend([
            self.system_load,
            self.time_budget,
            self.quality_requirement
        ])
        
        # Academic metrics
        for agent_id in agent_ids:
            features.extend([
                self.mrr_scores.get(agent_id, 0.0),
                self.ndcg_scores.get(agent_id, 0.0),
                self.response_times.get(agent_id, 1.0)
            ])
        
        return np.array(features, dtype=np.float32)

@dataclass
class MARLAction:
    """
    MARL action representation for agent selection
    
    Implements agent selection based on academic formulation:
    SelectionScore = α × similarity + β × popularity
    """
    selected_agents: List[str]           # List of selected agent IDs
    coordination_strategy: str           # Strategy for coordination
    priority_order: List[str]           # Execution order
    time_allocation: Dict[str, float]    # Time budget per agent
    quality_thresholds: Dict[str, float] # Quality requirements per agent
    
    # Academic parameters
    alpha: float = 0.7  # Similarity weight in SelectionScore formula
    beta: float = 0.3   # Popularity weight in SelectionScore formula
    
    def compute_selection_scores(self, state: MARLState) -> Dict[str, float]:
        """
        Compute agent selection scores using academic formula:
        SelectionScore = α × similarity + β × popularity
        
        Args:
            state: Current MARL state
            
        Returns:
            Dictionary of agent selection scores
        """
        scores = {}
        for agent_id in state.agent_similarities.keys():
            similarity = state.agent_similarities.get(agent_id, 0.0)
            popularity = state.agent_popularities.get(agent_id, 0.0)
            
            # Academic formula implementation
            selection_score = self.alpha * similarity + self.beta * popularity
            scores[agent_id] = selection_score
            
        return scores

@dataclass 
class MARLReward:
    """
    MARL reward computation based on academic formulations
    
    Implements multiple reward functions from the paper:
    1. r = γ × accuracy + δ × efficiency
    2. r = α × MRR + β × NDCG@5 - γ × ART
    """
    # Basic reward components
    accuracy: float     # Task completion accuracy [0,1]
    efficiency: float   # Resource/time efficiency [0,1]
    
    # Academic metrics
    mrr_score: float    # Mean Reciprocal Rank
    ndcg_at_5: float    # NDCG@5 score
    avg_response_time: float  # Average Response Time (normalized)
    
    # Reward function parameters
    gamma: float = 0.6   # Accuracy weight
    delta: float = 0.4   # Efficiency weight
    
    # Dynamic agent selection parameters
    alpha_marl: float = 0.5  # MRR weight
    beta_marl: float = 0.3   # NDCG weight  
    gamma_marl: float = 0.2  # ART penalty weight
    
    def compute_basic_reward(self) -> float:
        """
        Compute basic reward using formula: r = γ × accuracy + δ × efficiency
        
        Returns:
            Basic reward score
        """
        return self.gamma * self.accuracy + self.delta * self.efficiency
    
    def compute_dynamic_selection_reward(self) -> float:
        """
        Compute dynamic agent selection reward using formula:
        r = α × MRR + β × NDCG@5 - γ × ART
        
        Returns:
            Dynamic selection reward score
        """
        return (self.alpha_marl * self.mrr_score + 
                self.beta_marl * self.ndcg_at_5 - 
                self.gamma_marl * self.avg_response_time)
    
    def compute_total_reward(self) -> float:
        """
        Compute total reward combining both formulations
        
        Returns:
            Combined reward score
        """
        basic_reward = self.compute_basic_reward()
        dynamic_reward = self.compute_dynamic_selection_reward()
        
        # Weighted combination
        return 0.7 * basic_reward + 0.3 * dynamic_reward

class MARLEnvironment(gym.Env):
    """
    Multi-Agent Reinforcement Learning Environment for MAMA System
    
    Academic implementation following research paper specifications for
    dynamic agent selection and trust-aware coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MARL environment
        
        Args:
            config: Environment configuration parameters
        """
        super().__init__()
        
        self.config = config or {}
        
        # Agent configuration (compatible with MAMA system)
        self.agent_types = list(AgentType)
        self.task_types = list(TaskType)
        self.max_agents_per_selection = self.config.get('max_agents', 3)
        
        # State and action spaces
        self.state_dim = self._calculate_state_dimension()
        self.action_space = gym.spaces.MultiDiscrete([
            len(self.agent_types),  # Primary agent selection
            len(self.agent_types),  # Secondary agent selection  
            len(self.agent_types),  # Tertiary agent selection
            4,  # Coordination strategy (0-3)
            10  # Priority configuration (0-9)
        ])
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Academic parameters with learning rates
        self.learning_rates = {
            'popularity_update': self.config.get('eta', 0.1),  # η in popularity update
            'trust_update': self.config.get('trust_lr', 0.05),
            'quality_update': self.config.get('quality_lr', 0.1)
        }
        
        # Historical tracking for academic metrics
        self.agent_popularity_history = {}
        self.agent_performance_history = {}
        self.reward_history = []
        
        # Initialize agent popularities
        self._initialize_agent_popularities()
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.max_steps_per_episode = self.config.get('max_steps', 50)
        
        logger.info(f"MARL Environment initialized with {len(self.agent_types)} agents")
    
    def _calculate_state_dimension(self) -> int:
        """Calculate the dimension of state vector"""
        base_features = 3  # query_complexity, urgency, type
        agent_features = len(self.agent_types) * 5  # per-agent features
        system_features = 3  # system_load, time_budget, quality_requirement
        academic_features = len(self.agent_types) * 3  # MRR, NDCG, ART per agent
        
        return base_features + agent_features + system_features + academic_features
    
    def _initialize_agent_popularities(self):
        """Initialize agent popularity scores"""
        for agent_type in self.agent_types:
            agent_id = agent_type.value
            self.agent_popularity_history[agent_id] = {
                'popularity': 0.5,  # Initial popularity
                'success_count': 0,
                'failure_count': 0,
                'total_interactions': 0,
                'recent_rewards': []
            }
    
    def reset(self, seed: Optional[int] = None, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_episode += 1
        
        # Generate initial state
        self.state = self._generate_initial_state()
        
        info = {
            'episode': self.current_episode,
            'agent_popularities': {aid: data['popularity'] 
                                 for aid, data in self.agent_popularity_history.items()}
        }
        
        return self.state.to_vector(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute environment step
        
        Args:
            action: Agent selection action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Parse action
        marl_action = self._parse_action(action)
        
        # Simulate agent execution and get results
        execution_results = self._simulate_agent_execution(marl_action)
        
        # Compute reward using academic formulations
        reward_components = self._compute_reward_components(execution_results)
        marl_reward = MARLReward(**reward_components)
        reward = marl_reward.compute_total_reward()
        
        # Update agent popularities using academic formula
        self._update_agent_popularities(marl_action, marl_reward)
        
        # Update state
        self.state = self._update_state(execution_results)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Prepare info
        info = {
            'execution_results': execution_results,
            'reward_components': reward_components,
            'agent_popularities': {aid: data['popularity'] 
                                 for aid, data in self.agent_popularity_history.items()},
            'selection_scores': marl_action.compute_selection_scores(self.state)
        }
        
        self.reward_history.append(reward)
        
        return self.state.to_vector(), reward, terminated, truncated, info
    
    def _generate_initial_state(self) -> MARLState:
        """Generate initial MARL state"""
        agent_ids = [agent_type.value for agent_type in self.agent_types]
        
        # Generate realistic query characteristics
        query_complexity = np.random.uniform(0.3, 0.9)
        query_urgency = np.random.uniform(0.1, 0.8)
        query_type = np.random.choice(['simple_search', 'complex_analysis', 'comparison'])
        
        # Generate agent similarities (SBERT would compute these)
        agent_similarities = {
            aid: np.random.uniform(0.4, 0.95) for aid in agent_ids
        }
        
        # Get current popularities
        agent_popularities = {
            aid: self.agent_popularity_history[aid]['popularity'] 
            for aid in agent_ids
        }
        
        # Generate trust scores
        agent_trust_scores = {
            aid: np.random.uniform(0.6, 0.9) for aid in agent_ids
        }
        
        # Generate response times (normalized)
        agent_response_times = {
            aid: np.random.uniform(0.1, 0.8) for aid in agent_ids
        }
        
        # Generate load factors
        agent_load_factors = {
            aid: np.random.uniform(0.0, 0.6) for aid in agent_ids
        }
        
        # System metrics
        system_load = np.random.uniform(0.2, 0.7)
        time_budget = np.random.uniform(0.5, 1.0)
        quality_requirement = np.random.uniform(0.7, 0.95)
        
        # Academic metrics (would be computed from historical data)
        mrr_scores = {aid: np.random.uniform(0.3, 0.8) for aid in agent_ids}
        ndcg_scores = {aid: np.random.uniform(0.4, 0.9) for aid in agent_ids}
        response_times = {aid: np.random.uniform(0.1, 0.7) for aid in agent_ids}
        
        return MARLState(
            query_complexity=query_complexity,
            query_urgency=query_urgency,
            query_type=query_type,
            agent_similarities=agent_similarities,
            agent_popularities=agent_popularities,
            agent_trust_scores=agent_trust_scores,
            agent_response_times=agent_response_times,
            agent_load_factors=agent_load_factors,
            system_load=system_load,
            time_budget=time_budget,
            quality_requirement=quality_requirement,
            recent_agent_performance={},
            coordination_history=[],
            mrr_scores=mrr_scores,
            ndcg_scores=ndcg_scores,
            response_times=response_times
        )
    
    def _parse_action(self, action: np.ndarray) -> MARLAction:
        """Parse numerical action to MARLAction"""
        agent_ids = [agent_type.value for agent_type in self.agent_types]
        
        # Select agents based on action indices
        selected_agents = []
        for i in range(min(3, len(action))):
            if action[i] < len(agent_ids):
                agent_id = agent_ids[action[i]]
                if agent_id not in selected_agents:
                    selected_agents.append(agent_id)
        
        # Coordination strategy
        coordination_strategies = ['sequential', 'parallel', 'hierarchical', 'competitive']
        coord_strategy = coordination_strategies[action[3] % len(coordination_strategies)]
        
        # Priority order (same as selection order for simplicity)
        priority_order = selected_agents.copy()
        
        # Time allocation (equal distribution)
        time_allocation = {aid: 1.0 / len(selected_agents) for aid in selected_agents}
        
        # Quality thresholds
        quality_thresholds = {aid: 0.8 for aid in selected_agents}
        
        return MARLAction(
            selected_agents=selected_agents,
            coordination_strategy=coord_strategy,
            priority_order=priority_order,
            time_allocation=time_allocation,
            quality_thresholds=quality_thresholds
        )
    
    def _simulate_agent_execution(self, action: MARLAction) -> Dict[str, Any]:
        """Simulate agent execution and return results"""
        results = {}
        
        for agent_id in action.selected_agents:
            # Simulate execution based on agent characteristics
            base_quality = self.state.agent_trust_scores.get(agent_id, 0.5)
            load_penalty = self.state.agent_load_factors.get(agent_id, 0.0) * 0.2
            
            # Add noise for realism
            quality = max(0.0, min(1.0, base_quality - load_penalty + np.random.normal(0, 0.1)))
            response_time = self.state.agent_response_times.get(agent_id, 0.5) + np.random.uniform(-0.1, 0.1)
            
            results[agent_id] = {
                'quality': quality,
                'response_time': max(0.1, response_time),
                'success': quality > action.quality_thresholds.get(agent_id, 0.8),
                'output_size': np.random.uniform(0.5, 1.0)
            }
        
        return results
    
    def _compute_reward_components(self, execution_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute reward components from execution results"""
        if not execution_results:
            return {
                'accuracy': 0.0, 'efficiency': 0.0,
                'mrr_score': 0.0, 'ndcg_at_5': 0.0, 'avg_response_time': 1.0
            }
        
        # Accuracy: average quality across agents
        accuracy = np.mean([r['quality'] for r in execution_results.values()])
        
        # Efficiency: inverse of average response time
        avg_time = np.mean([r['response_time'] for r in execution_results.values()])
        efficiency = 1.0 / (1.0 + avg_time)
        
        # Academic metrics (simplified computation)
        success_agents = [aid for aid, r in execution_results.items() if r['success']]
        
        # MRR: Mean Reciprocal Rank
        mrr_score = 1.0 / (1.0 + len(execution_results) - len(success_agents)) if success_agents else 0.0
        
        # NDCG@5: Normalized Discounted Cumulative Gain (simplified)
        ndcg_at_5 = accuracy * 0.8 if len(success_agents) > 0 else 0.0
        
        # Average response time (normalized)
        avg_response_time = min(1.0, avg_time)
        
        return {
            'accuracy': accuracy,
            'efficiency': efficiency,
            'mrr_score': mrr_score,
            'ndcg_at_5': ndcg_at_5,
            'avg_response_time': avg_response_time
        }
    
    def _update_agent_popularities(self, action: MARLAction, reward: MARLReward):
        """
        Update agent popularities using academic formula:
        popularity_t = popularity_{t-1} + η × (reward - popularity_{t-1})
        """
        eta = self.learning_rates['popularity_update']
        total_reward = reward.compute_total_reward()
        
        for agent_id in action.selected_agents:
            if agent_id in self.agent_popularity_history:
                data = self.agent_popularity_history[agent_id]
                
                # Academic popularity update formula
                old_popularity = data['popularity']
                new_popularity = old_popularity + eta * (total_reward - old_popularity)
                
                # Ensure popularity stays in [0,1] range
                data['popularity'] = max(0.0, min(1.0, new_popularity))
                
                # Update statistics
                data['total_interactions'] += 1
                data['recent_rewards'].append(total_reward)
                
                # Keep only recent rewards (last 100)
                if len(data['recent_rewards']) > 100:
                    data['recent_rewards'] = data['recent_rewards'][-100:]
                
                if total_reward > 0.5:
                    data['success_count'] += 1
                else:
                    data['failure_count'] += 1
    
    def _update_state(self, execution_results: Dict[str, Any]) -> MARLState:
        """Update state based on execution results"""
        # Create new state with updated metrics
        new_state = copy.deepcopy(self.state)
        
        # Update time budget
        new_state.time_budget = max(0.0, new_state.time_budget - 0.1)
        
        # Update system load based on agent usage
        active_agents = len(execution_results)
        load_increase = active_agents * 0.05
        new_state.system_load = min(1.0, new_state.system_load + load_increase)
        
        # Update agent load factors
        for agent_id in execution_results.keys():
            current_load = new_state.agent_load_factors.get(agent_id, 0.0)
            new_state.agent_load_factors[agent_id] = min(1.0, current_load + 0.1)
        
        # Update academic metrics based on results
        for agent_id, result in execution_results.items():
            if result['success']:
                new_state.mrr_scores[agent_id] = min(1.0, new_state.mrr_scores[agent_id] + 0.05)
                new_state.ndcg_scores[agent_id] = min(1.0, new_state.ndcg_scores[agent_id] + 0.03)
            else:
                new_state.mrr_scores[agent_id] = max(0.0, new_state.mrr_scores[agent_id] - 0.02)
                new_state.ndcg_scores[agent_id] = max(0.0, new_state.ndcg_scores[agent_id] - 0.01)
        
        return new_state
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if time budget exhausted
        if self.state.time_budget <= 0.0:
            return True
        
        # Terminate if system overloaded
        if self.state.system_load >= 0.95:
            return True
        
        # Terminate if quality requirement met
        avg_quality = np.mean(list(self.state.ndcg_scores.values()))
        if avg_quality >= self.state.quality_requirement:
            return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'episode_count': self.current_episode,
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'agent_popularities': {aid: data['popularity'] 
                                 for aid, data in self.agent_popularity_history.items()},
            'agent_success_rates': {
                aid: data['success_count'] / max(1, data['total_interactions'])
                for aid, data in self.agent_popularity_history.items()
            },
            'recent_performance': {
                aid: np.mean(data['recent_rewards']) if data['recent_rewards'] else 0.0
                for aid, data in self.agent_popularity_history.items()
            }
        }

def create_marl_environment(config: Optional[Dict[str, Any]] = None) -> MARLEnvironment:
    """
    Factory function to create MARL environment
    
    Args:
        config: Environment configuration
        
    Returns:
        Configured MARLEnvironment instance
    """
    return MARLEnvironment(config)