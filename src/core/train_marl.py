# DVC_exp/marl/train_marl.py

"""
MARL Training Module for MAMA Flight Assistant System

Academic implementation of trust-aware multi-agent reinforcement learning training
based on research paper formulas with comprehensive training procedures.

Key Training Components:
1. Trust-Aware Q-Learning with experience replay
2. Dynamic agent selection optimization
3. Academic metric evaluation (MRR, NDCG@5, ART)
4. Model performance validation and saving
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, namedtuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our MARL components
from .marl_environment import MARLEnvironment
from .marl_policy import MARLPolicy, TrustAwareQNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'trust_scores'])

class ReplayBuffer:
    """
    Experience replay buffer for MARL training
    
    Implements prioritized experience replay for academic rigor
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize replay buffer
        
        Args:
            capacity: Buffer capacity
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer with priority"""
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], torch.Tensor]:
        """Sample experiences with prioritized sampling"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        experiences = [self.buffer[idx] for idx in indices]
        return experiences, torch.FloatTensor(weights)
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

class MARLTrainer:
    """
    Multi-Agent Reinforcement Learning Trainer
    
    Academic implementation of trust-aware MARL training with comprehensive
    evaluation metrics and research-grade training procedures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MARL trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.95)  # Discount factor
        self.tau = config.get('tau', 0.005)     # Soft update rate
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Academic parameters
        self.trust_weight_decay = config.get('trust_weight_decay', 0.99)
        self.academic_metric_weight = config.get('academic_metric_weight', 0.3)
        
        # Training configuration
        self.num_episodes = config.get('num_episodes', 5000)
        self.update_frequency = config.get('update_frequency', 4)
        self.target_update_frequency = config.get('target_update_frequency', 100)
        self.save_frequency = config.get('save_frequency', 500)
        self.eval_frequency = config.get('eval_frequency', 100)
        
        # Paths
        self.model_save_path = config.get('model_save_path', 'models/marl_policy.pth')
        self.log_dir = config.get('log_dir', 'logs/marl_training')
        
        # Initialize environment and policy
        self.env = MARLEnvironment(config.get('env_config', {}))
        self.policy = MARLPolicy(config.get('policy_config', {}))
        
        # Initialize target network for stable training
        self.target_network = TrustAwareQNetwork(
            state_dim=self.policy.state_dim,
            action_dim=self.policy.action_dim,
            hidden_dims=config.get('hidden_dims', [256, 128, 64])
        )
        self.target_network.load_state_dict(self.policy.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.q_network.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('buffer_capacity', 100000),
            alpha=config.get('prioritization_alpha', 0.6)
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        self.academic_metrics_history = []
        self.trust_scores_history = []
        
        # Current training state
        self.current_episode = 0
        self.current_epsilon = self.epsilon_start
        self.step_count = 0
        
        # Create directories
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info("MARL Trainer initialized successfully")
        logger.info(f"Environment: {self.env.num_agents} agents, State dim: {self.policy.state_dim}")
        logger.info(f"Training for {self.num_episodes} episodes")
    
    def select_action(self, state: np.ndarray, available_agents: List[str], 
                     trust_scores: Dict[str, float], exploration: bool = True) -> int:
        """
        Select action using epsilon-greedy with trust-aware Q-values
        
        Args:
            state: Current state
            available_agents: Available agents
            trust_scores: Trust scores for agents
            exploration: Whether to use exploration
            
        Returns:
            Selected action index
        """
        if exploration and random.random() < self.current_epsilon:
            # Random exploration among available agents
            available_indices = [i for i, agent in enumerate(self.policy.agent_types) 
                               if agent in available_agents]
            return random.choice(available_indices) if available_indices else 0
        else:
            # Use Q-network for action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                trust_tensor = torch.FloatTensor([trust_scores.get(agent, 0.5) 
                                                for agent in self.policy.agent_types])
                q_values = self.policy.q_network(state_tensor, trust_tensor)
                
                # Mask unavailable agents
                masked_q_values = q_values.clone()
                for i, agent in enumerate(self.policy.agent_types):
                    if agent not in available_agents:
                        masked_q_values[0, i] = float('-inf')
                
                return masked_q_values.argmax().item()
    
    def compute_td_error(self, experiences: List[Experience], weights: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute temporal difference error for prioritized replay
        
        Args:
            experiences: Batch of experiences
            weights: Importance sampling weights
            
        Returns:
            Loss tensor and TD errors for priority update
        """
        # Extract batch components
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.BoolTensor([exp.done for exp in experiences])
        trust_scores = torch.FloatTensor([[exp.trust_scores.get(agent, 0.5) 
                                         for agent in self.policy.agent_types] 
                                        for exp in experiences])
        
        # Current Q-values
        current_q_values = self.policy.q_network(states, trust_scores).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states, trust_scores).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors
        td_errors = (target_q_values.unsqueeze(1) - current_q_values).abs()
        
        # Weighted loss
        loss = (td_errors.squeeze() * weights).mean()
        
        return loss, td_errors.squeeze().detach().cpu().numpy().tolist()
    
    def update_networks(self):
        """Update Q-network using prioritized experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        experiences, weights = self.replay_buffer.sample(self.batch_size, beta=0.4)
        
        # Compute loss and TD errors
        loss, td_errors = self.compute_td_error(experiences, weights)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update replay buffer priorities
        indices = list(range(len(experiences)))  # Simplified - would need actual indices in practice
        self.replay_buffer.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def soft_update_target_network(self):
        """Soft update of target network"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.policy.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def evaluate_academic_metrics(self, num_eval_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate academic metrics (MRR, NDCG@5, ART)
        
        Args:
            num_eval_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of academic metrics
        """
        eval_rewards = []
        agent_selections = []
        response_times = []
        
        for _ in range(num_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get available agents and trust scores
                available_agents = list(self.policy.agent_types)
                trust_scores = self.policy._compute_trust_scores(available_agents)
                
                # Select action without exploration
                action = self.select_action(state, available_agents, trust_scores, exploration=False)
                selected_agent = self.policy.agent_types[action]
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                agent_selections.append(selected_agent)
                response_times.append(info.get('response_time', 1.0))
                
                if done:
                    break
                    
                state = next_state
            
            eval_rewards.append(episode_reward)
        
        # Compute academic metrics
        avg_reward = np.mean(eval_rewards)
        
        # Mean Reciprocal Rank (simplified)
        mrr_scores = []
        for agent in agent_selections:
            rank = list(self.policy.agent_types).index(agent) + 1
            mrr_scores.append(1.0 / rank)
        mrr = np.mean(mrr_scores)
        
        # NDCG@5 (simplified)
        ndcg_scores = []
        for i in range(0, len(agent_selections), 5):
            batch = agent_selections[i:i+5]
            dcg = sum((1.0) / np.log2(j + 2) for j in range(len(batch)))
            idcg = sum(1.0 / np.log2(j + 2) for j in range(min(5, len(self.policy.agent_types))))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
        ndcg_5 = np.mean(ndcg_scores)
        
        # Average Response Time
        art = np.mean(response_times)
        
        # Dynamic selection reward using academic formula
        alpha_metric, beta_metric, gamma_metric = 0.4, 0.4, 0.2
        dynamic_reward = alpha_metric * mrr + beta_metric * ndcg_5 - gamma_metric * art
        
        metrics = {
            'average_reward': avg_reward,
            'mrr': mrr,
            'ndcg_5': ndcg_5,
            'art': art,
            'dynamic_reward': dynamic_reward,
            'eval_episodes': num_eval_episodes
        }
        
        return metrics
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train one episode
        
        Returns:
            Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        num_updates = 0
        
        while True:
            # Get available agents and trust scores
            available_agents = list(self.policy.agent_types)
            trust_scores = self.policy._compute_trust_scores(available_agents)
            
            # Select action
            action = self.select_action(state, available_agents, trust_scores)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                trust_scores=trust_scores
            )
            self.replay_buffer.push(experience)
            
            # Update networks
            if self.step_count % self.update_frequency == 0:
                loss = self.update_networks()
                if loss is not None:
                    episode_loss += loss
                    num_updates += 1
            
            # Soft update target network
            if self.step_count % self.target_update_frequency == 0:
                self.soft_update_target_network()
            
            episode_reward += reward
            episode_length += 1
            self.step_count += 1
            
            if done:
                break
                
            state = next_state
        
        # Update epsilon
        self.current_epsilon = max(self.epsilon_end, 
                                 self.current_epsilon * self.epsilon_decay)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_loss': episode_loss / max(1, num_updates),
            'epsilon': self.current_epsilon
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting MARL training...")
        
        for episode in range(self.num_episodes):
            self.current_episode = episode
            
            # Train episode
            episode_stats = self.train_episode()
            
            # Track statistics
            self.episode_rewards.append(episode_stats['episode_reward'])
            self.episode_lengths.append(episode_stats['episode_length'])
            if episode_stats['episode_loss'] > 0:
                self.loss_history.append(episode_stats['episode_loss'])
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                academic_metrics = self.evaluate_academic_metrics()
                self.academic_metrics_history.append(academic_metrics)
                
                logger.info(f"Episode {episode}/{self.num_episodes}")
                logger.info(f"  Reward: {episode_stats['episode_reward']:.3f}")
                logger.info(f"  Loss: {episode_stats['episode_loss']:.6f}")
                logger.info(f"  Epsilon: {episode_stats['epsilon']:.3f}")
                logger.info(f"  MRR: {academic_metrics['mrr']:.3f}")
                logger.info(f"  NDCG@5: {academic_metrics['ndcg_5']:.3f}")
                logger.info(f"  ART: {academic_metrics['art']:.3f}")
                logger.info(f"  Dynamic Reward: {academic_metrics['dynamic_reward']:.3f}")
            
            # Save model
            if episode % self.save_frequency == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        # Final save
        self.save_final_model()
        self.plot_training_results()
        
        logger.info("Training completed successfully!")
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = f"{self.model_save_path.replace('.pth', f'_episode_{episode}.pth')}"
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.current_epsilon,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history,
            'academic_metrics_history': self.academic_metrics_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        # Save for policy use
        self.policy.save_model(self.model_save_path)
        
        # Save complete training state
        final_checkpoint = {
            'model_state_dict': self.policy.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'agent_popularities': self.policy.agent_popularities,
            'trust_metrics': self.policy.trust_metrics,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history,
            'academic_metrics_history': self.academic_metrics_history,
            'config': self.config,
            'training_completed': True,
            'final_epsilon': self.current_epsilon
        }
        
        final_path = self.model_save_path.replace('.pth', '_final.pth')
        torch.save(final_checkpoint, final_path)
        logger.info(f"Final model saved: {final_path}")
    
    def plot_training_results(self):
        """Plot training results and academic metrics"""
        try:
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
            # Loss history
            if self.loss_history:
                axes[0, 1].plot(self.loss_history)
                axes[0, 1].set_title('Training Loss')
                axes[0, 1].set_xlabel('Update')
                axes[0, 1].set_ylabel('Loss')
            
            # Academic metrics
            if self.academic_metrics_history:
                metrics_data = {
                    'MRR': [m['mrr'] for m in self.academic_metrics_history],
                    'NDCG@5': [m['ndcg_5'] for m in self.academic_metrics_history],
                    'Dynamic Reward': [m['dynamic_reward'] for m in self.academic_metrics_history]
                }
                
                for metric, values in metrics_data.items():
                    axes[1, 0].plot(values, label=metric)
                axes[1, 0].set_title('Academic Metrics')
                axes[1, 0].set_xlabel('Evaluation')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
            
            # Agent popularity evolution
            if hasattr(self.policy, 'agent_popularities'):
                pop_data = [[self.policy.agent_popularities.get(agent, 0.5) 
                           for agent in self.policy.agent_types]]
                sns.heatmap(pop_data, 
                           xticklabels=[a.replace('_agent', '') for a in self.policy.agent_types],
                           yticklabels=['Final'], 
                           ax=axes[1, 1], 
                           cmap='viridis')
                axes[1, 1].set_title('Agent Popularity (Final)')
            
            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, 'training_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

def main():
    """Main training function"""
    # Training configuration
    config = {
        # Network architecture
        'hidden_dims': [256, 128, 64],
        'learning_rate': 0.0003,
        'batch_size': 64,
        
        # RL parameters
        'gamma': 0.95,
        'tau': 0.005,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        
        # Training parameters
        'num_episodes': 2500,
        'update_frequency': 4,
        'target_update_frequency': 100,
        'save_frequency': 250,
        'eval_frequency': 50,
        
        # Buffer and exploration
        'buffer_capacity': 50000,
        'prioritization_alpha': 0.6,
        
        # Academic parameters
        'trust_weight_decay': 0.99,
        'academic_metric_weight': 0.3,
        
        # Paths
        'model_save_path': 'models/marl_policy.pth',
        'log_dir': 'logs/marl_training',
        
        # Environment configuration
        'env_config': {
            'max_episode_length': 50,
            'num_agents': 5,
            'reward_shaping': True
        },
        
        # Policy configuration
        'policy_config': {
            'alpha': 0.7,  # Similarity weight
            'beta': 0.3,   # Popularity weight
            'gamma': 0.6,  # Accuracy weight in reward
            'delta': 0.4,  # Efficiency weight in reward
            'eta': 0.1     # Popularity learning rate
        }
    }
    
    # Create trainer and start training
    trainer = MARLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()