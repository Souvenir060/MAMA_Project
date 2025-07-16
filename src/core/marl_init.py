"""
MARL (Multi-Agent Reinforcement Learning) Module

Academic implementation of trust-aware multi-agent reinforcement learning
for dynamic agent selection and coordination in MAMA Flight Assistant system.

This module implements the mathematical formulas from the research paper:

1. Agent Selection Score:
   SelectionScore = α × similarity + β × popularity

2. Reward Function:
   r = γ × accuracy + δ × efficiency

3. Popularity Update:
   popularity_t = popularity_{t-1} + η × (reward - popularity_{t-1})

4. Dynamic Agent Selection with MARL:
   r = α × MRR + β × NDCG@5 - γ × ART

Key Components:
- TrustAwareMARLEngine: Core MARL implementation
- MARLEnvironment: Reinforcement learning environment
- AgentSelectionPolicy: Policy for agent selection
- MARLTrainer: Training pipeline for MARL models
"""

from .environment import MARLEnvironment, MARLState, MARLAction, MARLReward
from .policy import (
    AgentSelectionPolicy, 
    TrustAwarePolicy,
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy
)
from .trainer import MARLTrainer, TrainingConfig
from .trust_aware_marl import TrustAwareMARLEngine
from .metrics import MARLMetrics, PerformanceEvaluator

__all__ = [
    'MARLEnvironment',
    'MARLState', 
    'MARLAction',
    'MARLReward',
    'AgentSelectionPolicy',
    'TrustAwarePolicy',
    'EpsilonGreedyPolicy', 
    'SoftmaxPolicy',
    'ThompsonSamplingPolicy',
    'MARLTrainer',
    'TrainingConfig',
    'TrustAwareMARLEngine',
    'MARLMetrics',
    'PerformanceEvaluator'
]

__version__ = "1.0.0"
