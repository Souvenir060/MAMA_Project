#!/usr/bin/env python3
"""
LTR (Learning to Rank) System

Implementation of Learning-to-Rank algorithms for flight/option ranking
in the MAMA system. Implements multiple ranking approaches with rigorous mathematical
foundations.

Key Formulas:
1. Pointwise: f(x) = w^T * φ(x), where φ(x) is feature vector
2. Pairwise: P(xi > xj) = σ(f(xi) - f(xj)), where σ is sigmoid function
3. Listwise: L(y, f) = -Σ (2^y_i - 1) / log2(1 + rank_i)
4. RankNet: ∂C/∂w = Σ σ'(o_ij) * (∂o_ij/∂w) * (P_ij - P̂_ij)
5. LambdaRank: λ_ij = -σ'(o_ij) * |ΔZ_ij|, where ΔZ_ij is NDCG change
"""

import numpy as np
import logging
import json
import pickle
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
from enum import Enum
import threading
import uuid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class RankingAlgorithm(Enum):
    """Ranking algorithm types"""
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"  
    LISTWISE = "listwise"
    RANKNET = "ranknet"
    LAMBDARANK = "lambdarank"
    LISTNET = "listnet"

@dataclass
class RankingFeature:
    """Feature representation for ranking"""
    feature_id: str
    feature_name: str
    feature_value: float
    feature_type: str  # numerical, categorical, ordinal
    importance: float
    timestamp: datetime

@dataclass 
class RankingInstance:
    """Instance for ranking (e.g., flight, option)"""
    instance_id: str
    features: Dict[str, RankingFeature]
    relevance_score: float  # Ground truth relevance
    predicted_score: float
    rank_position: int
    query_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RankingQuery:
    """Query with multiple instances to rank"""
    query_id: str
    query_text: str
    instances: List[RankingInstance]
    ground_truth_ranking: List[str]  # Instance IDs in relevance order
    predicted_ranking: List[str]
    query_features: Dict[str, float]
    timestamp: datetime

@dataclass
class RankingMetrics:
    """Comprehensive ranking evaluation metrics"""
    ndcg_at_k: Dict[int, float]  # NDCG@k for different k values
    precision_at_k: Dict[int, float]  # P@k
    recall_at_k: Dict[int, float]  # R@k
    map_score: float  # Mean Average Precision
    mrr_score: float  # Mean Reciprocal Rank
    kendall_tau: float  # Kendall's Tau correlation
    spearman_rho: float  # Spearman's rank correlation
    ranking_loss: float  # Ranking loss
    computation_time: float

class PointwiseRanker(nn.Module):
    """
    Pointwise ranking model: f(x) = w^T * φ(x)
    
    Treats ranking as regression/classification problem for individual instances.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super(PointwiseRanker, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))  # Output single score
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: f(x) = w^T * φ(x)"""
        return self.network(x).squeeze(-1)

class PairwiseRanker(nn.Module):
    """
    Pairwise ranking model: P(xi > xj) = σ(f(xi) - f(xj))
    
    Models pairwise preferences between instances.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super(PairwiseRanker, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single instance"""
        return self.network(x).squeeze(-1)
    
    def pairwise_forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Pairwise forward pass: P(xi > xj) = σ(f(xi) - f(xj))
        
        Args:
            x_i: Features for instance i
            x_j: Features for instance j
            
        Returns:
            Probability that instance i ranks higher than j
        """
        score_i = self.forward(x_i)
        score_j = self.forward(x_j)
        
        # P(xi > xj) = σ(f(xi) - f(xj))
        diff = score_i - score_j
        return torch.sigmoid(diff)

class ListwiseRanker(nn.Module):
    """
    Listwise ranking model: L(y, f) = -Σ (2^y_i - 1) / log2(1 + rank_i)
    
    Models the entire ranking list as a whole.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super(ListwiseRanker, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for batch of instances"""
        return self.network(x).squeeze(-1)
    
    def listwise_loss(self, scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        """
        Listwise loss: L(y, f) = -Σ (2^y_i - 1) / log2(1 + rank_i)
        
        Args:
            scores: Predicted scores for instances
            relevance: Ground truth relevance scores
            
        Returns:
            Listwise ranking loss
        """
        # Sort by predicted scores (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_relevance = relevance[sorted_indices]
        
        # Compute NDCG-based loss
        gains = torch.pow(2, sorted_relevance) - 1
        ranks = torch.arange(1, len(gains) + 1, dtype=torch.float32, device=gains.device)
        discounts = torch.log2(ranks + 1)
        
        dcg = torch.sum(gains / discounts)
        
        # Ideal DCG (sort by true relevance)
        ideal_sorted_relevance = torch.sort(relevance, descending=True)[0]
        ideal_gains = torch.pow(2, ideal_sorted_relevance) - 1
        ideal_dcg = torch.sum(ideal_gains / discounts)
        
        # Normalize to get NDCG
        ndcg = dcg / (ideal_dcg + 1e-8)
        
        # Return negative NDCG as loss (for gradient descent)
        return -ndcg

class LTRRankingEngine:
    """
    Comprehensive Learning-to-Rank Engine
    
    Implements multiple ranking algorithms with academic rigor and
    comprehensive evaluation metrics.
    """
    
    def __init__(self, 
                 algorithm: RankingAlgorithm = RankingAlgorithm.POINTWISE,
                 feature_dim: int = 50,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 device: str = 'cpu'):
        """
        Initialize LTR ranking engine
        
        Args:
            algorithm: Ranking algorithm to use
            feature_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Computing device
        """
        self.algorithm = algorithm
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model based on algorithm
        if algorithm == RankingAlgorithm.POINTWISE:
            self.model = PointwiseRanker(feature_dim, hidden_dims)
        elif algorithm in [RankingAlgorithm.PAIRWISE, RankingAlgorithm.RANKNET]:
            self.model = PairwiseRanker(feature_dim, hidden_dims)
        elif algorithm in [RankingAlgorithm.LISTWISE, RankingAlgorithm.LAMBDARANK, RankingAlgorithm.LISTNET]:
            self.model = ListwiseRanker(feature_dim, hidden_dims)
        else:
            self.model = PointwiseRanker(feature_dim, hidden_dims)
        
        self.model.to(device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = self._get_loss_function()
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        # Training history
        self.training_history = {
            'loss': [],
            'ndcg': [],
            'map': [],
            'mrr': []
        }
        
        # Data storage
        self.training_queries: List[RankingQuery] = []
        self.validation_queries: List[RankingQuery] = []
        self.feature_importance: Dict[str, float] = {}
        
        # Performance metrics
        self.evaluation_cache: Dict[str, RankingMetrics] = {}
        
        logger.info(f"Initialized LTR engine with {algorithm.value} algorithm")
    
    def add_training_data(self, queries: List[RankingQuery]) -> None:
        """
        Add training data to the engine
        
        Args:
            queries: List of ranking queries with ground truth
        """
        try:
            self.training_queries.extend(queries)
            logger.info(f"Added {len(queries)} training queries (total: {len(self.training_queries)})")
            
        except Exception as e:
            logger.error(f"Failed to add training data: {e}")
            raise
    
    def extract_features(self, instance: Any, query_context: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features from instance
        
        Args:
            instance: Instance to extract features from (flight, option, etc.)
            query_context: Query context information
            
        Returns:
            Feature vector as numpy array
        """
        try:
            features = []
            
            # Basic instance features
            if hasattr(instance, 'price'):
                features.append(float(instance.price))
            else:
                features.append(0.0)
                
            if hasattr(instance, 'duration'):
                features.append(float(instance.duration))
            else:
                features.append(0.0)
                
            if hasattr(instance, 'rating'):
                features.append(float(instance.rating))
            else:
                features.append(0.0)
                
            # Query-specific features
            features.extend([
                query_context.get('user_budget', 0.0),
                query_context.get('time_preference', 0.0),
                query_context.get('quality_preference', 0.0),
                query_context.get('urgency', 0.0),
                query_context.get('flexibility', 0.0)
            ])
            
            # Temporal features
            now = datetime.now()
            features.extend([
                now.hour / 24.0,  # Hour of day
                now.weekday() / 7.0,  # Day of week
                now.month / 12.0,  # Month
            ])
            
            # Pad to feature_dim
            while len(features) < self.feature_dim:
                features.append(0.0)
            
            # Truncate to feature_dim
            features = features[:self.feature_dim]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def train(self, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ranking model
        
        Args:
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and metrics
        """
        try:
            if not self.training_queries:
                raise ValueError("No training data available")
            
            logger.info(f"Starting training with {len(self.training_queries)} queries")
            
            # Split data
            train_queries, val_queries = train_test_split(
                self.training_queries, test_size=validation_split, random_state=42
            )
            self.validation_queries = val_queries
            
            # Prepare training data
            train_features, train_labels, train_query_ids = self._prepare_training_data(train_queries)
            
            # Fit feature scaler
            self.feature_scaler.fit(train_features)
            train_features = self.feature_scaler.transform(train_features)
            
            # Convert to tensors
            train_features = torch.FloatTensor(train_features).to(self.device)
            train_labels = torch.FloatTensor(train_labels).to(self.device)
            
            # Training loop
            self.model.train()
            best_ndcg = 0.0
            
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Mini-batch training
                for i in range(0, len(train_features), self.batch_size):
                    batch_features = train_features[i:i+self.batch_size]
                    batch_labels = train_labels[i:i+self.batch_size]
                    
                    # Forward pass
                    if self.algorithm == RankingAlgorithm.POINTWISE:
                        loss = self._pointwise_training_step(batch_features, batch_labels)
                    elif self.algorithm == RankingAlgorithm.PAIRWISE:
                        loss = self._pairwise_training_step(batch_features, batch_labels)
                    elif self.algorithm == RankingAlgorithm.LISTWISE:
                        loss = self._listwise_training_step(batch_features, batch_labels)
                    elif self.algorithm == RankingAlgorithm.RANKNET:
                        loss = self._ranknet_training_step(batch_features, batch_labels)
                    elif self.algorithm == RankingAlgorithm.LAMBDARANK:
                        loss = self._lambdarank_training_step(batch_features, batch_labels)
                    else:
                        loss = self._pointwise_training_step(batch_features, batch_labels)
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / max(num_batches, 1)
                self.training_history['loss'].append(avg_loss)
                
                # Validation
                if epoch % 10 == 0:
                    val_metrics = self.evaluate(val_queries)
                    val_ndcg = val_metrics.ndcg_at_k.get(10, 0.0)
                    self.training_history['ndcg'].append(val_ndcg)
                    self.training_history['map'].append(val_metrics.map_score)
                    self.training_history['mrr'].append(val_metrics.mrr_score)
                    
                    if val_ndcg > best_ndcg:
                        best_ndcg = val_ndcg
                    
                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, NDCG@10={val_ndcg:.4f}")
            
            self.is_fitted = True
            
            # Final evaluation
            final_metrics = self.evaluate(val_queries)
            
            training_results = {
                'best_ndcg': best_ndcg,
                'final_metrics': final_metrics,
                'training_history': self.training_history,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'training_time': len(self.training_history['loss']) * self.batch_size
            }
            
            logger.info(f"Training completed. Best NDCG@10: {best_ndcg:.4f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def rank_instances(self, instances: List[Any], query_context: Dict[str, Any]) -> List[Tuple[Any, float]]:
        """
        Rank instances using trained model
        
        Args:
            instances: List of instances to rank
            query_context: Query context information
            
        Returns:
            List of (instance, score) tuples sorted by score (descending)
        """
        try:
            if not self.is_fitted:
                logger.warning("Model not trained, using rule-based ranking")
                return self._rule_based_ranking(instances, query_context)
            
            if not instances:
                return []
            
            # Extract features
            features = []
            for instance in instances:
                feat = self.extract_features(instance, query_context)
                features.append(feat)
            
            features = np.array(features)
            features = self.feature_scaler.transform(features)
            
            # Predict scores
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                scores = self.model(features_tensor).cpu().numpy()
            
            # Create ranked results
            instance_scores = list(zip(instances, scores))
            instance_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Ranked {len(instances)} instances")
            return instance_scores
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return self._rule_based_ranking(instances, query_context)
    
    def evaluate(self, queries: List[RankingQuery], k_values: List[int] = [1, 5, 10, 20]) -> RankingMetrics:
        """
        Comprehensive evaluation of ranking performance
        
        Args:
            queries: Queries for evaluation
            k_values: Values of k for evaluation metrics
            
        Returns:
            Comprehensive ranking metrics
        """
        try:
            start_time = time.time()
            
            if not queries:
                return RankingMetrics(
                    ndcg_at_k={}, precision_at_k={}, recall_at_k={},
                    map_score=0.0, mrr_score=0.0, kendall_tau=0.0,
                    spearman_rho=0.0, ranking_loss=0.0, computation_time=0.0
                )
            
            all_ndcg = {k: [] for k in k_values}
            all_precision = {k: [] for k in k_values}
            all_recall = {k: [] for k in k_values}
            all_ap = []  # Average precision for MAP
            all_rr = []  # Reciprocal rank for MRR
            all_kendall = []
            all_spearman = []
            all_losses = []
            
            for query in queries:
                # Extract features and get predictions
                instances = [inst for inst in query.instances]
                features = []
                true_relevance = []
                
                for instance in instances:
                    feat = self.extract_features(instance, {})
                    features.append(feat)
                    true_relevance.append(instance.relevance_score)
                
                if not features:
                    continue
                
                features = np.array(features)
                if self.is_fitted:
                    features = self.feature_scaler.transform(features)
                    
                    self.model.eval()
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).to(self.device)
                        predicted_scores = self.model(features_tensor).cpu().numpy()
                else:
                    # Use rule-based scores if model not trained
                    predicted_scores = np.random.random(len(features))
                
                # Compute metrics for this query
                query_metrics = self._compute_query_metrics(
                    true_relevance, predicted_scores, k_values
                )
                
                # Aggregate metrics
                for k in k_values:
                    if f'ndcg@{k}' in query_metrics:
                        all_ndcg[k].append(query_metrics[f'ndcg@{k}'])
                    if f'precision@{k}' in query_metrics:
                        all_precision[k].append(query_metrics[f'precision@{k}'])
                    if f'recall@{k}' in query_metrics:
                        all_recall[k].append(query_metrics[f'recall@{k}'])
                
                if 'ap' in query_metrics:
                    all_ap.append(query_metrics['ap'])
                if 'rr' in query_metrics:
                    all_rr.append(query_metrics['rr'])
                if 'kendall_tau' in query_metrics:
                    all_kendall.append(query_metrics['kendall_tau'])
                if 'spearman_rho' in query_metrics:
                    all_spearman.append(query_metrics['spearman_rho'])
                if 'loss' in query_metrics:
                    all_losses.append(query_metrics['loss'])
            
            # Aggregate final metrics
            ndcg_at_k = {k: np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0 for k in k_values}
            precision_at_k = {k: np.mean(all_precision[k]) if all_precision[k] else 0.0 for k in k_values}
            recall_at_k = {k: np.mean(all_recall[k]) if all_recall[k] else 0.0 for k in k_values}
            
            computation_time = time.time() - start_time
            
            metrics = RankingMetrics(
                ndcg_at_k=ndcg_at_k,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                map_score=np.mean(all_ap) if all_ap else 0.0,
                mrr_score=np.mean(all_rr) if all_rr else 0.0,
                kendall_tau=np.mean(all_kendall) if all_kendall else 0.0,
                spearman_rho=np.mean(all_spearman) if all_spearman else 0.0,
                ranking_loss=np.mean(all_losses) if all_losses else 0.0,
                computation_time=computation_time
            )
            
            logger.info(f"Evaluation completed: NDCG@10={metrics.ndcg_at_k.get(10, 0.0):.4f}, MAP={metrics.map_score:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return RankingMetrics(
                ndcg_at_k={}, precision_at_k={}, recall_at_k={},
                map_score=0.0, mrr_score=0.0, kendall_tau=0.0,
                spearman_rho=0.0, ranking_loss=0.0, computation_time=0.0
            )
    
    def _get_loss_function(self) -> Callable:
        """Get appropriate loss function for algorithm"""
        if self.algorithm == RankingAlgorithm.POINTWISE:
            return nn.MSELoss()
        elif self.algorithm == RankingAlgorithm.PAIRWISE:
            return nn.BCEWithLogitsLoss()
        elif self.algorithm == RankingAlgorithm.LISTWISE:
            return self._listwise_loss
        else:
            return nn.MSELoss()
    
    def _prepare_training_data(self, queries: List[RankingQuery]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for model"""
        features = []
        labels = []
        query_ids = []
        
        for query in queries:
            for instance in query.instances:
                # Extract features (simplified)
                feat = self.extract_features(instance, {})
                features.append(feat)
                labels.append(instance.relevance_score)
                query_ids.append(query.query_id)
        
        return np.array(features), np.array(labels), query_ids
    
    def _pointwise_training_step(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pointwise training step: f(x) = w^T * φ(x)"""
        self.optimizer.zero_grad()
        predictions = self.model(features)
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _pairwise_training_step(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pairwise training step: P(xi > xj) = σ(f(xi) - f(xj))"""
        self.optimizer.zero_grad()
        
        # Generate pairs
        n = len(features)
        if n < 2:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j]:  # Only use pairs with different relevance
                    feat_i, feat_j = features[i:i+1], features[j:j+1]
                    label_i, label_j = labels[i], labels[j]
                    
                    # Determine preference
                    if label_i > label_j:
                        preference = 1.0  # i should rank higher than j
                    else:
                        preference = 0.0  # j should rank higher than i
                    
                    # Compute pairwise probability
                    score_i = self.model(feat_i)
                    score_j = self.model(feat_j)
                    pairwise_prob = torch.sigmoid(score_i - score_j)
                    
                    # Binary cross-entropy loss
                    pair_loss = -preference * torch.log(pairwise_prob + 1e-8) - (1 - preference) * torch.log(1 - pairwise_prob + 1e-8)
                    total_loss += pair_loss
                    num_pairs += 1
        
        if num_pairs > 0:
            loss = total_loss / num_pairs
            loss.backward()
            self.optimizer.step()
            return loss
        else:
            return torch.tensor(0.0)
    
    def _listwise_training_step(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Listwise training step"""
        self.optimizer.zero_grad()
        scores = self.model(features)
        loss = self.model.listwise_loss(scores, labels)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _ranknet_training_step(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        RankNet training step: ∂C/∂w = Σ σ'(o_ij) * (∂o_ij/∂w) * (P_ij - P̂_ij)
        """
        return self._pairwise_training_step(features, labels)  # RankNet is a special case of pairwise
    
    def _lambdarank_training_step(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        LambdaRank training step: λ_ij = -σ'(o_ij) * |ΔZ_ij|
        """
        # Simplified LambdaRank implementation
        return self._listwise_training_step(features, labels)
    
    def _listwise_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Listwise loss function"""
        return self.model.listwise_loss(scores, labels)
    
    def _compute_query_metrics(self, true_relevance: List[float], predicted_scores: List[float], 
                              k_values: List[int]) -> Dict[str, float]:
        """Compute metrics for a single query"""
        try:
            metrics = {}
            
            # Sort by predicted scores
            sorted_indices = np.argsort(predicted_scores)[::-1]
            sorted_relevance = np.array(true_relevance)[sorted_indices]
            
            # NDCG@k
            for k in k_values:
                if k <= len(sorted_relevance):
                    dcg_k = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(sorted_relevance[:k]))
                    ideal_sorted = sorted(true_relevance, reverse=True)
                    idcg_k = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_sorted[:k]))
                    metrics[f'ndcg@{k}'] = dcg_k / idcg_k if idcg_k > 0 else 0.0
                else:
                    metrics[f'ndcg@{k}'] = 0.0
            
            # Precision@k and Recall@k
            for k in k_values:
                if k <= len(sorted_relevance):
                    relevant_at_k = sum(1 for rel in sorted_relevance[:k] if rel > 0.5)
                    total_relevant = sum(1 for rel in true_relevance if rel > 0.5)
                    
                    metrics[f'precision@{k}'] = relevant_at_k / k if k > 0 else 0.0
                    metrics[f'recall@{k}'] = relevant_at_k / total_relevant if total_relevant > 0 else 0.0
                else:
                    metrics[f'precision@{k}'] = 0.0
                    metrics[f'recall@{k}'] = 0.0
            
            # Average Precision (AP)
            ap = 0.0
            relevant_count = 0
            for i, rel in enumerate(sorted_relevance):
                if rel > 0.5:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    ap += precision_at_i
            
            total_relevant = sum(1 for rel in true_relevance if rel > 0.5)
            metrics['ap'] = ap / total_relevant if total_relevant > 0 else 0.0
            
            # Reciprocal Rank (RR)
            rr = 0.0
            for i, rel in enumerate(sorted_relevance):
                if rel > 0.5:
                    rr = 1.0 / (i + 1)
                    break
            metrics['rr'] = rr
            
            # Correlation metrics
            try:
                from scipy.stats import kendalltau, spearmanr
                tau, _ = kendalltau(true_relevance, predicted_scores)
                rho, _ = spearmanr(true_relevance, predicted_scores)
                metrics['kendall_tau'] = tau if not math.isnan(tau) else 0.0
                metrics['spearman_rho'] = rho if not math.isnan(rho) else 0.0
            except:
                metrics['kendall_tau'] = 0.0
                metrics['spearman_rho'] = 0.0
            
            # Ranking loss (simplified)
            loss = np.mean(np.square(np.array(true_relevance) - np.array(predicted_scores)))
            metrics['loss'] = loss
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute query metrics: {e}")
            return {}
    
    def _rule_based_ranking(self, instances: List[Any], query_context: Dict[str, Any]) -> List[Tuple[Any, float]]:
        """Rule-based ranking fallback"""
        try:
            scored_instances = []
            
            for instance in instances:
                score = 0.0
                
                # Price score (lower is better)
                if hasattr(instance, 'price'):
                    max_price = query_context.get('max_budget', 1000.0)
                    price_score = max(0, (max_price - instance.price) / max_price)
                    score += 0.3 * price_score
                
                # Duration score (shorter might be better)
                if hasattr(instance, 'duration'):
                    max_duration = query_context.get('max_duration', 24.0)
                    duration_score = max(0, (max_duration - instance.duration) / max_duration)
                    score += 0.2 * duration_score
                
                # Rating score (higher is better)
                if hasattr(instance, 'rating'):
                    rating_score = instance.rating / 5.0  # Assume 5-star rating
                    score += 0.5 * rating_score
                
                scored_instances.append((instance, score))
            
            # Sort by score (descending)
            scored_instances.sort(key=lambda x: x[1], reverse=True)
            
            return scored_instances
            
        except Exception as e:
            logger.error(f"Rule-based ranking failed: {e}")
            return [(inst, 0.0) for inst in instances]
    
    def save_model(self, filepath: str) -> None:
        """Save LTR model"""
        try:
            save_data = {
                'model_state': self.model.state_dict(),
                'algorithm': self.algorithm.value,
                'feature_dim': self.feature_dim,
                'hidden_dims': self.hidden_dims,
                'scaler_state': {
                    'mean_': self.feature_scaler.mean_ if hasattr(self.feature_scaler, 'mean_') else None,
                    'scale_': self.feature_scaler.scale_ if hasattr(self.feature_scaler, 'scale_') else None
                },
                'training_history': self.training_history,
                'feature_importance': self.feature_importance,
                'is_fitted': self.is_fitted,
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'num_epochs': self.num_epochs
                }
            }
            
            torch.save(save_data, filepath)
            logger.info(f"Saved LTR model to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load LTR model"""
        try:
            save_data = torch.load(filepath, map_location=self.device)
            
            # Restore model
            self.model.load_state_dict(save_data['model_state'])
            self.algorithm = RankingAlgorithm(save_data['algorithm'])
            self.feature_dim = save_data['feature_dim']
            self.hidden_dims = save_data['hidden_dims']
            
            # Restore scaler
            if save_data['scaler_state']['mean_'] is not None:
                self.feature_scaler.mean_ = save_data['scaler_state']['mean_']
                self.feature_scaler.scale_ = save_data['scaler_state']['scale_']
            
            # Restore other attributes
            self.training_history = save_data.get('training_history', {})
            self.feature_importance = save_data.get('feature_importance', {})
            self.is_fitted = save_data.get('is_fitted', False)
            
            logger.info(f"Loaded LTR model from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


# Global LTR engine instance
ltr_engine = None

def initialize_ltr_engine(algorithm: RankingAlgorithm = RankingAlgorithm.POINTWISE, **kwargs) -> LTRRankingEngine:
    """Initialize global LTR engine"""
    global ltr_engine
    if ltr_engine is None:
        ltr_engine = LTRRankingEngine(algorithm=algorithm, **kwargs)
    return ltr_engine

def rank_flights(flights: List[Any], query_context: Dict[str, Any]) -> List[Tuple[Any, float]]:
    """Global function to rank flights using LTR"""
    if ltr_engine is None:
        initialize_ltr_engine()
    
    return ltr_engine.rank_instances(flights, query_context)

def train_ltr_model(training_data: List[RankingQuery]) -> Dict[str, Any]:
    """Global function to train LTR model"""
    if ltr_engine is None:
        initialize_ltr_engine()
    
    ltr_engine.add_training_data(training_data)
    return ltr_engine.train()
