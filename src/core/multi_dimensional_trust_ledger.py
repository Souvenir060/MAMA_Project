# MAMA_exp/core/multi_dimensional_trust_ledger.py

"""
Multi-Dimensional Trustworthiness Ledger

Core module implementing a dynamic, traceable reputation system for agents.
Quantifies performance across reliability, competence, fairness, security, and transparency
dimensions using distributed ledger technology principles.
"""

import json
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

def sigmoid(x):
    """标准的Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

# Handle milestone_connector import gracefully
try:
    from .milestone_connector import write_to_milestone, read_from_milestone
except ImportError:
    # Fallback mock functions for testing
    def write_to_milestone(entity_type: str, entity_id: str, entity_data: Dict[str, Any]) -> bool:
        return True
    
    def read_from_milestone(entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        return None

logger = logging.getLogger(__name__)

class TrustDimension(Enum):
    """Five dimensions of agent trustworthiness"""
    RELIABILITY = "reliability"
    COMPETENCE = "competence" 
    FAIRNESS = "fairness"
    SECURITY = "security"
    TRANSPARENCY = "transparency"

@dataclass
class TrustRecord:
    """Individual trust evaluation record"""
    agent_id: str
    timestamp: datetime
    dimension: TrustDimension
    score: float
    evidence: Dict[str, Any]
    evaluator: str
    transaction_hash: str

@dataclass
class DimensionMetrics:
    """Metrics for a specific trust dimension"""
    current_score: float
    historical_average: float
    trend: str  # "improving", "stable", "declining"
    last_updated: datetime
    evaluation_count: int
    confidence_level: float

class MultiDimensionalTrustLedger:
    """
    Manages multi-dimensional trust evaluation and storage using distributed ledger principles.
    
    Implements immutable trust records with comprehensive dimension-based scoring
    and historical analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the trust ledger"""
        self.trust_records: List[TrustRecord] = []
        self.dimension_weights = {
            TrustDimension.RELIABILITY: 0.25,
            TrustDimension.COMPETENCE: 0.25,
            TrustDimension.FAIRNESS: 0.20,
            TrustDimension.SECURITY: 0.15,
            TrustDimension.TRANSPARENCY: 0.15
        }
        self.score_decay_factor = 0.95  # Decay factor for old scores
        self.confidence_threshold = 0.7
        
    def record_trust_evaluation(self, agent_id: str, dimension: TrustDimension,
                              score: float, evidence: Dict[str, Any],
                              evaluator: str = "system") -> str:
        """
        Record a new trust evaluation in the ledger
        
        Args:
            agent_id: Agent being evaluated
            dimension: Trust dimension being assessed
            score: Score for this dimension (0.0-1.0)
            evidence: Supporting evidence for the evaluation
            evaluator: Entity performing the evaluation
            
        Returns:
            Transaction hash for the recorded evaluation
        """
        try:
            # Validate input
            if not 0.0 <= score <= 1.0:
                raise ValueError("Trust score must be between 0.0 and 1.0")
            
            # Create transaction hash
            timestamp = datetime.now()
            hash_input = f"{agent_id}:{dimension.value}:{score}:{timestamp.isoformat()}:{evaluator}"
            transaction_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
            # Create trust record
            record = TrustRecord(
                agent_id=agent_id,
                timestamp=timestamp,
                dimension=dimension,
                score=score,
                evidence=evidence,
                evaluator=evaluator,
                transaction_hash=transaction_hash
            )
            
            # Add to local ledger
            self.trust_records.append(record)
            
            # Persist to Milestone data space
            self._persist_to_milestone(record)
            
            logger.info(f"Recorded trust evaluation: {agent_id} - {dimension.value} = {score:.3f}")
            return transaction_hash
            
        except Exception as e:
            logger.error(f"Failed to record trust evaluation: {e}")
            raise
    
    def evaluate_reliability(self, agent_id: str, task_outcomes: List[Dict[str, Any]]) -> float:
        """
        Evaluate agent reliability based on task completion consistency
        
        Args:
            agent_id: Agent to evaluate
            task_outcomes: List of task completion records
            
        Returns:
            Reliability score (0.0-1.0)
        """
        try:
            if not task_outcomes:
                return 0.5  # Neutral score for no data
            
            successful_tasks = sum(1 for outcome in task_outcomes if outcome.get('success', False))
            total_tasks = len(task_outcomes)
            
            # Basic reliability calculation
            base_reliability = successful_tasks / total_tasks
            
            # Adjust for consistency patterns
            if len(task_outcomes) >= 5:
                recent_tasks = task_outcomes[-5:]
                recent_success_rate = sum(1 for outcome in recent_tasks if outcome.get('success', False)) / len(recent_tasks)
                # Weight recent performance more heavily
                reliability_score = 0.7 * base_reliability + 0.3 * recent_success_rate
            else:
                reliability_score = base_reliability
            
            # Record the evaluation
            evidence = {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": base_reliability,
                "evaluation_method": "task_completion_analysis"
            }
            
            self.record_trust_evaluation(
                agent_id, TrustDimension.RELIABILITY, reliability_score, evidence
            )
            
            return reliability_score
            
        except Exception as e:
            logger.error(f"Error evaluating reliability for {agent_id}: {e}")
            return 0.5
    
    def evaluate_competence(self, agent_id: str, system_reward: float, 
                          task_context: Optional[Dict[str, Any]] = None) -> float:
        """
        (最终修复版) 实现基于系统奖励、贡献度评估和平滑学习的综合能力评估。
        
        Args:
            agent_id (str): 被评估的智能体ID。
            system_reward (float): 本次交互后，整个MAMA系统获得的总奖励r。
            task_context (Optional[Dict[str, Any]]): 任务上下文，用于专长匹配。
            
        Returns:
            float: 更新后的能力分数。
        """
        LEARNING_RATE = 0.1  # 学习率
        REWARD_SCALING_FACTOR = 0.05  # 奖励缩放系数k，用于调整sigmoid函数的敏感度

        try:
            # 1. 将系统奖励r归一化到[0, 1]区间，作为本次任务的全局表现分
            # 一个正奖励会得到 > 0.5的分数，负奖励则 < 0.5
            task_performance_score = sigmoid(REWARD_SCALING_FACTOR * system_reward)
            
            # 2. 获取该智能体当前的能力分数 (旧能力)
            try:
                old_competence = self.get_dimension_metrics(agent_id, TrustDimension.COMPETENCE).current_score
                if old_competence is None or old_competence == 0.0:
                    old_competence = 0.5  # 如果没有历史分数，则从0.5开始
            except Exception:
                old_competence = 0.5

            # 3. "贡献度评估"
            agent_specialty = self._get_agent_specialty(agent_id)
            task_priority = None
            if task_context and 'preferences' in task_context:
                task_priority = task_context['preferences'].get('priority')
            
            effective_performance = task_performance_score
            if task_priority and task_priority != agent_specialty:
                # 如果专长不匹配，我们不使用本次表现来更新它的能力，而是让它维持原水平
                effective_performance = old_competence
                logger.debug(f"Agent '{agent_id}' 专长不符 (任务: {task_priority}), 其能力评估将维持原分数。")

            # 4. "平滑学习"
            new_competence = (1 - LEARNING_RATE) * old_competence + LEARNING_RATE * effective_performance

            # 确保分数在有效范围内
            new_competence = max(0.0, min(1.0, new_competence))
            
            # 5. 持久化新的能力分数
            evidence = {
                "system_reward": system_reward,
                "task_performance_score": task_performance_score,
                "old_competence": old_competence,
                "effective_performance": effective_performance,
                "new_competence": new_competence,
                "learning_rate": LEARNING_RATE,
                "reward_scaling_factor": REWARD_SCALING_FACTOR,
                "agent_specialty": agent_specialty,
                "task_priority": task_priority,
                "evaluation_method": "reward_driven_competence_evaluation"
            }
            
            self.record_trust_evaluation(
                agent_id=agent_id,
                dimension=TrustDimension.COMPETENCE,
                score=new_competence,
                evidence=evidence
            )
            
            logger.info(f"🎯 奖励驱动能力更新 - Agent '{agent_id}': {old_competence:.4f} → {new_competence:.4f} "
                       f"(系统奖励: {system_reward:.4f}, 表现分: {task_performance_score:.4f})")
            
            return new_competence
            
        except Exception as e:
            logger.error(f"Error in reward-driven competence evaluation for {agent_id}: {e}")
            return 0.5
    
    def _get_agent_specialty(self, agent_id: str) -> str:
        """获取智能体的专长领域"""
        specialty_mapping = {
            'safety_assessment_agent': 'safety',
            'economic_agent': 'cost', 
            'weather_agent': 'safety',  # 天气也与安全相关
            'flight_info_agent': 'time',  # 航班信息与时间相关
            'integration_agent': 'comfort'  # 集成智能体处理舒适度和综合评估
        }
        return specialty_mapping.get(agent_id, 'general')
    
    def evaluate_fairness(self, agent_id: str, decision_data: List[Dict[str, Any]]) -> float:
        """
        Evaluate agent fairness by detecting systematic biases in decisions
        
        Args:
            agent_id: Agent to evaluate
            decision_data: List of agent decisions with outcomes
            
        Returns:
            Fairness score (0.0-1.0)
        """
        try:
            if not decision_data or len(decision_data) < 10:
                return 0.7  # Neutral-positive score for insufficient data
            
            fairness_score = 1.0  # Start with perfect fairness
            bias_penalties = []
            
            # Check for airline bias
            if self._detect_airline_bias(decision_data):
                bias_penalties.append(0.2)
            
            # Check for price range bias
            if self._detect_price_bias(decision_data):
                bias_penalties.append(0.15)
            
            # Check for time preference bias
            if self._detect_time_bias(decision_data):
                bias_penalties.append(0.1)
            
            # Apply penalties
            total_penalty = sum(bias_penalties)
            fairness_score = max(0.0, fairness_score - total_penalty)
            
            # Record the evaluation
            evidence = {
                "decisions_analyzed": len(decision_data),
                "bias_penalties": bias_penalties,
                "total_penalty": total_penalty,
                "evaluation_method": "statistical_bias_detection"
            }
            
            self.record_trust_evaluation(
                agent_id, TrustDimension.FAIRNESS, fairness_score, evidence
            )
            
            return fairness_score
            
        except Exception as e:
            logger.error(f"Error evaluating fairness for {agent_id}: {e}")
            return 0.7
    
    def evaluate_security(self, agent_id: str, security_events: List[Dict[str, Any]]) -> float:
        """
        Evaluate agent security based on vulnerability and attack resistance
        
        Args:
            agent_id: Agent to evaluate
            security_events: List of security-related events
            
        Returns:
            Security score (0.0-1.0)
        """
        try:
            if not security_events:
                return 0.8  # High baseline security score
            
            security_score = 1.0
            
            # Analyze security events
            attack_attempts = sum(1 for event in security_events if event.get('type') == 'attack_attempt')
            successful_attacks = sum(1 for event in security_events if event.get('type') == 'successful_attack')
            security_violations = sum(1 for event in security_events if event.get('type') == 'security_violation')
            
            # Calculate penalties
            if attack_attempts > 0:
                defense_rate = 1.0 - (successful_attacks / attack_attempts)
                security_score *= defense_rate
            
            # Penalty for security violations
            violation_penalty = min(0.5, security_violations * 0.1)
            security_score -= violation_penalty
            
            # Ensure minimum score
            security_score = max(0.0, security_score)
            
            # Record the evaluation
            evidence = {
                "total_events": len(security_events),
                "attack_attempts": attack_attempts,
                "successful_attacks": successful_attacks,
                "security_violations": security_violations,
                "defense_rate": 1.0 - (successful_attacks / attack_attempts) if attack_attempts > 0 else 1.0,
                "evaluation_method": "security_event_analysis"
            }
            
            self.record_trust_evaluation(
                agent_id, TrustDimension.SECURITY, security_score, evidence
            )
            
            return security_score
            
        except Exception as e:
            logger.error(f"Error evaluating security for {agent_id}: {e}")
            return 0.8
    
    def evaluate_transparency(self, agent_id: str, explanation_data: List[Dict[str, Any]]) -> float:
        """
        Evaluate agent transparency based on explanation quality and completeness
        
        Args:
            agent_id: Agent to evaluate
            explanation_data: List of agent explanations and their completeness
            
        Returns:
            Transparency score (0.0-1.0)
        """
        try:
            if not explanation_data:
                return 0.6  # Moderate score for no explanation data
            
            total_explanations = len(explanation_data)
            complete_explanations = 0
            explanation_quality_sum = 0.0
            
            for explanation in explanation_data:
                # Check explanation completeness
                if explanation.get('has_reasoning', False) and explanation.get('has_evidence', False):
                    complete_explanations += 1
                
                # Sum explanation quality scores
                quality_score = explanation.get('quality_score', 0.5)
                explanation_quality_sum += quality_score
            
            # Calculate transparency components
            completeness_rate = complete_explanations / total_explanations
            average_quality = explanation_quality_sum / total_explanations
            
            # Weighted combination (60% quality, 40% completeness)
            transparency_score = 0.6 * average_quality + 0.4 * completeness_rate
            
            # Record the evaluation
            evidence = {
                "total_explanations": total_explanations,
                "complete_explanations": complete_explanations,
                "completeness_rate": completeness_rate,
                "average_quality": average_quality,
                "evaluation_method": "explanation_analysis"
            }
            
            self.record_trust_evaluation(
                agent_id, TrustDimension.TRANSPARENCY, transparency_score, evidence
            )
            
            return transparency_score
            
        except Exception as e:
            logger.error(f"Error evaluating transparency for {agent_id}: {e}")
            return 0.6
    
    def calculate_overall_trust_score(self, agent_id: str, 
                                    custom_weights: Optional[Dict[TrustDimension, float]] = None) -> Dict[str, Any]:
        """
        Calculate overall trust score combining all dimensions
        
        Args:
            agent_id: Agent to evaluate
            custom_weights: Optional custom dimension weights
            
        Returns:
            Comprehensive trust evaluation results
        """
        try:
            weights = custom_weights or self.dimension_weights
            dimension_scores = {}
            dimension_metrics_objects = {}
            
            # Get latest scores for each dimension
            for dimension in TrustDimension:
                metrics = self.get_dimension_metrics(agent_id, dimension)
                dimension_scores[dimension] = metrics.current_score
                dimension_metrics_objects[dimension] = metrics
            
            # Calculate weighted overall score
            overall_score = sum(
                weights[dimension] * score 
                for dimension, score in dimension_scores.items()
            )
            
            # Calculate confidence level from metrics objects
            confidence_values = [metrics.confidence_level for metrics in dimension_metrics_objects.values()]
            avg_confidence = np.mean(confidence_values) if confidence_values else 0.0
            
            # Determine trust level
            if overall_score >= 0.8 and avg_confidence >= self.confidence_threshold:
                trust_level = "high"
            elif overall_score >= 0.6 and avg_confidence >= 0.5:
                trust_level = "medium" 
            else:
                trust_level = "low"
            
            # Convert metrics objects to dict for serialization
            dimension_metrics_dict = {}
            for dimension, metrics in dimension_metrics_objects.items():
                dimension_metrics_dict[dimension.value] = asdict(metrics)
            
            return {
                "agent_id": agent_id,
                "overall_score": round(overall_score, 3),
                "trust_level": trust_level,
                "confidence_level": round(avg_confidence, 3),
                "dimension_scores": {dim.value: round(score, 3) for dim, score in dimension_scores.items()},
                "dimension_metrics": dimension_metrics_dict,
                "weights_used": {dim.value: weight for dim, weight in weights.items()},
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall trust score for {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "overall_score": 0.5,
                "trust_level": "unknown",
                "error": str(e)
            }
    
    def get_dimension_metrics(self, agent_id: str, dimension: TrustDimension) -> DimensionMetrics:
        """
        Get comprehensive metrics for a specific trust dimension
        
        Args:
            agent_id: Agent to analyze
            dimension: Trust dimension to analyze
            
        Returns:
            Dimension metrics object
        """
        try:
            # Get all records for this agent and dimension
            agent_records = [
                record for record in self.trust_records
                if record.agent_id == agent_id and record.dimension == dimension
            ]
            
            if not agent_records:
                return DimensionMetrics(
                    current_score=0.5,
                    historical_average=0.5,
                    trend="stable",
                    last_updated=datetime.now(),
                    evaluation_count=0,
                    confidence_level=0.0
                )
            
            # Sort by timestamp
            agent_records.sort(key=lambda x: x.timestamp)
            
            # Calculate current score (most recent)
            current_score = agent_records[-1].score
            
            # Calculate historical average with decay
            weighted_scores = []
            weights = []
            current_time = datetime.now()
            
            for record in agent_records:
                age_days = (current_time - record.timestamp).days
                decay_weight = self.score_decay_factor ** age_days
                weighted_scores.append(record.score * decay_weight)
                weights.append(decay_weight)
            
            historical_average = sum(weighted_scores) / sum(weights) if weights else current_score
            
            # Determine trend
            if len(agent_records) >= 3:
                recent_scores = [record.score for record in agent_records[-3:]]
                if recent_scores[-1] > recent_scores[0] + 0.05:
                    trend = "improving"
                elif recent_scores[-1] < recent_scores[0] - 0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Calculate confidence based on number of evaluations and recency
            evaluation_count = len(agent_records)
            recency_factor = min(1.0, 30 / max(1, (current_time - agent_records[-1].timestamp).days))
            confidence_level = min(1.0, (evaluation_count / 10) * recency_factor)
            
            return DimensionMetrics(
                current_score=current_score,
                historical_average=historical_average,
                trend=trend,
                last_updated=agent_records[-1].timestamp,
                evaluation_count=evaluation_count,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error getting dimension metrics for {agent_id} - {dimension.value}: {e}")
            return DimensionMetrics(
                current_score=0.5,
                historical_average=0.5,
                trend="unknown",
                last_updated=datetime.now(),
                evaluation_count=0,
                confidence_level=0.0
            )
    
    def get_trust_history(self, agent_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get trust evaluation history for an agent
        
        Args:
            agent_id: Agent to analyze
            days: Number of days of history to retrieve
            
        Returns:
            List of historical trust evaluations
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            relevant_records = [
                record for record in self.trust_records
                if record.agent_id == agent_id and record.timestamp >= cutoff_date
            ]
            
            # Sort by timestamp
            relevant_records.sort(key=lambda x: x.timestamp)
            
            # Convert to serializable format
            history = []
            for record in relevant_records:
                history.append({
                    "timestamp": record.timestamp.isoformat(),
                    "dimension": record.dimension.value,
                    "score": record.score,
                    "evidence": record.evidence,
                    "evaluator": record.evaluator,
                    "transaction_hash": record.transaction_hash
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting trust history for {agent_id}: {e}")
            return []
    
    def _persist_to_milestone(self, record: TrustRecord) -> bool:
        """
        Persist trust record to Milestone data space
        
        Args:
            record: Trust record to persist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create NGSI-LD entity
            entity_id = f"urn:ngsi-ld:TrustRecord:{record.transaction_hash}"
            entity_data = {
                "@context": [
                    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
                ],
                "id": entity_id,
                "type": "TrustRecord",
                "agentId": {
                    "type": "Property",
                    "value": record.agent_id
                },
                "dimension": {
                    "type": "Property", 
                    "value": record.dimension.value
                },
                "score": {
                    "type": "Property",
                    "value": record.score
                },
                "evidence": {
                    "type": "Property",
                    "value": json.dumps(record.evidence) if isinstance(record.evidence, dict) else str(record.evidence)
                },
                "evaluator": {
                    "type": "Property",
                    "value": record.evaluator
                },
                "timestamp": {
                    "type": "Property",
                    "value": record.timestamp.isoformat()
                },
                "transactionHash": {
                    "type": "Property",
                    "value": record.transaction_hash
                }
            }
            
            # Write to Milestone
            success = write_to_milestone("TrustRecord", entity_id, entity_data)
            
            if success:
                logger.debug(f"Persisted trust record {record.transaction_hash} to Milestone")
            else:
                logger.warning(f"Failed to persist trust record {record.transaction_hash}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error persisting trust record to Milestone: {e}")
            return False
    
    def _detect_airline_bias(self, decisions: List[Dict[str, Any]]) -> bool:
        """Detect systematic bias towards specific airlines"""
        try:
            airline_recommendations = {}
            for decision in decisions:
                airline = decision.get('airline', 'unknown')
                recommended = decision.get('recommended', False)
                
                if airline not in airline_recommendations:
                    airline_recommendations[airline] = {'recommended': 0, 'total': 0}
                
                airline_recommendations[airline]['total'] += 1
                if recommended:
                    airline_recommendations[airline]['recommended'] += 1
            
            # Check if any airline has a very high or very low recommendation rate
            # compared to others (strong bias indicator)
            rates = []
            for airline, stats in airline_recommendations.items():
                if stats['total'] >= 2:  # At least 2 decisions
                    recommendation_rate = stats['recommended'] / stats['total']
                    rates.append(recommendation_rate)
            
            if len(rates) >= 2:
                # If there's a large difference in recommendation rates, it indicates bias
                max_rate = max(rates)
                min_rate = min(rates)
                # Bias detected if difference is >0.6 (e.g., one airline 100% recommended, another 0%)
                return (max_rate - min_rate) > 0.6
            
            return False
            
        except Exception:
            return False
    
    def _detect_price_bias(self, decisions: List[Dict[str, Any]]) -> bool:
        """Detect systematic bias towards specific price ranges"""
        try:
            prices = [d.get('price', 0) for d in decisions if d.get('price')]
            if len(prices) < 10:
                return False
            
            # Get recommendations with prices
            recommended_prices = [d.get('price', 0) for d in decisions 
                                if d.get('recommended') and d.get('price')]
            not_recommended_prices = [d.get('price', 0) for d in decisions 
                                    if not d.get('recommended') and d.get('price')]
            
            if len(recommended_prices) == 0 or len(not_recommended_prices) == 0:
                return False
            
            # Calculate average prices for recommended vs not recommended
            avg_recommended_price = np.mean(recommended_prices)
            avg_not_recommended_price = np.mean(not_recommended_prices)
            overall_avg_price = np.mean(prices)
            
            # Check for significant bias (>50% difference from overall average)
            recommended_deviation = abs(avg_recommended_price - overall_avg_price) / overall_avg_price
            not_recommended_deviation = abs(avg_not_recommended_price - overall_avg_price) / overall_avg_price
            
            # Bias detected if there's a strong preference for high or low prices
            return recommended_deviation > 0.5 or not_recommended_deviation > 0.5
            
        except Exception:
            return False
    
    def _detect_time_bias(self, decisions: List[Dict[str, Any]]) -> bool:
        """Detect systematic bias towards specific departure times"""
        try:
            morning_recommended = sum(1 for d in decisions 
                                    if d.get('departure_hour', 12) < 12 and d.get('recommended'))
            total_recommended = sum(1 for d in decisions if d.get('recommended'))
            
            if total_recommended > 0:
                morning_rate = morning_recommended / total_recommended
                return morning_rate > 0.8 or morning_rate < 0.2  # Strong bias either way
            
            return False
            
        except Exception:
            return False

    def get_current_trust_score(self, agent_id: str) -> float:
        """
        Get current trust score for an agent (compatibility method)
        
        Args:
            agent_id: Agent to get trust score for
            
        Returns:
            Current overall trust score (0.0-1.0)
        """
        try:
            trust_result = self.calculate_overall_trust_score(agent_id)
            return trust_result.get('overall_score', 0.5)
        except Exception as e:
            logger.warning(f"Failed to get current trust score for {agent_id}: {e}")
            return 0.5
    
    def get_agent_trust_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive trust summary for an agent (compatibility method)
        
        Args:
            agent_id: Agent to get trust summary for
            
        Returns:
            Comprehensive trust summary including all dimensions
        """
        try:
            return self.calculate_overall_trust_score(agent_id)
        except Exception as e:
            logger.error(f"Failed to get agent trust summary for {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "overall_score": 0.5,
                "trust_level": "unknown",
                "error": str(e)
            }

# Global instance for system-wide use
trust_ledger = MultiDimensionalTrustLedger()

def record_agent_trust_evaluation(agent_id: str, dimension: str, score: float, 
                                evidence: Dict[str, Any]) -> str:
    """Convenience function to record trust evaluation"""
    dim = TrustDimension(dimension)
    return trust_ledger.record_trust_evaluation(agent_id, dim, score, evidence)

def get_agent_trust_score(agent_id: str) -> Dict[str, Any]:
    """Convenience function to get agent's overall trust score"""
    return trust_ledger.calculate_overall_trust_score(agent_id)

def evaluate_agent_reliability(agent_id: str, task_outcomes: List[Dict[str, Any]]) -> float:
    """Convenience function to evaluate agent reliability"""
    return trust_ledger.evaluate_reliability(agent_id, task_outcomes)

def evaluate_agent_competence(agent_id: str, performance_metrics: Dict[str, Any]) -> float:
    """Convenience function to evaluate agent competence"""
    return trust_ledger.evaluate_competence(agent_id, performance_metrics) 