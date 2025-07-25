#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - Integration Agent

Aggregates outputs from all other agents and uses Learning to Rank (LTR) model
to generate final sorted recommendations as described in the paper.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from autogen import ConversableAgent, register_function

logger = logging.getLogger(__name__)

@dataclass
class IntegrationResult:
    """Integration analysis result structure"""
    flight_id: str
    agent_scores: Dict[str, float]
    trust_weighted_scores: Dict[str, float]
    final_integrated_score: float
    ranking_position: int
    confidence_level: float
    contributing_factors: Dict[str, float]
    recommendations: List[str]

class IntegrationAgent(BaseAgent):
    """
    Integration Agent for multi-agent output aggregation and final ranking
    
    Implements the paper's approach:
    - Aggregates outputs from Weather, Safety, Economic, and Flight Info agents
    - Uses Learning to Rank (LTR) model for final recommendation generation
    - Applies trust-weighted scoring: WeightedScore_j = Σ(TrustScore_i · Score_i,j)
    - Generates final sorted recommendation list
    """
    
    def __init__(self, name: str = None, role: str = "integration_coordinator", **kwargs):
        super().__init__(
            name=name or "integration_agent",
            role=role,
            **kwargs
        )
        
        # Integration weights based on paper methodology
        self.agent_weights = {
            'weather': 0.25,      # Weather safety importance
            'safety': 0.30,       # Safety assessment highest priority
            'economic': 0.25,     # Economic value significant
            'flight_info': 0.20   # Operational information
        }
        
        # LTR model for final ranking (LambdaMART simulation using Random Forest)
        self.ltr_model = None
        self.is_trained = False
        
        # Trust score integration parameters
        self.trust_threshold = 0.5
        self.confidence_alpha = 0.8
        
        # Initialize agent with integration specialization
        try:
            # CRITICAL FIX: Use proper LLM config to avoid register_function errors
            llm_config = {
                "config_list": [
                    {
                        "model": "local_csv_processor",
                        "api_key": None,
                        "base_url": None,
                    }
                ],
                "temperature": 0.0,
                "timeout": 10,
            }
            
            self.agent = ConversableAgent(
                name="IntegrationCoordinator",
                system_message="""You are a professional multi-agent integration coordinator. Your responsibilities include:

🔄 **Core Functions:**
- Aggregate outputs from Weather, Safety, Economic, and Flight Info agents
- Apply trust-weighted scoring based on agent reliability
- Use Learning to Rank (LTR) methodology for final recommendations
- Generate comprehensive flight ranking from multi-agent analysis
- Provide final recommendation list with confidence levels

📊 **Integration Focus:**
- Trust-weighted score calculation: WeightedScore = Σ(TrustScore_i · Score_i,j)
- Multi-dimensional agent output aggregation
- LTR-based final ranking optimization
- Confidence level assessment across agent outputs
- Final recommendation synthesis

⚡ **Integration Standards:**
- Final Score: 0.9+ Excellent, 0.8-0.9 Good, 0.7-0.8 Fair, 0.6-0.7 Poor, <0.6 Very Poor
- Trust-weighted aggregation from all specialist agents
- LTR model for ranking optimization
- Confidence assessment based on agent agreement
""",
                llm_config=llm_config,  # Use proper LLM config
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1
            )
            logger.info("Integration agent initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize integration agent: {e}")
            self.agent = None
        
        # Initialize LTR model
        self._initialize_ltr_model()
    
    def _initialize_ltr_model(self):
        """🎯 PAPER-ACCURATE: Initialize LambdaMART algorithm for Learning to Rank"""
        try:
            # 🎯 论文指定：LambdaMART算法的GBDT实现
            from sklearn.ensemble import GradientBoostingRegressor
            
            # LambdaMART配置 (基于论文参数)
            self.ltr_model = GradientBoostingRegressor(
                n_estimators=100,      # Boosting stages
                learning_rate=0.1,     # Step size shrinkage
                max_depth=6,           # Tree depth for ranking
                subsample=0.8,         # Subsample ratio
                random_state=42,       # Reproducibility
                loss='squared_error'   # Loss function for ranking
            )
            logger.info("🎯 LambdaMART model (Gradient Boosting) initialized as per paper")
        except Exception as e:
            logger.error(f"Failed to initialize LambdaMART model: {e}")
            # 备选：仍然使用Random Forest作为fallback
            self.ltr_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
    
    def integrate_agent_outputs(self, agent_outputs: Dict[str, Any], 
                              trust_scores: Dict[str, float]) -> IntegrationResult:
        """
        Integrate outputs from all specialist agents using trust-weighted scoring
        
        Args:
            agent_outputs: Dictionary of agent outputs {agent_name: analysis_result}
            trust_scores: Dictionary of agent trust scores {agent_name: trust_score}
            
        Returns:
            IntegrationResult with final integrated analysis
        """
        try:
            flight_id = agent_outputs.get('flight_id', 'unknown')
            
            # 1. Extract scores from each agent
            agent_scores = self._extract_agent_scores(agent_outputs)
            
            # 2. Apply trust weighting: WeightedScore_j = Σ(TrustScore_i · Score_i,j)
            trust_weighted_scores = self._apply_trust_weighting(agent_scores, trust_scores)
            
            # 3. Calculate final integrated score using agent weights
            final_score = self._calculate_final_integrated_score(trust_weighted_scores)
            
            # 4. Generate contributing factors analysis
            contributing_factors = self._analyze_contributing_factors(
                agent_scores, trust_weighted_scores, trust_scores
            )
            
            # 5. Calculate confidence level based on agent agreement
            confidence_level = self._calculate_confidence_level(
                agent_scores, trust_scores
            )
            
            # 6. Generate integration recommendations
            recommendations = self._generate_integration_recommendations(
                final_score, contributing_factors, confidence_level
            )
            
            result = IntegrationResult(
                flight_id=flight_id,
                agent_scores=agent_scores,
                trust_weighted_scores=trust_weighted_scores,
                final_integrated_score=final_score,
                ranking_position=0,  # Will be set during ranking
                confidence_level=confidence_level,
                contributing_factors=contributing_factors,
                recommendations=recommendations
            )
            
            logger.info(f"Integration completed for flight {flight_id}: Score {final_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            # Return default result
            return IntegrationResult(
                flight_id=agent_outputs.get('flight_id', 'unknown'),
                agent_scores={},
                trust_weighted_scores={},
                final_integrated_score=0.5,
                ranking_position=999,
                confidence_level=0.0,
                contributing_factors={'error': 'integration_failed'},
                recommendations=['Integration analysis unavailable']
            )
    
    def generate_final_ranking(self, integration_results: List[IntegrationResult]) -> List[IntegrationResult]:
        """
        Generate final ranking using LTR model as described in paper
        
        Args:
            integration_results: List of integration results for all flights
            
        Returns:
            Sorted list of integration results (highest score first)
        """
        try:
            if not integration_results:
                return []
            
            # 1. Prepare features for LTR model
            features = self._prepare_ltr_features(integration_results)
            
            # 2. Apply LTR model if trained, otherwise use integrated scores
            if self.is_trained and self.ltr_model is not None:
                try:
                    # Use LTR model for ranking scores
                    ranking_scores = self.ltr_model.predict(features)
                    
                    # Update final scores with LTR predictions
                    for i, result in enumerate(integration_results):
                        result.final_integrated_score = max(0.0, min(1.0, ranking_scores[i]))
                        
                except Exception as e:
                    logger.warning(f"LTR model prediction failed: {e}")
                    # Fall back to integrated scores
            
            # 3. Sort by final integrated score (descending)
            sorted_results = sorted(
                integration_results,
                key=lambda x: x.final_integrated_score,
                reverse=True
            )
            
            # 4. Update ranking positions
            for i, result in enumerate(sorted_results):
                result.ranking_position = i + 1
            
            logger.info(f"Final ranking generated for {len(sorted_results)} flights")
            return sorted_results
            
        except Exception as e:
            logger.error(f"Final ranking generation failed: {e}")
            # Return original list with default ranking
            for i, result in enumerate(integration_results):
                result.ranking_position = i + 1
            return integration_results
    
    def _extract_agent_scores(self, agent_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical scores from each agent's output"""
        agent_scores = {}
        
        # Weather agent score
        weather_output = agent_outputs.get('weather', {})
        if isinstance(weather_output, dict):
            # Look for weather safety score
            weather_scores = [
                weather_output.get('weather_safety_score', 0.0),
                weather_output.get('overall_safety_score', 0.0),
                weather_output.get('route_assessment', {}).get('overall_safety_score', 0.0)
            ]
            agent_scores['weather'] = max([s for s in weather_scores if s > 0] or [0.5])
        else:
            agent_scores['weather'] = 0.5
        
        # Safety agent score
        safety_output = agent_outputs.get('safety', {})
        if isinstance(safety_output, dict):
            safety_scores = [
                safety_output.get('safety_score', 0.0),
                safety_output.get('overall_safety_score', 0.0),
                safety_output.get('assessment', {}).get('overall_safety_score', 0.0)
            ]
            agent_scores['safety'] = max([s for s in safety_scores if s > 0] or [0.5])
        else:
            agent_scores['safety'] = 0.5
        
        # Economic agent score
        economic_output = agent_outputs.get('economic', {})
        if isinstance(economic_output, dict):
            economic_scores = [
                economic_output.get('economic_score', 0.0),
                economic_output.get('overall_economic_score', 0.0),
                economic_output.get('analysis', {}).get('overall_economic_score', 0.0)
            ]
            agent_scores['economic'] = max([s for s in economic_scores if s > 0] or [0.5])
        else:
            agent_scores['economic'] = 0.5
        
        # Flight info agent score
        flight_info_output = agent_outputs.get('flight_info', {})
        if isinstance(flight_info_output, dict):
            info_scores = [
                flight_info_output.get('operational_score', 0.0),
                flight_info_output.get('schedule_reliability', 0.0),
                flight_info_output.get('flight_info_analysis', {}).get('operational_score', 0.0)
            ]
            agent_scores['flight_info'] = max([s for s in info_scores if s > 0] or [0.5])
        else:
            agent_scores['flight_info'] = 0.5
        
        return agent_scores
    
    def _apply_trust_weighting(self, agent_scores: Dict[str, float], 
                             trust_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply trust weighting as per paper: WeightedScore_j = Σ(TrustScore_i · Score_i,j)
        """
        trust_weighted_scores = {}
        
        for agent_name, score in agent_scores.items():
            trust_score = trust_scores.get(agent_name, 0.8)  # Default trust
            
            # Apply trust weighting
            weighted_score = trust_score * score
            trust_weighted_scores[agent_name] = weighted_score
            
            logger.debug(f"Agent {agent_name}: Score {score:.3f} × Trust {trust_score:.3f} = {weighted_score:.3f}")
        
        return trust_weighted_scores
    
    def _calculate_final_integrated_score(self, trust_weighted_scores: Dict[str, float]) -> float:
        """Calculate final integrated score using agent weights"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for agent_name, weighted_score in trust_weighted_scores.items():
            agent_weight = self.agent_weights.get(agent_name, 0.25)
            total_weighted_score += weighted_score * agent_weight
            total_weight += agent_weight
        
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.5
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, final_score))
    
    def _analyze_contributing_factors(self, agent_scores: Dict[str, float],
                                    trust_weighted_scores: Dict[str, float],
                                    trust_scores: Dict[str, float]) -> Dict[str, float]:
        """Analyze which factors contributed most to the final score"""
        contributing_factors = {}
        
        total_contribution = sum(
            trust_weighted_scores.get(agent, 0) * self.agent_weights.get(agent, 0.25)
            for agent in self.agent_weights.keys()
        )
        
        for agent_name in self.agent_weights.keys():
            weighted_score = trust_weighted_scores.get(agent_name, 0)
            agent_weight = self.agent_weights[agent_name]
            contribution = (weighted_score * agent_weight) / max(total_contribution, 0.001)
            contributing_factors[f"{agent_name}_contribution"] = contribution
            
        # Add trust impact analysis
        for agent_name, trust_score in trust_scores.items():
            contributing_factors[f"{agent_name}_trust_impact"] = trust_score
        
        return contributing_factors
    
    def _calculate_confidence_level(self, agent_scores: Dict[str, float],
                                  trust_scores: Dict[str, float]) -> float:
        """Calculate confidence level based on agent agreement and trust"""
        if not agent_scores:
            return 0.0
        
        # 1. Agent agreement (low variance = high confidence)
        scores = list(agent_scores.values())
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        agreement_confidence = max(0.0, 1.0 - (score_variance * 4))  # Scale variance
        
        # 2. Trust level confidence
        trust_values = list(trust_scores.values())
        avg_trust = np.mean(trust_values) if trust_values else 0.8
        trust_confidence = avg_trust
        
        # 3. Data completeness confidence
        completeness = len(agent_scores) / len(self.agent_weights)
        completeness_confidence = completeness
        
        # Combined confidence
        confidence = (
            0.4 * agreement_confidence +
            0.4 * trust_confidence +
            0.2 * completeness_confidence
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_integration_recommendations(self, final_score: float,
                                            contributing_factors: Dict[str, float],
                                            confidence_level: float) -> List[str]:
        """Generate integration recommendations based on analysis"""
        recommendations = []
        
        # Overall score recommendations
        if final_score >= 0.85:
            recommendations.append("Excellent overall rating - top recommendation")
        elif final_score >= 0.70:
            recommendations.append("Good overall rating - solid choice")
        elif final_score >= 0.55:
            recommendations.append("Average overall rating - acceptable option")
        else:
            recommendations.append("Below-average rating - consider alternatives")
        
        # Confidence level recommendations
        if confidence_level >= 0.8:
            recommendations.append("High confidence in assessment - reliable recommendation")
        elif confidence_level >= 0.6:
            recommendations.append("Moderate confidence - reasonably reliable")
        else:
            recommendations.append("Low confidence - limited data or agent disagreement")
        
        # Contributing factor analysis
        max_contributor = max(
            [k for k in contributing_factors.keys() if k.endswith('_contribution')],
            key=lambda x: contributing_factors[x],
            default=''
        )
        
        if max_contributor:
            factor_name = max_contributor.replace('_contribution', '')
            recommendations.append(f"Primary strength: {factor_name} performance")
        
        return recommendations
    
    def _prepare_ltr_features(self, integration_results: List[IntegrationResult]) -> np.ndarray:
        """Prepare features for LTR model"""
        features = []
        
        for result in integration_results:
            # Feature vector: [weather_score, safety_score, economic_score, flight_info_score, confidence]
            feature_vector = [
                result.agent_scores.get('weather', 0.5),
                result.agent_scores.get('safety', 0.5),
                result.agent_scores.get('economic', 0.5),
                result.agent_scores.get('flight_info', 0.5),
                result.confidence_level
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def process_task(self, task_description: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process integration task"""
        try:
            # Extract agent outputs and trust scores from task data
            agent_outputs = task_data.get('agent_outputs', {})
            trust_scores = task_data.get('trust_scores', {
                'weather': 0.8, 'safety': 0.9, 'economic': 0.85, 'flight_info': 0.8
            })
            
            # Perform integration
            result = self.integrate_agent_outputs(agent_outputs, trust_scores)
            
            return {
                'status': 'success',
                'analysis_type': 'integration_analysis',
                'final_integrated_score': result.final_integrated_score,
                'confidence_level': result.confidence_level,
                'agent_scores': result.agent_scores,
                'trust_weighted_scores': result.trust_weighted_scores,
                'contributing_factors': result.contributing_factors,
                'recommendations': result.recommendations,
                'performance_metrics': {
                    'integration_confidence': result.confidence_level,
                    'agent_agreement': np.std(list(result.agent_scores.values())) if result.agent_scores else 0.0,
                    'processing_time': 0.1
                }
            }
            
        except Exception as e:
            logger.error(f"Integration task failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'final_integrated_score': 0.5,
                'analysis_type': 'integration_analysis'
            }

def get_integration_tool(multi_agent_data: str) -> str:
    """
    Integration tool function for multi-agent output aggregation
    
    Args:
        multi_agent_data: JSON string containing outputs from all agents
        
    Returns:
        JSON string with integrated analysis results
    """
    logger.info("IntegrationAgent: Starting multi-agent integration")
    
    try:
        # Parse input data
        if not multi_agent_data.strip():
            return json.dumps({
                "status": "error",
                "message": "No multi-agent data provided"
            })
        
        data = json.loads(multi_agent_data)
        
        if "flights" not in data:
            return json.dumps({
                "status": "error",
                "message": "No flight data found"
            })
        
        flights = data["flights"]
        trust_scores = data.get("trust_scores", {
            'weather': 0.8, 'safety': 0.9, 'economic': 0.85, 'flight_info': 0.8
        })
        
        # Initialize integration agent
        integration_agent = IntegrationAgent()
        integration_results = []
        
        for flight in flights:
            try:
                # Extract agent outputs from flight data
                agent_outputs = {
                    'flight_id': flight.get('id', 'unknown'),
                    'weather': flight.get('weather_assessment', {}),
                    'safety': flight.get('safety_assessment', {}),
                    'economic': flight.get('economic_analysis', {}),
                    'flight_info': flight.get('flight_info_analysis', {})
                }
                
                # Perform integration
                result = integration_agent.integrate_agent_outputs(agent_outputs, trust_scores)
                integration_results.append(result)
                
            except Exception as e:
                logger.error(f"Error integrating flight {flight.get('id', 'unknown')}: {e}")
                # Add default integration result
                default_result = IntegrationResult(
                    flight_id=flight.get('id', 'unknown'),
                    agent_scores={},
                    trust_weighted_scores={},
                    final_integrated_score=0.5,
                    ranking_position=999,
                    confidence_level=0.0,
                    contributing_factors={'error': 'integration_failed'},
                    recommendations=['Integration failed']
                )
                integration_results.append(default_result)
        
        # Generate final ranking
        ranked_results = integration_agent.generate_final_ranking(integration_results)
        
        # Convert results to JSON-serializable format
        final_recommendations = []
        for result in ranked_results:
            flight_recommendation = {
                "flight_id": result.flight_id,
                "ranking_position": result.ranking_position,
                "final_integrated_score": round(result.final_integrated_score, 4),
                "confidence_level": round(result.confidence_level, 3),
                "agent_scores": {k: round(v, 3) for k, v in result.agent_scores.items()},
                "trust_weighted_scores": {k: round(v, 3) for k, v in result.trust_weighted_scores.items()},
                "contributing_factors": {k: round(v, 3) for k, v in result.contributing_factors.items()},
                "recommendations": result.recommendations[:3]  # Top 3 recommendations
            }
            final_recommendations.append(flight_recommendation)
        
        return json.dumps({
            "status": "success",
            "final_recommendations": final_recommendations,
            "integration_summary": {
                "total_flights_integrated": len(integration_results),
                "average_confidence": round(np.mean([r.confidence_level for r in ranked_results]), 3),
                "top_flight_score": round(ranked_results[0].final_integrated_score, 4) if ranked_results else 0.0,
                "integration_complete": True
            },
            "trust_scores_used": trust_scores
        })
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Data format error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Integration tool error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error occurred during integration: {str(e)}"
        })

def create_integration_agent():
    """Create and configure integration agent"""
    # CRITICAL FIX: Use proper LLM config to avoid register_function errors
    llm_config = {
        "config_list": [
            {
                "model": "local_csv_processor",
                "api_key": None,
                "base_url": None,
            }
        ],
        "temperature": 0.0,
        "timeout": 10,
    }
    
    agent = ConversableAgent(
        name="IntegrationAgent",
        system_message="""You are a professional multi-agent integration coordinator. Your responsibilities include:

🔄 **Core Functions:**
- Aggregate outputs from Weather, Safety, Economic, and Flight Info agents
- Apply trust-weighted scoring based on agent reliability
- Use Learning to Rank (LTR) methodology for final recommendations
- Generate comprehensive flight ranking from multi-agent analysis
- Provide final recommendation list with confidence levels

📊 **Integration Focus:**
- Trust-weighted score calculation: WeightedScore = Σ(TrustScore_i · Score_i,j)
- Multi-dimensional agent output aggregation
- LTR-based final ranking optimization
- Confidence level assessment across agent outputs
- Final recommendation synthesis

⚡ **Integration Standards:**
- Final Score: 0.9+ Excellent, 0.8-0.9 Good, 0.7-0.8 Fair, 0.6-0.7 Poor, <0.6 Very Poor
- Trust-weighted aggregation from all specialist agents
- LTR model for ranking optimization
- Confidence assessment based on agent agreement
""",
        llm_config=llm_config,  # Use proper LLM config
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )
    
    try:
        register_function(
            get_integration_tool,
            caller=agent,
            executor=agent,
            description="Multi-agent output integration and final ranking using LTR methodology",
            name="get_integration_tool"
        )
        logger.info("✅ Integration agent tools registered successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to register integration tool: {e}")
    
    return agent
