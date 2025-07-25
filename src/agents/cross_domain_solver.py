"""
Cross-Domain Problem Solving System

This module implements intelligent cross-domain problem solving with:
1. Multi-agent output integration
2. Conflict detection and resolution
3. Trust-weighted decision making
4. Standard conflict resolution analysis
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from enum import Enum
import requests
import openai
from config import OPENAI_API_KEY

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of conflicts between agent recommendations"""
    PRICE_DISCREPANCY = "price_discrepancy"
    SAFETY_CONCERN = "safety_concern"
    WEATHER_IMPACT = "weather_impact"
    TIMING_CONFLICT = "timing_conflict"
    ROUTE_DISAGREEMENT = "route_disagreement"
    RELIABILITY_ISSUE = "reliability_issue"

class SolutionConfidence(Enum):
    """Confidence levels for cross-domain solutions"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.6 - 0.8
    LOW = "low"        # < 0.6

@dataclass
class AgentRecommendation:
    """Single agent recommendation with metadata"""
    agent_id: str
    recommendation: Dict[str, Any]
    confidence: float
    trust_score: float
    timestamp: str
    domain_expertise: float
    supporting_evidence: List[str]
    
class ConflictDetector:
    """Detects conflicts between agent recommendations"""
    
    def __init__(self):
        self.conflict_thresholds = {
            'price_variance': 0.20,      # 20% price difference
            'safety_score_diff': 0.15,   # 15% safety score difference
            'weather_impact_diff': 0.25, # 25% weather impact difference
            'timing_window': 3600,       # 1 hour timing difference (seconds)
            'route_efficiency_diff': 0.10 # 10% route efficiency difference
        }
        
    def detect_conflicts(self, recommendations: List[AgentRecommendation]) -> List[Dict[str, Any]]:
        """Detect conflicts between agent recommendations"""
        conflicts = []
        
        if len(recommendations) < 2:
            return conflicts
        
        # Price-related conflicts
        price_conflicts = self._detect_price_conflicts(recommendations)
        conflicts.extend(price_conflicts)
        
        # Safety-related conflicts
        safety_conflicts = self._detect_safety_conflicts(recommendations)
        conflicts.extend(safety_conflicts)
        
        # Weather-related conflicts
        weather_conflicts = self._detect_weather_conflicts(recommendations)
        conflicts.extend(weather_conflicts)
        
        # Timing conflicts
        timing_conflicts = self._detect_timing_conflicts(recommendations)
        conflicts.extend(timing_conflicts)
        
        # Route efficiency conflicts
        route_conflicts = self._detect_route_conflicts(recommendations)
        conflicts.extend(route_conflicts)
        
        return conflicts
    
    def _detect_price_conflicts(self, recommendations: List[AgentRecommendation]) -> List[Dict[str, Any]]:
        """Detect price-related conflicts"""
        conflicts = []
        prices = []
        
        for rec in recommendations:
            if 'flight' in rec.recommendation and 'price' in rec.recommendation['flight']:
                prices.append({
                    'agent_id': rec.agent_id,
                    'price': rec.recommendation['flight']['price'],
                    'trust_score': rec.trust_score
                })
        
        if len(prices) < 2:
            return conflicts
        
        # Calculate price variance
        price_values = [p['price'] for p in prices]
        mean_price = np.mean(price_values)
        max_variance = max(abs(p - mean_price) / mean_price for p in price_values)
        
        if max_variance > self.conflict_thresholds['price_variance']:
            conflicts.append({
                'type': ConflictType.PRICE_DISCREPANCY,
                'severity': 'high' if max_variance > 0.4 else 'medium',
                'description': f"Price variance of {max_variance:.1%} detected between agents",
                'affected_agents': [p['agent_id'] for p in prices],
                'price_range': {'min': min(price_values), 'max': max(price_values)},
                'variance': max_variance,
                'evidence': f"Price range: ${min(price_values):.2f} - ${max(price_values):.2f}"
            })
        
        return conflicts
    
    def _detect_safety_conflicts(self, recommendations: List[AgentRecommendation]) -> List[Dict[str, Any]]:
        """Detect safety-related conflicts"""
        conflicts = []
        safety_scores = []
        
        for rec in recommendations:
            if 'safety_score' in rec.recommendation:
                safety_scores.append({
                    'agent_id': rec.agent_id,
                    'score': rec.recommendation['safety_score'],
                    'trust_score': rec.trust_score
                })
        
        if len(safety_scores) < 2:
            return conflicts
        
        # Check for significant safety score differences
        scores = [s['score'] for s in safety_scores]
        max_diff = max(scores) - min(scores)
        
        if max_diff > self.conflict_thresholds['safety_score_diff']:
            conflicts.append({
                'type': ConflictType.SAFETY_CONCERN,
                'severity': 'critical' if max_diff > 0.3 else 'high',
                'description': f"Safety score difference of {max_diff:.3f} detected",
                'affected_agents': [s['agent_id'] for s in safety_scores],
                'score_range': {'min': min(scores), 'max': max(scores)},
                'difference': max_diff,
                'evidence': f"Safety scores range from {min(scores):.3f} to {max(scores):.3f}"
            })
        
        return conflicts
    
    def _detect_weather_conflicts(self, recommendations: List[AgentRecommendation]) -> List[Dict[str, Any]]:
        """Detect weather-related conflicts"""
        conflicts = []
        weather_impacts = []
        
        for rec in recommendations:
            if 'weather_impact' in rec.recommendation:
                weather_impacts.append({
                    'agent_id': rec.agent_id,
                    'impact': rec.recommendation['weather_impact'],
                    'trust_score': rec.trust_score
                })
        
        if len(weather_impacts) < 2:
            return conflicts
        
        # Check for significant weather impact differences
        impacts = [w['impact'] for w in weather_impacts]
        max_diff = max(impacts) - min(impacts)
        
        if max_diff > self.conflict_thresholds['weather_impact_diff']:
            conflicts.append({
                'type': ConflictType.WEATHER_IMPACT,
                'severity': 'medium',
                'description': f"Weather impact difference of {max_diff:.1%} detected",
                'affected_agents': [w['agent_id'] for w in weather_impacts],
                'impact_range': {'min': min(impacts), 'max': max(impacts)},
                'difference': max_diff,
                'evidence': f"Weather impacts range from {min(impacts):.1%} to {max(impacts):.1%}"
            })
        
        return conflicts
    
    def _detect_timing_conflicts(self, recommendations: List[AgentRecommendation]) -> List[Dict[str, Any]]:
        """Detect timing-related conflicts"""
        conflicts = []
        departure_times = []
        
        for rec in recommendations:
            if 'flight' in rec.recommendation and 'departure_time' in rec.recommendation['flight']:
                try:
                    dep_time = datetime.fromisoformat(rec.recommendation['flight']['departure_time'])
                    departure_times.append({
                        'agent_id': rec.agent_id,
                        'time': dep_time,
                        'trust_score': rec.trust_score
                    })
                except:
                    continue
        
        if len(departure_times) < 2:
            return conflicts
        
        # Check for timing conflicts
        times = [d['time'] for d in departure_times]
        max_time_diff = max(times) - min(times)
        
        if max_time_diff.total_seconds() > self.conflict_thresholds['timing_window']:
            conflicts.append({
                'type': ConflictType.TIMING_CONFLICT,
                'severity': 'medium',
                'description': f"Departure time difference of {max_time_diff} detected",
                'affected_agents': [d['agent_id'] for d in departure_times],
                'time_range': {
                    'earliest': min(times).isoformat(),
                    'latest': max(times).isoformat()
                },
                'difference_hours': max_time_diff.total_seconds() / 3600,
                'evidence': f"Departure times span {max_time_diff}"
            })
        
        return conflicts
    
    def _detect_route_conflicts(self, recommendations: List[AgentRecommendation]) -> List[Dict[str, Any]]:
        """Detect route efficiency conflicts"""
        conflicts = []
        route_efficiencies = []
        
        for rec in recommendations:
            if 'route_efficiency' in rec.recommendation:
                route_efficiencies.append({
                    'agent_id': rec.agent_id,
                    'efficiency': rec.recommendation['route_efficiency'],
                    'trust_score': rec.trust_score
                })
        
        if len(route_efficiencies) < 2:
            return conflicts
        
        # Check for route efficiency conflicts
        efficiencies = [r['efficiency'] for r in route_efficiencies]
        max_diff = max(efficiencies) - min(efficiencies)
        
        if max_diff > self.conflict_thresholds['route_efficiency_diff']:
            conflicts.append({
                'type': ConflictType.ROUTE_DISAGREEMENT,
                'severity': 'medium',
                'description': f"Route efficiency difference of {max_diff:.1%} detected",
                'affected_agents': [r['agent_id'] for r in route_efficiencies],
                'efficiency_range': {'min': min(efficiencies), 'max': max(efficiencies)},
                'difference': max_diff,
                'evidence': f"Route efficiencies range from {min(efficiencies):.1%} to {max(efficiencies):.1%}"
            })
        
        return conflicts

class ConflictResolver:
    """Resolves conflicts using trust-weighted approaches and AI reasoning"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictType.PRICE_DISCREPANCY: self._resolve_price_conflict,
            ConflictType.SAFETY_CONCERN: self._resolve_safety_conflict,
            ConflictType.WEATHER_IMPACT: self._resolve_weather_conflict,
            ConflictType.TIMING_CONFLICT: self._resolve_timing_conflict,
            ConflictType.ROUTE_DISAGREEMENT: self._resolve_route_conflict,
            ConflictType.RELIABILITY_ISSUE: self._resolve_reliability_conflict
        }
        
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]], 
                         recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve detected conflicts and provide unified solution"""
        resolution_results = []
        
        for conflict in conflicts:
            conflict_type = conflict['type']
            if conflict_type in self.resolution_strategies:
                resolution = self.resolution_strategies[conflict_type](conflict, recommendations)
                resolution_results.append(resolution)
            else:
                # Default resolution strategy
                resolution = self._default_resolution(conflict, recommendations)
                resolution_results.append(resolution)
        
        # Synthesize final resolution
        final_resolution = self._synthesize_resolutions(resolution_results, recommendations)
        
        return final_resolution
    
    def _resolve_price_conflict(self, conflict: Dict[str, Any], 
                               recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve price discrepancy conflicts"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Trust-weighted price calculation
        weighted_prices = []
        total_trust = 0
        
        for rec in relevant_recs:
            if 'flight' in rec.recommendation and 'price' in rec.recommendation['flight']:
                price = rec.recommendation['flight']['price']
                trust = rec.trust_score
                weighted_prices.append(price * trust)
                total_trust += trust
        
        if total_trust > 0:
            trust_weighted_price = sum(weighted_prices) / total_trust
        else:
            # Fallback to simple average
            prices = [rec.recommendation['flight']['price'] for rec in relevant_recs 
                     if 'flight' in rec.recommendation and 'price' in rec.recommendation['flight']]
            trust_weighted_price = np.mean(prices) if prices else 0
        
        # Additional verification through economic analysis
        economic_factors = self._analyze_economic_factors(relevant_recs)
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'trust_weighted_average',
            'resolved_price': round(trust_weighted_price, 2),
            'confidence': self._calculate_resolution_confidence(relevant_recs),
            'economic_factors': economic_factors,
            'reasoning': f"Trust-weighted price calculation resulted in ${trust_weighted_price:.2f}",
            'supporting_agents': [r.agent_id for r in relevant_recs]
        }
    
    def _resolve_safety_conflict(self, conflict: Dict[str, Any], 
                                recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve safety-related conflicts with conservative approach"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Use most conservative (lowest) safety score when in doubt
        safety_scores = []
        for rec in relevant_recs:
            if 'safety_score' in rec.recommendation:
                safety_scores.append({
                    'score': rec.recommendation['safety_score'],
                    'agent': rec.agent_id,
                    'trust': rec.trust_score
                })
        
        # Conservative approach: prioritize lowest score but weight by trust
        if safety_scores:
            # Sort by safety score, then by trust
            safety_scores.sort(key=lambda x: (x['score'], -x['trust']))
            
            # Use lowest score from most trusted source
            resolved_score = safety_scores[0]['score']
            primary_agent = safety_scores[0]['agent']
        else:
            resolved_score = 0.5  # Default conservative score
            primary_agent = 'system_default'
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'conservative_safety_first',
            'resolved_safety_score': resolved_score,
            'primary_agent': primary_agent,
            'confidence': SolutionConfidence.HIGH.value,  # High confidence in safety-first approach
            'reasoning': f"Applied conservative safety-first principle, using lowest score {resolved_score:.3f}",
            'safety_recommendation': 'proceed_with_caution' if resolved_score < 0.7 else 'proceed_normally'
        }
    
    def _resolve_weather_conflict(self, conflict: Dict[str, Any], 
                                 recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve weather-related conflicts using real-time data integration"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Get weather-specific agents' recommendations
        weather_recs = [r for r in relevant_recs if 'Weather' in r.agent_id]
        
        if weather_recs:
            # Prioritize weather specialist agents
            primary_weather_rec = max(weather_recs, key=lambda x: x.trust_score)
            resolved_impact = primary_weather_rec.recommendation.get('weather_impact', 0.5)
            primary_agent = primary_weather_rec.agent_id
        else:
            # Use trust-weighted average from all agents
            impacts = []
            trusts = []
            for rec in relevant_recs:
                if 'weather_impact' in rec.recommendation:
                    impacts.append(rec.recommendation['weather_impact'])
                    trusts.append(rec.trust_score)
            
            if impacts and trusts:
                resolved_impact = np.average(impacts, weights=trusts)
                primary_agent = 'weighted_consensus'
            else:
                resolved_impact = 0.3  # Default moderate impact
                primary_agent = 'system_default'
        
        # Generate weather-specific recommendations
        weather_actions = self._generate_weather_actions(resolved_impact)
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'weather_specialist_priority',
            'resolved_weather_impact': resolved_impact,
            'primary_agent': primary_agent,
            'confidence': SolutionConfidence.MEDIUM.value,
            'weather_actions': weather_actions,
            'reasoning': f"Weather impact assessment: {resolved_impact:.1%} with recommended actions",
            'flight_recommendation': 'monitor_conditions' if resolved_impact > 0.4 else 'proceed_as_planned'
        }
    
    def _resolve_timing_conflict(self, conflict: Dict[str, Any], 
                                recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve timing conflicts through optimization"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Extract timing preferences with context
        timing_options = []
        for rec in relevant_recs:
            if 'flight' in rec.recommendation and 'departure_time' in rec.recommendation['flight']:
                try:
                    dep_time = datetime.fromisoformat(rec.recommendation['flight']['departure_time'])
                    timing_options.append({
                        'time': dep_time,
                        'agent': rec.agent_id,
                        'trust': rec.trust_score,
                        'preference_strength': rec.confidence
                    })
                except:
                    continue
        
        if timing_options:
            # Multi-criteria optimization: balance trust, preference strength, and practical constraints
            best_option = max(timing_options, key=lambda x: x['trust'] * x['preference_strength'])
            resolved_time = best_option['time']
            primary_agent = best_option['agent']
        else:
            resolved_time = datetime.now()  # Fallback
            primary_agent = 'system_default'
        
        # Generate timing optimization suggestions
        timing_alternatives = self._generate_timing_alternatives(timing_options)
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'multi_criteria_timing_optimization',
            'resolved_departure_time': resolved_time.isoformat(),
            'primary_agent': primary_agent,
            'confidence': SolutionConfidence.MEDIUM.value,
            'timing_alternatives': timing_alternatives,
                            'reasoning': f"Standard timing based on trust and preference strength",
            'flexibility_window': '±2 hours recommended for optimal alternatives'
        }
    
    def _resolve_route_conflict(self, conflict: Dict[str, Any], 
                               recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve route efficiency conflicts"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Evaluate route options with multiple criteria
        route_scores = []
        for rec in relevant_recs:
            if 'route_efficiency' in rec.recommendation:
                route_score = {
                    'agent': rec.agent_id,
                    'efficiency': rec.recommendation['route_efficiency'],
                    'trust': rec.trust_score,
                    'composite_score': rec.recommendation['route_efficiency'] * rec.trust_score
                }
                route_scores.append(route_score)
        
        if route_scores:
            # Select highest composite score
            best_route = max(route_scores, key=lambda x: x['composite_score'])
            resolved_efficiency = best_route['efficiency']
            primary_agent = best_route['agent']
        else:
            resolved_efficiency = 0.8  # Default efficiency
            primary_agent = 'system_default'
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'composite_route_scoring',
            'resolved_route_efficiency': resolved_efficiency,
            'primary_agent': primary_agent,
            'confidence': SolutionConfidence.MEDIUM.value,
            'reasoning': f"Selected route with {resolved_efficiency:.1%} efficiency based on composite scoring",
            'route_optimization': 'balanced_efficiency_trust'
        }
    
    def _resolve_reliability_conflict(self, conflict: Dict[str, Any], 
                                     recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Resolve reliability-related conflicts"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Assess reliability based on historical performance and current confidence
        reliability_assessments = []
        for rec in relevant_recs:
            reliability_score = rec.trust_score * rec.confidence
            reliability_assessments.append({
                'agent': rec.agent_id,
                'reliability': reliability_score,
                'trust': rec.trust_score,
                'confidence': rec.confidence
            })
        
        if reliability_assessments:
            # Use highest reliability agent's recommendation
            most_reliable = max(reliability_assessments, key=lambda x: x['reliability'])
            primary_agent = most_reliable['agent']
            resolved_reliability = most_reliable['reliability']
        else:
            primary_agent = 'system_default'
            resolved_reliability = 0.7
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'reliability_maximization',
            'resolved_reliability_score': resolved_reliability,
            'primary_agent': primary_agent,
            'confidence': SolutionConfidence.HIGH.value,
            'reasoning': f"Selected most reliable agent ({primary_agent}) with {resolved_reliability:.3f} reliability score",
            'reliability_threshold': 'above_acceptable_levels' if resolved_reliability > 0.7 else 'requires_monitoring'
        }
    
    def _default_resolution(self, conflict: Dict[str, Any], 
                           recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Default conflict resolution strategy"""
        affected_agents = conflict['affected_agents']
        relevant_recs = [r for r in recommendations if r.agent_id in affected_agents]
        
        # Simple trust-weighted approach for unknown conflict types
        if relevant_recs:
            highest_trust_rec = max(relevant_recs, key=lambda x: x.trust_score)
            primary_agent = highest_trust_rec.agent_id
        else:
            primary_agent = 'system_default'
        
        return {
            'conflict_type': conflict['type'],
            'resolution_method': 'trust_based_selection',
            'primary_agent': primary_agent,
            'confidence': SolutionConfidence.LOW.value,
            'reasoning': f"Applied default trust-based selection for unknown conflict type",
            'recommendation': 'review_manually'
        }
    
    def _synthesize_resolutions(self, resolutions: List[Dict[str, Any]], 
                               recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Synthesize multiple conflict resolutions into unified solution"""
        if not resolutions:
            return self._create_default_solution(recommendations)
        
        # Aggregate confidence scores
        confidences = [r.get('confidence', SolutionConfidence.LOW.value) for r in resolutions]
        confidence_values = {
            SolutionConfidence.HIGH.value: 0.9,
            SolutionConfidence.MEDIUM.value: 0.7,
            SolutionConfidence.LOW.value: 0.5
        }
        
        avg_confidence_score = np.mean([confidence_values.get(c, 0.5) for c in confidences])
        
        if avg_confidence_score > 0.8:
            overall_confidence = SolutionConfidence.HIGH.value
        elif avg_confidence_score > 0.6:
            overall_confidence = SolutionConfidence.MEDIUM.value
        else:
            overall_confidence = SolutionConfidence.LOW.value
        
        # Create unified solution
        unified_solution = {
            'solution_timestamp': datetime.now().isoformat(),
            'overall_confidence': overall_confidence,
            'conflict_resolutions': resolutions,
            'synthesis_method': 'trust_weighted_integration',
            'recommendations_considered': len(recommendations),
            'conflicts_resolved': len(resolutions)
        }
        
        # Extract key recommendations from resolutions
        if resolutions:
            key_recommendations = self._extract_key_recommendations(resolutions)
            unified_solution['key_recommendations'] = key_recommendations
        
        # Add standard reasoning analysis
        reasoning_summary = self._analyze_reasoning_summary(resolutions, recommendations)
        if reasoning_summary:
            unified_solution['reasoning_analysis'] = reasoning_summary
        
        return unified_solution
    
    def _create_default_solution(self, recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Create default solution when no conflicts detected"""
        if not recommendations:
            return {
                'solution_timestamp': datetime.now().isoformat(),
                'overall_confidence': SolutionConfidence.LOW.value,
                'status': 'no_recommendations_available',
                'recommendation': 'manual_review_required'
            }
        
        # Use highest trust recommendation as baseline
        best_rec = max(recommendations, key=lambda x: x.trust_score * x.confidence)
        
        return {
            'solution_timestamp': datetime.now().isoformat(),
            'overall_confidence': SolutionConfidence.HIGH.value,
            'status': 'no_conflicts_detected',
            'primary_recommendation': best_rec.recommendation,
            'primary_agent': best_rec.agent_id,
            'trust_score': best_rec.trust_score,
            'reasoning': 'All agents in agreement, using highest trust recommendation'
        }
    
    def _analyze_economic_factors(self, recommendations: List[AgentRecommendation]) -> Dict[str, Any]:
        """Analyze economic factors from recommendations"""
        economic_data = {
            'price_volatility': 0,
            'market_conditions': 'stable',
            'hidden_costs_detected': False,
            'cost_optimization_potential': 0
        }
        
        prices = []
        for rec in recommendations:
            if 'flight' in rec.recommendation and 'price' in rec.recommendation['flight']:
                prices.append(rec.recommendation['flight']['price'])
        
        if len(prices) > 1:
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            economic_data['price_volatility'] = price_std / price_mean if price_mean > 0 else 0
            
            if economic_data['price_volatility'] > 0.15:
                economic_data['market_conditions'] = 'volatile'
            elif economic_data['price_volatility'] < 0.05:
                economic_data['market_conditions'] = 'stable'
            else:
                economic_data['market_conditions'] = 'moderate'
        
        return economic_data
    
    def _generate_weather_actions(self, impact: float) -> List[str]:
        """Generate weather-specific action recommendations"""
        actions = []
        
        if impact > 0.6:
            actions.extend([
                'Monitor weather conditions closely',
                'Consider alternative departure times',
                'Have backup flight options ready',
                'Check airline rebooking policies'
            ])
        elif impact > 0.3:
            actions.extend([
                'Check weather updates before departure',
                'Allow extra travel time to airport',
                'Monitor flight status regularly'
            ])
        else:
            actions.extend([
                'Normal weather monitoring sufficient',
                'Proceed with planned schedule'
            ])
        
        return actions
    
    def _generate_timing_alternatives(self, timing_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alternative timing options"""
        alternatives = []
        
        if timing_options:
            # Sort by time
            sorted_options = sorted(timing_options, key=lambda x: x['time'])
            
            for i, option in enumerate(sorted_options[:3]):  # Top 3 alternatives
                alternative = {
                    'rank': i + 1,
                    'departure_time': option['time'].isoformat(),
                    'recommending_agent': option['agent'],
                    'trust_score': option['trust'],
                    'time_difference_hours': 0 if i == 0 else 
                        (option['time'] - sorted_options[0]['time']).total_seconds() / 3600
                }
                alternatives.append(alternative)
        
        return alternatives
    
    def _extract_key_recommendations(self, resolutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key actionable recommendations from resolutions"""
        key_recs = {
            'primary_actions': [],
            'secondary_actions': [],
            'monitoring_required': [],
            'risk_factors': []
        }
        
        for resolution in resolutions:
            conflict_type = resolution.get('conflict_type')
            
            if conflict_type == ConflictType.SAFETY_CONCERN:
                key_recs['primary_actions'].append(resolution.get('safety_recommendation', 'review_safety'))
                if resolution.get('resolved_safety_score', 1.0) < 0.7:
                    key_recs['risk_factors'].append('Safety score below acceptable threshold')
            
            elif conflict_type == ConflictType.WEATHER_IMPACT:
                weather_actions = resolution.get('weather_actions', [])
                key_recs['secondary_actions'].extend(weather_actions)
                key_recs['monitoring_required'].append('Weather conditions')
            
            elif conflict_type == ConflictType.PRICE_DISCREPANCY:
                key_recs['primary_actions'].append(f"Price verification required")
                if resolution.get('confidence') == SolutionConfidence.LOW.value:
                    key_recs['risk_factors'].append('High price uncertainty')
        
        return key_recs
    
    def _analyze_reasoning_summary(self, resolutions: List[Dict[str, Any]], 
                                  recommendations: List[AgentRecommendation]) -> Optional[Dict[str, Any]]:
        """Analyze reasoning summary using standard methods"""
        try:
            # Basic statistical analysis only
            if not recommendations:
                return None
                
            # Calculate basic statistics
            confidence_scores = [rec.confidence for rec in recommendations]
            trust_scores = [rec.trust_score for rec in recommendations]
            
            analysis_summary = {
                'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                'max_confidence': max(confidence_scores),
                'min_confidence': min(confidence_scores),
                'avg_trust': sum(trust_scores) / len(trust_scores),
                'num_resolutions': len(resolutions),
                'num_recommendations': len(recommendations),
                'source': 'statistical_analysis'
            }
            
            return analysis_summary
                
        except Exception as e:
            logger.warning(f"Failed to analyze reasoning: {e}")
            return None
    
    def _calculate_resolution_confidence(self, recommendations: List[AgentRecommendation]) -> str:
        """Calculate confidence in conflict resolution"""
        if not recommendations:
            return SolutionConfidence.LOW.value
        
        avg_trust = np.mean([rec.trust_score for rec in recommendations])
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        
        combined_score = (avg_trust + avg_confidence) / 2
        
        if combined_score > 0.8:
            return SolutionConfidence.HIGH.value
        elif combined_score > 0.6:
            return SolutionConfidence.MEDIUM.value
        else:
            return SolutionConfidence.LOW.value

class CrossDomainSolver:
    """Main cross-domain problem solving coordinator"""
    
    def __init__(self, trust_manager):
        self.trust_manager = trust_manager
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
        
    def solve_cross_domain_problem(self, agent_outputs: Dict[str, Dict[str, Any]], 
                                  user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for cross-domain problem solving"""
        
        # Convert agent outputs to structured recommendations
        recommendations = self._convert_to_recommendations(agent_outputs)
        
        # Detect conflicts between agent recommendations
        conflicts = self.conflict_detector.detect_conflicts(recommendations)
        
        # Resolve conflicts if any detected
        if conflicts:
            resolution = self.conflict_resolver.resolve_conflicts(conflicts, recommendations)
            solution_status = 'conflicts_resolved'
        else:
            resolution = self.conflict_resolver._create_default_solution(recommendations)
            solution_status = 'no_conflicts'
        
        # Apply user preferences if provided
        if user_preferences:
            resolution = self._apply_user_preferences(resolution, user_preferences)
        
        # Generate final integrated solution
        final_solution = {
            'timestamp': datetime.now().isoformat(),
            'status': solution_status,
            'conflicts_detected': len(conflicts),
            'conflicts_summary': [
                {
                    'type': c['type'].value if hasattr(c['type'], 'value') else str(c['type']),
                    'severity': c.get('severity', 'unknown'),
                    'description': c.get('description', '')
                }
                for c in conflicts
            ],
            'resolution': resolution,
            'recommendations_processed': len(recommendations),
            'agent_trust_scores': {
                rec.agent_id: rec.trust_score for rec in recommendations
            }
        }
        
        # Add performance metrics
        final_solution['performance_metrics'] = self._calculate_solution_metrics(
            recommendations, conflicts, resolution
        )
        
        return final_solution
    
    def _convert_to_recommendations(self, agent_outputs: Dict[str, Dict[str, Any]]) -> List[AgentRecommendation]:
        """Convert raw agent outputs to structured recommendations"""
        recommendations = []
        
        for agent_id, output in agent_outputs.items():
            # Get current trust score for agent
            trust_score = self.trust_manager.ledger.get_current_trust_score(agent_id)
            
            # Extract confidence from output
            confidence = output.get('confidence', 0.8)
            
            # Extract domain expertise (could be configured per agent)
            domain_expertise = self._get_domain_expertise(agent_id)
            
            # Extract supporting evidence
            evidence = output.get('supporting_evidence', [])
            if isinstance(evidence, str):
                evidence = [evidence]
            
            recommendation = AgentRecommendation(
                agent_id=agent_id,
                recommendation=output,
                confidence=confidence,
                trust_score=trust_score,
                timestamp=datetime.now().isoformat(),
                domain_expertise=domain_expertise,
                supporting_evidence=evidence
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_domain_expertise(self, agent_id: str) -> float:
        """Get domain expertise level for agent"""
        expertise_map = {
            'WeatherAgent': 0.9,
            'FlightInfoAgent': 0.85,
            'SafetyAgent': 0.95,
            'EconomicAgent': 0.8,
            'IntegrationAgent': 0.75
        }
        return expertise_map.get(agent_id, 0.7)
    
    def _apply_user_preferences(self, resolution: Dict[str, Any], 
                               preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to resolution"""
        # User preference weighting
        preference_weights = {
            'cost_importance': preferences.get('cost_importance', 0.3),
            'safety_importance': preferences.get('safety_importance', 0.4),
            'convenience_importance': preferences.get('convenience_importance', 0.2),
            'speed_importance': preferences.get('speed_importance', 0.1)
        }
        
        # Adjust resolution based on preferences
        if preference_weights['safety_importance'] > 0.5:
            # High safety preference - increase safety confidence
            if 'safety_recommendation' in resolution:
                resolution['safety_priority'] = 'standard'
        
        if preference_weights['cost_importance'] > 0.4:
            # High cost sensitivity - add cost optimization note
            resolution['cost_optimization'] = 'prioritized'
        
        resolution['user_preferences_applied'] = preference_weights
        
        return resolution
    
    def _calculate_solution_metrics(self, recommendations: List[AgentRecommendation], 
                                   conflicts: List[Dict[str, Any]], 
                                   resolution: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the solution"""
        metrics = {
            'agent_consensus_score': 0,
            'conflict_resolution_score': 0,
            'overall_confidence': 0,
            'trust_distribution': {},
            'processing_complexity': 'low'
        }
        
        if recommendations:
            # Calculate agent consensus
            trust_scores = [rec.trust_score for rec in recommendations]
            confidence_scores = [rec.confidence for rec in recommendations]
            
            metrics['agent_consensus_score'] = np.mean(trust_scores) * np.mean(confidence_scores)
            metrics['overall_confidence'] = np.mean(confidence_scores)
            
            # Trust distribution
            for rec in recommendations:
                metrics['trust_distribution'][rec.agent_id] = rec.trust_score
        
        # Conflict resolution scoring
        if conflicts:
            num_conflicts = len(conflicts)
            high_severity_conflicts = sum(1 for c in conflicts if c.get('severity') in ['high', 'critical'])
            
            if high_severity_conflicts > 0:
                metrics['conflict_resolution_score'] = 0.6  # Moderate success with high severity
                metrics['processing_complexity'] = 'high'
            else:
                metrics['conflict_resolution_score'] = 0.8  # Good resolution of lower severity
                metrics['processing_complexity'] = 'medium'
        else:
            metrics['conflict_resolution_score'] = 1.0  # No conflicts detected
            metrics['processing_complexity'] = 'low'
        
        return metrics

# Global cross-domain solver instance
def create_cross_domain_solver(trust_manager):
    """Factory function to create cross-domain solver"""
    return CrossDomainSolver(trust_manager) 