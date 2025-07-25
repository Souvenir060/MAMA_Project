# MAMA_exp/agents/trust_manager.py

"""
MAMA Flight Assistant - Trust Management System

This module implements:
1. Trust-Aware Adaptive Interaction Protocols
2. Multi-Dimensional Trustworthiness Ledgering  
3. Cross-Domain Problem Solving
4. Dynamic Trust Management and Decision Integration
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict
import requests
from config import JWT_TOKEN, PROTECTED_URL
from enum import Enum
import asyncio

# Import core module components
try:
    from core import AdaptiveInteractionProtocol, InteractionContext
    from core.mcp_integration import MCPClient, MCPMessage, MCPMessageType
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    AdaptiveInteractionProtocol = None
    InteractionContext = None
    MCPClient = None
    MCPMessage = None
    MCPMessageType = None

logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    """Trust level classifications"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8  
    LOW = "low"        # < 0.5

class InteractionProtocol(Enum):
    """Interaction protocol types"""
    SIMPLIFIED = "simplified"     # High trust
    PARTIAL = "partial"          # Medium trust
    STRICT = "strict"            # Low trust

@dataclass
class TrustDimensions:
    """Five-dimensional trust metrics"""
    reliability: float = 0.5      # Consistency in task completion
    competence: float = 0.5       # Task execution quality
    fairness: float = 0.5         # Decision bias detection
    security: float = 0.5         # Attack resistance
    transparency: float = 0.5     # Output explainability
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall trust score - 按论文公式1实现"""
        if weights is None:
            # 论文中的权重设置 (公式1)
            weights = {
                'reliability': 0.25,    # w1
                'competence': 0.25,     # w2
                'fairness': 0.2,        # w3
                'security': 0.15,       # w4
                'transparency': 0.15    # w5
            }
        
        # 论文公式1: TrustScore = w1·Reliability + w2·Competence + w3·Fairness + w4·Security + w5·Transparency
        score = (
            weights['reliability'] * self.reliability +
            weights['competence'] * self.competence +
            weights['fairness'] * self.fairness +
            weights['security'] * self.security +
            weights['transparency'] * self.transparency
        )
        return max(0.0, min(1.0, score))

@dataclass
class TrustRecord:
    """Individual trust record for ledger"""
    agent_id: str
    timestamp: str
    dimensions: TrustDimensions
    evidence: str
    task_id: str
    performance_metrics: Dict[str, float]
    
    def to_milestone_format(self) -> Dict[str, Any]:
        """Convert to NGSI-LD format for Milestone storage"""
        return {
            "id": f"urn:ngsi-ld:TrustRecord:{self.agent_id}:{int(time.time())}",
            "type": "TrustRecord",
            "agentId": {"type": "Property", "value": self.agent_id},
            "timestamp": {"type": "Property", "value": self.timestamp},
            "reliability": {"type": "Property", "value": self.dimensions.reliability},
            "competence": {"type": "Property", "value": self.dimensions.competence},
            "fairness": {"type": "Property", "value": self.dimensions.fairness},
            "security": {"type": "Property", "value": self.dimensions.security},
            "transparency": {"type": "Property", "value": self.dimensions.transparency},
            "overallScore": {"type": "Property", "value": self.dimensions.calculate_overall_score()},
            "evidence": {"type": "Property", "value": self.evidence},
            "taskId": {"type": "Property", "value": self.task_id},
            "performanceMetrics": {"type": "Property", "value": json.dumps(self.performance_metrics)},
            "@context": ["https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"]
        }

class TrustLedger:
    """Distributed trust ledger implementation"""
    
    def __init__(self):
        self.local_cache: Dict[str, List[TrustRecord]] = defaultdict(list)
        self.milestone_url = PROTECTED_URL
        self.jwt_token = JWT_TOKEN
        
    def add_record(self, record: TrustRecord) -> bool:
        """Add trust record to ledger"""
        try:
            # Add to local cache
            self.local_cache[record.agent_id].append(record)
            
            # Store to Milestone
            headers = {
                'Authorization': f'Bearer {self.jwt_token}',
                'Content-Type': 'application/ld+json'
            }
            
            milestone_data = record.to_milestone_format()
            response = requests.post(
                self.milestone_url,
                json=milestone_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Trust record stored for agent {record.agent_id}")
                return True
            else:
                logger.error(f"Failed to store trust record: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding trust record: {e}")
            return False
    
    def get_agent_history(self, agent_id: str, limit: int = 10) -> List[TrustRecord]:
        """Get trust history for specific agent"""
        return self.local_cache.get(agent_id, [])[-limit:]
    
    def get_current_trust_score(self, agent_id: str) -> float:
        """Get current trust score for agent"""
        records = self.local_cache.get(agent_id, [])
        if not records:
            return 0.5  # Default neutral trust
        
        # Use recent records with temporal decay
        current_time = datetime.now()
        weighted_scores = []
        
        for record in records[-10:]:  # Last 10 records
            record_time = datetime.fromisoformat(record.timestamp)
            days_old = (current_time - record_time).days
            decay_factor = 0.95 ** days_old  # 5% decay per day
            
            score = record.dimensions.calculate_overall_score()
            weighted_scores.append(score * decay_factor)
        
        return np.mean(weighted_scores) if weighted_scores else 0.5

class InteractionProtocolManager:
    """Manages trust-adaptive interaction protocols"""
    
    def __init__(self, trust_ledger: TrustLedger):
        self.trust_ledger = trust_ledger
        self.protocol_cache: Dict[Tuple[str, str], InteractionProtocol] = {}
        
    def get_interaction_protocol(self, source_agent: str, target_agent: str) -> InteractionProtocol:
        """Determine interaction protocol based on trust levels"""
        cache_key = (source_agent, target_agent)
        
        if cache_key in self.protocol_cache:
            return self.protocol_cache[cache_key]
        
        trust_score = self.trust_ledger.get_current_trust_score(target_agent)
        
        if trust_score > 0.8:
            protocol = InteractionProtocol.SIMPLIFIED
        elif trust_score > 0.5:
            protocol = InteractionProtocol.PARTIAL
        else:
            protocol = InteractionProtocol.STRICT
        
        self.protocol_cache[cache_key] = protocol
        return protocol
    
    def process_interaction(self, source_agent: str, target_agent: str, 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent interaction based on trust protocol"""
        protocol = self.get_interaction_protocol(source_agent, target_agent)
        
        if protocol == InteractionProtocol.SIMPLIFIED:
            return self._simplified_interaction(data)
        elif protocol == InteractionProtocol.PARTIAL:
            return self._partial_interaction(data)
        else:
            return self._strict_interaction(data)
    
    def _simplified_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """High trust - simplified processing"""
        # Direct data sharing with minimal validation
        result = {
            'status': 'success',
            'data': data,
            'validation_level': 'minimal',
            'processing_time': time.time()
        }
        return result
    
    def _partial_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Medium trust - partial data abstraction"""
        # Abstract sensitive data, lightweight validation
        processed_data = {}
        for key, value in data.items():
            if key in ['raw_data', 'detailed_metrics']:
                # Abstract detailed data to summary
                processed_data[f'{key}_summary'] = self._create_summary(value)
            else:
                processed_data[key] = value
        
        # Lightweight validation
        validation_result = self._validate_data_consistency(processed_data)
        
        result = {
            'status': 'success',
            'data': processed_data,
            'validation_level': 'partial',
            'validation_result': validation_result,
            'processing_time': time.time()
        }
        return result
    
    def _strict_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Low trust - strict verification"""
        # Multi-round verification and anomaly detection
        validation_results = []
        
        # Data integrity check
        integrity_check = self._check_data_integrity(data)
        validation_results.append(integrity_check)
        
        # Source verification
        source_check = self._verify_data_source(data)
        validation_results.append(source_check)
        
        # Anomaly detection
        anomaly_check = self._detect_anomalies(data)
        validation_results.append(anomaly_check)
        
        # Filter sensitive data
        filtered_data = self._filter_sensitive_data(data)
        
        result = {
            'status': 'success' if all(v['passed'] for v in validation_results) else 'warning',
            'data': filtered_data,
            'validation_level': 'strict',
            'validation_results': validation_results,
            'processing_time': time.time()
        }
        return result
    
    def _create_summary(self, data: Any) -> Dict[str, Any]:
        """Create data summary for partial interactions"""
        if isinstance(data, dict):
            return {
                'type': 'summary',
                'keys_count': len(data),
                'has_numeric_data': any(isinstance(v, (int, float)) for v in data.values()),
                'sample_keys': list(data.keys())[:3]
            }
        elif isinstance(data, list):
            return {
                'type': 'list_summary',
                'length': len(data),
                'sample_items': data[:2] if len(data) > 0 else []
            }
        else:
            return {'type': 'scalar_summary', 'data_type': type(data).__name__}
    
    def _validate_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight data consistency validation"""
        checks = {
            'has_required_fields': bool(data.get('agent_id') and data.get('timestamp')),
            'data_format_valid': isinstance(data, dict),
            'no_null_critical_fields': all(data.get(field) is not None 
                                          for field in ['agent_id', 'timestamp'])
        }
        
        return {
            'passed': all(checks.values()),
            'details': checks
        }
    
    def _check_data_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check data integrity for strict interactions"""
        try:
            # Calculate hash for integrity
            data_str = json.dumps(data, sort_keys=True)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Check for required fields
            required_fields = ['agent_id', 'timestamp', 'data']
            missing_fields = [field for field in required_fields if field not in data]
            
            return {
                'passed': len(missing_fields) == 0,
                'data_hash': data_hash,
                'missing_fields': missing_fields
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _verify_data_source(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data source authenticity"""
        agent_id = data.get('agent_id', '')
        timestamp = data.get('timestamp', '')
        
        # Basic source verification
        valid_agents = ['WeatherAgent', 'FlightInfoAgent', 'SafetyAgent', 
                       'EconomicAgent', 'IntegrationAgent']
        
        source_valid = agent_id in valid_agents
        timestamp_valid = self._validate_timestamp(timestamp)
        
        return {
            'passed': source_valid and timestamp_valid,
            'source_valid': source_valid,
            'timestamp_valid': timestamp_valid
        }
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in data"""
        anomalies = []
        
        # Check for unusual data patterns
        if 'scores' in data:
            scores = data['scores']
            if isinstance(scores, dict):
                for key, value in scores.items():
                    if isinstance(value, (int, float)):
                        if value < 0 or value > 1:
                            anomalies.append(f"Score {key} out of range: {value}")
        
        # Check timestamp freshness
        timestamp = data.get('timestamp', '')
        if timestamp:
            try:
                data_time = datetime.fromisoformat(timestamp)
                age_hours = (datetime.now() - data_time).total_seconds() / 3600
                if age_hours > 24:
                    anomalies.append(f"Data is {age_hours:.1f} hours old")
            except:
                anomalies.append("Invalid timestamp format")
        
        return {
            'passed': len(anomalies) == 0,
            'anomalies': anomalies
        }
    
    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data for low trust interactions"""
        sensitive_keys = ['api_key', 'credentials', 'raw_personal_data', 'internal_metrics']
        filtered_data = {}
        
        for key, value in data.items():
            if key not in sensitive_keys:
                filtered_data[key] = value
            else:
                filtered_data[f'{key}_filtered'] = '[SENSITIVE_DATA_FILTERED]'
        
        return filtered_data
    
    def _validate_timestamp(self, timestamp: str) -> bool:
        """Validate timestamp format and freshness"""
        try:
            dt = datetime.fromisoformat(timestamp)
            # Check if timestamp is within reasonable range (not too old or future)
            now = datetime.now()
            return (now - timedelta(days=7)) <= dt <= (now + timedelta(hours=1))
        except:
            return False

class TrustMetricsCalculator:
    """Calculate trust metrics for agents"""
    
    def __init__(self, trust_ledger: TrustLedger):
        self.trust_ledger = trust_ledger
        
    def evaluate_reliability(self, agent_id: str, task_results: List[Dict[str, Any]]) -> float:
        """Evaluate agent reliability based on task completion consistency"""
        if not task_results:
            return 0.5
        
        successful_tasks = sum(1 for result in task_results 
                             if result.get('status') == 'success')
        reliability_score = successful_tasks / len(task_results)
        
        return max(0.0, min(1.0, reliability_score))
    
    def evaluate_competence(self, agent_id: str, performance_data: Dict[str, Any]) -> float:
        """Evaluate agent competence based on output quality"""
        if not performance_data:
            return 0.5
        
        # Domain-specific competence evaluation
        if agent_id == 'WeatherAgent':
            return self._evaluate_weather_competence(performance_data)
        elif agent_id == 'SafetyAgent':
            return self._evaluate_safety_competence(performance_data)
        elif agent_id == 'EconomicAgent':
            return self._evaluate_economic_competence(performance_data)
        else:
            return self._evaluate_general_competence(performance_data)
    
    def evaluate_fairness(self, agent_id: str, decision_history: List[Dict[str, Any]]) -> float:
        """Evaluate agent fairness by detecting decision bias"""
        if not decision_history:
            return 0.5
        
        # Analyze decision distribution for bias
        airline_preferences = defaultdict(int)
        price_ranges = defaultdict(int)
        
        for decision in decision_history:
            if 'recommended_flight' in decision:
                flight = decision['recommended_flight']
                airline = flight.get('airline', 'unknown')
                price = flight.get('price', 0)
                
                airline_preferences[airline] += 1
                
                # Categorize price ranges
                if price < 500:
                    price_ranges['low'] += 1
                elif price < 1000:
                    price_ranges['medium'] += 1
                else:
                    price_ranges['high'] += 1
        
        # Calculate fairness based on distribution uniformity
        airline_fairness = self._calculate_distribution_fairness(airline_preferences)
        price_fairness = self._calculate_distribution_fairness(price_ranges)
        
        return (airline_fairness + price_fairness) / 2
    
    def evaluate_security(self, agent_id: str, attack_simulation_results: Dict[str, Any]) -> float:
        """Evaluate agent security based on attack resistance"""
        if not attack_simulation_results:
            return 0.5  # Default neutral score
        
        successful_defenses = attack_simulation_results.get('successful_defenses', 0)
        total_attacks = attack_simulation_results.get('total_attacks', 1)
        
        security_score = successful_defenses / total_attacks
        return max(0.0, min(1.0, security_score))
    
    def evaluate_transparency(self, agent_id: str, explanation_logs: List[str]) -> float:
        """Evaluate agent transparency based on explanation quality"""
        if not explanation_logs:
            return 0.5
        
        # Analyze explanation completeness
        total_score = 0
        for explanation in explanation_logs:
            score = self._score_explanation_quality(explanation)
            total_score += score
        
        transparency_score = total_score / len(explanation_logs)
        return max(0.0, min(1.0, transparency_score))
    
    def _evaluate_weather_competence(self, performance_data: Dict[str, Any]) -> float:
        """Evaluate weather agent specific competence"""
        accuracy = performance_data.get('weather_accuracy', 0.5)
        response_time = performance_data.get('avg_response_time', 5.0)
        
        # Combine accuracy and efficiency
        time_score = max(0, 1 - (response_time - 1) / 10)  # Penalty for slow response
        competence_score = 0.7 * accuracy + 0.3 * time_score
        
        return max(0.0, min(1.0, competence_score))
    
    def _evaluate_safety_competence(self, performance_data: Dict[str, Any]) -> float:
        """Evaluate safety agent specific competence"""
        risk_assessment_accuracy = performance_data.get('risk_accuracy', 0.5)
        correlation_with_actual = performance_data.get('actual_correlation', 0.5)
        
        competence_score = 0.6 * risk_assessment_accuracy + 0.4 * correlation_with_actual
        return max(0.0, min(1.0, competence_score))
    
    def _evaluate_economic_competence(self, performance_data: Dict[str, Any]) -> float:
        """Evaluate economic agent specific competence"""
        cost_prediction_accuracy = performance_data.get('cost_accuracy', 0.5)
        hidden_cost_detection = performance_data.get('hidden_cost_detection', 0.5)
        
        competence_score = 0.5 * cost_prediction_accuracy + 0.5 * hidden_cost_detection
        return max(0.0, min(1.0, competence_score))
    
    def _evaluate_general_competence(self, performance_data: Dict[str, Any]) -> float:
        """Evaluate general agent competence"""
        accuracy = performance_data.get('accuracy', 0.5)
        efficiency = performance_data.get('efficiency', 0.5)
        
        competence_score = 0.6 * accuracy + 0.4 * efficiency
        return max(0.0, min(1.0, competence_score))
    
    def _calculate_distribution_fairness(self, distribution: Dict[str, int]) -> float:
        """Calculate fairness score based on distribution uniformity"""
        if not distribution:
            return 1.0
        
        values = list(distribution.values())
        if len(values) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (lower is more fair)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 1.0
        
        cv = std_val / mean_val
        fairness_score = max(0, 1 - cv)  # Convert to fairness score
        
        return fairness_score
    
    def _score_explanation_quality(self, explanation: str) -> float:
        """Score explanation quality based on completeness"""
        if not explanation:
            return 0.0
        
        # Check for key explanation components
        quality_indicators = [
            'because' in explanation.lower(),
            'due to' in explanation.lower(),
            'based on' in explanation.lower(),
            len(explanation.split()) > 10,  # Sufficient detail
            any(word in explanation.lower() for word in ['score', 'metric', 'value']),
            any(word in explanation.lower() for word in ['weather', 'safety', 'cost', 'time'])
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        return quality_score

class TrustManager:
    """
    Trust Manager handles trust relationships between agents
    
    This class implements the multi-dimensional trust ledger and
    manages trust relationships between agents in the system.
    """
    
    def __init__(self, agent_id: str = "trust_manager", trust_ledger=None):
        """Initialize Trust Manager"""
        self.agent_id = agent_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize trust ledger
        self.trust_ledger = trust_ledger if trust_ledger else TrustLedger()
        
        # Initialize trust metrics calculator
        self.metrics_calculator = TrustMetricsCalculator(self.trust_ledger)
        
        # Initialize interaction protocol manager
        self.protocol_manager = InteractionProtocolManager(self.trust_ledger)
        
        # Trust scores between agents
        # Format: {agent_a: {agent_b: score, agent_c: score, ...}, ...}
        self.trust_scores = {}
        
        # Interaction history
        self.interaction_history = []
        
        # MCP client for message passing
        self.mcp_client = None
        
        # Agent capabilities
        self.agent_capabilities = {}
        
        # Trust update policies
        self.trust_policies = {}
        
        # Initialize default policies
        self._initialize_default_policies()
        
        self.logger.info(f"Trust Manager initialized with ID: {self.agent_id}")
    
    async def initialize_mcp_connection(self, server_url: str = "ws://localhost:8765"):
        """Initialize MCP connection for agent communication"""
        if self.mcp_client:
            try:
                await self.mcp_client.connect()
                
                # Register message handlers
                self.mcp_client.message_handlers[MCPMessageType.AGENT_RESPONSE] = self._handle_agent_response
                self.mcp_client.message_handlers[MCPMessageType.NOTIFICATION] = self._handle_notification
                self.mcp_client.message_handlers[MCPMessageType.ERROR] = self._handle_error
                
                # Subscribe to relevant contexts
                await self._subscribe_to_contexts()
                
                self.logger.info("MCP connection established successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to establish MCP connection: {e}")
                return False
        return False
    
    async def _subscribe_to_contexts(self):
        """Subscribe to relevant MCP contexts"""
        if self.mcp_client:
            try:
                # Subscribe to agent coordination context
                context_request = MCPMessage(
                    message_id=f"sub_{int(datetime.now().timestamp())}",
                    message_type=MCPMessageType.CONTEXT_REQUEST,
                    sender_id=self.agent_id,
                    recipient_id="mcp_server",
                    payload={
                        "action": "subscribe",
                        "context_id": "agent_coordination"
                    },
                    timestamp=datetime.now().isoformat(),
                    context_id="agent_coordination"
                )
                await self.mcp_client.send_message(context_request)
                
                self.logger.debug("Subscribed to agent coordination context")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to contexts: {e}")
    
    async def _handle_agent_response(self, message: MCPMessage):
        """Handle agent response messages"""
        try:
            self.logger.debug(f"Received agent response from {message.sender_id}")
            
            # Update trust metrics based on response quality
            await self._update_trust_from_response(message)
            
            # Process response for coordination
            await self._process_coordination_response(message)
            
        except Exception as e:
            self.logger.error(f"Error handling agent response: {e}")
    
    async def _handle_notification(self, message: MCPMessage):
        """Handle notification messages"""
        try:
            event = message.payload.get("event")
            
            if event == "context_updated":
                context_id = message.payload.get("context_id")
                updates = message.payload.get("updates", {})
                await self._handle_context_update(context_id, updates)
            
            elif event == "agent_status_change":
                agent_id = message.payload.get("agent_id")
                status = message.payload.get("status")
                await self._handle_agent_status_change(agent_id, status)
            
            self.logger.debug(f"Processed notification: {event}")
            
        except Exception as e:
            self.logger.error(f"Error handling notification: {e}")
    
    async def _handle_error(self, message: MCPMessage):
        """Handle error messages"""
        error_msg = message.payload.get("error", "Unknown error")
        self.logger.warning(f"Received MCP error: {error_msg}")
        
        # Update trust scores for unreliable agents
        if message.sender_id != "mcp_server":
            await self._penalize_agent_trust(message.sender_id, "mcp_error")

    def _initialize_default_policies(self):
        """Initialize default trust policies for the system"""
        self.trust_policies = {
            "interaction_thresholds": {
                "high_trust": 0.8,
                "medium_trust": 0.5,
                "low_trust": 0.3
            },
            "trust_decay": {
                "daily_decay_rate": 0.01,  # Small daily decrease if no interactions
                "max_decay": 0.1  # Maximum decay per period
            },
            "trust_degradation": {
                "description": "Trust degradation due to extended unavailability",
                "severity": "medium",
                "degradation_factor": 0.1
                },
            "agent_recovery": {
                "description": "Agent recovery protocol activated",
                "status": "monitoring"
            }
        }
        
        self.logger.info("Default trust policies initialized")

    async def process_agent_interaction(self, agent_a: str, agent_b: str, 
                                      interaction_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process interaction between agents using trust protocols."""
        
        # Try to use core module's adaptive interaction protocol first
        if self.adaptive_protocol:
            try:
                if CORE_AVAILABLE and InteractionContext:
                    interaction_context = InteractionContext(
                        agents=[agent_a, agent_b],
                        interaction_type=interaction_type,
                        context_data=context,
                        timestamp=datetime.now()
                    )
                    
                    result = await self.adaptive_protocol.process_interaction(interaction_context)
                    
                    # Update trust scores based on interaction result
                    if result.get('success', False):
                        self._update_trust_score(agent_a, agent_b, 0.1)
                        self._update_trust_score(agent_b, agent_a, 0.1)
                    
                    # Log interaction through MCP if available
                    if self.mcp_client:
                        await self._log_interaction_to_mcp(agent_a, agent_b, interaction_type, result)
                    
                    return result
            except Exception as e:
                self.logger.error(f"Core protocol interaction failed: {e}")
                # Fall through to local implementation
        
        # Local implementation as fallback
        self.logger.info(f"Processing interaction between {agent_a} and {agent_b}")
        
        # Get trust scores
        trust_a_to_b = self.get_trust_score(agent_a, agent_b)
        trust_b_to_a = self.get_trust_score(agent_b, agent_a)
        
        # Determine interaction outcome based on trust
        success_probability = (trust_a_to_b + trust_b_to_a) / 2
        
        result = {
            'success': success_probability > 0.5,
            'trust_factor': success_probability,
            'interaction_type': interaction_type,
            'participants': [agent_a, agent_b],
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        # Update interaction history
        self.interaction_history.append({
            'agents': [agent_a, agent_b],
            'type': interaction_type,
            'result': result,
            'timestamp': datetime.now()
        })
        
        # Log interaction through MCP if available
        if self.mcp_client:
            await self._log_interaction_to_mcp(agent_a, agent_b, interaction_type, result)
        
        return result
    
    async def _log_interaction_to_mcp(self, agent_a: str, agent_b: str, 
                                    interaction_type: str, result: Dict[str, Any]):
        """Log interaction result to MCP for system-wide coordination"""
        if self.mcp_client:
            try:
                log_message = MCPMessage(
                    message_id=f"log_{int(datetime.now().timestamp())}",
                    message_type=MCPMessageType.NOTIFICATION,
                    sender_id=self.agent_id,
                    recipient_id="all",
                    payload={
                        "event": "interaction_logged",
                        "participants": [agent_a, agent_b],
                        "interaction_type": interaction_type,
                        "result": result,
                        "trust_manager": self.agent_id
                    },
                    timestamp=datetime.now().isoformat(),
                    context_id="agent_coordination"
                )
                
                await self.mcp_client.send_message(log_message)
                self.logger.debug("Interaction logged to MCP")
                
            except Exception as e:
                self.logger.error(f"Failed to log interaction to MCP: {e}")
    
    async def request_agent_collaboration(self, target_agent: str, 
                                        task_description: str, 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Request collaboration from another agent through MCP"""
        if not self.mcp_client:
            self.logger.warning("MCP client not available for collaboration request")
            return {"success": False, "error": "MCP client not available"}
        
        try:
            collaboration_request = MCPMessage(
                message_id=f"collab_{int(datetime.now().timestamp())}",
                message_type=MCPMessageType.AGENT_CALL,
                sender_id=self.agent_id,
                recipient_id=target_agent,
                payload={
                    "action": "collaboration_request",
                    "task_description": task_description,
                    "context": context,
                    "trust_score": self.get_trust_score(self.agent_id, target_agent)
                },
                timestamp=datetime.now().isoformat()
            )
            
            await self.mcp_client.send_message(collaboration_request)
            
            self.logger.info(f"Collaboration request sent to {target_agent}")
            
            return {
                "success": True,
                "message_id": collaboration_request.message_id,
                "target_agent": target_agent
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send collaboration request: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_trust_from_response(self, message: MCPMessage):
        """Update trust metrics based on agent response quality"""
        try:
            sender = message.sender_id
            payload = message.payload
            
            # Evaluate response quality (simplified heuristic)
            quality_score = 0.5  # Default neutral score
            
            if payload.get("success", False):
                quality_score += 0.3
            
            if "error" in payload:
                quality_score -= 0.2
            
            # Response time evaluation (if timestamp available)
            response_time = payload.get("response_time", 0)
            if response_time < 1.0:  # Fast response
                quality_score += 0.1
            elif response_time > 5.0:  # Slow response
                quality_score -= 0.1
            
            # Update trust score
            self._update_trust_score(self.agent_id, sender, quality_score * 0.1)
            
        except Exception as e:
            self.logger.error(f"Failed to update trust from response: {e}")
    
    async def _handle_context_update(self, context_id: str, updates: Dict[str, Any]):
        """Handle context updates from MCP"""
        try:
            if context_id == "agent_coordination":
                # Update local state based on coordination context changes
                if "trust_metrics" in updates:
                    self._merge_external_trust_metrics(updates["trust_metrics"])
                
                if "active_agents" in updates:
                    await self._update_active_agents(updates["active_agents"])
            
            self.logger.debug(f"Processed context update for {context_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle context update: {e}")
    
    async def _handle_agent_status_change(self, agent_id: str, status: str):
        """Handle agent status changes"""
        try:
            if status == "offline":
                # Penalize trust for agents that go offline unexpectedly
                await self._penalize_agent_trust(agent_id, "unexpected_offline")
            # Record agent coming back online - NO artificial boosts
            self.record_agent_interaction(
                agent_id=agent_id,
                interaction_type="agent_recovery",
                success=True,
                context={
                    "previous_status": "unavailable",
                    "current_status": "available",
                    "recovery_timestamp": time.time()
                }
            )
            
            self.logger.info(f"Handled status change for {agent_id}: {status}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle agent status change: {e}")
    
    async def _penalize_agent_trust(self, agent_id: str, reason: str):
        """Apply trust penalty to an agent"""
        penalty_map = {
            "mcp_error": -0.1,
            "unexpected_offline": -0.2,
            "failed_response": -0.15,
            "timeout": -0.1
        }
        
        penalty = penalty_map.get(reason, -0.05)
        self._update_trust_score(self.agent_id, agent_id, penalty)
        
        self.logger.warning(f"Applied trust penalty {penalty} to {agent_id} for {reason}")
    
    async def disconnect_mcp(self):
        """Disconnect from MCP server"""
        if self.mcp_client:
            try:
                await self.mcp_client.disconnect()
                self.logger.info("Disconnected from MCP server")
            except Exception as e:
                self.logger.error(f"Error disconnecting from MCP: {e}")
    
    def get_trust_score(self, agent_a: str, agent_b: str) -> float:
        """Get trust score from agent_a to agent_b"""
        if agent_a not in self.trust_scores:
            self.trust_scores[agent_a] = {}
        
        return self.trust_scores[agent_a].get(agent_b, 0.5)  # Default neutral trust
    
    def _update_trust_score(self, agent_a: str, agent_b: str, delta: float):
        """Update trust score between two agents"""
        if agent_a not in self.trust_scores:
            self.trust_scores[agent_a] = {}
        
        current_score = self.trust_scores[agent_a].get(agent_b, 0.5)
        new_score = max(0.0, min(1.0, current_score + delta))  # Clamp to [0, 1]
        
        self.trust_scores[agent_a][agent_b] = new_score
        
        self.logger.debug(f"Updated trust score {agent_a} -> {agent_b}: {current_score:.3f} + {delta:.3f} = {new_score:.3f}")
    
    def _merge_external_trust_metrics(self, external_metrics: Dict[str, Any]):
        """Merge external trust metrics from MCP"""
        try:
            for agent_pair, score in external_metrics.items():
                if "->" in agent_pair:
                    agent_a, agent_b = agent_pair.split("->")
                    agent_a, agent_b = agent_a.strip(), agent_b.strip()
                    
                    current_score = self.get_trust_score(agent_a, agent_b)
                    # Weighted average: 70% local, 30% external
                    merged_score = 0.7 * current_score + 0.3 * score
                    
                    if agent_a not in self.trust_scores:
                        self.trust_scores[agent_a] = {}
                    self.trust_scores[agent_a][agent_b] = merged_score
            
            self.logger.debug("Merged external trust metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to merge external trust metrics: {e}")
    
    async def _update_active_agents(self, active_agents: Dict[str, Any]):
        """Update information about active agents"""
        try:
            # Initialize trust scores for new agents
            for agent_id, agent_info in active_agents.items():
                if agent_id not in self.trust_scores:
                    self.trust_scores[agent_id] = {}
                
                # Set initial trust scores for new agents
                for other_agent in active_agents:
                    if other_agent != agent_id:
                        if other_agent not in self.trust_scores[agent_id]:
                            self.trust_scores[agent_id][other_agent] = 0.5  # Neutral initial trust
            
            self.logger.debug(f"Updated active agents: {list(active_agents.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to update active agents: {e}")
    
    async def _process_coordination_response(self, message: MCPMessage):
        """Process response for coordination purposes"""
        try:
            # Extract coordination-relevant information
            payload = message.payload
            
            if "coordination_success" in payload:
                # Update coordination metrics
                success = payload["coordination_success"]
                if success:
                    self._update_trust_score(self.agent_id, message.sender_id, 0.05)
                else:
                    self._update_trust_score(self.agent_id, message.sender_id, -0.05)
            
            self.logger.debug(f"Processed coordination response from {message.sender_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process coordination response: {e}")
    
    def get_trust_summary(self) -> Dict[str, Any]:
        """Get summary of current trust state"""
        summary = {
            "total_agents": len(self.trust_scores),
            "total_interactions": len(self.interaction_history),
            "average_trust_scores": {},
            "trust_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        # Calculate average trust scores for each agent
        for agent_id, scores in self.trust_scores.items():
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                summary["average_trust_scores"][agent_id] = avg_score
                
                # Categorize trust level
                if avg_score > 0.7:
                    summary["trust_distribution"]["high"] += 1
                elif avg_score > 0.4:
                    summary["trust_distribution"]["medium"] += 1
                else:
                    summary["trust_distribution"]["low"] += 1
        
        return summary
    
    async def broadcast_trust_update(self, agent_id: str, trust_change: float, reason: str):
        """Broadcast trust update to other agents via MCP"""
        if self.mcp_client:
            try:
                broadcast_message = MCPMessage(
                    message_id=f"trust_update_{int(datetime.now().timestamp())}",
                    message_type=MCPMessageType.NOTIFICATION,
                    sender_id=self.agent_id,
                    recipient_id="all",
                    payload={
                        "event": "trust_update",
                        "target_agent": agent_id,
                        "trust_change": trust_change,
                        "reason": reason,
                        "new_trust_level": self.get_trust_score(self.agent_id, agent_id)
                    },
                    timestamp=datetime.now().isoformat(),
                    context_id="agent_coordination"
                )
                
                await self.mcp_client.send_message(broadcast_message)
                self.logger.debug(f"Broadcasted trust update for {agent_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to broadcast trust update: {e}")

    def evaluate_agent_performance(self, agent_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate agent performance across trust dimensions
        
        Args:
            agent_id: ID of the agent to evaluate
            performance_data: Dictionary containing performance metrics
            
        Returns:
            Dict with trust evaluation results
        """
        try:
            # Extract performance data
            task_results = performance_data.get('task_results', [])
            perf_metrics = performance_data.get('performance_data', {})
            decision_history = performance_data.get('decision_history', [])
            attack_simulation = performance_data.get('attack_simulation', {})
            explanations = performance_data.get('explanations', [])
            
            # Calculate trust dimensions
            reliability = self.metrics_calculator.evaluate_reliability(agent_id, task_results)
            competence = self.metrics_calculator.evaluate_competence(agent_id, perf_metrics)
            fairness = self.metrics_calculator.evaluate_fairness(agent_id, decision_history)
            security = self.metrics_calculator.evaluate_security(agent_id, attack_simulation)
            transparency = self.metrics_calculator.evaluate_transparency(agent_id, explanations)
            
            # Create dimensions object
            dimensions = TrustDimensions(
                reliability=reliability,
                competence=competence,
                fairness=fairness,
                security=security,
                transparency=transparency
            )
            
            # Calculate overall score
            overall_score = dimensions.calculate_overall_score()
            
            return {
                'agent_id': agent_id,
                'reliability': reliability,
                'competence': competence,
                'fairness': fairness,
                'security': security,
                'transparency': transparency,
                'overall_score': overall_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating agent performance: {e}")
            # Return default good evaluation for new agents
            return {
                'agent_id': agent_id,
                'reliability': 0.7,
                'competence': 0.7,
                'fairness': 0.7,
                'security': 0.7,
                'transparency': 0.7,
                'overall_score': 0.7,  # Increased from 0.5 to 0.7
                'trust_level': 'medium',
                'risk_factors': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def update_trust_from_feedback(self, agent_id: str, feedback: Dict[str, Any]):
        """
        Update trust dimensions based on user or system feedback
        
        Args:
            agent_id: ID of the agent to update
            feedback: Dictionary with feedback metrics
        """
        try:
            # Extract feedback metrics
            satisfaction = feedback.get('satisfaction_score', 0.5)
            response_time = feedback.get('response_time', 3.0)
            explanation = feedback.get('explanation', '')
            
            # Calculate dimensional impacts
            reliability_impact = 0.1 if response_time < 2.0 else -0.05
            competence_impact = 0.2 * (satisfaction - 0.5)  # Scale to [-0.1, 0.1]
            transparency_impact = 0.05 if explanation else 0.0
            
            # Default neutral impacts for dimensions without direct feedback
            fairness_impact = 0.0
            security_impact = 0.0
            
            # Get current trust dimensions
            current_trust = self.get_trust_score(self.agent_id, agent_id)
            
            # Initialize default trust score if not exists
            if agent_id not in self.trust_scores:
                self.trust_scores[agent_id] = {}
            
            # Update trust score with this agent
            if self.agent_id not in self.trust_scores[agent_id]:
                self.trust_scores[agent_id][self.agent_id] = 0.5  # Initialize
            
            # Apply impacts to trust dimensions
            self._update_trust_dimension(agent_id, 'reliability', reliability_impact)
            self._update_trust_dimension(agent_id, 'competence', competence_impact)
            self._update_trust_dimension(agent_id, 'transparency', transparency_impact)
            
            self.logger.debug(f"Updated trust for {agent_id} based on feedback")
            
            # Record feedback in interaction history
            self.interaction_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'feedback': feedback,
                'trust_impact': {
                    'reliability': reliability_impact,
                    'competence': competence_impact,
                    'transparency': transparency_impact
                }
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update trust from feedback: {e}")
    
    def _update_trust_dimension(self, agent_id: str, dimension: str, impact: float):
        """Update a specific trust dimension with impact value"""
        try:
            # Create TrustRecord for the ledger
            dims = TrustDimensions()
            
            # Set the specific dimension
            if hasattr(dims, dimension):
                setattr(dims, dimension, max(0.0, min(1.0, 0.5 + impact)))
                
            # Create record
            record = TrustRecord(
                agent_id=agent_id,
                timestamp=datetime.now().isoformat(),
                dimensions=dims,
                evidence=f"Feedback impact on {dimension}: {impact}",
                task_id=f"feedback_{int(datetime.now().timestamp())}",
                performance_metrics={dimension: impact}
            )
            
            # Add to ledger
            self.trust_ledger.add_record(record)
            
        except Exception as e:
            self.logger.error(f"Error updating trust dimension: {e}")

# Global trust orchestrator instance
trust_orchestrator = TrustManager()

def get_trust_evaluation(agent_id: str, operation_type: str = 'general') -> Dict[str, Any]:
    """Get trust evaluation for an agent"""
    try:
        return trust_orchestrator.evaluate_agent_performance(agent_id, {'task_results': [], 'performance_data': {}, 'decision_history': [], 'attack_simulation': {}, 'explanations': []})
    except Exception as e:
        logger.error(f"Error getting trust evaluation for {agent_id}: {e}")
        # Return default good evaluation for new agents
        return {
            'agent_id': agent_id,
            'reliability': 0.7,
            'competence': 0.7,
            'fairness': 0.7,
            'security': 0.7,
            'transparency': 0.7,
            'overall_score': 0.7,  # Increased from 0.5 to 0.7 for better initial trust
            'trust_level': 'medium',
            'risk_factors': [],
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def record_agent_outcome(agent_id: str, operation_type: str, success: bool, 
                        details: Dict[str, Any], user_feedback: Optional[str] = None):
    """Record agent operation outcome for trust learning"""
    try:
        trust_orchestrator.update_trust_from_feedback(agent_id, {
            'satisfaction_score': 1.0 if success else 0.0,
            'response_time': details.get('response_time', 3.0),
            'explanation': user_feedback
        })
    except Exception as e:
        logger.error(f"Error recording agent outcome for {agent_id}: {e}")
        # If we can't update trust through the normal mechanism,
        # update the basic trust score directly
        if agent_id not in trust_orchestrator.trust_scores:
            trust_orchestrator.trust_scores[agent_id] = {}
        
        if "trust_manager" not in trust_orchestrator.trust_scores[agent_id]:
            trust_orchestrator.trust_scores[agent_id]["trust_manager"] = 0.5
        
        # Apply a simple update based on success
        delta = 0.05 if success else -0.05
        current = trust_orchestrator.trust_scores[agent_id]["trust_manager"]
        trust_orchestrator.trust_scores[agent_id]["trust_manager"] = max(0.0, min(1.0, current + delta)) 