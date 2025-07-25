# MAMA_exp/core/trust_adaptive_interaction.py

"""
Trust-Aware Adaptive Interaction Protocols

Core module implementing dynamic agent interaction rules based on mutual trust levels.
High-trust interactions use simplified protocols, low-trust interactions trigger strict auditing.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    """Trust level enumeration for agent interactions"""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class InteractionProtocol(Enum):
    """Available interaction protocols"""
    SIMPLIFIED = "simplified"
    STANDARD = "standard"
    STRICT_AUDIT = "strict_audit"

@dataclass
class InteractionState:
    """State object for agent interaction protocols"""
    agent_id: str
    trust_score: float
    trust_level: TrustLevel
    protocol: InteractionProtocol
    last_update: datetime
    transition_count: int = 0

@dataclass 
class ProtocolMetrics:
    """Metrics for protocol performance tracking"""
    protocol_type: str
    execution_time: float
    data_volume: int
    verification_steps: int
    success_rate: float

class TrustAdaptiveInteractionManager:
    """
    Manages trust-aware adaptive interaction protocols between agents.
    
    Implements dynamic protocol adjustment based on trust scores with
    state machine management and feedback loops for continuous optimization.
    """
    
    def __init__(self):
        """Initialize the interaction manager"""
        self.agent_states: Dict[str, InteractionState] = {}
        self.protocol_metrics: List[ProtocolMetrics] = []
        self.trust_thresholds = {
            TrustLevel.HIGH: 0.8,
            TrustLevel.MEDIUM: 0.5,
            TrustLevel.LOW: 0.0
        }
        self.protocol_mapping = {
            TrustLevel.HIGH: InteractionProtocol.SIMPLIFIED,
            TrustLevel.MEDIUM: InteractionProtocol.STANDARD,
            TrustLevel.LOW: InteractionProtocol.STRICT_AUDIT
        }
        
    def update_agent_trust(self, agent_id: str, trust_score: float) -> bool:
        """
        Update agent trust score and adjust interaction protocol if needed
        
        Args:
            agent_id: Agent identifier
            trust_score: New trust score (0.0-1.0)
            
        Returns:
            True if protocol changed, False otherwise
        """
        try:
            new_trust_level = self._determine_trust_level(trust_score)
            new_protocol = self.protocol_mapping[new_trust_level]
            
            current_state = self.agent_states.get(agent_id)
            protocol_changed = False
            
            if current_state is None:
                # First time initialization
                self.agent_states[agent_id] = InteractionState(
                    agent_id=agent_id,
                    trust_score=trust_score,
                    trust_level=new_trust_level,
                    protocol=new_protocol,
                    last_update=datetime.now()
                )
                protocol_changed = True
                logger.info(f"Initialized agent {agent_id} with {new_trust_level.value} trust level")
                
            elif current_state.trust_level != new_trust_level:
                # Trust level changed - update protocol
                old_protocol = current_state.protocol
                current_state.trust_score = trust_score
                current_state.trust_level = new_trust_level
                current_state.protocol = new_protocol
                current_state.last_update = datetime.now()
                current_state.transition_count += 1
                protocol_changed = True
                
                logger.info(f"Agent {agent_id} trust transition: {old_protocol.value} -> {new_protocol.value}")
                
            else:
                # Same trust level - just update score
                current_state.trust_score = trust_score
                current_state.last_update = datetime.now()
                
            return protocol_changed
            
        except Exception as e:
            logger.error(f"Error updating agent trust for {agent_id}: {e}")
            return False
    
    def get_interaction_protocol(self, agent_id: str) -> InteractionProtocol:
        """
        Get current interaction protocol for an agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Current interaction protocol
        """
        state = self.agent_states.get(agent_id)
        if state is None:
            logger.warning(f"No state found for agent {agent_id}, using strict audit protocol")
            return InteractionProtocol.STRICT_AUDIT
        return state.protocol
    
    def execute_interaction(self, sender_id: str, receiver_id: str, 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute interaction between two agents using appropriate protocol
        
        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID  
            data: Data to be transmitted
            
        Returns:
            Processed interaction result with metrics
        """
        start_time = time.time()
        
        # Determine interaction protocol based on receiver's trust level
        protocol = self.get_interaction_protocol(receiver_id)
        
        try:
            if protocol == InteractionProtocol.SIMPLIFIED:
                result = self._execute_simplified_protocol(sender_id, receiver_id, data)
            elif protocol == InteractionProtocol.STANDARD:
                result = self._execute_standard_protocol(sender_id, receiver_id, data)
            else:  # STRICT_AUDIT
                result = self._execute_strict_audit_protocol(sender_id, receiver_id, data)
            
            # Record protocol metrics
            execution_time = time.time() - start_time
            metrics = ProtocolMetrics(
                protocol_type=protocol.value,
                execution_time=execution_time,
                data_volume=len(json.dumps(data).encode('utf-8')),
                verification_steps=result.get('verification_steps', 0),
                success_rate=1.0 if result.get('success') else 0.0
            )
            self.protocol_metrics.append(metrics)
            
            logger.debug(f"Interaction {sender_id}->{receiver_id} completed using {protocol.value} protocol")
            return result
            
        except Exception as e:
            logger.error(f"Error during interaction {sender_id}->{receiver_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'protocol': protocol.value,
                'data': {}
            }
    
    def _execute_simplified_protocol(self, sender_id: str, receiver_id: str, 
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute simplified high-trust interaction protocol
        
        - Direct data sharing without abstraction
        - Minimal verification steps
        - Designed for speed
        """
        try:
            # Basic format validation only
            if not isinstance(data, dict):
                raise ValueError("Data must be dictionary format")
            
            # Direct transmission with minimal processing
            processed_data = data.copy()
            processed_data['protocol_info'] = {
                'type': 'simplified',
                'sender': sender_id,
                'timestamp': datetime.now().isoformat(),
                'trust_level': 'high'
            }
            
            return {
                'success': True,
                'data': processed_data,
                'protocol': 'simplified',
                'verification_steps': 1,
                'processing_time': 0.001  # Minimal processing
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'protocol': 'simplified',
                'data': {}
            }
    
    def _execute_standard_protocol(self, sender_id: str, receiver_id: str,
                                 data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute standard medium-trust interaction protocol
        
        - Partial data abstraction
        - Lightweight verification
        - Balance between security and efficiency
        """
        try:
            # Format and source validation
            if not isinstance(data, dict):
                raise ValueError("Data must be dictionary format")
            
            # Data abstraction - create summary for sensitive fields
            abstracted_data = self._abstract_sensitive_data(data)
            
            # Lightweight verification
            verification_passed = self._verify_data_consistency(abstracted_data)
            if not verification_passed:
                logger.warning(f"Data consistency check failed for {sender_id}->{receiver_id}")
            
            # Add protocol metadata
            abstracted_data['protocol_info'] = {
                'type': 'standard',
                'sender': sender_id,
                'timestamp': datetime.now().isoformat(),
                'trust_level': 'medium',
                'verification_passed': verification_passed
            }
            
            return {
                'success': True,
                'data': abstracted_data,
                'protocol': 'standard',
                'verification_steps': 2,
                'processing_time': 0.005
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'protocol': 'standard',
                'data': {}
            }
    
    def _execute_strict_audit_protocol(self, sender_id: str, receiver_id: str,
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strict audit low-trust interaction protocol
        
        - Multiple verification rounds
        - Data integrity checks
        - Anomaly detection
        - Minimal data exposure
        """
        try:
            # Comprehensive validation
            validation_results = []
            
            # Step 1: Format validation
            if not isinstance(data, dict):
                raise ValueError("Data must be dictionary format")
            validation_results.append("format_check_passed")
            
            # Step 2: Source verification
            source_verified = self._verify_source_authenticity(sender_id, data)
            validation_results.append(f"source_verification_{source_verified}")
            
            # Step 3: Data integrity check
            integrity_passed = self._check_data_integrity(data)
            validation_results.append(f"integrity_check_{integrity_passed}")
            
            # Step 4: Anomaly detection
            anomaly_detected = self._detect_anomalies(data)
            validation_results.append(f"anomaly_detection_{not anomaly_detected}")
            
            # Step 5: Data minimization
            minimal_data = self._minimize_data_exposure(data)
            
            # Step 6: Encryption for sensitive fields (simulated)
            encrypted_data = self._encrypt_sensitive_fields(minimal_data)
            
            # Add comprehensive audit trail
            encrypted_data['protocol_info'] = {
                'type': 'strict_audit',
                'sender': sender_id,
                'timestamp': datetime.now().isoformat(),
                'trust_level': 'low',
                'validation_results': validation_results,
                'security_level': 'high'
            }
            
            all_checks_passed = all('_True' in result or '_passed' in result for result in validation_results)
            
            return {
                'success': all_checks_passed,
                'data': encrypted_data,
                'protocol': 'strict_audit',
                'verification_steps': 6,
                'processing_time': 0.02,
                'audit_trail': validation_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'protocol': 'strict_audit',
                'data': {},
                'audit_trail': ['error_occurred']
            }
    
    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """Determine trust level based on score"""
        if trust_score >= self.trust_thresholds[TrustLevel.HIGH]:
            return TrustLevel.HIGH
        elif trust_score >= self.trust_thresholds[TrustLevel.MEDIUM]:
            return TrustLevel.MEDIUM
        else:
            return TrustLevel.LOW
    
    def _abstract_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create abstracted version of data for medium trust interactions"""
        abstracted = data.copy()
        
        # Abstract potentially sensitive fields
        sensitive_fields = ['price', 'total_cost', 'personal_info', 'api_key']
        for field in sensitive_fields:
            if field in abstracted:
                if isinstance(abstracted[field], (int, float)):
                    # Convert to range
                    value = abstracted[field]
                    abstracted[field] = f"range_{int(value//100)*100}-{int(value//100)*100+100}"
                else:
                    abstracted[field] = "abstracted"
        
        return abstracted
    
    def _verify_data_consistency(self, data: Dict[str, Any]) -> bool:
        """Verify data consistency for medium trust interactions"""
        try:
            # Basic consistency checks
            required_fields = ['timestamp', 'sender']
            for field in required_fields:
                if field not in data.get('protocol_info', {}):
                    return False
            
            # Check for reasonable data ranges
            if 'price' in data and isinstance(data['price'], (int, float)):
                if data['price'] < 0 or data['price'] > 10000:
                    return False
            
            return True
        except:
            return False
    
    def _verify_source_authenticity(self, sender_id: str, data: Dict[str, Any]) -> bool:
        """Verify the authenticity of data source for strict audit"""
        try:
            # Check if sender_id is in approved agents list
            approved_agents = ['WeatherAgent', 'SafetyAgent', 'EconomicAgent', 'FlightInfoAgent']
            if sender_id not in approved_agents:
                return False
            
            # Verify data structure matches expected format for sender
            expected_structures = {
                'WeatherAgent': ['weather_safety_score', 'weather_condition'],
                'SafetyAgent': ['overall_safety_score', 'risk_factors'],
                'EconomicAgent': ['total_cost', 'value_for_money_score'],
                'FlightInfoAgent': ['flight_number', 'airline']
            }
            
            if sender_id in expected_structures:
                expected_fields = expected_structures[sender_id]
                for field in expected_fields:
                    if field not in data:
                        return False
            
            return True
        except:
            return False
    
    def _check_data_integrity(self, data: Dict[str, Any]) -> bool:
        """Check data integrity for strict audit protocol"""
        try:
            # Check for data completeness
            if not data or len(data) == 0:
                return False
            
            # Check for suspicious patterns
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 10000:
                    return False  # Suspiciously large string
                if isinstance(value, (int, float)) and (value < -999999 or value > 999999):
                    return False  # Suspiciously large number
            
            return True
        except:
            return False
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> bool:
        """Detect anomalies in data for strict audit protocol"""
        try:
            anomalies_detected = False
            
            # Check for unusual patterns
            if 'flights' in data and isinstance(data['flights'], list):
                for flight in data['flights']:
                    # Check for impossible values
                    if 'price' in flight and isinstance(flight['price'], (int, float)):
                        if flight['price'] < 10 or flight['price'] > 50000:
                            anomalies_detected = True
                    
                    # Check for duplicate flight numbers
                    flight_numbers = [f.get('flight_number') for f in data['flights'] if f.get('flight_number')]
                    if len(flight_numbers) != len(set(flight_numbers)):
                        anomalies_detected = True
            
            return anomalies_detected
        except:
            return True  # Assume anomaly if check fails
    
    def _minimize_data_exposure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimize data exposure for strict audit protocol"""
        minimal_data = {}
        
        # Only include essential fields
        essential_fields = ['flights', 'status', 'total_flights', 'timestamp']
        for field in essential_fields:
            if field in data:
                minimal_data[field] = data[field]
        
        # For flights, only include critical information
        if 'flights' in minimal_data and isinstance(minimal_data['flights'], list):
            for flight in minimal_data['flights']:
                # Remove sensitive personal or detailed financial data
                sensitive_keys = ['passenger_details', 'payment_info', 'api_raw_data']
                for key in sensitive_keys:
                    flight.pop(key, None)
        
        return minimal_data
    
    def _encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate encryption of sensitive fields for strict audit"""
        encrypted_data = data.copy()
        
        # Mark sensitive fields as encrypted (in real implementation, use actual encryption)
        sensitive_fields = ['total_cost', 'price', 'personal_info']
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = f"encrypted_{hash(str(encrypted_data[field])) % 10000}"
        
        return encrypted_data
    
    def get_protocol_analytics(self) -> Dict[str, Any]:
        """
        Get analytics on protocol performance
        
        Returns:
            Analytics dictionary with performance metrics
        """
        if not self.protocol_metrics:
            return {
                'total_interactions': 0,
                'average_execution_time': 0.0,
                'protocol_distribution': {},
                'success_rate': 0.0
            }
        
        total_interactions = len(self.protocol_metrics)
        avg_execution_time = sum(m.execution_time for m in self.protocol_metrics) / total_interactions
        success_rate = sum(m.success_rate for m in self.protocol_metrics) / total_interactions
        
        # Protocol usage distribution
        protocol_counts = {}
        for metric in self.protocol_metrics:
            protocol_counts[metric.protocol_type] = protocol_counts.get(metric.protocol_type, 0) + 1
        
        return {
            'total_interactions': total_interactions,
            'average_execution_time': avg_execution_time,
            'protocol_distribution': protocol_counts,
            'success_rate': success_rate,
            'total_data_volume': sum(m.data_volume for m in self.protocol_metrics)
        }

    async def update_with_feedback(self, query_id: str, selected_agents: list, 
                                  collaboration_result: Dict[str, Any], 
                                  integration_result: Dict[str, Any]) -> None:
        """
        Update trust adaptive system based on query processing feedback
        
        Args:
            query_id: Query identifier
            selected_agents: List of selected agents
            collaboration_result: Result from agent collaboration
            integration_result: Result from decision integration
        """
        try:
            # Extract agent performance metrics from results
            agent_outputs = collaboration_result.get('agent_outputs', {})
            coordination_metrics = collaboration_result.get('coordination_metrics', {})
            trust_scores = coordination_metrics.get('trust_scores', {})
            
            # Update trust scores for each agent based on performance
            for agent_tuple in selected_agents:
                if isinstance(agent_tuple, tuple) and len(agent_tuple) >= 3:
                    agent_id, similarity_score, trust_score = agent_tuple[0], agent_tuple[1], agent_tuple[2]
                    
                    # Get agent output performance
                    agent_output = agent_outputs.get(agent_id, {})
                    success = agent_output.get('success', True)
                    confidence = agent_output.get('confidence', 0.5)
                    
                    # Calculate trust adjustment based on performance
                    performance_factor = 1.0 if success else 0.7
                    confidence_factor = min(1.2, max(0.8, confidence + 0.2))
                    
                    # Update agent trust with feedback
                    new_trust_score = min(1.0, trust_score * performance_factor * confidence_factor)
                    self.update_agent_trust(agent_id, new_trust_score)
            
            logger.info(f"Trust adaptive system updated for query {query_id}")
            
        except Exception as e:
            logger.error(f"Failed to update trust adaptive system: {e}")
    
    def get_agent_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of agent interaction states"""
        summary = {}
        for agent_id, state in self.agent_states.items():
            summary[agent_id] = {
                "trust_score": state.trust_score,
                "trust_level": state.trust_level.value,
                "current_protocol": state.protocol.value,
                "last_update": state.last_update.isoformat(),
                "transition_count": state.transition_count
            }
        
        return summary

# Global instance for system-wide use
trust_interaction_manager = TrustAdaptiveInteractionManager()

def update_agent_trust_level(agent_id: str, trust_score: float) -> bool:
    """Convenience function to update agent trust level"""
    return trust_interaction_manager.update_agent_trust(agent_id, trust_score)

def execute_trusted_interaction(sender_id: str, receiver_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to execute trusted interaction between agents"""
    return trust_interaction_manager.execute_interaction(sender_id, receiver_id, data)

def get_agent_protocol(agent_id: str) -> str:
    """Convenience function to get agent's current interaction protocol"""
    return trust_interaction_manager.get_interaction_protocol(agent_id).value 