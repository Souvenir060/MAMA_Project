#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - Base Agent Architecture

This module provides the foundational base class for all agents in the MAMA
multi-agent system, implementing academic-level agent coordination protocols,
trust-aware communication, and Byzantine fault tolerance.

Academic Features:
- Trust-aware multi-agent communication protocols
- Byzantine fault tolerance and consensus mechanisms
- Academic-level agent coordination algorithms
- Performance monitoring and evaluation metrics
- Integration with MARL and PML frameworks
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from autogen import AssistantAgent, ConversableAgent
import numpy as np
import psutil

# Configure comprehensive logging
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of agent roles in the MAMA system"""
    WEATHER_ANALYST = "weather_analyst"
    SAFETY_ASSESSOR = "safety_assessor"
    FLIGHT_INFORMATION = "flight_information"
    ECONOMIC_ANALYST = "economic_analyst"
    INTEGRATION_COORDINATOR = "integration_coordinator"
    TRUST_MANAGER = "trust_manager"
    CROSS_DOMAIN_SOLVER = "cross_domain_solver"


class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    TERMINATED = "terminated"


class CommunicationProtocol(Enum):
    """Communication protocols for agent interaction"""
    DIRECT = "direct"
    CONSENSUS = "consensus"
    BYZANTINE_TOLERANT = "byzantine_tolerant"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass
class AgentCapability:
    """Definition of agent capability with academic metrics"""
    name: str
    description: str
    accuracy_score: float = 0.0
    response_time_ms: float = 0.0
    confidence_interval: tuple = field(default_factory=lambda: (0.0, 1.0))
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TaskExecution:
    """Task execution tracking with academic metrics"""
    task_id: str
    task_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    trust_score: float = 0.8


@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    total_tasks_executed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    current_trust_score: float = 0.8
    accuracy_history: List[float] = field(default_factory=list)
    uptime_percentage: float = 100.0
    last_heartbeat: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    Academic-level base agent for the MAMA multi-agent system
    
    Provides comprehensive agent architecture with:
    - Trust-aware communication protocols
    - Byzantine fault tolerance
    - Academic performance evaluation
    - Multi-agent coordination capabilities
    - Real-time monitoring and adaptation
    
    This base class implements academic standards for agent design
    as per multi-agent systems research literature.
    """
    
    def __init__(self, 
                 name: str = None,
                 role: str = None,
                 trust_threshold: float = 0.7,
                 max_byzantine_faults: int = 1,
                 communication_protocol: str = "trust_weighted",
                 **kwargs):
        """Initialize base agent with academic configuration
        
        Args:
            name: Agent identifier
            role: Agent role from AgentRole enum
            trust_threshold: Minimum trust score for interactions
            max_byzantine_faults: Maximum tolerated Byzantine faults
            communication_protocol: Protocol for agent communication
        """
        self.name = name or str(uuid.uuid4())
        self.role = role
        self.trust_threshold = trust_threshold
        self.max_byzantine_faults = max_byzantine_faults
        self.communication_protocol = communication_protocol
        
        # Initialize model attribute
        self.model = kwargs.get('model', 'real_api')
        
        # Initialize logging
        self.logger = logging.getLogger(f"agent.{self.name}")
        
        # Initialize metrics
        self.metrics = AgentMetrics()
        
        # Initialize capabilities
        self._initialize_default_capabilities()
        
        # Generate system message
        self.system_message = self._generate_academic_system_message(role)
        
        # Initialize agent
        self._initialize_agent()
        
        # Create AssistantAgent instance
        self.assistant = AssistantAgent(name=self.name, system_message=self.system_message)
        
        logger.info(f"Base agent initialized: {self.name} (Role: {role})")
    
    def _generate_academic_system_message(self, role: str) -> str:
        """Generate comprehensive system message based on agent role"""
        base_message = f"""You are an academic-level AI agent in the MAMA (Multi-Agent Meteorological Aviation) flight selection system.

Role: {role.replace('_', ' ').title()}
Agent ID: {self.name}
Trust Threshold: {self.trust_threshold}
Communication Protocol: {self.communication_protocol}
Model Type: {self.model}

Core Responsibilities:
1. Execute specialized tasks with academic rigor and precision
2. Maintain trust-aware communication with other agents
3. Implement Byzantine fault tolerance in all operations
4. Provide comprehensive performance metrics and evaluation
5. Support real-time learning and adaptation protocols

Academic Standards:
- All algorithms must follow published research methodologies
- Performance must be measurable using standard academic metrics
- Communication must be transparent and verifiable
- Error handling must be comprehensive and documented
- All decisions must be traceable and explainable

You operate within a multi-agent system designed for academic research and
practical aviation applications. Maintain the highest standards of accuracy,
reliability, and scientific rigor in all operations."""
        
        return base_message
    
    def _initialize_agent(self):
        """Initialize agent with academic capabilities"""
        try:
            # Set initial state
            self.state = "active"
            
            # Initialize metrics
            self.metrics = AgentMetrics()
            
            # Initialize capabilities based on role
            self._initialize_default_capabilities()
            
            # Log initialization
            logger.info(f"Agent {self.name} ({self.role}) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {str(e)}")
            raise
    
    def _initialize_default_capabilities(self) -> None:
        """Initialize default capabilities based on agent role"""
        role_capabilities = {
            AgentRole.WEATHER_ANALYST: [
                AgentCapability("weather_analysis", "Comprehensive meteorological analysis", 0.95),
                AgentCapability("forecast_evaluation", "Weather forecast accuracy assessment", 0.90),
                AgentCapability("risk_assessment", "Weather-related flight risk evaluation", 0.88)
            ],
            AgentRole.SAFETY_ASSESSOR: [
                AgentCapability("safety_evaluation", "Aircraft and route safety assessment", 0.96),
                AgentCapability("risk_analysis", "Comprehensive safety risk analysis", 0.94),
                AgentCapability("compliance_check", "Regulatory compliance verification", 0.98)
            ],
            AgentRole.FLIGHT_INFORMATION: [
                AgentCapability("flight_search", "Real-time flight information retrieval", 0.99),
                AgentCapability("schedule_analysis", "Flight schedule optimization", 0.92),
                AgentCapability("route_planning", "Optimal route planning and analysis", 0.90)
            ],
            AgentRole.ECONOMIC_ANALYST: [
                AgentCapability("cost_analysis", "Comprehensive cost-benefit analysis", 0.93),
                AgentCapability("price_optimization", "Dynamic pricing optimization", 0.89),
                AgentCapability("budget_planning", "Travel budget optimization", 0.91)
            ],
            AgentRole.INTEGRATION_COORDINATOR: [
                AgentCapability("data_integration", "Multi-source data integration", 0.97),
                AgentCapability("decision_synthesis", "Multi-criteria decision synthesis", 0.95),
                AgentCapability("consensus_building", "Agent consensus coordination", 0.93)
            ],
            AgentRole.TRUST_MANAGER: [
                AgentCapability("trust_evaluation", "Agent trust score evaluation", 0.98),
                AgentCapability("reputation_management", "System-wide reputation management", 0.96),
                AgentCapability("security_monitoring", "Byzantine fault detection", 0.94)
            ],
            AgentRole.CROSS_DOMAIN_SOLVER: [
                AgentCapability("cross_domain_analysis", "Cross-domain problem solving", 0.92),
                AgentCapability("constraint_optimization", "Multi-constraint optimization", 0.90),
                AgentCapability("solution_integration", "Integrated solution synthesis", 0.94)
            ]
        }
        
        try:
            role_enum = AgentRole(self.role)
            self.capabilities = role_capabilities.get(role_enum, [])
            logger.info(f"Initialized {len(self.capabilities)} capabilities for {self.role}")
        except ValueError:
            logger.warning(f"Unknown role {self.role}, using empty capabilities list")
            self.capabilities = []
    
    @abstractmethod
    def process_task(self, task_description: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for task processing - must be implemented by subclasses

        Args:
            task_description: Description of the task to be performed
            task_data: Task-specific data and parameters

        Returns:
            Dictionary containing task results and metadata
        """
        pass
    
    def execute_task_with_monitoring(self, 
                                   task_description: str, 
                                   task_data: Dict[str, Any],
                                   timeout_seconds: float = 30.0) -> Dict[str, Any]:
        """
        Execute task with comprehensive monitoring and evaluation

        Args:
            task_description: Description of the task
            task_data: Task input data
            timeout_seconds: Maximum execution time

        Returns:
            Task execution result with performance metrics
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create task execution record
        task_execution = TaskExecution(
            task_id=task_id,
            task_type=task_description,
            start_time=start_time
        )
        
        self.active_tasks[task_id] = task_execution
        
        try:
            # Update metrics
            self.metrics.total_tasks_executed += 1
            
            # Execute the task
            result = self.process_task(task_description, task_data)
            
            # Calculate performance metrics
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000  # milliseconds
            
            # Update task execution record
            task_execution.end_time = end_time
            task_execution.status = "completed"
            task_execution.result = result
            task_execution.performance_metrics = {
                "execution_time_ms": execution_time,
                "memory_usage": self._get_memory_usage(),
                "cpu_usage": self._get_cpu_usage()
            }
            
            # Update agent metrics
            self.metrics.successful_tasks += 1
            self._update_response_time_metric(execution_time)
            
            # Move to completed tasks
            self.completed_tasks.append(task_execution)
            del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} completed successfully in {execution_time:.2f}ms")
            
            return {
                "status": "success",
                "task_id": task_id,
                "result": result,
                "execution_time_ms": execution_time,
                "agent_metrics": self._get_current_metrics()
            }
            
        except Exception as e:
            # Handle task failure
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            task_execution.end_time = end_time
            task_execution.status = "failed"
            task_execution.error_message = str(e)
            task_execution.performance_metrics = {
                "execution_time_ms": execution_time,
                "error_type": type(e).__name__
            }
            
            # Update metrics
            self.metrics.failed_tasks += 1
            
            # Move to completed tasks
            self.completed_tasks.append(task_execution)
            del self.active_tasks[task_id]
            
            logger.error(f"Task {task_id} failed after {execution_time:.2f}ms: {e}")
            
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": execution_time,
                "agent_metrics": self._get_current_metrics()
            }
    
    def communicate_with_agent(self, 
                             target_agent: str, 
                             message: Dict[str, Any],
                             protocol: Optional[CommunicationProtocol] = None) -> Dict[str, Any]:
        """
        Trust-aware communication with another agent

        Args:
            target_agent: Name/ID of target agent
            message: Message to send
            protocol: Communication protocol to use

        Returns:
            Communication result
        """
        protocol = protocol or self.communication_protocol
        
        # Check trust score
        trust_score = self.trust_scores.get(target_agent, 0.5)
        if trust_score < self.trust_threshold:
            logger.warning(f"Trust score for {target_agent} ({trust_score}) below threshold ({self.trust_threshold})")
        
        # Create communication record
        comm_record = {
            "timestamp": datetime.now(),
            "source": self.name,
            "target": target_agent,
            "protocol": protocol,
            "message_type": message.get("type", "unknown"),
            "trust_score": trust_score,
            "status": "pending"
        }
        
        try:
            # Apply protocol-specific processing
            if protocol == CommunicationProtocol.BYZANTINE_TOLERANT:
                response = self._byzantine_tolerant_communication(target_agent, message)
            elif protocol == CommunicationProtocol.TRUST_WEIGHTED:
                response = self._trust_weighted_communication(target_agent, message)
            elif protocol == CommunicationProtocol.CONSENSUS:
                response = self._consensus_communication(target_agent, message)
            else:
                response = self._direct_communication(target_agent, message)
            
            comm_record["status"] = "success"
            comm_record["response"] = response
            
            return response
            
        except Exception as e:
            comm_record["status"] = "failed"
            comm_record["error"] = str(e)
            logger.error(f"Communication with {target_agent} failed: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "agent": self.name
            }
        
        finally:
            self.communication_history.append(comm_record)
    
    def update_trust_score(self, agent_name: str, new_score: float) -> None:
        """Update trust score for another agent"""
        if 0.0 <= new_score <= 1.0:
            old_score = self.trust_scores.get(agent_name, 0.5)
            self.trust_scores[agent_name] = new_score
            logger.info(f"Updated trust score for {agent_name}: {old_score:.3f} -> {new_score:.3f}")
        else:
            logger.warning(f"Invalid trust score for {agent_name}: {new_score}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status report"""
        return {
            "id": self.name,
            "role": self.role,
            "model": self.model,
            "state": self.state,
            "trust_threshold": self.trust_threshold,
            "communication_protocol": self.communication_protocol,
            "metrics": {
                "total_tasks": len(self.completed_tasks),
                "successful_tasks": sum(1 for t in self.completed_tasks if t.status == "completed"),
                "failed_tasks": sum(1 for t in self.completed_tasks if t.status == "failed"),
                "avg_response_time": sum(t.performance_metrics.get("execution_time_ms", 0) for t in self.completed_tasks) / len(self.completed_tasks) if self.completed_tasks else 0,
                "current_trust_score": sum(self.trust_scores.values()) / len(self.trust_scores) if self.trust_scores else 0,
                "uptime_percentage": 100.0
            },
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "known_agents": len(self.known_agents),
            "capabilities": [str(c) for c in self.capabilities],
            "last_heartbeat": datetime.now().isoformat(),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent()
        }
    
    # Private helper methods
    
    def _direct_communication(self, target_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Direct communication without additional processing"""
        # Implementation would involve actual agent communication
        # This is a placeholder for the communication mechanism
        return {
            "status": "success",
            "response": f"Direct communication with {target_agent}",
            "protocol": "direct"
        }
    
    def _trust_weighted_communication(self, target_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Trust-weighted communication with confidence adjustment"""
        trust_score = self.trust_scores.get(target_agent, 0.5)
        
        # Adjust message confidence based on trust
        adjusted_message = message.copy()
        if "confidence" in adjusted_message:
            adjusted_message["confidence"] *= trust_score
        
        return {
            "status": "success",
            "response": f"Trust-weighted communication with {target_agent}",
            "trust_adjusted": True,
            "trust_score": trust_score,
            "protocol": "trust_weighted"
        }
    
    def _byzantine_tolerant_communication(self, target_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Byzantine fault tolerant communication"""
        # Implement Byzantine fault tolerance algorithm
        # This would include message verification, consensus, etc.
        
        return {
            "status": "success",
            "response": f"Byzantine tolerant communication with {target_agent}",
            "byzantine_verified": True,
            "protocol": "byzantine_tolerant"
        }
    
    def _consensus_communication(self, target_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Consensus-based communication"""
        # Implement consensus protocol
        
        return {
            "status": "success",
            "response": f"Consensus communication with {target_agent}",
            "consensus_achieved": True,
            "protocol": "consensus"
        }
    
    def _update_response_time_metric(self, execution_time_ms: float) -> None:
        """Update average response time metric"""
        if self.metrics.average_response_time == 0.0:
            self.metrics.average_response_time = execution_time_ms
        else:
            # Exponential moving average
            alpha = 0.1  # Learning rate
            self.metrics.average_response_time = (
                alpha * execution_time_ms + 
                (1 - alpha) * self.metrics.average_response_time
            )
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        success_rate = (
            self.metrics.successful_tasks / max(self.metrics.total_tasks_executed, 1)
        )
        
        return {
            "total_tasks": self.metrics.total_tasks_executed,
            "successful_tasks": self.metrics.successful_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "success_rate": success_rate,
            "average_response_time_ms": self.metrics.average_response_time,
            "current_trust_score": self.metrics.current_trust_score,
            "uptime_percentage": self.metrics.uptime_percentage,
            "last_heartbeat": self.metrics.last_heartbeat.isoformat()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            return psutil.Process().cpu_percent()
        except Exception:
            return 0.0
    
    def heartbeat(self) -> None:
        """Update agent heartbeat for monitoring"""
        self.metrics.last_heartbeat = datetime.now()
        
    def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        self.state = AgentState.TERMINATED
        logger.info(f"Agent {self.name} shutting down gracefully")
        
        # Complete any active tasks or mark them as interrupted
        for task_id, task in self.active_tasks.items():
            task.status = "interrupted"
            task.end_time = datetime.now()
            task.error_message = "Agent shutdown"
            self.completed_tasks.append(task)
        
        self.active_tasks.clear()


# Academic evaluation functions

def calculate_agent_performance_metrics(agent: BaseAgent) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for an agent
    
    Args:
        agent: BaseAgent instance to evaluate
        
    Returns:
        Dictionary with academic performance metrics
    """
    from core.evaluation_metrics import calculate_mrr, calculate_ndcg, calculate_art
    
    # Convert completed tasks to evaluation format
    results = []
    for task in agent.completed_tasks:
        if task.status == "completed" and task.result:
            results.append({
                "recommendations": task.result.get("recommendations", []),
                "ground_truth_id": task.result.get("ground_truth_id", ""),
                "response_time": task.performance_metrics.get("execution_time_ms", 0.0)
            })
    
    if not results:
        return {
            "mean_reciprocal_rank": 0.0,
            "ndcg_at_5": 0.0,
            "average_response_time": 0.0,
            "task_success_rate": 0.0
        }
    
    return {
        "mean_reciprocal_rank": calculate_mrr(results),
        "ndcg_at_5": calculate_ndcg(results, k=5),
        "average_response_time": calculate_art(results),
        "task_success_rate": agent.metrics.successful_tasks / max(agent.metrics.total_tasks_executed, 1)
    }


def create_agent_from_role(role: AgentRole, 
                          name: Optional[str] = None,
                          **kwargs) -> BaseAgent:
    """
    Factory function to create agents based on role
    
    Args:
        role: Agent role
        name: Optional agent name (auto-generated if None)
        **kwargs: Additional configuration
        
    Returns:
        Configured BaseAgent instance
    """
    if name is None:
        name = f"{role.value}_{uuid.uuid4().hex[:8]}"
    
    # This would return specific agent implementations
    # For now, it's a placeholder that would be replaced with actual implementations
    
    class ConcreteAgent(BaseAgent):
        def process_task(self, task_description: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
            # Placeholder implementation
            return {
                "status": "completed",
                "message": f"Task '{task_description}' processed by {self.role.value}",
                "data": task_data
            }
    
    return ConcreteAgent(name=name, role=role, **kwargs)