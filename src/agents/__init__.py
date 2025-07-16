#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - Agent Management System

This module provides a comprehensive multi-agent system for flight selection
and analysis, implementing academic-level algorithms and methodologies.

Agent Architecture:
- BaseAgent: Foundation class with trust-aware communication protocols
- FlightInformationAgent: Multi-source flight data aggregation and analysis
- SafetyAssessmentAgent: ICAO-compliant aviation safety risk assessment
- EconomicAgent: Comprehensive cost analysis and optimization
- WeatherAgent: Meteorological analysis and weather risk assessment
- TrustManager: Byzantine fault tolerance and agent trust management
- IntegrationAgent: Multi-agent coordination and result synthesis
- CrossDomainSolver: Cross-domain optimization and constraint satisfaction

Academic Features:
- Trust-aware multi-agent coordination protocols
- Byzantine fault tolerance mechanisms
- Real-time learning and adaptation algorithms
- Academic-validated assessment methodologies
- Comprehensive performance metrics and evaluation
"""

import logging
from typing import Dict, Any, List, Optional

# Configure module logging
logger = logging.getLogger(__name__)

# Import base agent framework
from .base_agent import (
    BaseAgent,
    AgentRole,
    AgentState,
    CommunicationProtocol,
    AgentCapability,
    TaskExecution,
    AgentMetrics
)

# Import specialized agents
from .flight_info_agent import (
    FlightInformationAgent,
    FlightDataAggregator,
    FlightDetails,
    FlightRoute,
    FlightStatus,
    DataSource,
    create_flight_info_agent,
    get_flight_information_tool
)

from .safety_assessment_agent import (
    SafetyAssessmentAgent,
    SafetyDataAggregator,
    SafetyAssessment,
    SafetyMetrics,
    SafetyIncident,
    SafetyRiskLevel,
    SafetyCategory,
    IncidentSeverity,
    create_safety_assessment_agent,
    get_safety_assessment_tool
)

from .economic_agent import (
    EconomicAgent,
    create_economic_agent,
    calculate_total_cost_tool,
    get_city_accommodation_cost
)

from .weather_agent import (
    WeatherAgent,
    get_weather_safety_tool,
    get_weather_conditions,
    calculate_weather_safety_score,
    generate_weather_recommendation,
    create_weather_agent
)

# Import trust and coordination systems
from .trust_manager import (
    TrustManager,
    TrustLedger,
    TrustLevel,
    InteractionProtocol,
    TrustDimensions,
    TrustRecord,
    InteractionProtocolManager,
    TrustMetricsCalculator,
    get_trust_evaluation,
    record_agent_outcome
)

from .integration_agent import (
    LTRRankingEngine
)

from .cross_domain_solver import (
    CrossDomainSolver,
    create_cross_domain_solver
)

from .manager import (
    MAMAFlightManager,
    TOOL_FUNCTIONS
)

# Import trust simulation for academic validation
from .trust_simulation import (
    TaskSimulation,
    AgentPersonality,
    WeatherDataSimulator,
    FlightSearchSimulator,
    SafetyAssessmentSimulator,
    EconomicAnalysisSimulator,
    SecurityAttackSimulator,
    DecisionBiasSimulator,
    ExplanationGenerator,
    TrustSimulationEngine,
    generate_test_data,
    simulate_trust_scenarios
)

# Import TrustMetrics from correct location
from core.marl_policy import TrustMetrics

# Agent factory functions
def create_complete_agent_system() -> Dict[str, BaseAgent]:
    """
    Create a complete MAMA agent system with all specialized agents
    
    Returns:
        Dictionary mapping agent names to configured agent instances
    """
    logger.info("Initializing complete MAMA agent system")
    
    try:
        # Create specialized agents
        agents = {
            "flight_info": create_flight_info_agent(),
            "safety_assessment": create_safety_assessment_agent(),
            "economic": create_economic_agent(),
            "weather": create_weather_agent(),
            "integration": create_integration_agent(),
            "cross_domain": create_cross_domain_solver(),
            "manager": create_agent_manager()
        }
        
        # Initialize trust manager with all agents
        trust_manager = create_trust_manager()
        for agent_name, agent in agents.items():
            trust_manager.register_agent(agent_name, agent)
        
        agents["trust_manager"] = trust_manager
        
        logger.info(f"Successfully initialized {len(agents)} agents")
        return agents
        
    except Exception as e:
        logger.error(f"Failed to initialize agent system: {e}")
        raise


def validate_agent_system(agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
    """
    Validate the complete agent system for academic compliance
    
    Args:
        agents: Dictionary of agent instances
        
    Returns:
        Validation report with compliance metrics
    """
    logger.info("Validating MAMA agent system")
    
    validation_report = {
        "system_status": "validating",
        "agent_count": len(agents),
        "validation_results": {},
        "compliance_metrics": {},
        "academic_standards": {
            "trust_protocols": False,
            "byzantine_tolerance": False,
            "performance_monitoring": False,
            "real_time_learning": False
        }
    }
    
    try:
        # Validate each agent
        for agent_name, agent in agents.items():
            agent_validation = {
                "status": "valid",
                "capabilities": [],
                "trust_enabled": hasattr(agent, 'trust_score'),
                "academic_compliant": True
            }
            
            # Check agent capabilities
            if hasattr(agent, 'capabilities'):
                agent_validation["capabilities"] = [cap.value for cap in agent.capabilities]
            
            # Check for required methods
            required_methods = ['process_task', 'get_status', 'update_performance']
            for method in required_methods:
                if not hasattr(agent, method):
                    agent_validation["status"] = "incomplete"
                    agent_validation["academic_compliant"] = False
            
            validation_report["validation_results"][agent_name] = agent_validation
        
        # Check system-level features
        if "trust_manager" in agents:
            validation_report["academic_standards"]["trust_protocols"] = True
            validation_report["academic_standards"]["byzantine_tolerance"] = True
        
        if all(hasattr(agent, 'performance_metrics') for agent in agents.values()):
            validation_report["academic_standards"]["performance_monitoring"] = True
        
        # Overall system status
        all_valid = all(
            result["status"] == "valid" 
            for result in validation_report["validation_results"].values()
        )
        
        validation_report["system_status"] = "valid" if all_valid else "requires_attention"
        
        # Calculate compliance score
        compliance_score = sum(validation_report["academic_standards"].values()) / len(validation_report["academic_standards"])
        validation_report["compliance_metrics"]["overall_score"] = compliance_score
        validation_report["compliance_metrics"]["academic_readiness"] = compliance_score >= 0.75
        
        logger.info(f"Agent system validation completed: {validation_report['system_status']}")
        return validation_report
        
    except Exception as e:
        logger.error(f"Agent system validation failed: {e}")
        validation_report["system_status"] = "error"
        validation_report["error"] = str(e)
        return validation_report


def get_agent_system_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the MAMA agent system
    
    Returns:
        System information including capabilities and academic features
    """
    return {
        "system_name": "MAMA Flight Selection Assistant",
        "version": "1.0.0",
        "architecture": "Multi-Agent Trust-Aware System",
        "academic_features": [
            "Byzantine fault tolerance",
            "Trust-aware communication protocols",
            "Real-time learning and adaptation",
            "Multi-source data fusion",
            "ICAO-compliant safety assessment",
            "Academic algorithm implementations",
            "Performance monitoring and evaluation"
        ],
        "agent_types": [
            "Flight Information Agent",
            "Safety Assessment Agent", 
            "Economic Analysis Agent",
            "Weather Assessment Agent",
            "Trust Management Agent",
            "Integration Agent",
            "Cross-Domain Solver",
            "Agent Manager"
        ],
        "compliance_standards": [
            "ICAO Safety Management Systems",
            "Academic research methodologies",
            "Multi-agent system protocols",
            "Real-time data processing standards"
        ],
        "supported_operations": [
            "Flight information retrieval and analysis",
            "Aviation safety risk assessment",
            "Economic cost optimization",
            "Weather impact analysis", 
            "Multi-criteria decision making",
            "Trust-based agent coordination",
            "Cross-domain constraint satisfaction"
        ]
    }


# Export all classes and functions for external use
__all__ = [
    # Base agent framework
    "BaseAgent",
    "AgentRole", 
    "AgentState",
    "CommunicationProtocol",
    "AgentCapability",
    "TaskExecution",
    "AgentMetrics",
    
    # Specialized agents
    "FlightInformationAgent",
    "SafetyAssessmentAgent", 
    "EconomicAgent",
    "WeatherAgent",
    "TrustManager",
    "IntegrationAgent",
    "CrossDomainSolver",
    "AgentManager",
    
    # Data structures
    "FlightDetails",
    "FlightRoute",
    "FlightStatus",
    "DataSource",
    "SafetyAssessment",
    "SafetyMetrics", 
    "SafetyIncident",
    "SafetyRiskLevel",
    "SafetyCategory",
    "IncidentSeverity",
    "TrustMetrics",
    "TrustLevel",
    
    # Aggregators and processors
    "FlightDataAggregator",
    "SafetyDataAggregator",
    "TrustSimulation",
    
    # Factory functions
    "create_flight_info_agent",
    "create_safety_assessment_agent",
    "create_economic_agent", 
    "create_weather_agent",
    "create_trust_manager",
    "create_integration_agent",
    "create_cross_domain_solver",
    "create_agent_manager",
    "create_complete_agent_system",
    
    # Tool functions
    "get_flight_information_tool",
    "get_safety_assessment_tool",
    "calculate_total_cost_tool",
    "get_weather_safety_tool",
    "get_city_accommodation_cost",
    
    # System functions
    "validate_agent_system",
    "get_agent_system_info",
    "simulate_trust_scenarios"
]

# Initialize logging for the agents module
logger.info("MAMA Flight Selection Assistant - Agent Management System initialized")
logger.info(f"Available agents: {len(__all__)} classes and functions exported")
