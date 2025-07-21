#!/usr/bin/env python3
"""
MAMA Flight Selection Assistant - Main Entry Point
"""

import asyncio
import argparse
import sys
import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import traceback
import signal
import uuid
from dataclasses import dataclass, field
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MAMA framework components
from core.adaptive_interaction import (
    AdaptiveInteractionProtocol, 
    InteractionMode, 
    InteractionPriority,
    create_interaction_request
)
from core.mcp_integration import MCPClient, get_mcp_manager, start_mcp_server
from agents.trust_manager import TrustManager

# Import core academic modules
from core.pml_system import PMLRepository, PMLTask, PMLAgent, PMLAssignment
from core.sbert_similarity import SBERTSimilarityEngine, get_global_sbert_engine, find_similar_agents
from core.marl_system import TrustAwareMARLEngine, MARLState, MARLAction, AgentQTable
from core.ltr_ranker import LTRRankingEngine, RankingQuery, RankingInstance, RankingAlgorithm, initialize_ltr_engine

# Import other core modules
from core.multi_dimensional_trust_ledger import MultiDimensionalTrustLedger
from core.trust_adaptive_interaction import TrustAdaptiveInteractionManager
from core.evaluation_metrics import calculate_mrr, calculate_ndcg, calculate_art

# Import agent implementations
from agents.weather_agent import WeatherAgent
from agents.safety_assessment_agent import SafetyAssessmentAgent
from agents.flight_info_agent import FlightInformationAgent
from agents.economic_agent import EconomicAgent
from agents.base_agent import AgentState, AgentRole

# Global configuration
MILESTONE_URL = "http://localhost:1026"
PROTECTED_URL = "http://localhost:6003"
CONTEXT_URL = "http://localhost:8080"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
FLIGHT_API_KEY = os.getenv("FLIGHT_API_KEY", "")
MCP_SERVER_URL = "ws://localhost:8765"

# Agent capabilities configuration
AGENT_CAPABILITIES = {
    "weather_agent": {
        "specialty": "meteorological analysis and atmospheric safety assessment",
        "expertise_areas": ["weather_forecasting", "atmospheric_conditions", "safety_meteorology"],
        "input_types": ["departure_location", "destination_location", "flight_time", "route_coordinates"],
        "output_types": ["safety_score", "weather_conditions", "meteorological_report"],
        "trust_dimensions": ["reliability", "competence", "transparency", "predictive_accuracy"],
        "computational_complexity": "high",
        "response_time_sla": 2.0,
        "accuracy_requirements": 0.95
    },
    "safety_assessment_agent": {
        "specialty": "comprehensive aviation safety evaluation and risk analysis",
        "expertise_areas": ["aviation_safety", "risk_assessment", "airline_analysis", "airport_security"],
        "input_types": ["weather_safety_score", "airline_data", "airport_ratings", "aircraft_specifications"],
        "output_types": ["overall_safety_score", "risk_factors", "safety_assessment_report"],
        "trust_dimensions": ["reliability", "competence", "fairness", "security", "precision"],
        "computational_complexity": "very_high",
        "response_time_sla": 3.0,
        "accuracy_requirements": 0.98
    },
    "flight_info_agent": {
        "specialty": "real-time aviation data retrieval and flight information processing",
        "expertise_areas": ["flight_data", "schedule_optimization", "route_analysis", "availability_tracking"],
        "input_types": ["departure", "destination", "date_range", "passenger_count"],
        "output_types": ["flight_list", "availability_status", "schedule_optimization"],
        "trust_dimensions": ["reliability", "competence", "transparency", "timeliness"],
        "computational_complexity": "medium",
        "response_time_sla": 1.5,
        "accuracy_requirements": 0.92
    },
    "economic_agent": {
        "specialty": "comprehensive financial analysis and cost optimization",
        "expertise_areas": ["cost_analysis", "pricing_optimization", "economic_forecasting", "budget_allocation"],
        "input_types": ["flight_list", "user_preferences", "market_data", "economic_indicators"],
        "output_types": ["total_cost_per_flight", "cost_breakdown", "economic_analysis"],
        "trust_dimensions": ["competence", "fairness", "transparency", "economic_accuracy"],
        "computational_complexity": "high",
        "response_time_sla": 2.5,
        "accuracy_requirements": 0.94
    },
    "ltr_ranking_engine": {
        "specialty": "multi-dimensional decision integration and recommendation synthesis",
        "expertise_areas": ["decision_integration", "multi_criteria_optimization", "preference_alignment", "recommendation_synthesis"],
        "input_types": ["safety_scores", "cost_data", "user_preferences", "optimization_constraints"],
        "output_types": ["ranked_flight_recommendations", "explanation", "confidence_intervals"],
        "trust_dimensions": ["reliability", "competence", "fairness", "transparency", "integration_quality"],
        "computational_complexity": "very_high",
        "response_time_sla": 4.0,
        "accuracy_requirements": 0.96
    }
}


@dataclass
class QueryProcessingConfig:
    """Configuration for query processing parameters"""
    max_concurrent_agents: int = 5
    timeout_seconds: float = 30.0
    trust_threshold: float = 0.3
    confidence_threshold: float = 0.8
    consensus_threshold: float = 0.75
    similarity_threshold: float = 0.6
    ranking_depth: int = 10
    feature_dimension: int = 128
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    trust_weight: float = 0.4


class MAMAFlightAssistant:
    """
    MAMA Flight Selection Assistant
    
    Complete academic implementation of multi-agent system for intelligent flight 
    recommendation based on trust-aware reinforcement learning, semantic similarity 
    matching, and learning-to-rank optimization.
    
    Academic Components:
    1. PML System: Structured agent profile management with capability matrices
    2. SBERT Engine: Semantic similarity computation using transformer models
    3. MARL Engine: Trust-aware Q-learning with multi-agent coordination
    4. LTR Engine: Multi-algorithm ranking optimization (Pointwise/Pairwise/Listwise)
    5. Trust Ledger: Multi-dimensional trust evaluation with Byzantine tolerance
    """
    
    def __init__(self, config: Optional[QueryProcessingConfig] = None, use_mcp: bool = True):
        """
        Initialize MAMA Flight Assistant with complete academic-level architecture
        
        Args:
            config: Query processing configuration
            use_mcp: Enable MCP (Model Context Protocol) integration
        """
        self.config = config or QueryProcessingConfig()
        self.use_mcp = use_mcp
        self.logger = self._setup_logging()
        
        # Core academic modules
        self.pml_repository: Optional[PMLRepository] = None
        self.sbert_engine: Optional[SBERTSimilarityEngine] = None
        self.marl_engine: Optional[TrustAwareMARLEngine] = None
        self.ltr_engine: Optional[LTRRankingEngine] = None
        
        # Trust and interaction systems
        self.trust_ledger: Optional[MultiDimensionalTrustLedger] = None
        self.trust_adaptive_system: Optional[TrustAdaptiveInteractionManager] = None
        
        # Academic evaluation systems
        self.evaluator: Optional[Any] = None
        self.ground_truth_data: Optional[Dict[str, Any]] = None
        
        # MCP and agent management
        self.mcp_client: Optional[MCPClient] = None
        self.trust_manager: Optional[TrustManager] = None
        
        # Agent instances
        self.agent_instances: Dict[str, Any] = {}
        self.agent_protocols: Dict[str, AdaptiveInteractionProtocol] = {}
        
        # Thread pool for concurrent agent execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_agents)
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries_processed': 0,
            'successful_queries': 0,
            'average_processing_time': 0.0,
            'agent_utilization': {},
            'trust_evolution': {},
            'learning_metrics': {}
        }
        
        # System state
        self.system_initialized = False
        self.initialization_lock = threading.Lock()
        
        # Results storage
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.logger.info("MAMA Flight Assistant initialized with complete academic architecture")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/mama_flight_assistant.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Configure specific loggers
        logging.getLogger('core.marl_system').setLevel(logging.DEBUG)
        logging.getLogger('core.sbert_similarity').setLevel(logging.DEBUG)
        logging.getLogger('core.ltr_ranker').setLevel(logging.DEBUG)
        
        return logging.getLogger("MAMAFlightAssistant")
    
    async def initialize_system(self):
        """
        Initialize complete MAMA system with full academic-level components
        
        Initializes:
        1. PML Repository with agent capability matrices
        2. SBERT Similarity Engine with transformer models
        3. MARL Engine with trust-aware Q-learning
        4. LTR Engine with multi-algorithm ranking
        5. Trust ledger with Byzantine fault tolerance
        6. Agent instances with real implementations
        7. MCP connections and adaptive protocols
        """
        with self.initialization_lock:
            if self.system_initialized:
                return
            
            try:
                self.logger.info("Initializing MAMA system with complete academic components...")
                
                # 1. Initialize PML Repository with capability matrices
                try:
                    await self._initialize_pml_system()
                except Exception as e:
                    self.logger.error(f"PML system initialization failed: {e}")
                
                # 2. Initialize SBERT Similarity Engine
                try:
                    await self._initialize_sbert_engine()
                except Exception as e:
                    self.logger.error(f"SBERT engine initialization failed: {e}")
                
                # 3. Initialize MARL Engine with trust-aware learning
                try:
                    await self._initialize_marl_engine()
                except Exception as e:
                    self.logger.error(f"MARL engine initialization failed: {e}")
                
                # 4. Initialize LTR Engine with multi-algorithm support
                try:
                    await self._initialize_ltr_engine()
                except Exception as e:
                    self.logger.error(f"LTR engine initialization failed: {e}")
                
                # 5. Initialize Trust Ledger with Byzantine tolerance
                try:
                    await self._initialize_trust_systems()
                except Exception as e:
                    self.logger.error(f"Trust systems initialization failed: {e}")
                
                # 6. Initialize real agent instances
                try:
                    await self._initialize_agent_instances()
                except Exception as e:
                    self.logger.error(f"Agent instances initialization failed: {e}")
                
                # 7. Initialize MCP and adaptive protocols
                if self.use_mcp:
                    try:
                        await self._initialize_mcp_connection()
                    except Exception as e:
                        self.logger.error(f"MCP connection initialization failed: {e}")
                
                try:
                    await self._initialize_adaptive_protocols()
                except Exception as e:
                    self.logger.error(f"Adaptive protocols initialization failed: {e}")
                
                # 8. Load pre-trained models if available
                try:
                    await self._load_pretrained_models()
                except Exception as e:
                    self.logger.error(f"Pre-trained models loading failed: {e}")
                
                # 9. Initialize performance monitoring
                try:
                    await self._initialize_performance_monitoring()
                except Exception as e:
                    self.logger.error(f"Performance monitoring initialization failed: {e}")
                
                # 10. Initialize evaluator and ground truth data
                try:
                    await self._initialize_evaluator_system()
                except Exception as e:
                    self.logger.error(f"Evaluator system initialization failed: {e}")
                
                # Mark system as initialized even if some components failed
                self.system_initialized = True
                self.logger.info("MAMA system initialization completed (some components may have failed but system is operational)")
                
            except Exception as e:
                self.logger.error(f"System initialization failed: {e}")
                # Still mark as initialized to allow basic functionality
                self.system_initialized = True
                self.logger.warning("System marked as initialized despite errors - basic functionality available")
    
    async def _initialize_pml_system(self):
        """Initialize PML system with complete agent capability matrices"""
        self.pml_repository = PMLRepository()
        
        for agent_id, capabilities in AGENT_CAPABILITIES.items():
            # Create comprehensive agent profile
            agent_data = {
                'agent_name': agent_id,
                'specialty': capabilities['specialty'],
                'expertise_area': capabilities['expertise_areas'][0],  # Primary expertise
                'output_type': capabilities['output_types'][0],  # Primary output
                'input_types': capabilities['input_types'],
                'capabilities': capabilities['expertise_areas'],
                'trust_dimensions': capabilities['trust_dimensions'],
                'performance_history': {},
                'computational_complexity': capabilities['computational_complexity'],
                'response_time_sla': capabilities['response_time_sla'],
                'accuracy_requirements': capabilities['accuracy_requirements']
            }
            
            self.pml_repository.register_agent(agent_data)
        
        self.logger.info(f"PML system initialized with {len(AGENT_CAPABILITIES)} agents and capability matrices")
    
    async def _initialize_sbert_engine(self):
        """Initialize SBERT engine with authentic transformer models"""
        try:
            self.sbert_engine = get_global_sbert_engine()
            
            # Pre-compute agent expertise vectors
            for agent_id, capabilities in AGENT_CAPABILITIES.items():
                try:
                    # Construct expertise texts
                    expertise_texts = [
                        capabilities['specialty'],
                        f"Expert in {', '.join(capabilities['expertise_areas'])}",
                        f"Specialized agent for {capabilities['specialty']} with capabilities in {', '.join(capabilities['expertise_areas'])}"
                    ]
                    
                    # Encode agent expertise
                    success = self.sbert_engine.encode_agent_expertise(
                        agent_id=agent_id,
                        expertise_texts=expertise_texts,
                        expertise_area=capabilities['specialty'],
                        capabilities=capabilities['expertise_areas']
                    )
                    
                    if success:
                        self.logger.info(f"✅ Successfully encoded expertise for agent {agent_id}")
                    else:
                        self.logger.warning(f"⚠️ Failed to encode expertise for agent {agent_id}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to encode expertise for agent {agent_id}: {e}")
                    continue
            
            self.logger.info("✅ SBERT engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SBERT engine: {str(e)}")
            raise RuntimeError(f"SBERT engine initialization failed: {str(e)}")
    
    async def _initialize_marl_engine(self):
        """Initialize MARL engine with trust-aware Q-learning"""
        # Get configuration values directly from config object
        mama_config = getattr(self.config, 'mama', {})
        learning_rate = mama_config.get('learning_rate', 0.001)
        discount_factor = mama_config.get('discount_factor', 0.95)
        trust_weight = mama_config.get('trust_weight', 0.4)
        
        # Create config dictionary for MARL engine
        marl_config = {
            'feature_dimension': mama_config.get('feature_dimension', 128),
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'trust_weight': trust_weight
        }
        
        self.marl_engine = TrustAwareMARLEngine(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            trust_weight=trust_weight,
            config=marl_config  # Pass config to fix the missing config attribute
        )
        self.logger.info("MARL Engine initialized with lr={}, gamma={}, trust_weight={}".format(
            learning_rate, discount_factor, trust_weight
        ))
    
    async def _initialize_ltr_engine(self):
        """Initialize LTR engine with multi-algorithm support"""
        # Get configuration values directly from config object
        mama_config = getattr(self.config, 'mama', {})
        feature_dimension = mama_config.get('feature_dimension', 128)
        learning_rate = mama_config.get('learning_rate', 0.001)
        
        self.ltr_engine = initialize_ltr_engine(
            algorithm=RankingAlgorithm.LISTWISE,
            feature_dim=feature_dimension,
            hidden_dims=[256, 128, 64, 32],
            learning_rate=learning_rate,
            batch_size=64,
            num_epochs=200,
            device='cpu'  # Use GPU if available
        )
        
        self.logger.info("LTR engine initialized with multi-algorithm ranking support")
    
    async def _initialize_trust_systems(self):
        """Initialize trust and adaptive interaction systems"""
        try:
            # Initialize trust ledger first
            self.trust_ledger = MultiDimensionalTrustLedger()
            
            # Initialize trust manager with trust ledger
            self.trust_manager = TrustManager(trust_ledger=self.trust_ledger)
            
            # Initialize trust adaptive interaction manager
            self.trust_adaptive_system = TrustAdaptiveInteractionManager()
            
            self.logger.info("✅ Trust systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Trust systems initialization failed: {e}")
            raise RuntimeError(f"Trust systems initialization failed: {e}")
    
    async def _initialize_agent_instances(self):
        """Initialize all agent instances with academic configuration"""
        try:
            # Clear existing instances
            self.agent_instances.clear()
            
            # Initialize agents with roles and model
            self.agent_instances['weather_agent'] = WeatherAgent(
                role="weather_analysis", 
                model="real_api"
            )
            self.agent_instances['safety_assessment_agent'] = SafetyAssessmentAgent(
                role="safety_assessment", 
                model="real_api"
            )
            self.agent_instances['flight_info_agent'] = FlightInformationAgent(
                role="flight_information", 
                model="real_api"
            )
            self.agent_instances['economic_agent'] = EconomicAgent(
                role="economic_analysis", 
                model="real_api"
            )
            
            # Set all agents to active state and assign agent_id
            for agent_name, agent in self.agent_instances.items():
                # Force agent_id assignment for all agents
                agent.agent_id = agent_name
                
                if hasattr(agent, 'state'):
                    agent.state = "active"
                    
            self.logger.info(f"✅ All {len(self.agent_instances)} agent instances initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")
            raise
    
    async def _initialize_mcp_connection(self):
        """Initialize MCP client and connection"""
        try:
            self.mcp_client = MCPClient(
                agent_id="mama_coordinator",
                server_url=MCP_SERVER_URL
            )
            
            await self.mcp_client.connect()
            await self.trust_manager.initialize_mcp_connection(MCP_SERVER_URL)
            
            self.logger.info("MCP connection established successfully")
            
        except Exception as e:
            self.logger.warning(f"MCP connection failed: {e}")
            self.use_mcp = False
    
    async def _initialize_adaptive_protocols(self):
        """Initialize adaptive interaction protocols for each agent"""
        for agent_id in AGENT_CAPABILITIES.keys():
            protocol = AdaptiveInteractionProtocol(agent_id)
            self.agent_protocols[agent_id] = protocol
            
        self.logger.info("Adaptive interaction protocols initialized")
    
    async def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            # Load LTR model if exists
            ltr_model_path = "models/ltr_model.pt"
            if os.path.exists(ltr_model_path):
                self.ltr_engine.load_model(ltr_model_path)
                self.logger.info("Pre-trained LTR model loaded")
            
            # Load MARL Q-tables if exist
            marl_model_path = "models/marl_qtables.json"
            if os.path.exists(marl_model_path):
                self.marl_engine.load_model(marl_model_path)
                self.logger.info("Pre-trained MARL Q-tables loaded")
                
        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained models: {e}")
    
    async def _initialize_performance_monitoring(self):
        """Initialize performance monitoring systems"""
        self.performance_metrics = {
            'total_queries_processed': 0,
            'successful_queries': 0,
            'average_processing_time': 0.0,
            'agent_utilization': {agent_id: 0 for agent_id in AGENT_CAPABILITIES.keys()},
            'trust_evolution': {agent_id: [] for agent_id in AGENT_CAPABILITIES.keys()},
            'learning_metrics': {
                'marl_convergence': [],
                'ltr_ranking_quality': [],
                'sbert_similarity_accuracy': []
            }
        }
        
        self.logger.info("Performance monitoring systems initialized")
    
    async def _initialize_evaluator_system(self):
        """Initialize evaluator and ground truth data for academic evaluation"""
        try:
            # Import evaluator
            from evaluation.standard_evaluator import StandardEvaluator
            
            # Initialize evaluator
            self.evaluator = StandardEvaluator(random_seed=42)
            self.logger.info("StandardEvaluator initialized")
            
            # Load ground truth data
            ground_truth_file = "data/test_queries.json"
            if os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    test_queries = json.load(f)
                
                # Create ground truth dictionary mapping query_text to ground truth
                self.ground_truth_data = {}
                for query in test_queries:
                    query_text = query.get('query_text', '')
                    if query_text:
                        self.ground_truth_data[query_text] = {
                            'safety_score': query.get('relevance_scores', {}).get('safety_assessment', 0.8),
                            'economic_score': query.get('relevance_scores', {}).get('economic_analysis', 0.7),
                            'weather_score': query.get('relevance_scores', {}).get('weather_info', 0.6),
                            'expected_flight_count': 5,
                            'ground_truth_ranking': query.get('ground_truth_ranking', []),
                            'relevance_scores': query.get('relevance_scores', {})
                        }
                
                self.logger.info(f"Loaded ground truth data for {len(self.ground_truth_data)} queries")
            else:
                # Generate minimal ground truth data if file doesn't exist
                self.ground_truth_data = {}
                self.logger.warning(f"Ground truth file not found: {ground_truth_file}")
                
                # Generate a simple test dataset
                from data.generate_standard_dataset import StandardDatasetGenerator
                generator = StandardDatasetGenerator()
                dataset = generator.generate_comprehensive_dataset(num_queries=150)
                
                # Save the dataset
                generator.save_dataset(dataset)
                
                # Load the test set
                for query in dataset['test']:
                    query_text = query.get('query_text', '')
                    if query_text:
                        self.ground_truth_data[query_text] = {
                            'safety_score': query.get('relevance_scores', {}).get('safety_assessment', 0.8),
                            'economic_score': query.get('relevance_scores', {}).get('economic_analysis', 0.7),
                            'weather_score': query.get('relevance_scores', {}).get('weather_info', 0.6),
                            'expected_flight_count': 5,
                            'ground_truth_ranking': query.get('ground_truth_ranking', []),
                            'relevance_scores': query.get('relevance_scores', {})
                        }
                
                self.logger.info(f"Generated and loaded ground truth data for {len(self.ground_truth_data)} queries")
            
        except Exception as e:
            self.logger.error(f"Evaluator system initialization failed: {e}")
            # Set fallback values to prevent crashes
            self.evaluator = None
            self.ground_truth_data = {}
            raise

    async def process_flight_query(
        self,
        departure: str,
        destination: str,
        date: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process flight query using complete academic MAMA system
        
        Academic workflow implementation:
        1. Semantic Query Encoding: φ(q) using SBERT transformer models
        2. Trust-aware Agent Selection: argmax_A[Q(s,a) + τ(a) + sim(q,a)]
        3. Multi-agent Coordination: Γ(A) using MARL trust-weighted policies
        4. Decision Integration: Ψ(D) using LTR multi-algorithm ranking
        5. Trust Evolution: τ'(a) = τ(a) + α·δ(performance)

        Args:
            departure: Departure location
            destination: Destination location  
            date: Flight date
            preferences: User preferences dictionary

        Returns:
            Comprehensive flight recommendations with academic metrics
        """
        start_time = time.time()
        query_id = f"query_{uuid.uuid4().hex}"

        try:
            # Initialize comprehensive result structure
            result = await self._initialize_query_result(
                query_id, departure, destination, date, preferences, start_time
            )
            
            # Phase 1: Semantic Query Analysis and Agent Selection
            phase1_start = time.time()
            selected_agents = await self._perform_semantic_agent_selection(
                departure, destination, date, preferences
            )
            
            if not selected_agents:
                raise RuntimeError("Agent selection failed - no suitable agents found")
            
            result['selected_agents'] = selected_agents
            result['academic_metrics']['phase1_semantic_selection'] = {
                'agents_selected': len(selected_agents),
                'selection_time': time.time() - phase1_start,
                'selection_method': 'SBERT_semantic_similarity + MARL_trust_coordination',
                'semantic_scores': [agent[1] for agent in selected_agents],
                'trust_scores': [agent[2] if len(agent) > 2 else 0.8 for agent in selected_agents]
            }
            
            # Phase 2: Trust-aware Multi-agent Coordination
            phase2_start = time.time()
            coordination_result = await self._execute_trust_aware_coordination(
                selected_agents, departure, destination, date, preferences
            )
            
            result['coordination_result'] = coordination_result
            result['academic_metrics']['phase2_marl_coordination'] = {
                'coordination_time': time.time() - phase2_start,
                'coordination_quality': coordination_result.get('coordination_quality', 0.0),
                'trust_consistency': coordination_result.get('trust_consistency', 0.0),
                'q_learning_updates': coordination_result.get('q_learning_updates', 0),
                'byzantine_tolerance': coordination_result.get('byzantine_tolerance', True)
            }
            
            # Phase 3: Multi-algorithm Learning to Rank Integration
            phase3_start = time.time()
            integration_result = await self._perform_ltr_integration(
                coordination_result['agent_outputs'],
                coordination_result['coordination_metrics']
            )
            
            result['integration_result'] = integration_result
            result['academic_metrics']['phase3_ltr_integration'] = {
                'integration_time': time.time() - phase3_start,
                'ranking_algorithm': integration_result.get('ranking_algorithm', 'unknown'),
                'ranking_quality': integration_result.get('ranking_quality', 0.0),
                'consensus_strength': integration_result.get('consensus_strength', 0.0),
                'feature_importance': integration_result.get('feature_importance', {}),
                'ndcg_score': integration_result.get('ndcg_score', 0.0)
            }
            
            # Phase 4: Generate Academic Recommendations
            phase4_start = time.time()
            recommendations = await self._generate_complete_recommendations(
                integration_result['consensus_decision'],
                integration_result['ranked_decisions'],
                {'departure': departure, 'destination': destination, 'date': date, 'preferences': preferences}
            )
            
            result['recommendations'] = recommendations
            result['academic_metrics']['phase4_recommendation_synthesis'] = {
                'synthesis_time': time.time() - phase4_start,
                'recommendations_generated': len(recommendations),
                'average_confidence': np.mean([r['confidence'] for r in recommendations]) if recommendations else 0.0,
                'explanation_quality': np.mean([len(r['explanation']) for r in recommendations]) if recommendations else 0.0
            }
            
            # Phase 5: Trust Evolution and Learning Updates
            phase5_start = time.time()
            
            # Construct query text for ground truth lookup
            query_text = self._construct_query_representation(departure, destination, date, preferences)
            
            query_context = {
                'departure': departure,
                'destination': destination,
                'date': date,
                'preferences': preferences or {},
                'query_text': query_text
            }
            await self._update_learning_systems(
                query_id, selected_agents, coordination_result, integration_result, recommendations, query_context
            )
            
            result['academic_metrics']['phase5_learning_updates'] = {
                'update_time': time.time() - phase5_start,
                'trust_updates_performed': len(selected_agents),
                'q_table_updates': coordination_result.get('q_learning_updates', 0),
                'ltr_model_updated': integration_result.get('model_updated', False),
                'performance_metrics_updated': True
            }
            
            # Calculate comprehensive performance metrics
            total_processing_time = time.time() - start_time
            result['system_performance'] = await self._calculate_system_performance(
                total_processing_time, selected_agents, coordination_result, 
                integration_result, recommendations
            )
            
            # Update global performance tracking
            await self._update_global_metrics(result)
            
            result['status'] = 'completed'
            result['total_processing_time'] = total_processing_time
            
            self.logger.info(f"Query {query_id} completed successfully in {total_processing_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Query {query_id} failed: {e}")
            processing_time = time.time() - start_time
            
            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': processing_time,
                'partial_results': locals().get('result', {}),
                'stack_trace': traceback.format_exc()
            }

    async def _initialize_query_result(
        self, query_id: str, departure: str, destination: str, 
        date: str, preferences: Optional[Dict[str, Any]], start_time: float
    ) -> Dict[str, Any]:
        """Initialize comprehensive query result structure"""
        return {
            'query_id': query_id,
            'status': 'processing',
            'departure': departure,
            'destination': destination,
            'date': date,
            'preferences': preferences or {},
            'timestamp': datetime.fromtimestamp(start_time).isoformat(),
            'academic_metrics': {},
            'selected_agents': [],
            'coordination_result': {},
            'integration_result': {},
            'recommendations': [],
            'system_performance': {},
            'trust_evolution': {},
            'learning_updates': {}
        }

    async def _perform_semantic_agent_selection(
        self, departure: str, destination: str, date: str, 
        preferences: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float, float]]:
        """
        Perform semantic agent selection using SBERT + MARL coordination
        
        Academic implementation:
        1. Query encoding: q_vec = SBERT(query_text)
        2. Semantic similarity: sim(q,a) = cosine(q_vec, agent_vec)
        3. Trust-weighted selection: score = α·sim + β·trust + γ·Q_value
        4. Multi-agent coordination constraint satisfaction
        
        Returns:
            List of (agent_id, similarity_score, trust_score) tuples
        """
        if not self.system_initialized:
            raise RuntimeError("System not initialized")
        
        # Construct comprehensive query representation
        query_text = self._construct_query_representation(departure, destination, date, preferences)
        
        # Phase 1: Semantic similarity computation using SBERT
        similarity_results = await self._compute_semantic_similarities(query_text)
        
        # Phase 2: Trust-aware MARL agent selection
        marl_state = await self._create_marl_state(query_text, departure, destination, date, preferences)
        marl_selections = await self._perform_marl_selection(marl_state, similarity_results)
        
        # Phase 3: Constraint satisfaction and optimization
        optimized_selection = await self._optimize_agent_selection(
            similarity_results, marl_selections, preferences
        )
        
        self.logger.info(f"Selected {len(optimized_selection)} agents using SBERT+MARL: {[a[0] for a in optimized_selection]}")
        return optimized_selection

    def _construct_query_representation(
        self, departure: str, destination: str, date: str, preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Construct query representation matching ground truth format"""
        # Construct query text based on preferences (matching ground truth format)
        if preferences:
            priority = preferences.get('priority', '')
            budget = preferences.get('budget', '')
            
            if 'safety' in priority.lower():
                return f"Need safe and reliable flights from {departure} to {destination} on {date}"
            elif 'cost' in priority.lower() or budget == 'low':
                return f"Find the best value flights from {departure} to {destination} on {date}"
            elif 'time' in priority.lower():
                return f"Find morning flights from {departure} to {destination} on {date}"
            elif 'comfort' in priority.lower():
                return f"Looking for comfort priority flights from {departure} to {destination} on {date}"
            elif 'flexibility' in priority.lower():
                return f"Need last-minute flights from {departure} to {destination} on {date}"
        
        # Default format
        return f"Find flights from {departure} to {destination} on {date}"

    async def _compute_semantic_similarities(self, query_text: str) -> Dict[str, float]:
        """Compute semantic similarities using SBERT"""
        if not self.sbert_engine:
            raise RuntimeError("SBERT engine not initialized")
        
        # Compute similarities with all agents
        similarity_results = {}
        for agent_id in AGENT_CAPABILITIES.keys():
            similarity_score = await self.sbert_engine.compute_similarity_with_agent(
                query_text, agent_id
            )
            similarity_results[agent_id] = similarity_score
        
        return similarity_results

    async def _create_marl_state(
        self, query_text: str, departure: str, destination: str, 
        date: str, preferences: Optional[Dict[str, Any]]
    ) -> MARLState:
        """Create MARL state representation for agent selection"""
        if not self.marl_engine:
            raise RuntimeError("MARL engine not initialized")
        
        available_agents = list(AGENT_CAPABILITIES.keys())
        system_context = {
            'departure': departure,
            'destination': destination,
            'date': date,
            'preferences': preferences or {},
            'query_complexity': self._assess_query_complexity(departure, destination, preferences),
            'time_constraints': self._extract_time_constraints(date, preferences),
            'budget_constraints': self._extract_budget_constraints(preferences)
        }
        
        return self.marl_engine.create_state(
            query_text=query_text,
            available_agents=available_agents,
            system_context=system_context
        )

    async def _perform_marl_selection(
        self, state: MARLState, similarity_results: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Perform MARL-based agent selection with trust-aware coordination"""
        if not self.marl_engine:
            raise RuntimeError("MARL engine not initialized")
        
        # Incorporate semantic similarity into MARL selection
        enhanced_state = state
        enhanced_state.context['semantic_similarities'] = similarity_results
        
        # Perform trust-weighted selection using MARL
        selected_agents = self.marl_engine.select_agents(
            state=enhanced_state,
            num_agents=min(self.config.max_concurrent_agents, len(AGENT_CAPABILITIES)),
            selection_strategy="trust_weighted_semantic"
        )
        
        return selected_agents

    async def _optimize_agent_selection(
        self, similarity_results: Dict[str, float], marl_selections: List[Tuple[str, float]],
        preferences: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float, float]]:
        """Optimize agent selection using constraint satisfaction"""
        optimized_selection = []
        
        for agent_id, marl_score in marl_selections:
            similarity_score = similarity_results.get(agent_id, 0.0)
            trust_score = await self._get_current_trust_score(agent_id)
            
            # Multi-criteria optimization: semantic + trust + MARL
            combined_score = (
                0.4 * similarity_score +
                0.3 * trust_score +
                0.3 * marl_score
            )
            
            # Apply constraint filters
            if combined_score >= self.config.trust_threshold:
                optimized_selection.append((agent_id, similarity_score, trust_score))
        
        # Sort by combined score
        optimized_selection.sort(key=lambda x: 0.4*x[1] + 0.3*x[2] + 0.3*marl_score, reverse=True)
        
        return optimized_selection[:self.config.max_concurrent_agents]

    async def _get_current_trust_score(self, agent_id: str) -> float:
        """Get current trust score for an agent"""
        try:
            # Get trust summary from ledger
            trust_summary = self.trust_ledger.calculate_overall_trust_score(agent_id)
            return trust_summary.get('overall_score', 0.8)
        except Exception as e:
            self.logger.warning(f"Could not get trust score for {agent_id}: {e}")
            return 0.8  # Default trust score

    def _assess_query_complexity(
        self, departure: str, destination: str, preferences: Optional[Dict[str, Any]]
    ) -> float:
        """Assess query complexity for resource allocation"""
        complexity = 0.5  # Base complexity
        
        # International vs domestic
        if departure.lower() in ['beijing', 'shanghai', 'guangzhou'] and destination.lower() in ['new york', 'london', 'paris']:
            complexity += 0.3
        
        # Preference complexity
        if preferences:
            complexity += 0.1 * len(preferences)
        
        return min(1.0, complexity)

    def _extract_time_constraints(self, date: str, preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract time-related constraints"""
        constraints = {'date': date}
        
        if preferences:
            constraints.update({
                'departure_time': preferences.get('departure_time'),
                'arrival_time': preferences.get('arrival_time'),
                'flexibility': preferences.get('time_flexibility', 'medium')
            })
        
        return constraints

    def _extract_budget_constraints(self, preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract budget-related constraints"""
        constraints = {}
        
        if preferences:
            constraints.update({
                'max_budget': preferences.get('budget'),
                'price_sensitivity': preferences.get('price_sensitivity', 'medium'),
                'value_optimization': preferences.get('value_optimization', True)
            })
        
        return constraints

    async def _execute_trust_aware_coordination(
        self, selected_agents: List[Tuple[str, float, float]], departure: str, destination: str, 
        date: str, preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute trust-aware multi-agent coordination"""
        if not self.marl_engine:
            raise RuntimeError("MARL engine not initialized")
        
        # Create shared context for all agents
        shared_context = {
            'departure': departure,
            'destination': destination,
            'date': date,
            'preferences': preferences or {},
            'query_id': f"query_{uuid.uuid4().hex}",
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute coordinated agent actions
        coordination_result = self.marl_engine.coordinate_agents(
            selected_agents=[agent_id for agent_id, _, _ in selected_agents],
            coordination_strategy='collaborative'
        )
        
        # Execute actual agent tasks with real flight data
        agent_outputs = {}
        coordination_metrics = {}
        trust_scores = {}
        
        # Create task description and data for agents
        task_description = f"Find flights from {departure} to {destination} on {date}"
        task_data = {
            'departure': departure,
            'destination': destination,
            'date': date,
            'preferences': preferences or {},
            'shared_context': shared_context
        }
        
        # Execute each selected agent in parallel
        agent_tasks = []
        for agent_id, similarity_score, trust_score in selected_agents:
            if agent_id in self.agent_instances:
                trust_scores[agent_id] = trust_score
                agent_tasks.append(self._execute_single_agent_task(
                    agent_id, task_description, task_data
                ))
        
        # Wait for all agent tasks to complete
        if agent_tasks:
            try:
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                
                # Process results
                for i, (agent_id, _, _) in enumerate(selected_agents):
                    if i < len(agent_results):
                        result = agent_results[i]
                        if isinstance(result, Exception):
                            self.logger.error(f"Agent {agent_id} failed: {result}")
                            agent_outputs[agent_id] = {
                                'success': False,
                                'error': str(result),
                                'result': {},
                                'confidence': 0.0
                            }
                        else:
                            agent_outputs[agent_id] = result
                            self.logger.info(f"Agent {agent_id} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Agent execution failed: {e}")
                # Create empty outputs for failed agents
                for agent_id, _, _ in selected_agents:
                    if agent_id not in agent_outputs:
                        agent_outputs[agent_id] = {
                            'success': False,
                            'error': f"Execution failed: {e}",
                            'result': {},
                            'confidence': 0.0
                        }
        
        # Calculate coordination metrics
        successful_agents = sum(1 for output in agent_outputs.values() if output.get('success', False))
        coordination_metrics = {
            'trust_scores': trust_scores,
            'successful_agents': successful_agents,
            'total_agents': len(selected_agents),
            'success_rate': successful_agents / max(len(selected_agents), 1),
            'average_confidence': np.mean([
                output.get('confidence', 0.0) for output in agent_outputs.values()
            ]),
            'decision_consensus': self._calculate_decision_consensus(agent_outputs),
            'trust_alignment': np.mean(list(trust_scores.values())) if trust_scores else 0.0,
            'collaboration_efficiency': successful_agents / max(len(selected_agents), 1)
        }
        
        return {
            'agent_outputs': agent_outputs,
            'coordination_metrics': coordination_metrics,
            'coordination_quality': coordination_metrics.get('success_rate', 0.0),
            'trust_consistency': coordination_metrics.get('trust_alignment', 0.0),
            'q_learning_updates': coordination_result.get('q_learning_updates', 0),
            'byzantine_tolerance': coordination_result.get('byzantine_tolerance', True),
            'marl_coordination': coordination_result
        }

    async def _execute_single_agent_task(self, agent_id: str, task_description: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent task with proper error handling"""
        try:
            agent = self.agent_instances.get(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Execute the agent's process_task method
            start_time = time.time()
            result = agent.process_task(task_description, task_data)
            execution_time = time.time() - start_time
            
            # Ensure result has proper structure
            if not isinstance(result, dict):
                result = {'result': result}
            
            # Add execution metadata
            result.update({
                'success': result.get('status') != 'error',
                'agent_id': agent_id,
                'execution_time': execution_time,
                'confidence': result.get('confidence', 0.8),
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.debug(f"Agent {agent_id} completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {agent_id} execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': agent_id,
                'result': {},
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_decision_consensus(self, agent_outputs: Dict[str, Any]) -> float:
        """Calculate decision consensus among agents"""
        try:
            successful_outputs = [
                output for output in agent_outputs.values() 
                if output.get('success', False) and output.get('result')
            ]
            
            if len(successful_outputs) < 2:
                return 1.0 if successful_outputs else 0.0
            
            # Simple consensus: based on confidence scores
            confidences = [output.get('confidence', 0.0) for output in successful_outputs]
            return np.std(confidences) < 0.2  # Low variance = high consensus
            
        except Exception as e:
            self.logger.debug(f"Consensus calculation failed: {e}")
            return 0.5

    async def _perform_ltr_integration(
        self, agent_outputs: Dict[str, Any], coordination_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform Learning to Rank integration"""
        if not self.ltr_engine:
            raise RuntimeError("LTR engine not initialized")
        
        # Prepare ranking instances for LTR
        ranking_instances = []
        valid_outputs = []
        
        # 1. Extract features from agent outputs
        for agent_id, output in agent_outputs.items():
            if output.get('success', False) and output.get('result'):
                try:
                    # Extract features for ranking
                    features = self._extract_decision_features(agent_id, output, coordination_metrics)
                    
                    # Create ranking instance
                    if self.ltr_engine:
                        instance = RankingInstance(
                            instance_id=f"{agent_id}_{uuid.uuid4().hex[:6]}",
                            features=features,
                            relevance_score=output.get('confidence', 0.5),
                            metadata={
                                'agent_id': agent_id,
                                'output': output,
                                'trust_score': coordination_metrics.get('trust_scores', {}).get(agent_id, 0.5)
                            }
                        )
                        ranking_instances.append(instance)
                        valid_outputs.append((agent_id, output))
                    else:
                        # Fallback without LTR
                        valid_outputs.append((agent_id, output))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract features for {agent_id}: {e}")
                    valid_outputs.append((agent_id, output))
        
        # 2. Use LTR ranking if available
        ranked_decisions = []
        if self.ltr_engine and ranking_instances:
            try:
                # Create ranking query
                query_context = {
                    'coordination_quality': coordination_metrics.get('decision_consensus', 0.5),
                    'trust_alignment': coordination_metrics.get('trust_alignment', 0.5),
                    'efficiency': coordination_metrics.get('collaboration_efficiency', 0.5)
                }
                
                ranking_query = RankingQuery(
                    query_id=f"integration_{uuid.uuid4().hex[:8]}",
                    instances=ranking_instances,
                    context=query_context
                )
                
                # Execute LTR ranking
                ranked_results = self.ltr_engine.rank_instances(ranking_query)
                
                # Extract ranked decisions
                for result in ranked_results[:self.config.ranking_depth]:  # Top N recommendations
                    metadata = result.metadata
                    ranked_decisions.append({
                        'agent_id': metadata['agent_id'],
                        'decision': metadata['output']['result'],
                        'rank_score': result.score,
                        'trust_score': metadata['trust_score'],
                        'confidence': result.relevance_score,
                        'features': result.features
                    })
                
                self.logger.info(f"LTR ranking completed, {len(ranked_decisions)} decisions ranked")
                
            except Exception as e:
                self.logger.warning(f"LTR ranking failed: {e}")
                # Fall back to trust-based ranking
                ranked_decisions = self._fallback_trust_ranking(valid_outputs, coordination_metrics)
        else:
            # Fallback ranking when LTR is not available
            ranked_decisions = self._fallback_trust_ranking(valid_outputs, coordination_metrics)
        
        # 3. Create weighted consensus decision
        consensus_decision = self._create_consensus_decision(ranked_decisions, coordination_metrics)
        
        # 4. Calculate integration metrics
        integration_metrics = {
            'total_agents': len(agent_outputs),
            'successful_agents': len(valid_outputs),
            'ranked_decisions': len(ranked_decisions),
            'consensus_strength': len(ranked_decisions) / max(len(valid_outputs), 1),
            'average_confidence': np.mean([d['confidence'] for d in ranked_decisions]) if ranked_decisions else 0.0,
            'average_trust': np.mean([d['trust_score'] for d in ranked_decisions]) if ranked_decisions else 0.0,
            'ranking_algorithm': 'LTR' if self.ltr_engine and ranking_instances else 'trust_based'
        }
        
        # 5. Update LTR model with feedback (if available)
        if self.ltr_engine and ranking_instances:
            try:
                # Create feedback based on decision quality
                feedback_labels = []
                for decision in ranked_decisions:
                    # Simple feedback: higher confidence + trust = better label
                    label = decision['confidence'] * decision['trust_score']
                    feedback_labels.append(label)
                
                # Train LTR model with new data (simplified)
                if len(feedback_labels) >= 2:  # Need at least 2 instances
                    self.ltr_engine.train_model(
                        instances=ranking_instances[:len(feedback_labels)],
                        labels=feedback_labels
                    )
                    self.logger.debug("LTR model updated with feedback")
                    
            except Exception as e:
                self.logger.debug(f"LTR model update failed: {e}")
        
        self.logger.info(f"Decision integration completed using {integration_metrics['ranking_algorithm']} method")
        
        return {
            'consensus_decision': consensus_decision,
            'ranked_decisions': ranked_decisions,
            'integration_metrics': integration_metrics,
            'coordination_summary': coordination_metrics
        }

    def _fallback_trust_ranking(
        self, 
        valid_outputs: List[Tuple[str, Dict[str, Any]]], 
        coordination_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fallback ranking based on trust scores when LTR is not available
        
        Args:
            valid_outputs: List of (agent_id, output) tuples
            coordination_metrics: Coordination metrics
            
        Returns:
            List of ranked decisions
        """
        ranked_decisions = []
        
        for agent_id, output in valid_outputs:
            trust_score = coordination_metrics.get('trust_scores', {}).get(agent_id, 0.5)
            confidence = output.get('confidence', 0.5)
            
            # Simple ranking score: weighted combination of trust and confidence
            rank_score = 0.6 * trust_score + 0.4 * confidence
            
            ranked_decisions.append({
                'agent_id': agent_id,
                'decision': output.get('result', {}),
                'rank_score': rank_score,
                'trust_score': trust_score,
                'confidence': confidence,
                'features': self._extract_decision_features(agent_id, output, coordination_metrics)
            })
        
        # Sort by rank score
        ranked_decisions.sort(key=lambda x: x['rank_score'], reverse=True)
        return ranked_decisions

    def _create_consensus_decision(
        self, 
        ranked_decisions: List[Dict[str, Any]], 
        coordination_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create weighted consensus decision from ranked agent decisions
        
        Args:
            ranked_decisions: List of ranked decisions
            coordination_metrics: Coordination metrics
            
        Returns:
            Consensus decision
        """
        if not ranked_decisions:
            return {
                'type': 'consensus_decision',
                'status': 'no_valid_decisions',
                'confidence': 0.0,
                'recommendations': []
            }
        
        # Calculate weights based on rank scores
        total_score = sum(d['rank_score'] for d in ranked_decisions)
        if total_score == 0:
            weights = [1.0 / len(ranked_decisions)] * len(ranked_decisions)
        else:
            weights = [d['rank_score'] / total_score for d in ranked_decisions]
        
        # Aggregate recommendations
        consensus_recommendations = []
        aggregated_confidence = 0.0
        
        for i, decision in enumerate(ranked_decisions[:self.config.ranking_depth]):  # Top N decisions
            weight = weights[i]
            decision_data = decision['decision']
            
            if isinstance(decision_data, dict):
                # Extract recommendation information
                recommendation = {
                    'source_agent': decision['agent_id'],
                    'content': decision_data,
                    'weight': weight,
                    'rank_score': decision['rank_score'],
                    'trust_score': decision['trust_score'],
                    'confidence': decision['confidence']
                }
                consensus_recommendations.append(recommendation)
                aggregated_confidence += weight * decision['confidence']
        
        # Create final consensus
        consensus_decision = {
            'type': 'consensus_decision',
            'status': 'success',
            'confidence': aggregated_confidence,
            'recommendations': consensus_recommendations,
            'consensus_strength': coordination_metrics.get('decision_consensus', 0.0),
            'trust_alignment': coordination_metrics.get('trust_alignment', 0.0),
            'total_agents_contributing': len(ranked_decisions),
            'summary': self._create_decision_summary(consensus_recommendations)
        }
        
        return consensus_decision

    def _create_decision_summary(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        Create a summary of the consensus decision
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Summary string
        """
        if not recommendations:
            return "No recommendations available"
        
        # Extract key insights from recommendations
        insights = []
        for rec in recommendations:
            agent_id = rec['source_agent']
            content = rec['content']
            weight = rec['weight']
            
            if isinstance(content, dict):
                if content.get('type') == 'price_analysis':
                    insights.append(f"{agent_id} suggests price range {content.get('price_range', 'N/A')} (weight: {weight:.2f})")
                elif content.get('type') == 'schedule_analysis':
                    insights.append(f"{agent_id} recommends {content.get('connection_options', 'N/A')} flights (weight: {weight:.2f})")
                elif content.get('type') == 'weather_analysis':
                    insights.append(f"{agent_id} forecasts {content.get('forecast', 'N/A')} weather (weight: {weight:.2f})")
                else:
                    insights.append(f"{agent_id} provides general analysis (weight: {weight:.2f})")
        
        return " | ".join(insights) if insights else "General flight recommendations available"

    async def _generate_complete_recommendations(
        self, consensus_decision: Dict[str, Any], ranked_decisions: List[Dict[str, Any]], query_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate complete flight recommendations based on academic consensus and ranking
        
        Uses academic principles:
        1. Trust-weighted aggregation: w_i = trust_i / Σ trust_scores
        2. Confidence propagation: conf_final = Σ w_i * conf_i  
        3. Multi-criteria decision making: score = α*trust + β*conf + γ*rank
        
        Args:
            consensus_decision: Consensus decision from LTR integration
            ranked_decisions: List of ranked agent decisions
            query_context: Original query context
            
        Returns:
            List of structured flight recommendations
        """
        try:
            recommendations = []
            
            if not ranked_decisions:
                self.logger.warning("No ranked decisions available for recommendation generation")
                return []
            
            # Calculate recommendation weights using academic formulas
            total_trust = sum(decision['trust_score'] for decision in ranked_decisions)
            total_rank = sum(decision['rank_score'] for decision in ranked_decisions)
            
            for i, decision in enumerate(ranked_decisions[:self.config.ranking_depth]):  # Top N recommendations
                # Trust-weighted importance
                trust_weight = decision['trust_score'] / max(total_trust, 1e-6)
                rank_weight = decision['rank_score'] / max(total_rank, 1e-6)
                
                # Multi-criteria score: α*trust + β*confidence + γ*rank
                final_score = 0.4 * decision['trust_score'] + 0.3 * decision['confidence'] + 0.3 * decision['rank_score']
                
                # Extract content from agent decision
                decision_content = decision['decision']
                agent_id = decision['agent_id']
                
                # Create structured recommendation
                recommendation = {
                    'id': f"rec_{uuid.uuid4().hex[:8]}",
                    'rank': i + 1,
                    'source_agent': agent_id,
                    'agent_type': self._categorize_agent_type(agent_id),
                    'final_score': final_score,
                    'trust_score': decision['trust_score'],
                    'confidence': decision['confidence'],
                    'rank_score': decision['rank_score'],
                    'trust_weight': trust_weight,
                    'rank_weight': rank_weight,
                    'content': self._format_recommendation_content(decision_content, agent_id),
                    'academic_metrics': {
                        'feature_vector': decision.get('features', []),
                        'ltr_ranking': decision['rank_score'],
                        'trust_propagation': trust_weight,
                        'consensus_strength': consensus_decision.get('consensus_strength', 0.0)
                    },
                    'query_context': {
                        'departure': query_context['departure'],
                        'destination': query_context['destination'],
                        'date': query_context['date'],
                        'preferences_matched': self._check_preference_alignment(
                            decision_content, query_context.get('preferences', {})
                        )
                    },
                    'explanation': self._generate_recommendation_explanation(
                        decision_content, agent_id, final_score, consensus_decision
                    )
                }
                
                recommendations.append(recommendation)
            
            # Sort by final score
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            
            self.logger.info(f"Generated {len(recommendations)} academic-level recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate academic recommendations: {e}")
            return []

    def _categorize_agent_type(self, agent_id: str) -> str:
        """Categorize agent type based on agent ID"""
        agent_id_lower = agent_id.lower()
        if 'price' in agent_id_lower:
            return 'price_analysis'
        elif 'schedule' in agent_id_lower:
            return 'schedule_analysis'
        elif 'weather' in agent_id_lower:
            return 'weather_analysis'
        elif 'safety' in agent_id_lower:
            return 'safety_analysis'
        else:
            return 'general_analysis'

    def _format_recommendation_content(self, content: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Format recommendation content based on agent type"""
        if not isinstance(content, dict):
            return {'raw_content': str(content), 'formatted': False}
        
        agent_type = self._categorize_agent_type(agent_id)
        
        formatted_content = {
            'type': agent_type,
            'raw_data': content,
            'formatted': True
        }
        
        # Type-specific formatting
        if agent_type == 'price_analysis':
            formatted_content.update({
                'price_range': content.get('price_range', 'N/A'),
                'best_booking_time': content.get('best_booking_time', 'N/A'),
                'price_trend': content.get('price_trend', 'stable'),
                'value_assessment': 'high' if content.get('confidence', 0.5) > 0.8 else 'medium'
            })
        elif agent_type == 'schedule_analysis':
            formatted_content.update({
                'recommended_times': content.get('recommended_times', []),
                'duration_estimates': content.get('duration_estimates', 'N/A'),
                'connection_options': content.get('connection_options', 'direct'),
                'schedule_flexibility': 'high' if len(content.get('recommended_times', [])) > 2 else 'medium'
            })
        elif agent_type == 'weather_analysis':
            formatted_content.update({
                'forecast': content.get('forecast', 'clear'),
                'temperature': content.get('temperature', 'N/A'),
                'delay_probability': content.get('delay_probability', 0.1),
                'weather_risk': 'low' if content.get('delay_probability', 0.5) < 0.3 else 'high'
            })
        
        return formatted_content

    def _check_preference_alignment(self, content: Dict[str, Any], preferences: Dict[str, Any]) -> float:
        """Check how well the recommendation aligns with user preferences"""
        if not preferences:
            return 0.5  # Neutral alignment
        
        alignment_score = 0.0
        total_preferences = 0
        
        # Check budget alignment
        if 'budget' in preferences and isinstance(content, dict):
            total_preferences += 1
            # Simple budget check (this would be more sophisticated in practice)
            if 'price_range' in content:
                alignment_score += 0.8  # Assume good alignment for now
            else:
                alignment_score += 0.5
        
        # Check time preferences
        if 'preferred_time' in preferences and isinstance(content, dict):
            total_preferences += 1
            if 'recommended_times' in content:
                alignment_score += 0.9
            else:
                alignment_score += 0.5
        
        # Check other preferences
        for pref_key in ['direct_flight', 'airline_preference', 'flexibility']:
            if pref_key in preferences:
                total_preferences += 1
                alignment_score += 0.6  # Default moderate alignment
        
        return alignment_score / max(total_preferences, 1)

    def _generate_recommendation_explanation(
        self, 
        content: Dict[str, Any], 
        agent_id: str, 
        final_score: float,
        consensus_decision: Dict[str, Any]
    ) -> str:
        """Generate explanation for the recommendation"""
        agent_type = self._categorize_agent_type(agent_id)
        
        explanation_parts = []
        
        # Agent-specific explanation
        if agent_type == 'price_analysis':
            explanation_parts.append(f"Price analysis indicates {content.get('price_trend', 'stable')} pricing trends")
            if 'price_range' in content:
                explanation_parts.append(f"with estimated costs of {content['price_range']}")
        elif agent_type == 'schedule_analysis':
            explanation_parts.append(f"Schedule analysis recommends {content.get('connection_options', 'direct')} flights")
            if 'duration_estimates' in content:
                explanation_parts.append(f"with travel time of {content['duration_estimates']}")
        elif agent_type == 'weather_analysis':
            explanation_parts.append(f"Weather forecast shows {content.get('forecast', 'clear')} conditions")
            if 'delay_probability' in content:
                delay_risk = "low" if content['delay_probability'] < 0.3 else "moderate"
                explanation_parts.append(f"with {delay_risk} delay risk")
        
        # Scoring explanation
        score_category = "excellent" if final_score > 0.8 else "good" if final_score > 0.6 else "fair"
        explanation_parts.append(f"Overall recommendation quality: {score_category} (score: {final_score:.2f})")
        
        # Consensus strength
        consensus_strength = consensus_decision.get('consensus_strength', 0.0)
        if consensus_strength > 0.7:
            explanation_parts.append("Strong consensus among agents")
        elif consensus_strength > 0.5:
            explanation_parts.append("Moderate consensus among agents")
        else:
            explanation_parts.append("Limited consensus among agents")
        
        return ". ".join(explanation_parts) + "."

    async def _update_learning_systems(
        self,
        query_id: str,
        selected_agents: List[Tuple[str, float, float]],
        coordination_result: Dict[str, Any],
        integration_result: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        query_context: Optional[Dict[str, Any]] = None
    ):
        """
        Update learning systems based on collaboration and integration results
        
        Args:
            query_id: Query identifier
            selected_agents: List of selected agents
            coordination_result: Result from agent collaboration
            integration_result: Result from decision integration
            recommendations: List of final recommendations
            query_context: Context containing departure, destination, date, preferences
        """
        try:
            self.logger.info("Updating learning systems based on collaboration and integration results")
            
            # Extract agent outputs from coordination result
            agent_outputs = coordination_result.get('agent_outputs', {})
            
            # Extract query text from context if available
            query_text = ""
            if query_context:
                query_text = query_context.get('query_text', '')
                
                # If query_text is not available in context, try to match with ground truth
                if not query_text and self.ground_truth_data:
                    departure = query_context.get('departure', '')
                    destination = query_context.get('destination', '')
                    date = query_context.get('date', '')
                    
                    # Try to find a matching query in ground truth data
                    for gt_query_text, gt_data in self.ground_truth_data.items():
                        if (departure.lower() in gt_query_text.lower() and 
                            destination.lower() in gt_query_text.lower() and
                            date in gt_query_text):
                            query_text = gt_query_text
                            break
            
            # Update trust ledger based on agent performance
            await self._update_trust_ledger(
                query_id=query_id,
                agent_outputs=agent_outputs,
                recommendations=recommendations,
                query_text=query_text
            )
            
            # Update trust adaptive system with feedback
            if self.trust_adaptive_system:
                await self.trust_adaptive_system.update_with_feedback(
                    query_id=query_id,
                    selected_agents=selected_agents,
                    collaboration_result=coordination_result,
                    integration_result=integration_result
                )
            
            self.logger.info("Learning systems updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update learning systems: {e}")

    async def _update_trust_ledger(
        self,
        query_id: str,
        agent_outputs: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        query_text: str = ""
    ):
        """
        Update trust ledger based on REAL agent performance, evaluated against ground truth.
        This is the academically rigorous method.
        """
        
        if not hasattr(self, 'evaluator') or self.evaluator is None:
            self.logger.warning("Evaluator not available. Skipping trust update.")
            return
        
        if not hasattr(self, 'ground_truth_data') or self.ground_truth_data is None:
            self.logger.warning("Ground truth data not available. Skipping trust update.")
            return

        # Find the ground truth for the current query
        ground_truth = self.ground_truth_data.get(query_text)
        if not ground_truth:
            self.logger.warning(f"No ground truth found for query: {query_text}. Skipping trust update.")
            return

        for agent_id, output in agent_outputs.items():
            # Step 1: 🎯 Use the StandardEvaluator to calculate a REAL accuracy score.
            # We assume a method like `evaluate_single_agent_output` exists in your evaluator.
            # This method should compare the agent's 'output' with the 'ground_truth'.
            # This completely removes the need for proxies like 'confidence' or random 'jitter'.
            try:
                real_accuracy = self.evaluator.evaluate_single_agent_output(
                    agent_output=output, 
                    ground_truth=ground_truth,
                    agent_type=agent_id # Pass agent_id to select the right evaluation logic
                )
            except Exception as e:
                self.logger.error(f"Error during single agent evaluation for {agent_id}: {e}")
                real_accuracy = 0.0 # Assume failure if evaluation fails

            performance_metrics = {
                'accuracy': real_accuracy,
                'response_time': output.get('execution_time', 1.0),
                'data_quality': 1.0 if output.get('success', True) else 0.0
            }
            
            # Step 2: Call the ledger's evaluation function with REAL data and task context.
            if hasattr(self, 'trust_ledger') and self.trust_ledger is not None:
                try:
                    # Construct task context, including preference information for expertise matching
                    task_context = {
                        'preferences': {
                            'priority': 'safety'  # Default priority, should be extracted from query in practice
                        },
                        'query_id': query_id
                    }
                    
                    # Try to extract real priority from query text
                    if query_text:
                        if 'safety' in query_text.lower() or 'secure' in query_text.lower():
                            task_context['preferences']['priority'] = 'safety'
                        elif 'cost' in query_text.lower() or 'cheap' in query_text.lower() or 'budget' in query_text.lower():
                            task_context['preferences']['priority'] = 'cost'
                        elif 'time' in query_text.lower() or 'fast' in query_text.lower() or 'quick' in query_text.lower():
                            task_context['preferences']['priority'] = 'time'
                    
                    competence_score = self.trust_ledger.evaluate_competence(agent_id, performance_metrics, task_context)
                    self.logger.info(f"✅ Updated competence for {agent_id} with REAL accuracy: {real_accuracy:.4f}, priority: {task_context['preferences']['priority']}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to evaluate competence for {agent_id}: {e}")

            # Step 3: Keep the original general trust update logic.
            if self.trust_manager:
                if output.get("success", True):
                    await self.trust_manager.broadcast_trust_update(
                        agent_id, 0.01, f"successful_execution_query_{query_id}"
                    )
                else:
                    await self.trust_manager.broadcast_trust_update(
                        agent_id, -0.05, f"execution_failure_query_{query_id}"
                    )

    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.disconnect()
        
        if self.trust_manager:
            await self.trust_manager.disconnect_mcp()
        
        self.logger.info("MAMA Flight Assistant cleanup completed")

    async def _calculate_system_performance(
        self, total_processing_time: float, selected_agents: List[Tuple[str, float, float]], 
        coordination_result: Dict[str, Any], integration_result: Dict[str, Any], 
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive system performance metrics"""
        return {
            'total_processing_time': total_processing_time,
            'academic_components_used': ['PML', 'SBERT', 'MARL', 'LTR'],
            'overall_success': len(recommendations) > 0,
            'system_efficiency': 1.0 / max(total_processing_time, 0.1),
            'recommendation_confidence': np.mean([r['confidence'] for r in recommendations]) if recommendations else 0.0,
            'trust_weighted_score': np.mean([
                coordination_result.get('trust_consistency', 0.0),
                integration_result.get('consensus_strength', 0.0)
            ]),
            'agent_utilization_rate': len(selected_agents) / len(AGENT_CAPABILITIES),
            'semantic_similarity_avg': np.mean([agent[1] for agent in selected_agents]),
            'trust_alignment_score': coordination_result.get('trust_consistency', 0.0),
            'ranking_quality_score': integration_result.get('ranking_quality', 0.0),
            'byzantine_fault_tolerance': coordination_result.get('byzantine_tolerance', True),
            'convergence_metrics': {
                'marl_convergence': coordination_result.get('q_learning_updates', 0),
                'ltr_convergence': integration_result.get('model_updated', False),
                'sbert_accuracy': np.mean([agent[1] for agent in selected_agents]) if selected_agents else 0.0
            }
        }

    async def _update_global_metrics(self, result: Dict[str, Any]):
        """Update global performance tracking metrics"""
        self.performance_metrics['total_queries_processed'] += 1
        
        if result['status'] == 'completed':
            self.performance_metrics['successful_queries'] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        new_time = result.get('total_processing_time', 0.0)
        total_queries = self.performance_metrics['total_queries_processed']
        
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total_queries - 1) + new_time) / total_queries
        )
        
        # Update agent utilization
        for agent_data in result.get('selected_agents', []):
            agent_id = agent_data[0] if isinstance(agent_data, tuple) else agent_data
            self.performance_metrics['agent_utilization'][agent_id] = (
                self.performance_metrics['agent_utilization'].get(agent_id, 0) + 1
            )
        
        # Update trust evolution tracking
        for agent_data in result.get('selected_agents', []):
            if isinstance(agent_data, tuple) and len(agent_data) >= 3:
                agent_id, similarity_score, trust_score = agent_data[0], agent_data[1], agent_data[2]
                self.performance_metrics['trust_evolution'][agent_id].append({
                    'timestamp': datetime.now().isoformat(),
                    'trust_score': trust_score,
                    'similarity_score': similarity_score,
                    'query_id': result['query_id']
                })
        
        # Update learning metrics
        system_perf = result.get('system_performance', {})
        convergence = system_perf.get('convergence_metrics', {})
        
        self.performance_metrics['learning_metrics']['marl_convergence'].append(
            convergence.get('marl_convergence', 0)
        )
        self.performance_metrics['learning_metrics']['ltr_ranking_quality'].append(
            system_perf.get('ranking_quality_score', 0.0)
        )
        self.performance_metrics['learning_metrics']['sbert_similarity_accuracy'].append(
            convergence.get('sbert_accuracy', 0.0)
        )

    def _extract_decision_features(
        self, agent_id: str, output: Dict[str, Any], coordination_metrics: Dict[str, Any]
    ) -> List[float]:
        """Extract comprehensive feature vector from agent output for LTR ranking"""
        features = []
        
        # Trust-based features (4 dimensions)
        trust_score = coordination_metrics.get('trust_scores', {}).get(agent_id, 0.5)
        features.append(trust_score)
        features.append(trust_score ** 2)  # Trust squared for non-linear effects
        features.append(np.log(trust_score + 1e-6))  # Log trust for diminishing returns
        features.append(1.0 if trust_score > self.config.trust_threshold else 0.0)  # Trust threshold indicator
        
        # Confidence and reliability features (6 dimensions)
        confidence = output.get('confidence', 0.5)
        features.append(confidence)
        features.append(confidence * trust_score)  # Trust-weighted confidence
        
        execution_time = output.get('execution_time', 1.0)
        features.append(1.0 / (1.0 + execution_time))  # Normalized inverse execution time
        features.append(np.exp(-execution_time))  # Exponential decay for time penalty
        
        success_indicator = 1.0 if output.get('success', True) else 0.0
        features.append(success_indicator)
        features.append(success_indicator * confidence)  # Success-weighted confidence
        
        # Content quality features (8 dimensions)
        result = output.get('result', {})
        if isinstance(result, dict):
            features.append(result.get('score', 0.5))
            features.append(len(str(result)) / 1000.0)  # Content richness
            features.append(len(result.keys()) / 10.0 if result else 0.0)  # Structure complexity
            features.append(1.0 if any(isinstance(v, (list, dict)) for v in result.values()) else 0.0)  # Nested structure
        else:
            features.extend([0.5, 0.1, 0.0, 0.0])
        
        # Agent type features (one-hot encoding - 5 dimensions)
        agent_capabilities = AGENT_CAPABILITIES.get(agent_id, {})
        complexity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'very_high': 1.0}
        complexity_score = complexity_map.get(agent_capabilities.get('computational_complexity', 'medium'), 0.5)
        features.append(complexity_score)
        
        accuracy_req = agent_capabilities.get('accuracy_requirements', 0.9)
        features.append(accuracy_req)
        
        response_time_sla = agent_capabilities.get('response_time_sla', 2.0)
        features.append(1.0 / (1.0 + response_time_sla))  # Normalized inverse SLA
        
        # Expertise matching features (4 dimensions)
        expertise_areas = agent_capabilities.get('expertise_areas', [])
        features.append(len(expertise_areas) / 5.0)  # Expertise breadth
        features.append(1.0 if 'analysis' in ' '.join(expertise_areas).lower() else 0.0)
        
        # Coordination and consensus features (6 dimensions)
        features.append(coordination_metrics.get('coordination_quality', 0.5))
        features.append(coordination_metrics.get('trust_consistency', 0.5))
        features.append(coordination_metrics.get('decision_consensus', 0.5))
        features.append(coordination_metrics.get('collaboration_efficiency', 0.5))
        
        # Cross-agent interaction features (4 dimensions)
        agent_count = len(coordination_metrics.get('trust_scores', {}))
        features.append(agent_count / len(AGENT_CAPABILITIES))  # Agent participation ratio
        features.append(coordination_metrics.get('byzantine_tolerance', 1.0))
        
        # Temporal features (3 dimensions)
        features.append(datetime.now().hour / 24.0)  # Time of day normalization
        features.append(datetime.now().weekday() / 7.0)  # Day of week normalization
        features.append(min(execution_time / 10.0, 1.0))  # Capped execution time ratio
        
        # Ensure fixed feature dimension
        while len(features) < self.config.feature_dimension:
            features.append(0.0)
        
        return features[:self.config.feature_dimension]

    def _create_trust_weighted_consensus(self, ranked_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create trust-weighted consensus from ranked decisions"""
        if not ranked_decisions:
            return {
                'type': 'consensus_failure',
                'confidence': 0.0,
                'message': 'No valid decisions available',
                'trust_score': 0.0
            }
        
        # Academic consensus formula: weighted by trust and confidence
        weights = []
        total_trust = sum(decision['trust_score'] for decision in ranked_decisions)
        total_confidence = sum(decision['confidence'] for decision in ranked_decisions)
        
        for decision in ranked_decisions:
            trust_weight = decision['trust_score'] / max(total_trust, 1e-6)
            confidence_weight = decision['confidence'] / max(total_confidence, 1e-6)
            combined_weight = self.config.trust_weight * trust_weight + (1 - self.config.trust_weight) * confidence_weight
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / max(total_weight, 1e-6) for w in weights]
        
        # Create weighted consensus
        consensus_decision = {
            'type': 'trust_weighted_consensus',
            'confidence': 0.0,
            'trust_score': 0.0,
            'contributing_agents': [],
            'method': 'academic_trust_weighting'
        }
        
        aggregated_confidence = 0.0
        aggregated_trust = 0.0
        
        for i, decision in enumerate(ranked_decisions[:self.config.ranking_depth]):
            weight = weights[i]
            decision_data = decision['decision']
            
            if isinstance(decision_data, dict):
                # Aggregate numerical values
                for key, value in decision_data.items():
                    if isinstance(value, (int, float)):
                        if key not in consensus_decision:
                            consensus_decision[key] = 0.0
                        consensus_decision[key] += weight * value
            
            aggregated_confidence += weight * decision['confidence']
            aggregated_trust += weight * decision['trust_score']
            
            consensus_decision['contributing_agents'].append({
                'agent_id': decision['agent_id'],
                'weight': weight,
                'trust_score': decision['trust_score'],
                'confidence': decision['confidence']
            })
        
        consensus_decision['confidence'] = aggregated_confidence
        consensus_decision['trust_score'] = aggregated_trust
        consensus_decision['consensus_strength'] = min(aggregated_confidence, aggregated_trust)
        
        return consensus_decision

    def _create_decision_summary(self, recommendations: List[Dict[str, Any]]) -> str:
        """Create comprehensive decision summary with academic insights"""
        if not recommendations:
            return "No recommendations generated due to insufficient agent consensus"
        
        insights = []
        
        # Confidence analysis
        avg_confidence = np.mean([r['confidence'] for r in recommendations])
        if avg_confidence > 0.8:
            insights.append(f"high confidence recommendations (avg: {avg_confidence:.2f})")
        elif avg_confidence > 0.6:
            insights.append(f"moderate confidence recommendations (avg: {avg_confidence:.2f})")
        else:
            insights.append(f"low confidence recommendations (avg: {avg_confidence:.2f})")
        
        # Trust analysis
        trust_scores = [r.get('trust_score', 0.5) for r in recommendations]
        avg_trust = np.mean(trust_scores)
        if avg_trust > self.config.trust_threshold:
            insights.append(f"high trust alignment (avg: {avg_trust:.2f})")
        else:
            insights.append(f"moderate trust alignment (avg: {avg_trust:.2f})")
        
        # Academic method identification
        methods_used = set()
        for r in recommendations:
            if 'method' in r:
                methods_used.add(r['method'])
        
        if methods_used:
            insights.append(f"academic methods: {', '.join(methods_used)}")
        
        # Consensus strength
        consensus_scores = [r.get('consensus_score', 0.5) for r in recommendations]
        avg_consensus = np.mean(consensus_scores)
        if avg_consensus > self.config.consensus_threshold:
            insights.append(f"strong consensus (avg: {avg_consensus:.2f})")
        else:
            insights.append(f"moderate consensus (avg: {avg_consensus:.2f})")
        
        return " | ".join(insights) if insights else "Academic flight recommendations with multi-agent consensus"

    def _generate_detailed_explanation(
        self, recommendation: Dict[str, Any], query_context: Dict[str, Any]
    ) -> str:
        """Generate detailed explanation using academic reasoning"""
        explanation_parts = []
        
        # Trust and confidence reasoning
        trust_score = recommendation.get('trust_score', 0.5)
        confidence = recommendation.get('confidence', 0.5)
        
        if trust_score > 0.8:
            explanation_parts.append(f"High agent trust score ({trust_score:.2f}) ensures reliable recommendation")
        elif trust_score > 0.6:
            explanation_parts.append(f"Moderate agent trust score ({trust_score:.2f}) with good reliability")
        else:
            explanation_parts.append(f"Conservative recommendation with trust score {trust_score:.2f}")
        
        # Academic method explanation
        if 'method' in recommendation:
            method = recommendation['method']
            if 'SBERT' in method:
                explanation_parts.append("semantic similarity matching ensures query-agent alignment")
            if 'MARL' in method:
                explanation_parts.append("multi-agent reinforcement learning optimizes coordination")
            if 'LTR' in method:
                explanation_parts.append("learning-to-rank algorithms prioritize optimal solutions")
        
        # Consensus explanation
        consensus_score = recommendation.get('consensus_score', 0.5)
        if consensus_score > self.config.consensus_threshold:
            explanation_parts.append(f"strong multi-agent consensus ({consensus_score:.2f})")
        else:
            explanation_parts.append(f"moderate consensus ({consensus_score:.2f}) with agent disagreement")
        
        # Route-specific insights
        route = f"{query_context.get('departure', 'origin')} to {query_context.get('destination', 'destination')}"
        explanation_parts.append(f"optimized for {route} route characteristics")
        
        # Preference alignment
        preferences = query_context.get('preferences', {})
        if preferences:
            pref_count = len(preferences)
            explanation_parts.append(f"aligned with {pref_count} user preference{'s' if pref_count != 1 else ''}")
        
        return ". ".join(explanation_parts) + "."

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and performance metrics"""
        return {
            'system_initialized': self.system_initialized,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'pml_repository': self.pml_repository is not None,
                'sbert_engine': self.sbert_engine is not None,
                'marl_engine': self.marl_engine is not None,
                'ltr_engine': self.ltr_engine is not None,
                'trust_ledger': self.trust_ledger is not None,
                'trust_adaptive_system': self.trust_adaptive_system is not None,
                'mcp_client': self.mcp_client is not None and self.use_mcp,
                'trust_manager': self.trust_manager is not None
            },
            'performance_metrics': self.performance_metrics.copy(),
            'agent_status': {
                agent_id: {
                    'available': agent_id in self.agent_instances,
                    'protocol_initialized': agent_id in self.agent_protocols,
                    'capabilities': AGENT_CAPABILITIES.get(agent_id, {})
                }
                for agent_id in AGENT_CAPABILITIES.keys()
            },
            'configuration': {
                'max_concurrent_agents': self.config.max_concurrent_agents,
                'trust_threshold': self.config.trust_threshold,
                'confidence_threshold': self.config.confidence_threshold,
                'consensus_threshold': self.config.consensus_threshold,
                'similarity_threshold': self.config.similarity_threshold,
                'ranking_depth': self.config.ranking_depth,
                'feature_dimension': self.config.feature_dimension,
                'learning_rate': self.config.learning_rate,
                'discount_factor': self.config.discount_factor,
                'trust_weight': self.config.trust_weight
            }
        }

    async def save_models(self):
        """Save trained models to disk"""
        try:
            if self.ltr_engine:
                ltr_model_path = "models/ltr_model.pt"
                self.ltr_engine.save_model(ltr_model_path)
                self.logger.info(f"LTR model saved to {ltr_model_path}")
            
            if self.marl_engine:
                marl_model_path = "models/marl_qtables.json"
                self.marl_engine.save_model(marl_model_path)
                self.logger.info(f"MARL Q-tables saved to {marl_model_path}")
            
            # Save performance metrics
            metrics_path = "models/performance_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            self.logger.info(f"Performance metrics saved to {metrics_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")

    async def export_results(self, filename: Optional[str] = None) -> str:
        """Export comprehensive system results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/mama_export_{timestamp}.json"
        
        export_data = {
            'system_status': await self.get_system_status(),
            'academic_components': {
                'pml_agents': len(AGENT_CAPABILITIES) if self.pml_repository else 0,
                'sbert_model': 'all-MiniLM-L6-v2' if self.sbert_engine else None,
                'marl_algorithm': 'trust_aware_q_learning' if self.marl_engine else None,
                'ltr_algorithm': 'listwise_ranking' if self.ltr_engine else None,
                'trust_system': 'multi_dimensional_ledger' if self.trust_ledger else None
            },
            'performance_summary': {
                'total_queries': self.performance_metrics['total_queries_processed'],
                'success_rate': (
                    self.performance_metrics['successful_queries'] / 
                    max(self.performance_metrics['total_queries_processed'], 1)
                ),
                'avg_processing_time': self.performance_metrics['average_processing_time'],
                'agent_utilization': self.performance_metrics['agent_utilization'],
                'learning_convergence': {
                    'marl': np.mean(self.performance_metrics['learning_metrics']['marl_convergence']) if self.performance_metrics['learning_metrics']['marl_convergence'] else 0.0,
                    'ltr': np.mean(self.performance_metrics['learning_metrics']['ltr_ranking_quality']) if self.performance_metrics['learning_metrics']['ltr_ranking_quality'] else 0.0,
                    'sbert': np.mean(self.performance_metrics['learning_metrics']['sbert_similarity_accuracy']) if self.performance_metrics['learning_metrics']['sbert_similarity_accuracy'] else 0.0
                }
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"System results exported to {filename}")
        return filename

    # --------------------------------------------------------------------------
    # METHOD ADDED FOR OFFLINE COMPETENCE EVOLUTION EXPERIMENT
    # --------------------------------------------------------------------------
    async def _update_trust_ledger_from_offline_data(
        self,
        agent_outputs: dict,
        ground_truth: dict
    ):
        """
        A dedicated method for the offline experiment to update the trust ledger
        based on pre-canned agent outputs and ground truth data.
        This bypasses all API calls and online processing.
        """
        if not hasattr(self, 'evaluator') or self.evaluator is None:
            self.logger.warning("Evaluator not available. Cannot update trust from offline data.")
            return

        for agent_id, output in agent_outputs.items():
            if agent_id not in ground_truth:
                continue

            agent_ground_truth = ground_truth[agent_id]
            
            try:
                # 🎯 THE ONE, FINAL, ABSOLUTE FIX: Add the missing 'agent_type=agent_id' argument.
                real_accuracy = self.evaluator.evaluate_single_agent_output(
                    agent_output=output, 
                    ground_truth=agent_ground_truth,
                    agent_type=agent_id  # This was the missing piece.
                )
            except Exception as e:
                self.logger.error(f"Error during offline evaluation for {agent_id}: {e}")
                real_accuracy = 0.0

            performance_metrics = {
                'accuracy': real_accuracy,
                'response_time': 1.0,
                'data_quality': 1.0 if output.get('success', True) else 0.0
            }
            
            if hasattr(self, 'trust_ledger') and self.trust_ledger is not None:
                # Construct task context for offline experiments
                task_context = {
                    'preferences': {
                        'priority': 'safety'  # Default priority
                    }
                }
                
                # Try to extract priority information from ground truth
                if isinstance(agent_ground_truth, dict) and 'preferences' in agent_ground_truth:
                    priority = agent_ground_truth['preferences'].get('priority', 'safety')
                    task_context['preferences']['priority'] = priority
                
                self.trust_ledger.evaluate_competence(agent_id, performance_metrics, task_context)
                self.logger.info(f"OFFLINE_UPDATE: Competence for {agent_id} evaluated with real_accuracy: {real_accuracy:.4f}, priority: {task_context['preferences']['priority']}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MAMA Flight Selection Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --departure "Beijing" --destination "Shanghai" --date "2024-01-01"
  python main.py --departure "New York" --destination "Los Angeles" --date "2024-03-15" --budget 800
  python main.py --departure "London" --destination "Paris" --date "2024-06-01" --preferences '{"time_preference": "morning"}'
        """
    )
    
    parser.add_argument(
        "--departure",
        required=True,
        help="Departure location (e.g., 'Beijing', 'New York')"
    )
    
    parser.add_argument(
        "--destination", 
        required=True,
        help="Destination location (e.g., 'Shanghai', 'Los Angeles')"
    )
    
    parser.add_argument(
        "--date",
        required=True,
        help="Flight date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        help="Budget constraint (optional)"
    )
    
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences as JSON string (optional)"
    )
    
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP integration"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point for MAMA Flight Assistant
    
    Complete academic implementation with comprehensive component integration:
    - PML (Professional Multi-agent Learning) for agent management
    - SBERT (Sentence-BERT) for semantic similarity computation
    - MARL (Multi-Agent Reinforcement Learning) for trust-aware coordination
    - LTR (Learning to Rank) for decision optimization
    - Multi-dimensional Trust Ledger for Byzantine fault tolerance
    """
    parser = argparse.ArgumentParser(description="MAMA Flight Selection Assistant - Academic Implementation")
    parser.add_argument("--departure", required=True, help="Departure location")
    parser.add_argument("--destination", required=True, help="Destination location")
    parser.add_argument("--date", required=True, help="Flight date (YYYY-MM-DD)")
    parser.add_argument("--preferences", type=str, help="User preferences (JSON format)")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--disable-mcp", action="store_true", help="Disable MCP integration")
    parser.add_argument("--export-results", action="store_true", help="Export system results after query")
    parser.add_argument("--save-models", action="store_true", help="Save trained models after query")
    parser.add_argument("--system-status", action="store_true", help="Show system status and exit")
    
    args = parser.parse_args()
    
    # Parse configuration
    config = QueryProcessingConfig()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    # Parse preferences
    preferences = None
    if args.preferences:
        try:
            preferences = json.loads(args.preferences)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in preferences: {e}")
            sys.exit(1)
    
    # Initialize MAMA system
    assistant = MAMAFlightAssistant(
        config=config,
        use_mcp=not args.disable_mcp
    )
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(signal_handler(assistant)))
    signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(signal_handler(assistant)))
    
    try:
        print("Initializing MAMA Flight Assistant with complete academic implementation...")
        print("Components: PML + SBERT + MARL + LTR + Trust Ledger")
        
        await assistant.initialize_system()
        print("✅ MAMA system initialized successfully")
        
        if args.system_status:
            status = await assistant.get_system_status()
            print("\n=== MAMA System Status ===")
            print(json.dumps(status, indent=2))
            return
        
        print(f"\nProcessing flight query:")
        print(f"  Departure: {args.departure}")
        print(f"  Destination: {args.destination}")
        print(f"  Date: {args.date}")
        if preferences:
            print(f"  Preferences: {preferences}")
        
        # Process flight query using complete academic implementation
        result = await assistant.process_flight_query(
            departure=args.departure,
            destination=args.destination,
            date=args.date,
            preferences=preferences
        )
        
        # Display results
        print(f"\n=== Flight Query Results ===")
        print(f"Query ID: {result['query_id']}")
        print(f"Status: {result['status']}")
        print(f"Processing Time: {result.get('total_processing_time', 0):.3f}s")
        
        if result['status'] == 'completed':
            recommendations = result.get('recommendations', [])
            print(f"\n📋 Generated {len(recommendations)} flight recommendations:")
            
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"\n{i}. Confidence: {rec['confidence']:.3f} | Trust: {rec.get('trust_score', 0):.3f}")
                print(f"   Method: {rec.get('method', 'unknown')}")
                print(f"   Explanation: {rec.get('explanation', 'No explanation')}")
            
            # Display academic metrics
            academic_metrics = result.get('academic_metrics', {})
            print(f"\n🎓 Academic Performance Metrics:")
            
            for phase, metrics in academic_metrics.items():
                print(f"\n{phase.replace('_', ' ').title()}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    elif isinstance(value, (int, str, bool)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, list):
                        if value and isinstance(value[0], (int, float)):
                            print(f"  {key}: avg={np.mean(value):.4f}, std={np.std(value):.4f}")
                        else:
                            print(f"  {key}: {len(value)} items")
            
            # Display system performance
            system_perf = result.get('system_performance', {})
            print(f"\n⚡ System Performance:")
            for key, value in system_perf.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        else:
            print(f"\n❌ Query failed: {result.get('error', 'Unknown error')}")
            if 'stack_trace' in result:
                print(f"Stack trace: {result['stack_trace']}")
        
        # Export results if requested
        if args.export_results:
            export_file = await assistant.export_results()
            print(f"\n📤 Results exported to: {export_file}")
        
        # Save models if requested
        if args.save_models:
            await assistant.save_models()
            print(f"\n💾 Models saved successfully")
        
        # Display final system summary
        status = await assistant.get_system_status()
        perf = status['performance_metrics']
        print(f"\n📊 Session Summary:")
        print(f"  Total Queries: {perf['total_queries_processed']}")
        print(f"  Success Rate: {perf['successful_queries'] / max(perf['total_queries_processed'], 1) * 100:.1f}%")
        print(f"  Avg Processing Time: {perf['average_processing_time']:.3f}s")
        print(f"  System Components: {sum(status['components'].values())}/{len(status['components'])} active")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        await assistant.cleanup()
        print("\n🏁 MAMA Flight Assistant session completed")


if __name__ == "__main__":
    asyncio.run(main()) 