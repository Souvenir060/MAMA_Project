#!/usr/bin/env python3
"""
PML (Prompt Markup Language) System

Academic implementation of Prompt Markup Language for agent role definition and task assignment.
Provides structured recognition of named items and categorization of AI agent capabilities.

Formula: PML = {agent: agent_name, specialty: expertise_area, output: output_type}
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class PMLAgent:
    """PML Agent definition structure"""
    agent_name: str
    specialty: str
    expertise_area: str
    output_type: str
    input_types: List[str]
    capabilities: List[str]
    trust_dimensions: List[str]
    performance_history: Dict[str, float]
    created_at: datetime
    updated_at: datetime

@dataclass
class PMLTask:
    """PML Task definition structure"""
    task_id: str
    task_type: str
    required_expertise: List[str]
    input_data: Dict[str, Any]
    expected_output_type: str
    complexity_level: float
    priority: int
    created_at: datetime

@dataclass
class PMLAssignment:
    """PML Task assignment structure"""
    assignment_id: str
    task_id: str
    agent_name: str
    assignment_score: float
    confidence: float
    reasoning: str
    created_at: datetime

class PMLRepository:
    """Central repository for PML agent profiles and task definitions"""
    
    def __init__(self):
        """Initialize PML repository"""
        self.agents: Dict[str, PMLAgent] = {}
        self.tasks: Dict[str, PMLTask] = {}
        self.assignments: Dict[str, PMLAssignment] = {}
        self.expertise_index: Dict[str, Set[str]] = defaultdict(set)
        self.capability_matrix: Dict[str, Dict[str, float]] = {}
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        
    def register_agent(self, agent_data: Dict[str, Any]) -> str:
        """
        Register a new agent in the PML system
        
        Args:
            agent_data: Agent configuration dictionary
            
        Returns:
            Agent name/ID
        """
        try:
            agent = PMLAgent(
                agent_name=agent_data['agent_name'],
                specialty=agent_data['specialty'],
                expertise_area=agent_data['expertise_area'],
                output_type=agent_data['output_type'],
                input_types=agent_data.get('input_types', []),
                capabilities=agent_data.get('capabilities', []),
                trust_dimensions=agent_data.get('trust_dimensions', []),
                performance_history=agent_data.get('performance_history', {}),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.agents[agent.agent_name] = agent
            
            # Update expertise index
            self.expertise_index[agent.expertise_area].add(agent.agent_name)
            for capability in agent.capabilities:
                self.expertise_index[capability].add(agent.agent_name)
            
            # Initialize capability matrix
            self._update_capability_matrix(agent)
            
            logger.info(f"Registered PML agent: {agent.agent_name}")
            return agent.agent_name
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            raise
    
    def create_task(self, task_data: Dict[str, Any]) -> str:
        """
        Create a new task in the PML system
        
        Args:
            task_data: Task definition dictionary
            
        Returns:
            Task ID
        """
        try:
            task_id = f"task_{int(time.time())}_{hashlib.md5(str(task_data).encode()).hexdigest()[:8]}"
            
            task = PMLTask(
                task_id=task_id,
                task_type=task_data['task_type'],
                required_expertise=task_data.get('required_expertise', []),
                input_data=task_data.get('input_data', {}),
                expected_output_type=task_data['expected_output_type'],
                complexity_level=task_data.get('complexity_level', 0.5),
                priority=task_data.get('priority', 1),
                created_at=datetime.now()
            )
            
            self.tasks[task_id] = task
            logger.info(f"Created PML task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            raise
    
    def assign_optimal_agent(self, task_id: str) -> Optional[PMLAssignment]:
        """
        Assign optimal agent to task using PML matching algorithm
        
        Args:
            task_id: Task identifier
            
        Returns:
            PML assignment object
        """
        try:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            # Calculate assignment scores for all agents
            agent_scores = {}
            for agent_name, agent in self.agents.items():
                score = self._calculate_assignment_score(agent, task)
                agent_scores[agent_name] = score
            
            # Select best agent
            if not agent_scores:
                return None
                
            best_agent = max(agent_scores, key=agent_scores.get)
            best_score = agent_scores[best_agent]
            
            # Create assignment
            assignment_id = f"assign_{task_id}_{best_agent}_{int(time())}"
            assignment = PMLAssignment(
                assignment_id=assignment_id,
                task_id=task_id,
                agent_name=best_agent,
                assignment_score=best_score,
                confidence=self._calculate_confidence(best_agent, task),
                reasoning=self._generate_assignment_reasoning(best_agent, task, best_score),
                created_at=datetime.now()
            )
            
            self.assignments[assignment_id] = assignment
            logger.info(f"Assigned task {task_id} to agent {best_agent} (score: {best_score:.3f})")
            
            return assignment
            
        except Exception as e:
            logger.error(f"Failed to assign agent to task {task_id}: {e}")
            return None
    
    def _calculate_assignment_score(self, agent: PMLAgent, task: PMLTask) -> float:
        """
        Calculate assignment score between agent and task
        Implementation of PML formula components
        
        Args:
            agent: PML agent
            task: PML task
            
        Returns:
            Assignment score [0, 1]
        """
        try:
            score = 0.0
            
            # Expertise matching (40% weight)
            expertise_score = self._calculate_expertise_match(agent, task)
            score += 0.4 * expertise_score
            
            # Output type compatibility (25% weight)
            output_score = self._calculate_output_compatibility(agent, task)
            score += 0.25 * output_score
            
            # Input type compatibility (20% weight)
            input_score = self._calculate_input_compatibility(agent, task)
            score += 0.2 * input_score
            
            # Performance history (15% weight)
            performance_score = self._calculate_performance_score(agent, task)
            score += 0.15 * performance_score
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating assignment score: {e}")
            return 0.0
    
    def _calculate_expertise_match(self, agent: PMLAgent, task: PMLTask) -> float:
        """Calculate expertise matching score"""
        try:
            if not task.required_expertise:
                return 1.0  # No specific requirements
            
            agent_expertise = set([agent.expertise_area] + agent.capabilities)
            required_expertise = set(task.required_expertise)
            
            # Calculate Jaccard similarity
            intersection = len(agent_expertise.intersection(required_expertise))
            union = len(agent_expertise.union(required_expertise))
            
            if union == 0:
                return 0.0
            
            jaccard_score = intersection / union
            
            # Apply specialty bonus if agent specialty directly matches
            if agent.specialty in task.required_expertise:
                jaccard_score = min(1.0, jaccard_score + 0.1)
            
            return jaccard_score
            
        except Exception as e:
            logger.error(f"Error calculating expertise match: {e}")
            return 0.0
    
    def _calculate_output_compatibility(self, agent: PMLAgent, task: PMLTask) -> float:
        """Calculate output type compatibility score"""
        try:
            if agent.output_type == task.expected_output_type:
                return 1.0
            
            # Check for compatible output types
            compatibility_map = {
                'safety_score': ['numerical_score', 'assessment_report'],
                'flight_list': ['structured_data', 'json_array'],
                'cost_breakdown': ['financial_analysis', 'structured_data'],
                'ranked_recommendations': ['ranking_list', 'structured_data']
            }
            
            agent_outputs = compatibility_map.get(agent.output_type, [agent.output_type])
            if task.expected_output_type in agent_outputs:
                return 0.8
            
            return 0.2  # Partial compatibility
            
        except Exception as e:
            logger.error(f"Error calculating output compatibility: {e}")
            return 0.0
    
    def _calculate_input_compatibility(self, agent: PMLAgent, task: PMLTask) -> float:
        """Calculate input type compatibility score"""
        try:
            if not agent.input_types or not task.input_data:
                return 1.0
            
            task_input_types = set(task.input_data.keys())
            agent_input_types = set(agent.input_types)
            
            # Calculate overlap
            overlap = len(task_input_types.intersection(agent_input_types))
            total_required = len(task_input_types)
            
            if total_required == 0:
                return 1.0
            
            return overlap / total_required
            
        except Exception as e:
            logger.error(f"Error calculating input compatibility: {e}")
            return 0.0
    
    def _calculate_performance_score(self, agent: PMLAgent, task: PMLTask) -> float:
        """Calculate performance-based score"""
        try:
            if not agent.performance_history:
                return 0.5  # Neutral score for new agents
            
            # Get relevant performance metrics
            task_type_performance = agent.performance_history.get(task.task_type, 0.5)
            overall_performance = np.mean(list(agent.performance_history.values()))
            
            # Weight recent performance more heavily
            performance_score = 0.7 * task_type_performance + 0.3 * overall_performance
            
            return min(1.0, max(0.0, performance_score))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def _calculate_confidence(self, agent_name: str, task: PMLTask) -> float:
        """Calculate confidence in assignment"""
        try:
            agent = self.agents[agent_name]
            
            # Base confidence from performance history
            if agent.performance_history:
                base_confidence = np.mean(list(agent.performance_history.values()))
            else:
                base_confidence = 0.5
            
            # Adjust based on task complexity
            complexity_factor = 1.0 - (task.complexity_level * 0.3)
            
            # Adjust based on expertise match
            expertise_factor = self._calculate_expertise_match(agent, task)
            
            confidence = base_confidence * complexity_factor * (0.5 + 0.5 * expertise_factor)
            
            return min(1.0, max(0.1, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_assignment_reasoning(self, agent_name: str, task: PMLTask, score: float) -> str:
        """Generate human-readable reasoning for assignment"""
        try:
            agent = self.agents[agent_name]
            reasons = []
            
            # Expertise reasoning
            expertise_score = self._calculate_expertise_match(agent, task)
            if expertise_score > 0.8:
                reasons.append(f"High expertise match ({expertise_score:.2f})")
            elif expertise_score > 0.6:
                reasons.append(f"Good expertise alignment ({expertise_score:.2f})")
            
            # Output compatibility
            if agent.output_type == task.expected_output_type:
                reasons.append("Exact output type match")
            
            # Performance history
            if agent.performance_history:
                avg_performance = np.mean(list(agent.performance_history.values()))
                if avg_performance > 0.8:
                    reasons.append(f"Strong performance history ({avg_performance:.2f})")
            
            # Specialty match
            if agent.specialty in task.required_expertise:
                reasons.append("Specialty directly matches requirements")
            
            if not reasons:
                reasons.append("Best available option")
            
            return f"Selected {agent_name}: " + ", ".join(reasons) + f" (Overall score: {score:.3f})"
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"Agent {agent_name} assigned with score {score:.3f}"
    
    def _update_capability_matrix(self, agent: PMLAgent):
        """Update the capability matrix for the agent"""
        try:
            if agent.agent_name not in self.capability_matrix:
                self.capability_matrix[agent.agent_name] = {}
            
            # Initialize capabilities
            for capability in agent.capabilities:
                self.capability_matrix[agent.agent_name][capability] = 1.0
            
            # Add expertise area
            self.capability_matrix[agent.agent_name][agent.expertise_area] = 1.0
            
        except Exception as e:
            logger.error(f"Error updating capability matrix: {e}")
    
    def update_agent_performance(self, agent_name: str, task_type: str, performance_score: float):
        """Update agent performance metrics"""
        try:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                agent.performance_history[task_type] = performance_score
                agent.updated_at = datetime.now()
                
                # Update capability matrix
                if agent_name in self.capability_matrix:
                    self.capability_matrix[agent_name][task_type] = performance_score
                
                logger.info(f"Updated performance for {agent_name}: {task_type} = {performance_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating agent performance: {e}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the PML repository"""
        try:
            stats = {
                'total_agents': len(self.agents),
                'total_tasks': len(self.tasks),
                'total_assignments': len(self.assignments),
                'expertise_areas': len(self.expertise_index),
                'avg_agent_capabilities': 0.0,
                'agent_distribution': {},
                'task_distribution': {},
                'performance_summary': {}
            }
            
            if self.agents:
                # Calculate average capabilities per agent
                total_capabilities = sum(len(agent.capabilities) for agent in self.agents.values())
                stats['avg_agent_capabilities'] = total_capabilities / len(self.agents)
                
                # Agent distribution by expertise
                for agent in self.agents.values():
                    area = agent.expertise_area
                    stats['agent_distribution'][area] = stats['agent_distribution'].get(area, 0) + 1
                
                # Performance summary
                for agent_name, agent in self.agents.items():
                    if agent.performance_history:
                        avg_perf = np.mean(list(agent.performance_history.values()))
                        stats['performance_summary'][agent_name] = avg_perf
            
            if self.tasks:
                # Task distribution by type
                for task in self.tasks.values():
                    task_type = task.task_type
                    stats['task_distribution'][task_type] = stats['task_distribution'].get(task_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}
    
    def export_pml_data(self) -> Dict[str, Any]:
        """Export all PML data for persistence or transfer"""
        try:
            export_data = {
                'agents': {name: asdict(agent) for name, agent in self.agents.items()},
                'tasks': {task_id: asdict(task) for task_id, task in self.tasks.items()},
                'assignments': {assign_id: asdict(assignment) for assign_id, assignment in self.assignments.items()},
                'expertise_index': {area: list(agents) for area, agents in self.expertise_index.items()},
                'capability_matrix': self.capability_matrix,
                'export_timestamp': datetime.now().isoformat()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting PML data: {e}")
            return {}


# Global PML repository instance
pml_repository = PMLRepository()

def register_pml_agent(agent_data: Dict[str, Any]) -> str:
    """Global function to register PML agent"""
    return pml_repository.register_agent(agent_data)

def create_pml_task(task_data: Dict[str, Any]) -> str:
    """Global function to create PML task"""
    return pml_repository.create_task(task_data)

def assign_pml_agent(task_id: str) -> Optional[PMLAssignment]:
    """Global function to assign optimal agent"""
    return pml_repository.assign_optimal_agent(task_id)

def update_pml_performance(agent_name: str, task_type: str, performance_score: float):
    """Global function to update agent performance"""
    pml_repository.update_agent_performance(agent_name, task_type, performance_score) 