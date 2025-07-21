#!/usr/bin/env python3
"""
MAMA framework scalability stress test experiment

Objective: To measure the performance of semantic matching, registration service, and MARL decision modules when the number of agents is expanded from 10 to 5000.

"""

import asyncio
import concurrent.futures
import json
import logging
import numpy as np
import os
import pickle
import random
import time
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class SyntheticAgentProfileGenerator:
    """Synthetic agent profile generator"""
    
    def __init__(self):
        # Professional domain descriptions sampled from Wikipedia sample texts
        self.expertise_templates = [
            "Expert in machine learning algorithms and deep neural networks for predictive modeling",
            "Specializes in natural language processing and computational linguistics research",
            "Advanced knowledge in computer vision and image recognition systems",
            "Professional experience in distributed systems and cloud computing architectures",
            "Expertise in cybersecurity protocols and network security analysis",
            "Specialist in data mining techniques and big data analytics platforms",
            "Advanced understanding of robotics control systems and autonomous navigation",
            "Professional background in financial modeling and quantitative analysis",
            "Expert in bioinformatics and computational biology research methods",
            "Specializes in software engineering best practices and system design patterns",
            "Advanced knowledge in signal processing and digital communications",
            "Professional experience in web development and full-stack technologies",
            "Expertise in database management systems and data warehouse optimization",
            "Specialist in artificial intelligence ethics and responsible AI development",
            "Advanced understanding of blockchain technology and cryptocurrency systems",
            "Professional background in mobile application development and user experience",
            "Expert in scientific computing and numerical analysis methods",
            "Specializes in game development and interactive entertainment systems",
            "Advanced knowledge in embedded systems and IoT device programming",
            "Professional experience in quality assurance and software testing methodologies",
            "Expertise in human-computer interaction and interface design principles",
            "Specialist in operations research and optimization algorithms",
            "Advanced understanding of compiler design and programming language theory",
            "Professional background in project management and agile development methodologies",
            "Expert in environmental modeling and climate change simulation systems"
        ]
        
    def generate_profiles(self, N: int) -> List[Dict[str, Any]]:
        """
        Generate N synthetic agent profiles
        
        Args:
            N: Number of agents
            
        Returns:
            List of agent profile dictionaries
        """
        profiles = []
        
        for i in range(N):
            # Randomly select expertise description
            expertise = random.choice(self.expertise_templates)
            
            # Generate random competence scores
            competence_scores = {
                'safety_assessment': np.random.uniform(0.3, 0.9),
                'economic_analysis': np.random.uniform(0.3, 0.9),
                'weather_prediction': np.random.uniform(0.3, 0.9),
                'flight_information': np.random.uniform(0.3, 0.9),
                'integration_coordination': np.random.uniform(0.3, 0.9)
            }
            
            profile = {
                'agent_id': f"synthetic_agent_{i+1:05d}",
                'name': f"Agent {i+1}",
                'expertise_description': expertise,
                'specialization': random.choice(['safety', 'economic', 'weather', 'flight_info', 'integration']),
                'competence_scores': competence_scores,
                'registration_time': time.time(),
                'last_active': time.time(),
                'total_tasks_completed': random.randint(0, 100),
                'average_response_time': np.random.uniform(0.5, 3.0),
                'trust_score': np.random.uniform(0.4, 0.95),
                'metadata': {
                    'version': '1.0',
                    'created_for_scalability_test': True,
                    'synthetic': True
                }
            }
            
            profiles.append(profile)
        
        return profiles

class ScalabilityBenchmarkSuite:
    """Scalability benchmark test suite"""
    
    def __init__(self):
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize SBERT model
        self.sbert_model = SentenceTransformer('models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf')
        
        # Test configurations
        self.test_scales = [10, 50, 100, 500, 1000, 2000, 5000]
        self.num_test_queries = 50
        
        # Profile generator
        self.profile_generator = SyntheticAgentProfileGenerator()
        
        # Results storage
        self.benchmark_results = {}
        
    def generate_test_queries(self, num_queries: int = 50) -> List[str]:
        """Generate test queries for semantic matching"""
        
        query_templates = [
            "Find flights with highest safety ratings for business travel",
            "Recommend cost-effective flights for budget-conscious travelers",
            "Search for flights with minimal weather-related delays",
            "Identify flights with best on-time performance records",
            "Find integrated travel solutions with hotel and car rental",
            "Search for flights with premium amenities and services",
            "Recommend flights with flexible cancellation policies",
            "Find eco-friendly flights with carbon offset programs",
            "Search for flights with shortest total travel time",
            "Identify flights with best customer service ratings",
            "Find flights suitable for elderly passengers",
            "Recommend flights with wheelchair accessibility",
            "Search for flights with pet-friendly policies",
            "Find flights with best loyalty program benefits",
            "Recommend flights with premium lounge access",
            "Search for flights with in-flight entertainment systems",
            "Find flights with healthy meal options",
            "Recommend flights with WiFi connectivity",
            "Search for flights with power outlets at seats",
            "Find flights with spacious legroom options"
        ]
        
        queries = []
        for i in range(num_queries):
            base_query = random.choice(query_templates)
            # Add some variation
            variation = random.choice([
                " for international travel",
                " for domestic routes",
                " for weekend trips",
                " for business meetings",
                " for family vacations",
                " for solo travelers",
                " for group bookings",
                " for last-minute travel",
                " for advance bookings",
                " for connecting flights"
            ])
            queries.append(base_query + variation)
        
        return queries
    
    def benchmark_semantic_matching(self, agent_profiles: List[Dict[str, Any]], 
                                  test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark semantic matching performance"""
        
        N = len(agent_profiles)
        logger.info(f"  Benchmarking semantic matching with {N} agents...")
        
        # Extract agent expertise descriptions
        agent_descriptions = [profile['expertise_description'] for profile in agent_profiles]
        
        # Pre-compute agent embeddings
        start_time = time.time()
        agent_embeddings = self.sbert_model.encode(agent_descriptions)
        embedding_time = time.time() - start_time
        
        # Benchmark query processing
        matching_times = []
        similarity_computation_times = []
        
        for query in test_queries:
            # Encode query
            query_start = time.time()
            query_embedding = self.sbert_model.encode([query])
            query_encoding_time = time.time() - query_start
            
            # Compute similarities
            similarity_start = time.time()
            similarities = cosine_similarity(query_embedding, agent_embeddings)[0]
            similarity_time = time.time() - similarity_start
            
            # Find top-k matches
            matching_start = time.time()
            top_k_indices = np.argsort(similarities)[-10:][::-1]  # Top 10 matches
            matching_time = time.time() - matching_start
            
            matching_times.append(query_encoding_time + matching_time)
            similarity_computation_times.append(similarity_time)
        
        # Calculate statistics
        avg_matching_time = np.mean(matching_times)
        avg_similarity_time = np.mean(similarity_computation_times)
        total_time = embedding_time + np.sum(matching_times) + np.sum(similarity_computation_times)
        
        # Memory usage estimation
        embedding_memory = agent_embeddings.nbytes / (1024 * 1024)  # MB
        
        return {
            'num_agents': N,
            'num_queries': len(test_queries),
            'avg_matching_time_ms': avg_matching_time * 1000,
            'avg_similarity_computation_time_ms': avg_similarity_time * 1000,
            'embedding_preprocessing_time_s': embedding_time,
            'total_processing_time_s': total_time,
            'embedding_memory_mb': embedding_memory,
            'throughput_queries_per_second': len(test_queries) / total_time,
            'scalability_factor': avg_matching_time * N,  # O(N) complexity indicator
            'memory_per_agent_kb': (embedding_memory * 1024) / N
        }
    
    def benchmark_registration_service(self, agent_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark agent registration service performance"""
        
        N = len(agent_profiles)
        logger.info(f"  Benchmarking registration service with {N} agents...")
        
        # Simulate registration service operations
        registration_times = []
        lookup_times = []
        update_times = []
        
        # Simulate agent registry (in-memory dictionary)
        agent_registry = {}
        
        # Benchmark registration operations
        for profile in agent_profiles:
            # Registration
            reg_start = time.time()
            agent_registry[profile['agent_id']] = profile
            reg_time = time.time() - reg_start
            registration_times.append(reg_time)
            
            # Lookup operation
            lookup_start = time.time()
            _ = agent_registry.get(profile['agent_id'])
            lookup_time = time.time() - lookup_start
            lookup_times.append(lookup_time)
            
            # Update operation
            update_start = time.time()
            agent_registry[profile['agent_id']]['last_active'] = time.time()
            update_time = time.time() - update_start
            update_times.append(update_time)
        
        # Calculate statistics
        avg_registration_time = np.mean(registration_times)
        avg_lookup_time = np.mean(lookup_times)
        avg_update_time = np.mean(update_times)
        
        # Memory usage estimation
        import sys
        registry_memory = sys.getsizeof(agent_registry) / (1024 * 1024)  # MB
        
        return {
            'num_agents': N,
            'avg_registration_time_ms': avg_registration_time * 1000,
            'avg_lookup_time_ms': avg_lookup_time * 1000,
            'avg_update_time_ms': avg_update_time * 1000,
            'total_registration_time_s': np.sum(registration_times),
            'registry_memory_mb': registry_memory,
            'registration_throughput_ops_per_second': N / np.sum(registration_times),
            'lookup_throughput_ops_per_second': N / np.sum(lookup_times),
            'memory_per_agent_kb': (registry_memory * 1024) / N
        }
    
    def benchmark_marl_decision_module(self, agent_profiles: List[Dict[str, Any]], 
                                     test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark MARL decision module performance"""
        
        N = len(agent_profiles)
        logger.info(f"  Benchmarking MARL decision module with {N} agents...")
        
        # Initialize Q-tables for all agents
        num_states = 20
        num_actions = 5
        q_tables = {}
        
        q_table_init_start = time.time()
        for profile in agent_profiles:
            agent_id = profile['agent_id']
            q_tables[agent_id] = np.random.uniform(0.0, 1.0, size=(num_states, num_actions))
        q_table_init_time = time.time() - q_table_init_start
        
        # Benchmark decision making
        decision_times = []
        update_times = []
        
        for query in test_queries:
            # Simulate decision making for random subset of agents
            selected_agents = random.sample(agent_profiles, min(10, N))
            
            for agent_profile in selected_agents:
                agent_id = agent_profile['agent_id']
                
                # Decision making
                decision_start = time.time()
                state = random.randint(0, num_states - 1)
                action = np.argmax(q_tables[agent_id][state])
                decision_time = time.time() - decision_start
                decision_times.append(decision_time)
                
                # Q-table update
                update_start = time.time()
                reward = np.random.uniform(-1, 1)
                learning_rate = 0.1
                discount_factor = 0.95
                next_state = random.randint(0, num_states - 1)
                
                # Q-learning update
                q_tables[agent_id][state][action] += learning_rate * (
                    reward + discount_factor * np.max(q_tables[agent_id][next_state]) - 
                    q_tables[agent_id][state][action]
                )
                update_time = time.time() - update_start
                update_times.append(update_time)
        
        # Calculate statistics
        avg_decision_time = np.mean(decision_times)
        avg_update_time = np.mean(update_times)
        
        # Memory usage estimation
        q_table_memory = sum(q_table.nbytes for q_table in q_tables.values()) / (1024 * 1024)  # MB
        
        return {
            'num_agents': N,
            'num_queries': len(test_queries),
            'q_table_initialization_time_s': q_table_init_time,
            'avg_decision_time_ms': avg_decision_time * 1000,
            'avg_update_time_ms': avg_update_time * 1000,
            'total_decision_operations': len(decision_times),
            'total_update_operations': len(update_times),
            'q_table_memory_mb': q_table_memory,
            'decision_throughput_ops_per_second': len(decision_times) / np.sum(decision_times),
            'update_throughput_ops_per_second': len(update_times) / np.sum(update_times),
            'memory_per_agent_kb': (q_table_memory * 1024) / N
        }
    
    def run_comprehensive_scalability_test(self) -> Dict[str, Any]:
        """Run comprehensive scalability test across all scales"""
        
        logger.info("🚀 Starting MAMA Framework Scalability Stress Test")
        logger.info("=" * 80)
        
        # Generate test queries once
        test_queries = self.generate_test_queries(self.num_test_queries)
        logger.info(f"📝 Generated {len(test_queries)} test queries")
        
        # Results storage
        scalability_results = {
            'semantic_matching': {},
            'registration_service': {},
            'marl_decision_module': {},
            'test_configuration': {
                'test_scales': self.test_scales,
                'num_test_queries': self.num_test_queries,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Run tests for each scale
        for scale in self.test_scales:
            logger.info(f"\n🔬 Testing scale: {scale} agents")
            
            # Generate agent profiles
            logger.info(f"  Generating {scale} synthetic agent profiles...")
            agent_profiles = self.profile_generator.generate_profiles(scale)
            
            # Benchmark semantic matching
            semantic_results = self.benchmark_semantic_matching(agent_profiles, test_queries)
            scalability_results['semantic_matching'][scale] = semantic_results
            
            # Benchmark registration service
            registration_results = self.benchmark_registration_service(agent_profiles)
            scalability_results['registration_service'][scale] = registration_results
            
            # Benchmark MARL decision module
            marl_results = self.benchmark_marl_decision_module(agent_profiles, test_queries)
            scalability_results['marl_decision_module'][scale] = marl_results
            
            # Progress summary
            logger.info(f"  ✅ Scale {scale} completed:")
            logger.info(f"    Semantic matching: {semantic_results['avg_matching_time_ms']:.2f}ms avg")
            logger.info(f"    Registration: {registration_results['avg_registration_time_ms']:.2f}ms avg")
            logger.info(f"    MARL decision: {marl_results['avg_decision_time_ms']:.2f}ms avg")
        
        # Save results
        self.save_scalability_results(scalability_results)
        
        # Generate analysis and visualizations
        self.analyze_scalability_results(scalability_results)
        
        return scalability_results
    
    def save_scalability_results(self, results: Dict[str, Any]):
        """Save scalability test results to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'scalability_stress_test_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Scalability test results saved to: {results_file}")
    
    def analyze_scalability_results(self, results: Dict[str, Any]):
        """Analyze scalability results and generate insights"""
        
        logger.info("\n📊 SCALABILITY ANALYSIS")
        logger.info("=" * 80)
        
        # Extract data for analysis
        scales = self.test_scales
        
        # Semantic matching analysis
        semantic_times = [results['semantic_matching'][scale]['avg_matching_time_ms'] for scale in scales]
        semantic_memory = [results['semantic_matching'][scale]['embedding_memory_mb'] for scale in scales]
        
        # Registration service analysis
        registration_times = [results['registration_service'][scale]['avg_registration_time_ms'] for scale in scales]
        registration_memory = [results['registration_service'][scale]['registry_memory_mb'] for scale in scales]
        
        # MARL decision module analysis
        marl_decision_times = [results['marl_decision_module'][scale]['avg_decision_time_ms'] for scale in scales]
        marl_memory = [results['marl_decision_module'][scale]['q_table_memory_mb'] for scale in scales]
        
        # Performance analysis
        logger.info("\n🔍 PERFORMANCE SCALING ANALYSIS:")
        
        # Semantic matching scalability
        semantic_growth_rate = np.polyfit(np.log(scales), np.log(semantic_times), 1)[0]
        logger.info(f"  Semantic Matching:")
        logger.info(f"    Growth rate: O(N^{semantic_growth_rate:.2f})")
        logger.info(f"    10→5000 agents: {semantic_times[0]:.2f}ms → {semantic_times[-1]:.2f}ms")
        
        # Registration service scalability
        registration_growth_rate = np.polyfit(np.log(scales), np.log(registration_times), 1)[0]
        logger.info(f"  Registration Service:")
        logger.info(f"    Growth rate: O(N^{registration_growth_rate:.2f})")
        logger.info(f"    10→5000 agents: {registration_times[0]:.2f}ms → {registration_times[-1]:.2f}ms")
        
        # MARL decision module scalability
        marl_growth_rate = np.polyfit(np.log(scales), np.log(marl_decision_times), 1)[0]
        logger.info(f"  MARL Decision Module:")
        logger.info(f"    Growth rate: O(N^{marl_growth_rate:.2f})")
        logger.info(f"    10→5000 agents: {marl_decision_times[0]:.2f}ms → {marl_decision_times[-1]:.2f}ms")
        
        # Memory analysis
        logger.info(f"\n💾 MEMORY SCALING ANALYSIS:")
        logger.info(f"  Semantic Matching Memory: {semantic_memory[0]:.2f}MB → {semantic_memory[-1]:.2f}MB")
        logger.info(f"  Registration Memory: {registration_memory[0]:.2f}MB → {registration_memory[-1]:.2f}MB")
        logger.info(f"  MARL Q-tables Memory: {marl_memory[0]:.2f}MB → {marl_memory[-1]:.2f}MB")
        
        # Generate scalability plots
        self.generate_scalability_plots(results)
        
        # Bottleneck analysis
        logger.info(f"\n🎯 BOTTLENECK ANALYSIS:")
        
        # Identify bottlenecks at largest scale
        largest_scale = scales[-1]
        semantic_time_5000 = results['semantic_matching'][largest_scale]['avg_matching_time_ms']
        registration_time_5000 = results['registration_service'][largest_scale]['avg_registration_time_ms']
        marl_time_5000 = results['marl_decision_module'][largest_scale]['avg_decision_time_ms']
        
        bottlenecks = [
            ('Semantic Matching', semantic_time_5000),
            ('Registration Service', registration_time_5000),
            ('MARL Decision Module', marl_time_5000)
        ]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"  At {largest_scale} agents:")
        for i, (component, time_ms) in enumerate(bottlenecks):
            status = "🔴 BOTTLENECK" if i == 0 else "🟡 MODERATE" if i == 1 else "🟢 EFFICIENT"
            logger.info(f"    {component}: {time_ms:.2f}ms {status}")
        
        # Recommendations
        logger.info(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        if semantic_growth_rate > 1.5:
            logger.info("  • Consider implementing approximate nearest neighbor search for semantic matching")
            logger.info("  • Implement agent embedding caching and incremental updates")
        
        if registration_growth_rate > 1.2:
            logger.info("  • Consider using distributed hash tables for agent registration")
            logger.info("  • Implement connection pooling and batch operations")
        
        if marl_growth_rate > 1.3:
            logger.info("  • Consider hierarchical MARL or agent clustering")
            logger.info("  • Implement Q-table compression and approximation methods")
        
        logger.info("=" * 80)
    
    def generate_scalability_plots(self, results: Dict[str, Any]):
        """Generate scalability visualization plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scales = self.test_scales
        
        # Plot 1: Response Time Scaling
        ax1 = axes[0, 0]
        
        semantic_times = [results['semantic_matching'][scale]['avg_matching_time_ms'] for scale in scales]
        registration_times = [results['registration_service'][scale]['avg_registration_time_ms'] for scale in scales]
        marl_times = [results['marl_decision_module'][scale]['avg_decision_time_ms'] for scale in scales]
        
        ax1.loglog(scales, semantic_times, 'o-', label='Semantic Matching', linewidth=2, markersize=6)
        ax1.loglog(scales, registration_times, 's-', label='Registration Service', linewidth=2, markersize=6)
        ax1.loglog(scales, marl_times, '^-', label='MARL Decision Module', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Average Response Time (ms)')
        ax1.set_title('Response Time Scalability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage Scaling
        ax2 = axes[0, 1]
        
        semantic_memory = [results['semantic_matching'][scale]['embedding_memory_mb'] for scale in scales]
        registration_memory = [results['registration_service'][scale]['registry_memory_mb'] for scale in scales]
        marl_memory = [results['marl_decision_module'][scale]['q_table_memory_mb'] for scale in scales]
        
        ax2.loglog(scales, semantic_memory, 'o-', label='Semantic Embeddings', linewidth=2, markersize=6)
        ax2.loglog(scales, registration_memory, 's-', label='Registration Registry', linewidth=2, markersize=6)
        ax2.loglog(scales, marl_memory, '^-', label='MARL Q-tables', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Scalability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput Scaling
        ax3 = axes[1, 0]
        
        semantic_throughput = [results['semantic_matching'][scale]['throughput_queries_per_second'] for scale in scales]
        registration_throughput = [results['registration_service'][scale]['registration_throughput_ops_per_second'] for scale in scales]
        marl_throughput = [results['marl_decision_module'][scale]['decision_throughput_ops_per_second'] for scale in scales]
        
        ax3.semilogx(scales, semantic_throughput, 'o-', label='Semantic Matching', linewidth=2, markersize=6)
        ax3.semilogx(scales, registration_throughput, 's-', label='Registration Service', linewidth=2, markersize=6)
        ax3.semilogx(scales, marl_throughput, '^-', label='MARL Decision Module', linewidth=2, markersize=6)
        
        ax3.set_xlabel('Number of Agents')
        ax3.set_ylabel('Throughput (ops/second)')
        ax3.set_title('Throughput Scalability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scalability Factor (Response Time × N)
        ax4 = axes[1, 1]
        
        semantic_scalability = [results['semantic_matching'][scale]['scalability_factor'] for scale in scales]
        
        ax4.loglog(scales, semantic_scalability, 'o-', label='Semantic Matching', linewidth=2, markersize=6)
        ax4.loglog(scales, [t * n for t, n in zip(registration_times, scales)], 's-', 
                  label='Registration Service', linewidth=2, markersize=6)
        ax4.loglog(scales, [t * n for t, n in zip(marl_times, scales)], '^-', 
                  label='MARL Decision Module', linewidth=2, markersize=6)
        
        ax4.set_xlabel('Number of Agents')
        ax4.set_ylabel('Scalability Factor (Time × N)')
        ax4.set_title('Scalability Factor Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.results_dir / f'scalability_analysis_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Scalability plots saved to: {plot_file}")

def main():
    """Main function to run scalability stress test"""
    
    # Create benchmark suite
    benchmark = ScalabilityBenchmarkSuite()
    
    # Run comprehensive scalability test
    results = benchmark.run_comprehensive_scalability_test()
    
    logger.info("\n🎉 SCALABILITY STRESS TEST COMPLETED!")
    logger.info(f"📊 Results saved and analyzed")
    logger.info(f"🔬 Tested scales: {benchmark.test_scales}")
    logger.info(f"📝 Test queries: {benchmark.num_test_queries}")
    
    return results

if __name__ == "__main__":
    main() 