import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup environment and logging ---
# Ensure proper import of MAMA framework modules from your project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import MAMAFlightAssistant, QueryProcessingConfig
    from core.multi_dimensional_trust_ledger import TrustDimension
except ImportError as e:
    print(f"CRITICAL ERROR: Unable to import MAMA framework components: {e}")
    print("Please ensure this script is at the same level as your project folder.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalCompetenceExperiment:
    """An independent script for running and plotting competence evolution curves in final experiments."""
    
    def __init__(self, testbed_file):
        self.testbed_file = Path(testbed_file)
        self.config = QueryProcessingConfig()
        self.assistant = None
        self.competence_log = []
        self.results_dir = Path('results')
        self.figures_dir = Path('figures')
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

    async def _prepare_offline_testbed(self, source_file, output_file):
        """Use real experimental result files to create a self-consistent testbed for offline competence evolution analysis."""
        logging.info(f"Loading real experimental data from {source_file} to create testbed...")
        source_path = Path(source_file)
        if not source_path.exists():
            logging.error(f"FATAL: Source data file not found at {source_path}.")
            raise FileNotFoundError(f"Real query file {source_path} not found.")

        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        # Extract query data from real experimental results
        if 'raw_results' in source_data:
            raw_results = source_data['raw_results']
            logging.info(f"Found {len(raw_results)} raw experimental results")
        else:
            logging.error("No 'raw_results' key found in source data")
            raise KeyError("Source data does not contain 'raw_results'")

        # Create testbed structure
        testbed = {
            'metadata': {
                'source_file': str(source_path),
                'creation_time': datetime.now().isoformat(),
                'total_queries': len(raw_results),
                'description': 'Offline testbed created from real experimental data'
            },
            'test_queries': []
        }

        # Convert raw results to testbed queries
        for i, result in enumerate(raw_results[:150]):  # Limit to 150 queries
            query = {
                'query_id': result.get('query_id', f'query_{i+1:03d}'),
                'query_text': result.get('query_text', f'Test query {i+1}'),
                'expected_performance': {
                    'MRR': result.get('MRR', 0.5),
                    'NDCG': result.get('NDCG', 0.5),
                    'ART': result.get('ART', 1.0)
                },
                'preferences': result.get('preferences', {
                    'priority': 'safety',
                    'budget': 'medium'
                })
            }
            testbed['test_queries'].append(query)

        # Save testbed
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(testbed, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Testbed created successfully: {output_path}")
        logging.info(f"Total test queries: {len(testbed['test_queries'])}")

    async def run_competence_experiment(self, num_interactions=150):
        """Run complete competence evolution experiment"""
        logging.info("🚀 Starting final competence evolution experiment")
        
        # Initialize MAMA system
        self.assistant = MAMAFlightAssistant(config=self.config)
        await self.assistant.initialize_system()
        logging.info("✅ MAMA system initialized successfully")

        # Load testbed
        if not self.testbed_file.exists():
            logging.error(f"Testbed file not found: {self.testbed_file}")
            raise FileNotFoundError(f"Testbed file {self.testbed_file} not found")

        with open(self.testbed_file, 'r', encoding='utf-8') as f:
            testbed = json.load(f)

        test_queries = testbed['test_queries'][:num_interactions]
        logging.info(f"Loaded {len(test_queries)} test queries from testbed")

        # Agent IDs for competence tracking
        agent_ids = [
            'safety_assessment_agent',
            'economic_agent',
            'weather_agent',
            'flight_info_agent',
            'integration_agent'
        ]

        # Run experiment
        for i, query in enumerate(test_queries):
            logging.info(f"Processing query {i+1}/{len(test_queries)}: {query['query_id']}")
            
            try:
                # Process query
                start_time = datetime.now()
                result = await self.assistant.process_query(query['query_text'])
                end_time = datetime.now()
                
                response_time = (end_time - start_time).total_seconds()
                
                # Get current competence scores
                competence_scores = {}
                for agent_id in agent_ids:
                    try:
                        score = self.assistant.trust_ledger.get_competence_score(
                            agent_id, TrustDimension.COMPETENCE
                        )
                        competence_scores[agent_id] = score
                    except Exception as e:
                        logging.warning(f"Failed to get competence score for {agent_id}: {e}")
                        competence_scores[agent_id] = 0.5  # Default score

                # Log competence evolution
                log_entry = {
                    'interaction': i + 1,
                    'query_id': query['query_id'],
                    'response_time': response_time,
                    'competence_scores': competence_scores,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.competence_log.append(log_entry)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    avg_scores = {
                        agent_id: np.mean([entry['competence_scores'][agent_id] 
                                         for entry in self.competence_log[-10:]])
                        for agent_id in agent_ids
                    }
                    logging.info(f"Progress: {i+1}/{len(test_queries)}, recent avg scores: {avg_scores}")
                
            except Exception as e:
                logging.error(f"Error processing query {i+1}: {e}")
                continue

        # Generate results
        await self._generate_results()
        
        # Cleanup
        await self.assistant.cleanup()
        logging.info("✅ Final competence evolution experiment completed successfully!")

    async def _generate_results(self):
        """Generate experiment results and visualizations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.results_dir / f'final_competence_experiment_{timestamp}.json'
        
        results = {
            'experiment_info': {
                'type': 'final_competence_evolution',
                'timestamp': timestamp,
                'total_interactions': len(self.competence_log),
                'testbed_file': str(self.testbed_file)
            },
            'competence_evolution': self.competence_log
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to: {results_file}")
        
        # Generate visualization
        self._plot_competence_evolution(timestamp)

    def _plot_competence_evolution(self, timestamp):
        """Plot competence evolution curves"""
        if not self.competence_log:
            logging.warning("No competence data to plot")
            return

        plt.figure(figsize=(14, 8))
        
        # Extract data for plotting
        interactions = [entry['interaction'] for entry in self.competence_log]
        
        agent_colors = {
            'safety_assessment_agent': '#FF6B6B',
            'economic_agent': '#4ECDC4',
            'weather_agent': '#45B7D1',
            'flight_info_agent': '#96CEB4',
            'integration_agent': '#FFEAA7'
        }
        
        agent_names = {
            'safety_assessment_agent': 'Safety Assessment',
            'economic_agent': 'Economic Agent',
            'weather_agent': 'Weather Agent',
            'flight_info_agent': 'Flight Info Agent',
            'integration_agent': 'Integration Agent'
        }
        
        # Plot competence evolution for each agent
        for agent_id, color in agent_colors.items():
            scores = [entry['competence_scores'].get(agent_id, 0.5) 
                     for entry in self.competence_log]
            
            plt.plot(interactions, scores, 
                    label=agent_names.get(agent_id, agent_id),
                    color=color,
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    alpha=0.8)
        
        plt.title('MAMA Framework: Final Competence Evolution Experiment\n'
                 'Agent Competence Scores Over Time', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Interaction Number', fontsize=12)
        plt.ylabel('Competence Score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.figures_dir / f'final_competence_evolution_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Competence evolution plot saved to: {plot_file}")

async def main():
    """Main function"""
    # Configuration
    source_data_file = 'results/final_experiment_results_20250705_210817.json'
    testbed_file = 'results/offline_competence_testbed.json'
    
    try:
        experiment = FinalCompetenceExperiment(testbed_file)
        
        # Create testbed if it doesn't exist
        if not Path(testbed_file).exists():
            logging.info("Creating offline testbed from real experimental data...")
            await experiment._prepare_offline_testbed(source_data_file, testbed_file)
        
        # Run experiment
        await experiment.run_competence_experiment(num_interactions=150)
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 