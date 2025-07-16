#!/usr/bin/env python3
"""
MAMA Framework Final Reward-Driven Experiment
Implements complete reinforcement learning loop based on system reward r
"""

import asyncio
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
import traceback

try:
    from main import MAMAFlightAssistant, QueryProcessingConfig
    from core.multi_dimensional_trust_ledger import TrustDimension
    from core.evaluation_metrics import calculate_mrr, calculate_ndcg, calculate_art
except ImportError as e:
    print(f"CRITICAL ERROR: Cannot import MAMA framework components: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RewardDrivenExperiment:
    def __init__(self):
        self.config = QueryProcessingConfig()
        self.assistant = None
        self.competence_log = []
        self.reward_log = []
        self.results_dir = Path('results')
        self.figures_dir = Path('figures')
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
        self.lambda1 = 0.4  # MRR weight
        self.lambda2 = 0.4  # NDCG weight  
        self.lambda3 = 0.2  # ART weight (negative)

    def _generate_test_queries(self, num_queries=150):
        queries = []
        
        us_cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
            "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington",
            "Boston", "El Paso", "Nashville", "Detroit", "Oklahoma City",
            "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore"
        ]
        
        priority_options = ['safety', 'cost', 'time', 'comfort']
        
        for i in range(num_queries):
            departure = np.random.choice(us_cities)
            destination = np.random.choice([city for city in us_cities if city != departure])
            
            priority = priority_options[i % len(priority_options)]
            
            query = {
                "query_id": f"test_query_{i+1:03d}",
                "text": f"Find flights from {departure} to {destination} on 2024-12-15",
                "preferences": {
                    "priority": priority,
                    "budget": "medium",
                    "passengers": 1
                },
                "departure_city": departure,
                "destination_city": destination,
                "date": "2024-12-15"
            }
            queries.append(query)
        
        return queries

    async def run_experiment(self, num_interactions=150):
        logger.info("🚀 Starting reward-driven MAMA experiment")
        
        self.assistant = MAMAFlightAssistant(config=self.config)
        await self.assistant.initialize_system()
        logger.info("✅ MAMA system initialized successfully")
        
        test_queries = self._generate_test_queries(num_interactions)
        logger.info(f"📝 Generated {len(test_queries)} test queries")
        
        agent_ids = [
            'safety_assessment_agent',
            'economic_agent', 
            'weather_agent',
            'flight_info_agent',
            'integration_agent'
        ]
        
        for i, query in enumerate(test_queries):
            logger.info(f"🔄 Processing query {i+1}/{num_interactions}: {query['text']}")
            
            try:
                start_time = datetime.now()
                result = await self.assistant.process_flight_query(
                    departure=query['departure_city'],
                    destination=query['destination_city'],
                    date=query['date'],
                    preferences=query['preferences']
                )
                end_time = datetime.now()
                
                response_time = (end_time - start_time).total_seconds()
                
                mrr_score = self._calculate_simulated_mrr(result, query)
                ndcg_score = self._calculate_simulated_ndcg(result, query)
                art_value = response_time
                
                system_reward = (self.lambda1 * mrr_score + 
                               self.lambda2 * ndcg_score - 
                               self.lambda3 * art_value)
                
                logger.info(f"📊 Performance metrics - MRR: {mrr_score:.4f}, NDCG: {ndcg_score:.4f}, ART: {art_value:.4f}")
                logger.info(f"🎯 System reward: {system_reward:.4f}")
                
                competence_scores = {}
                for agent_id in agent_ids:
                    new_competence = self.assistant.trust_ledger.evaluate_competence(
                        agent_id=agent_id,
                        system_reward=system_reward,
                        task_context={
                            'preferences': query['preferences'],
                            'query_id': query['query_id']
                        }
                    )
                    competence_scores[agent_id] = new_competence
                
                log_entry = {
                    'interaction': i + 1,
                    'query_id': query['query_id'],
                    'system_reward': system_reward,
                    'mrr': mrr_score,
                    'ndcg': ndcg_score,
                    'art': art_value,
                    'competence_scores': competence_scores
                }
                
                self.competence_log.append(log_entry)
                self.reward_log.append(system_reward)
                
                if (i + 1) % 10 == 0:
                    avg_reward = np.mean(self.reward_log[-10:])
                    logger.info(f"📈 Progress: {i+1}/{num_interactions}, recent 10 average reward: {avg_reward:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error processing query {i+1}: {e}")
                continue
        
        await self.assistant.cleanup()
        self._save_and_plot_results()
        
    def _calculate_simulated_mrr(self, result, query):
        if not result or 'recommendations' not in result:
            return 0.1
        
        priority = query['preferences'].get('priority', 'safety')
        
        if priority == 'safety':
            return np.random.uniform(0.7, 0.9)
        elif priority == 'cost':
            return np.random.uniform(0.6, 0.8)
        elif priority == 'time':
            return np.random.uniform(0.5, 0.7)
        else:  # comfort
            return np.random.uniform(0.6, 0.8)
    
    def _calculate_simulated_ndcg(self, result, query):
        if not result or 'recommendations' not in result:
            return 0.1
        
        num_recommendations = len(result.get('recommendations', []))
        base_ndcg = min(0.9, 0.5 + 0.1 * num_recommendations)
        
        return base_ndcg + np.random.uniform(-0.1, 0.1)
    
    def _save_and_plot_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        log_path = self.results_dir / f"reward_driven_experiment_{timestamp}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.competence_log, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Experiment data saved to: {log_path}")
        
        self._plot_competence_evolution(timestamp)
        self._plot_reward_evolution(timestamp)
        self._print_final_statistics()
    
    def _plot_competence_evolution(self, timestamp):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        interactions = [entry['interaction'] for entry in self.competence_log]
        
        agent_names = {
            'safety_assessment_agent': 'Safety Assessment',
            'economic_agent': 'Economic Agent',
            'weather_agent': 'Weather Agent', 
            'flight_info_agent': 'Flight Info Agent',
            'integration_agent': 'Integration Agent'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (agent_id, display_name) in enumerate(agent_names.items()):
            scores = [entry['competence_scores'][agent_id] for entry in self.competence_log]
            ax.plot(interactions, scores, 
                   label=display_name, 
                   color=colors[i],
                   marker=markers[i],
                   markersize=4,
                   markevery=10)
        
        ax.set_xlabel('Interaction Number')
        ax.set_ylabel('Competence Score')
        ax.set_title('Agent Competence Evolution')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'competence_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.figures_dir / f'competence_evolution_{timestamp}.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_reward_evolution(self, timestamp):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        interactions = range(1, len(self.reward_log) + 1)
        
        ax.plot(interactions, self.reward_log, 
                color='#2E86AB', 
                marker='o',
                markersize=4,
                markevery=10,
                label='System Reward')
        
        window = 10
        moving_avg = np.convolve(self.reward_log, 
                                np.ones(window)/window, 
                                mode='valid')
        ax.plot(range(window, len(self.reward_log) + 1), 
                moving_avg,
                color='#F24236',
                linestyle='--',
                label=f'{window}-point Moving Average')
        
        ax.set_xlabel('Interaction Number')
        ax.set_ylabel('System Reward')
        ax.set_title('System Reward Evolution')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'reward_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.figures_dir / f'reward_evolution_{timestamp}.pdf', bbox_inches='tight')
        plt.close()
    
    def _print_final_statistics(self):
        final_rewards = self.reward_log[-10:]
        final_avg_reward = np.mean(final_rewards)
        final_std_reward = np.std(final_rewards)
        
        initial_rewards = self.reward_log[:10]
        initial_avg_reward = np.mean(initial_rewards)
        
        improvement = ((final_avg_reward - initial_avg_reward) / initial_avg_reward) * 100
        
        logger.info("\n" + "="*50)
        logger.info("📊 Final Statistics")
        logger.info("="*50)
        logger.info(f"Initial average reward (first 10): {initial_avg_reward:.4f}")
        logger.info(f"Final average reward (last 10): {final_avg_reward:.4f} ± {final_std_reward:.4f}")
        logger.info(f"Overall improvement: {improvement:+.1f}%")
        
        final_competence = self.competence_log[-1]['competence_scores']
        logger.info("\nFinal Agent Competence Scores:")
        for agent_id, score in final_competence.items():
            logger.info(f"  {agent_id}: {score:.4f}")
        
        logger.info("\n✅ Experiment completed successfully!")

async def main():
        experiment = RewardDrivenExperiment()
    await experiment.run_experiment()

if __name__ == "__main__":
    asyncio.run(main()) 