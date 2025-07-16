#!/usr/bin/env python3
"""
Ground Truth Robustness Analysis Experiment - Real Model Version
Validates that MAMA framework's performance advantage is insensitive to Ground Truth generator parameter variations

Using real MAMA system models:
- MAMA (Full) - Complete multi-agent system
- MAMA (No Trust) - Version without trust mechanism
- Single Agent - Single agent baseline
- Traditional Ranking - Traditional ranking baseline
"""

import json
import numpy as np
import time
import logging
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import ndcg_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mama_full import MAMAFull
from models.base_model import ModelConfig
from main import MAMAFlightAssistant, QueryProcessingConfig
from models.traditional_ranking import generate_decision_tree_ground_truth

np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealMAMARobustnessExperiment:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        self.filter_modes = {
            'Normal': {
                'safety_threshold': 0.4,
                'budget_multiplier': 1.0,
                'description': 'Baseline mode - Paper parameters'
            },
            'Loose': {
                'safety_threshold': 0.3,
                'budget_multiplier': 1.5,
                'description': 'Relaxed mode - More candidates enter ranking'
            },
            'Strict': {
                'safety_threshold': 0.5,
                'budget_multiplier': 0.8,
                'description': 'Strict mode - Fewer candidates, simpler ranking'
            }
        }
        
        self.models = self._initialize_real_models()
        
    def _initialize_real_models(self) -> Dict[str, Any]:
        logger.info("🔄 Initializing real MAMA system models...")
        
        models = {}
        
        try:
            mama_config = ModelConfig(
                alpha=0.7,  # SBERT weight
                beta=0.2,   # Trust weight  
                gamma=0.1,  # Historical performance weight
                max_agents=3,
                trust_threshold=0.5
            )
            models['MAMA_Full'] = MAMAFull(config=mama_config)
            logger.info("✅ MAMA Full System initialized")
            
            no_trust_config = ModelConfig(
                alpha=0.8,
                beta=0.0,
                gamma=0.2,
                max_agents=3,
                trust_threshold=0.0
            )
            models['MAMA_NoTrust'] = MAMAFull(config=no_trust_config)
            models['MAMA_NoTrust'].trust_enabled = False
            logger.info("✅ MAMA No Trust System initialized")
            
            single_config = ModelConfig(
                alpha=1.0,
                beta=0.0,
                gamma=0.0,
                max_agents=1,
                trust_threshold=0.0
            )
            models['SingleAgent'] = MAMAFull(config=single_config)
            models['SingleAgent'].trust_enabled = False
            models['SingleAgent'].historical_enabled = False
            models['SingleAgent'].marl_enabled = False
            logger.info("✅ Single Agent System initialized")
            
            traditional_config = ModelConfig(
                alpha=0.0,
                beta=0.0,
                gamma=0.0,
                max_agents=1,
                trust_threshold=0.0
            )
            models['Traditional'] = MAMAFull(config=traditional_config)
            models['Traditional'].sbert_enabled = False
            models['Traditional'].trust_enabled = False
            models['Traditional'].historical_enabled = False
            models['Traditional'].marl_enabled = False
            logger.info("✅ Traditional Ranking System initialized")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
        
        logger.info(f"✅ Successfully initialized {len(models)} real models")
        return models
    
    def generate_modified_ground_truth(self, flight_options: List[Dict[str, Any]], 
                                     user_preferences: Dict[str, str],
                                     mode: str) -> List[str]:
        mode_params = self.filter_modes[mode]
        safety_threshold = mode_params['safety_threshold']
        budget_multiplier = mode_params['budget_multiplier']
        
        modified_flight_options = []
        
        for flight in flight_options:
            modified_flight = {
                'flight_id': flight.get('flight_id', f"flight_{len(modified_flight_options)+1:03d}"),
                'safety_score': flight.get('safety_score', np.random.uniform(0.2, 0.95)),
                'price': flight.get('price', np.random.uniform(300, 1200)),
                'duration': flight.get('duration', np.random.uniform(2.0, 8.0)),
                'availability': flight.get('availability', True)
            }
            
            if modified_flight['safety_score'] <= safety_threshold:
                continue
                
            budget = user_preferences.get('budget', 'medium')
            price = modified_flight['price']
            
            if budget == 'low' and price >= (500 * budget_multiplier):
                continue
            elif budget == 'medium' and price >= (1000 * budget_multiplier):
                continue
                
            modified_flight_options.append(modified_flight)
        
        if len(modified_flight_options) < 3:
            logger.warning(f"Mode {mode}: Too few flights after filtering, using original list")
            modified_flight_options = flight_options
        
        try:
            priority = user_preferences.get('priority', 'safety')
            
            if priority == 'safety':
                modified_flight_options.sort(key=lambda x: x.get('safety_score', 0.5), reverse=True)
            elif priority == 'cost':
                modified_flight_options.sort(key=lambda x: x.get('price', 1000), reverse=False)
            elif priority == 'time':
                modified_flight_options.sort(key=lambda x: x.get('duration', 5.0), reverse=False)
            else:
                modified_flight_options.sort(key=lambda x: x.get('safety_score', 0.5), reverse=True)
            
            ground_truth_ranking = [flight['flight_id'] for flight in modified_flight_options]
            
            all_flight_ids = [f.get('flight_id', f"flight_{i:03d}") for i, f in enumerate(flight_options)]
            for flight_id in all_flight_ids:
                if flight_id not in ground_truth_ranking:
                    ground_truth_ranking.append(flight_id)
            
            return ground_truth_ranking[:10]
            
        except Exception as e:
            logger.error(f"Ground Truth generation failed: {e}")
            return [f.get('flight_id', f"flight_{i:03d}") for i, f in enumerate(flight_options[:10])]
    
    def load_test_set(self) -> List[Dict[str, Any]]:
        dataset_path = Path('data/standard_dataset.json')
        
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                
                if 'test' in dataset and len(dataset['test']) >= 150:
                    test_queries = dataset['test'][:150]
                    logger.info(f"✅ Loaded 150 test queries from existing dataset")
                    return test_queries
                    
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}")
        
        logger.info("📊 Generating 150 new test queries...")
        test_queries = []
        
        cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
            "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington"
        ]
        
        priorities = ['safety', 'cost', 'time', 'comfort']
        budgets = ['low', 'medium', 'high']
        
        for i in range(150):
            departure = np.random.choice(cities)
            destination = np.random.choice([city for city in cities if city != departure])
            
            query = {
                'query_id': f'test_{i+1:03d}',
                'departure': departure,
                'destination': destination,
                'date': '2024-12-15',
                'preferences': {
                    'priority': priorities[i % len(priorities)],
                    'budget': budgets[i % len(budgets)]
                }
            }
            test_queries.append(query)
        
        logger.info("✅ Generated 150 test queries")
        return test_queries
    
    def evaluate_model_on_query(self, model: Any, query: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            result = model.process_query(query)
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'ranking': result.get('ranking', []),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def calculate_mrr(self, predicted_ranking: List[str], ground_truth: List[str]) -> float:
        if not predicted_ranking or not ground_truth:
            return 0.0
        
        try:
            optimal_item = ground_truth[0]
            rank = predicted_ranking.index(optimal_item) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    def calculate_ndcg_5(self, predicted_ranking: List[str], ground_truth: List[str]) -> float:
        if not predicted_ranking or not ground_truth:
            return 0.0
        
        try:
            k = 5
            relevance = np.zeros(len(predicted_ranking))
            
            for i, item in enumerate(predicted_ranking[:k]):
                if item in ground_truth[:3]:
                    relevance[i] = 1.0 if i >= 3 else 3.0 - i
                    
            ideal_relevance = np.zeros(len(ground_truth))
            for i in range(min(3, len(ground_truth))):
                ideal_relevance[i] = 3.0 - i
                
            return ndcg_score([ideal_relevance], [relevance])
            
        except Exception as e:
            logger.error(f"NDCG calculation failed: {e}")
            return 0.0
    
    def calculate_model_performance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {'mrr': 0.0, 'ndcg': 0.0, 'avg_time': 0.0}
        
        mrr_scores = []
        ndcg_scores = []
        processing_times = []
        
        for result in results:
            if result.get('success', False):
                mrr_scores.append(result['mrr'])
                ndcg_scores.append(result['ndcg'])
                processing_times.append(result['processing_time'])
        
        return {
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'avg_time': np.mean(processing_times) if processing_times else 0.0
        }
    
    def run_robustness_analysis(self) -> Dict[str, Any]:
        logger.info("🚀 Starting robustness analysis experiment")
        
        test_queries = self.load_test_set()
        all_mode_results = {}
        
        for mode_name, mode_config in self.filter_modes.items():
            logger.info(f"\n📊 Processing mode: {mode_name}")
            logger.info(f"Description: {mode_config['description']}")
            
            mode_results = {model_name: [] for model_name in self.models.keys()}
            
            for i, query in enumerate(test_queries):
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(test_queries)} queries processed")
                
                try:
                    ground_truth = self.generate_modified_ground_truth(
                        query.get('flight_options', []),
                        query.get('preferences', {}),
                        mode_name
                    )
                    
                    for model_name, model in self.models.items():
                        evaluation_result = self.evaluate_model_on_query(model, query)
                        
                        if evaluation_result['success']:
                            predicted_ranking = evaluation_result['ranking']
                            
                            result = {
                                'query_id': query['query_id'],
                                'success': True,
                                'mrr': self.calculate_mrr(predicted_ranking, ground_truth),
                                'ndcg': self.calculate_ndcg_5(predicted_ranking, ground_truth),
                                'processing_time': evaluation_result['processing_time']
                            }
                        else:
                            result = {
                                'query_id': query['query_id'],
                                'success': False,
                                'error': evaluation_result.get('error', 'Unknown error')
                            }
                            
                        mode_results[model_name].append(result)
                        
                except Exception as e:
                    logger.error(f"Error processing query {query['query_id']}: {e}")
                    continue
            
            mode_summary = {}
            for model_name, results in mode_results.items():
                performance = self.calculate_model_performance(results)
                mode_summary[model_name] = performance
                
                logger.info(f"\n{model_name} Performance:")
                logger.info(f"MRR: {performance['mrr']:.4f}")
                logger.info(f"NDCG@5: {performance['ndcg']:.4f}")
                logger.info(f"Avg. Processing Time: {performance['avg_time']:.4f}s")
            
            all_mode_results[mode_name] = {
                'config': mode_config,
                'results': mode_results,
                'summary': mode_summary
            }
        
        self.save_results(all_mode_results)
        return all_mode_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        results_file = self.results_dir / f'robustness_analysis_{self.timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Results saved to: {results_file}")
        
        report_file = self.results_dir / f'robustness_analysis_report_{self.timestamp}.md'
        report_content = self.generate_results_table(results)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"📄 Report saved to: {report_file}")
    
    def generate_results_table(self, all_mode_results: Dict[str, Any]) -> str:
        table = "# Ground Truth Robustness Analysis Results\n\n"
        table += "## Performance Comparison Across Different Filter Modes\n\n"
        table += "| Filter Mode | Model | MRR | NDCG@5 | Avg. Time (s) |\n"
        table += "|-------------|-------|-----|---------|---------------|\n"
        
        for mode_name, mode_data in all_mode_results.items():
            summary = mode_data['summary']
            first_row = True
            
            for model_name, metrics in summary.items():
                if first_row:
                    mode_cell = f"**{mode_name}**<br>({mode_data['config']['description']})"
                    first_row = False
                else:
                    mode_cell = ""
                
                table += f"| {mode_cell} | {model_name} | {metrics['mrr']:.4f} | {metrics['ndcg']:.4f} | {metrics['avg_time']:.4f} |\n"
        
        return table

def main():
    experiment = RealMAMARobustnessExperiment()
    results = experiment.run_robustness_analysis()
    logger.info("✅ Robustness analysis completed successfully!")

if __name__ == "__main__":
    main() 