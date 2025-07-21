#!/usr/bin/env python3
"""
Final Experiment Runner - Based on 150 test queries
Strictly using the standard split of the original 1000 query dataset: 700 training/150 validation/150 testing
"""

import json
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any
import os
import logging

# Set random seed to ensure reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Final150TestExperiment:
    """Final experiment based on 150 test queries"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
    def load_test_set(self):
        """Load 150 test queries"""
        logger.info("📊 Loading test query set...")
        
        # Try different possible paths
        paths_to_try = [
            Path('data/test_queries_150.json'),
            Path('../data/test_queries_150.json'),
            Path('../../data/test_queries_150.json'),
            Path('src/data/test_queries_150.json')
        ]
        
        test_set = None
        for path in paths_to_try:
            if path.exists():
                logger.info(f"Found test set at: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    test_set = json.load(f)
                break
        
        if test_set is None:
            raise FileNotFoundError("Could not find test query set. Please ensure test_queries_150.json exists in data directory.")
            
        logger.info(f"✅ Loaded {len(test_set)} test queries")
        return test_set
    
    def _import_models(self):
        """Import model classes dynamically"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from models.mama_full import MAMAFull
        from models.mama_no_trust import MAMANoTrust
        from models.single_agent_system import SingleAgentSystemModel
        from models.traditional_ranking import TraditionalRanking
        from evaluation.standard_evaluator import StandardEvaluator
        
        return {
            'mama_full': MAMAFull(),
            'mama_no_trust': MAMANoTrust(),
            'single_agent': SingleAgentSystemModel(),
            'traditional': TraditionalRanking(),
            'evaluator': StandardEvaluator()
        }
    
    def run_experiment(self):
        """Run the final experiment on 150 test queries"""
        logger.info("🚀 Starting Final 150 Test Experiment")
        start_time = time.time()
        
        try:
            # Load test queries
            test_queries = self.load_test_set()
            
            # Import models
            logger.info("🔧 Importing models...")
            components = self._import_models()
            
            # Models
            mama_full = components['mama_full']
            mama_no_trust = components['mama_no_trust']
            single_agent = components['single_agent']
            traditional = components['traditional']
            evaluator = components['evaluator']
            
            # Run evaluations on all models
            logger.info("🧪 Running model evaluations...")
            results = {}
            
            # MAMA Full
            logger.info("Evaluating MAMA Full...")
            mama_full_results = evaluator.evaluate_model(mama_full, test_queries)
            results['mama_full'] = mama_full_results
            logger.info(f"MAMA Full - MRR: {mama_full_results['mrr']:.4f}, NDCG@5: {mama_full_results['ndcg@5']:.4f}")
            
            # MAMA No Trust
            logger.info("Evaluating MAMA No Trust...")
            mama_no_trust_results = evaluator.evaluate_model(mama_no_trust, test_queries)
            results['mama_no_trust'] = mama_no_trust_results
            logger.info(f"MAMA No Trust - MRR: {mama_no_trust_results['mrr']:.4f}, NDCG@5: {mama_no_trust_results['ndcg@5']:.4f}")
            
            # Single Agent
            logger.info("Evaluating Single Agent...")
            single_agent_results = evaluator.evaluate_model(single_agent, test_queries)
            results['single_agent'] = single_agent_results
            logger.info(f"Single Agent - MRR: {single_agent_results['mrr']:.4f}, NDCG@5: {single_agent_results['ndcg@5']:.4f}")
            
            # Traditional
            logger.info("Evaluating Traditional Ranking...")
            traditional_results = evaluator.evaluate_model(traditional, test_queries)
            results['traditional'] = traditional_results
            logger.info(f"Traditional - MRR: {traditional_results['mrr']:.4f}, NDCG@5: {traditional_results['ndcg@5']:.4f}")
            
            # Calculate relative improvements
            mama_full_mrr = mama_full_results['mrr']
            mama_no_trust_mrr = mama_no_trust_results['mrr']
            single_agent_mrr = single_agent_results['mrr']
            traditional_mrr = traditional_results['mrr']
            
            trust_improvement = ((mama_full_mrr - mama_no_trust_mrr) / mama_no_trust_mrr) * 100
            multi_agent_improvement = ((mama_full_mrr - single_agent_mrr) / single_agent_mrr) * 100
            overall_improvement = ((mama_full_mrr - traditional_mrr) / traditional_mrr) * 100
            
            # Save results
            output_data = {
                'timestamp': self.timestamp,
                'dataset_size': len(test_queries),
                'results': results,
                'improvements': {
                    'trust_improvement': float(trust_improvement),
                    'multi_agent_improvement': float(multi_agent_improvement),
                    'overall_improvement': float(overall_improvement)
                },
                'statistical_tests': self._run_statistical_tests(results)
            }
            
            output_file = self.results_dir / f'final_150_test_experiment_{self.timestamp}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("=" * 50)
            logger.info("📊 EXPERIMENT RESULTS SUMMARY")
            logger.info("=" * 50)
            logger.info(f"MAMA Full - MRR: {mama_full_mrr:.4f}")
            logger.info(f"MAMA No Trust - MRR: {mama_no_trust_mrr:.4f}")
            logger.info(f"Single Agent - MRR: {single_agent_mrr:.4f}")
            logger.info(f"Traditional - MRR: {traditional_mrr:.4f}")
            logger.info("-" * 50)
            logger.info(f"Trust Improvement: {trust_improvement:.1f}%")
            logger.info(f"Multi-agent Improvement: {multi_agent_improvement:.1f}%")
            logger.info(f"Overall Improvement: {overall_improvement:.1f}%")
            logger.info("-" * 50)
            logger.info(f"Runtime: {duration:.2f} seconds")
            logger.info(f"Results saved to: {output_file}")
            logger.info("=" * 50)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _run_statistical_tests(self, results):
        """Run statistical significance tests between models"""
        mama_full = np.array(results['mama_full']['mrr_per_query'])
        mama_no_trust = np.array(results['mama_no_trust']['mrr_per_query'])
        single_agent = np.array(results['single_agent']['mrr_per_query'])
        traditional = np.array(results['traditional']['mrr_per_query'])
        
        # Paired t-tests
        _, pval_trust = stats.ttest_rel(mama_full, mama_no_trust)
        _, pval_agent = stats.ttest_rel(mama_full, single_agent)
        _, pval_trad = stats.ttest_rel(mama_full, traditional)
        
        # Effect sizes (Cohen's d)
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
        
        d_trust = cohens_d(mama_full, mama_no_trust)
        d_agent = cohens_d(mama_full, single_agent)
        d_trad = cohens_d(mama_full, traditional)
        
        return {
            'mama_full_vs_mama_no_trust': {
                'p_value': float(pval_trust),
                'cohens_d': float(d_trust),
                'significant': bool(pval_trust < 0.05)
            },
            'mama_full_vs_single_agent': {
                'p_value': float(pval_agent),
                'cohens_d': float(d_agent),
                'significant': bool(pval_agent < 0.05)
            },
            'mama_full_vs_traditional': {
                'p_value': float(pval_trad),
                'cohens_d': float(d_trad),
                'significant': bool(pval_trad < 0.05)
            }
        }


def main():
    """Run the final experiment"""
    experiment = Final150TestExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main() 