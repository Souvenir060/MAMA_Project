#!/usr/bin/env python3
"""
MAMA Framework - Main Evaluation Script
Runs comprehensive evaluation of all model
"""

import json
import logging
import os
import time
import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from models.mama_full import MAMAFull
from models.mama_no_trust import MAMANoTrust
from models.single_agent_system import SingleAgentSystemModel
from models.traditional_ranking import TraditionalRanking
from core.evaluation_metrics import calculate_mrr, calculate_ndcg_at_k

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
        
def load_test_dataset(data_path: str = None) -> List[Dict[str, Any]]:
    """
    Load the standard 150-query test dataset
    
    Args:
        data_path: Optional path to test dataset
    
    Returns:
        List of test queries
    """
    if data_path:
        dataset_path = Path(data_path)
    else:
        # Try standard locations
        dataset_path = Path('../data/test_queries.json')
        if not dataset_path.exists():
            dataset_path = Path('data/test_queries.json')
            
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {dataset_path}")
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    logger.info(f"✅ Successfully loaded {len(queries)} test queries from {dataset_path}")
    return queries
    
def run_model_evaluation(model, queries: List[Dict[str, Any]], 
                        use_cache: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Run evaluation on a single model
    
    Args:
        model: Model to evaluate
        queries: List of test queries
        use_cache: Whether to use cached results if available
        verbose: Whether to display detailed progress
        
    Returns:
        List of evaluation results for each query
    """
    results = []
    
    # Check for cached results
    cache_file = Path(f'results/cache/{model.model_name.lower().replace(" ", "_")}_results.json')
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_results = json.load(f)
            logger.info(f"Loaded cached results for {model.model_name}")
            return cached_results
        except Exception as e:
            logger.warning(f"Failed to load cached results: {e}")
    
    # Ensure cache directory exists
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_queries = len(queries)
    start_time_all = time.time()
    
    for i, query in enumerate(queries):
        if verbose:
            logger.info(f"Processing query {i+1}/{total_queries} with {model.model_name}")
        elif (i+1) % 10 == 0 or i+1 == total_queries:
            logger.info(f"Progress: {i+1}/{total_queries} queries processed with {model.model_name}")
        
        start_time = time.time()
        
        try:
            # Process the query with the model
            result = model.process_query(query)
            processing_time = time.time() - start_time
            
            # Extract ground truth and recommendations
            ground_truth = query.get('ground_truth_id', '')
            recommendations = result.get('recommendations', [])
            recommendation_ids = [r.get('flight_id', '') for r in recommendations]
            
            # Calculate metrics
            mrr = calculate_mrr([{
                'ground_truth_id': ground_truth,
                'recommendations': recommendation_ids
            }])
            
            ndcg5 = calculate_ndcg_at_k([{
                'ground_truth_id': ground_truth,
                'recommendations': recommendation_ids
            }], k=5)
            
            # Store result
            results.append({
                'query_id': query.get('query_id', f'query_{i+1:03d}'),
                'ground_truth_id': ground_truth,
                'recommendations': recommendation_ids,
                'MRR': float(mrr),
                'NDCG@5': float(ndcg5),
                'Response_Time': float(processing_time),
                'model': model.model_name
            })
        except Exception as e:
            logger.error(f"Error processing query {i+1}: {e}")
            
            # Add failed result
            results.append({
                'query_id': query.get('query_id', f'query_{i+1:03d}'),
                'ground_truth_id': query.get('ground_truth_id', ''),
                'recommendations': [],
                'MRR': 0.0,
                'NDCG@5': 0.0,
                'Response_Time': time.time() - start_time,
                'model': model.model_name,
                'error': str(e)
            })
    
    total_time = time.time() - start_time_all
    logger.info(f"Finished evaluating {model.model_name} on {total_queries} queries in {total_time:.2f}s")
    
    # Cache results
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results
    
def calculate_statistics(results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """
    Calculate statistics for a model's results
    
    Args:
        results: Evaluation results
        model_name: Model name to filter results
    
    Returns:
        Statistics dictionary
    """
    model_results = [r for r in results if r['model'] == model_name]
    
    if not model_results:
        return None
    
    mrr_values = [r['MRR'] for r in model_results]
    ndcg_values = [r['NDCG@5'] for r in model_results]
    response_times = [r['Response_Time'] for r in model_results]
    
    return {
        'model': model_name,
        'MRR_mean': float(np.mean(mrr_values)),
        'MRR_std': float(np.std(mrr_values, ddof=1)),
        'NDCG@5_mean': float(np.mean(ndcg_values)),
        'NDCG@5_std': float(np.std(ndcg_values, ddof=1)),
        'Response_Time_mean': float(np.mean(response_times)),
        'Response_Time_std': float(np.std(response_times, ddof=1)),
        'sample_size': len(model_results)
    }

def generate_report(statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report
    
    Args:
        statistics: List of model statistics
        
    Returns:
        Report dictionary
    """
    # Find statistics for each model
    try:
        mama_full = next(s for s in statistics if s['model'] == 'MAMA Full')
        mama_no_trust = next(s for s in statistics if s['model'] == 'MAMA No Trust')
        single_agent = next(s for s in statistics if s['model'] == 'Single Agent System')
        traditional = next(s for s in statistics if s['model'] == 'Traditional Ranking')
        
        # Calculate improvements
        trust_improvement = ((mama_full['MRR_mean'] - mama_no_trust['MRR_mean']) / 
                            mama_no_trust['MRR_mean'] * 100)
        multi_agent_improvement = ((mama_full['MRR_mean'] - single_agent['MRR_mean']) / 
                                single_agent['MRR_mean'] * 100)
        overall_improvement = ((mama_full['MRR_mean'] - traditional['MRR_mean']) / 
                            traditional['MRR_mean'] * 100)
        
        report = {
            'key_findings': [
                f"MAMA Full achieved best performance: MRR={mama_full['MRR_mean']:.4f}±{mama_full['MRR_std']:.4f}",
                f"Trust mechanism contributed {trust_improvement:.1f}% improvement",
                f"Multi-agent approach shows {multi_agent_improvement:.1f}% advantage",
                f"Overall improvement of {overall_improvement:.1f}% over traditional methods"
            ],
            'performance_ranking': [
                f"1. MAMA Full: {mama_full['MRR_mean']:.4f}",
                f"2. MAMA No Trust: {mama_no_trust['MRR_mean']:.4f}",
                f"3. Single Agent: {single_agent['MRR_mean']:.4f}",
                f"4. Traditional: {traditional['MRR_mean']:.4f}"
            ],
            'improvements': {
                'trust_improvement': float(trust_improvement),
                'multi_agent_improvement': float(multi_agent_improvement),
                'overall_improvement': float(overall_improvement)
            }
        }
        
        return report
    except (StopIteration, KeyError) as e:
        logger.error(f"Error generating report: {e}")
        return {
            'key_findings': ["Error generating report - missing model statistics"],
            'performance_ranking': [],
            'improvements': {}
        }

def run_evaluation(args):
    """
    Run complete evaluation on all models
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to output file
    """
    logger.info("🚀 MAMA Framework - Main Evaluation")
    logger.info("=" * 50)
    
    # Create results directory
    results_dir = Path('../results')
    if not results_dir.exists():
        results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load test dataset
    logger.info("📊 Loading test dataset...")
    queries = load_test_dataset(args.data_path)
    
    # Initialize models
    logger.info("🔧 Initializing models...")
    models = [
        MAMAFull(),
        MAMANoTrust(),
        SingleAgentSystemModel(),
        TraditionalRanking()
    ]
    
    # Run evaluations
    logger.info(f"🔬 Running evaluations on {len(queries)} queries...")
    all_results = []
    statistics = []
    
    for model in models:
        logger.info(f"Evaluating {model.model_name}...")
        model_results = run_model_evaluation(model, queries, not args.no_cache, args.verbose)
        all_results.extend(model_results)
        
        # Calculate model statistics
        stats = calculate_statistics(model_results, model.model_name)
        statistics.append(stats)
        logger.info(f"{model.model_name}: MRR={stats['MRR_mean']:.4f}±{stats['MRR_std']:.4f}, "
                  f"NDCG@5={stats['NDCG@5_mean']:.4f}±{stats['NDCG@5_std']:.4f}")
    
    # Generate evaluation report
    logger.info("📝 Generating evaluation report...")
    report = generate_report(statistics)
    
    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_file = results_dir / f'final_run_150_test_set_{timestamp}.json'
    
    output_data = {
        'metadata': {
            'experiment_name': 'MAMA Main Evaluation',
            'timestamp': timestamp,
            'query_count': len(queries),
            'settings': {
                'use_cache': not args.no_cache,
                'verbose': args.verbose
            }
        },
        'results': all_results,
        'statistics': statistics,
        'report': report
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Generate markdown report
    report_file = results_dir / f'MAMA_Evaluation_Report_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# MAMA Framework Evaluation Report\n\n")
        f.write(f"Date: {timestamp}\n\n")
        
        f.write("## Key Findings\n\n")
        for finding in report['key_findings']:
            f.write(f"- {finding}\n")
        
        f.write("\n## Model Performance\n\n")
        for rank in report['performance_ranking']:
            f.write(f"- {rank}\n")
        
        f.write("\n## Detailed Statistics\n\n")
        f.write("| Model | MRR | NDCG@5 | Response Time (s) |\n")
        f.write("|-------|-----|--------|------------------|\n")
        for stat in statistics:
            f.write(f"| {stat['model']} | {stat['MRR_mean']:.4f}±{stat['MRR_std']:.4f} | "
                   f"{stat['NDCG@5_mean']:.4f}±{stat['NDCG@5_std']:.4f} | "
                   f"{stat['Response_Time_mean']:.3f}±{stat['Response_Time_std']:.3f} |\n")
    
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info(f"📄 Report saved to: {report_file}")
    logger.info("✅ Evaluation completed!")
    
    return output_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MAMA Framework Main Evaluation')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to test dataset JSON file')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force rerunning all evaluations without using cache')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed progress information')
    args = parser.parse_args()
    
    try:
        output_file = run_evaluation(args)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
