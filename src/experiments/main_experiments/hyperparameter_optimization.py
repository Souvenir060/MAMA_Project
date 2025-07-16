#!/usr/bin/env python3
"""
Hyperparameter Optimizer - MAMA System Academic Experiments
Solve randomness in hyperparameter selection in papers, perform grid search and sensitivity analysis
"""

import json
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Iterator
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import models and evaluators
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import ModelConfig
from models.mama_full import MAMAFull
from evaluation.standard_evaluator import StandardEvaluator

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter Optimizer"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Hyperparameter Optimizer
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Evaluator
        self.evaluator = StandardEvaluator(random_seed)
        
        # Optimization history
        self.optimization_history = []
        
        # Best configuration
        self.best_config = None
        self.best_score = -np.inf
        
        # Hyperparameter search space
        self.search_space = {
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'trust_decay': [0.9, 0.95, 0.99, 0.995],
            'reward_scale': [0.1, 0.5, 1.0, 2.0],
            'exploration_rate': [0.1, 0.2, 0.3, 0.5],
            'batch_size': [16, 32, 64, 128],
            'hidden_dim': [64, 128, 256, 512],
            'num_layers': [2, 3, 4, 5],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3]
        }
        
        # Results storage
        self.results_dir = Path('results/hyperparameter_optimization')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_hyperparameter_configurations(self, 
                                             strategy: str = 'grid',
                                             max_configs: int = 100) -> List[Dict[str, Any]]:
        """Generate hyperparameter configurations"""
        
        if strategy == 'grid':
            return self._generate_grid_search_configs(max_configs)
        elif strategy == 'random':
            return self._generate_random_search_configs(max_configs)
        elif strategy == 'focused':
            return self._generate_focused_search_configs(max_configs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _generate_grid_search_configs(self, max_configs: int) -> List[Dict[str, Any]]:
        """Generate grid search configurations"""
        # For full grid search, we need to limit the search space
        reduced_space = {
            'learning_rate': [0.001, 0.01, 0.1],
            'trust_decay': [0.9, 0.95, 0.99],
            'reward_scale': [0.5, 1.0, 2.0],
            'exploration_rate': [0.1, 0.3, 0.5],
            'batch_size': [32, 64, 128],
            'hidden_dim': [128, 256, 512],
            'num_layers': [2, 3, 4],
            'dropout_rate': [0.0, 0.1, 0.2]
        }
        
        # Generate all combinations
        param_names = list(reduced_space.keys())
        param_values = list(reduced_space.values())
        
        configurations = []
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            configurations.append(config)
            
            if len(configurations) >= max_configs:
                break
        
        logger.info(f"Generated {len(configurations)} grid search configurations")
        return configurations
    
    def _generate_random_search_configs(self, max_configs: int) -> List[Dict[str, Any]]:
        """Generate random search configurations"""
        configurations = []
        
        for _ in range(max_configs):
            config = {}
            for param_name, param_values in self.search_space.items():
                config[param_name] = np.random.choice(param_values)
            configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} random search configurations")
        return configurations
    
    def _generate_focused_search_configs(self, max_configs: int) -> List[Dict[str, Any]]:
        """Generate focused search around known good configurations"""
        # Start with known good baseline
        baseline_config = {
            'learning_rate': 0.01,
            'trust_decay': 0.95,
            'reward_scale': 1.0,
            'exploration_rate': 0.2,
            'batch_size': 64,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout_rate': 0.1
        }
        
        configurations = [baseline_config]
        
        # Generate variations around baseline
        for _ in range(max_configs - 1):
            config = baseline_config.copy()
            
            # Randomly select 1-3 parameters to vary
            params_to_vary = np.random.choice(
                list(self.search_space.keys()), 
                size=np.random.randint(1, 4),
                replace=False
            )
            
            for param in params_to_vary:
                # Choose from nearby values
                current_value = config[param]
                available_values = self.search_space[param]
                
                if current_value in available_values:
                    current_idx = available_values.index(current_value)
                    # Select from nearby indices
                    nearby_indices = [
                        max(0, current_idx - 1),
                        current_idx,
                        min(len(available_values) - 1, current_idx + 1)
                    ]
                    selected_idx = np.random.choice(nearby_indices)
                    config[param] = available_values[selected_idx]
                else:
                    config[param] = np.random.choice(available_values)
            
            configurations.append(config)
        
        logger.info(f"Generated {len(configurations)} focused search configurations")
        return configurations
    
    def evaluate_configuration(self, config: Dict[str, Any], 
                             test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single hyperparameter configuration"""
        
        try:
            # Create model with configuration
            model_config = ModelConfig(**config)
            model = MAMAFull(model_config)
            
            # Evaluate on test queries
            start_time = time.time()
            results = self.evaluator.evaluate_model(model, test_queries, 
                                                   f"MAMA_config_{len(self.optimization_history)}")
            evaluation_time = time.time() - start_time
            
            # Extract key metrics
            metrics = results['metrics']
            score = metrics['MRR']  # Use MRR as primary optimization metric
            
            evaluation_result = {
                'config': config,
                'score': score,
                'metrics': metrics,
                'evaluation_time': evaluation_time,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                logger.info(f"New best configuration found! Score: {score:.4f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating configuration: {e}")
            return {
                'config': config,
                'score': -1.0,
                'metrics': {},
                'evaluation_time': 0.0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_optimization(self, 
                        test_queries: List[Dict[str, Any]], 
                        strategy: str = 'grid',
                        max_configs: int = 50) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        logger.info(f"Starting hyperparameter optimization with {strategy} strategy")
        logger.info(f"Max configurations: {max_configs}")
        
        # Generate configurations
        configurations = self.generate_hyperparameter_configurations(strategy, max_configs)
        
        # Evaluate each configuration
        for i, config in enumerate(configurations):
            logger.info(f"Evaluating configuration {i+1}/{len(configurations)}")
            
            result = self.evaluate_configuration(config, test_queries)
            self.optimization_history.append(result)
            
            # Log progress
            if result['success']:
                logger.info(f"Config {i+1}: Score={result['score']:.4f}, "
                          f"Time={result['evaluation_time']:.2f}s")
            else:
                logger.warning(f"Config {i+1}: Failed - {result.get('error', 'Unknown error')}")
        
        # Analyze results
        analysis = self._analyze_optimization_results()
        
        # Generate final report
        optimization_report = {
            'metadata': {
                'strategy': strategy,
                'max_configs': max_configs,
                'total_evaluated': len(self.optimization_history),
                'successful_evaluations': sum(1 for r in self.optimization_history if r['success']),
                'best_score': self.best_score,
                'best_config': self.best_config,
                'timestamp': datetime.now().isoformat()
            },
            'optimization_history': self.optimization_history,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        # Save results
        self._save_optimization_results(optimization_report)
        
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return optimization_report
    
    def _analyze_optimization_results(self) -> Dict[str, Any]:
        """Analyze optimization results"""
        
        successful_results = [r for r in self.optimization_history if r['success']]
        
        if not successful_results:
            return {'error': 'No successful evaluations'}
        
        # Extract scores and configurations
        scores = [r['score'] for r in successful_results]
        configs = [r['config'] for r in successful_results]
        
        # Basic statistics
        analysis = {
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'parameter_importance': self._calculate_parameter_importance(configs, scores),
            'top_configurations': self._get_top_configurations(successful_results, top_k=10),
            'convergence_analysis': self._analyze_convergence(scores)
        }
        
        return analysis
    
    def _calculate_parameter_importance(self, configs: List[Dict[str, Any]], 
                                      scores: List[float]) -> Dict[str, float]:
        """Calculate parameter importance using correlation analysis"""
        
        importance = {}
        
        for param_name in self.search_space.keys():
            param_values = []
            param_scores = []
            
            for config, score in zip(configs, scores):
                if param_name in config:
                    param_values.append(config[param_name])
                    param_scores.append(score)
            
            if len(set(param_values)) > 1:  # Need variation to calculate correlation
                # Convert to numeric if needed
                if isinstance(param_values[0], (int, float)):
                    correlation = np.corrcoef(param_values, param_scores)[0, 1]
                    importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    # For categorical parameters, use variance analysis
                    unique_values = list(set(param_values))
                    if len(unique_values) > 1:
                        score_groups = {}
                        for val, score in zip(param_values, param_scores):
                            if val not in score_groups:
                                score_groups[val] = []
                            score_groups[val].append(score)
                        
                        # Calculate between-group variance
                        group_means = [np.mean(scores) for scores in score_groups.values()]
                        overall_mean = np.mean(param_scores)
                        between_var = np.var(group_means)
                        total_var = np.var(param_scores)
                        
                        importance[param_name] = between_var / total_var if total_var > 0 else 0.0
                    else:
                        importance[param_name] = 0.0
            else:
                importance[param_name] = 0.0
        
        return importance
    
    def _get_top_configurations(self, results: List[Dict[str, Any]], 
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top k configurations by score"""
        
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        top_configs = []
        for i, result in enumerate(sorted_results[:top_k]):
            top_configs.append({
                'rank': i + 1,
                'score': result['score'],
                'config': result['config'],
                'metrics': result['metrics']
            })
        
        return top_configs
    
    def _analyze_convergence(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze convergence of optimization"""
        
        if len(scores) < 10:
            return {'error': 'Insufficient data for convergence analysis'}
        
        # Calculate running best
        running_best = []
        current_best = -np.inf
        
        for score in scores:
            if score > current_best:
                current_best = score
            running_best.append(current_best)
        
        # Calculate improvement rate
        improvements = []
        for i in range(1, len(running_best)):
            if running_best[i] > running_best[i-1]:
                improvements.append(i)
        
        # Convergence metrics
        convergence_analysis = {
            'total_improvements': len(improvements),
            'improvement_rate': len(improvements) / len(scores),
            'final_score': running_best[-1],
            'score_range': max(scores) - min(scores),
            'convergence_point': improvements[-1] if improvements else 0,
            'plateau_length': len(scores) - (improvements[-1] if improvements else 0)
        }
        
        return convergence_analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        
        if 'error' in analysis:
            return {'error': 'Cannot generate recommendations due to analysis error'}
        
        recommendations = {
            'best_parameters': {},
            'parameter_insights': {},
            'optimization_insights': []
        }
        
        # Best parameter recommendations
        if self.best_config:
            recommendations['best_parameters'] = self.best_config.copy()
        
        # Parameter insights
        param_importance = analysis.get('parameter_importance', {})
        sorted_importance = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        for param, importance in sorted_importance[:5]:  # Top 5 most important
            if importance > 0.1:  # Significant importance threshold
                recommendations['parameter_insights'][param] = {
                    'importance_score': importance,
                    'recommendation': f"This parameter significantly affects performance (importance: {importance:.3f})"
                }
        
        # Optimization insights
        convergence = analysis.get('convergence_analysis', {})
        
        if convergence.get('improvement_rate', 0) > 0.3:
            recommendations['optimization_insights'].append(
                "High improvement rate suggests more configurations could be beneficial"
            )
        
        if convergence.get('plateau_length', 0) > len(self.optimization_history) * 0.5:
            recommendations['optimization_insights'].append(
                "Long plateau suggests convergence - current best may be near optimal"
            )
        
        score_stats = analysis.get('score_statistics', {})
        if score_stats.get('std', 0) > 0.1:
            recommendations['optimization_insights'].append(
                "High score variance suggests hyperparameters have significant impact"
            )
        
        return recommendations
    
    def _save_optimization_results(self, report: Dict[str, Any]) -> None:
        """Save optimization results to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'hyperparameter_optimization_{timestamp}.json'
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimization results saved to: {filepath}")
    
    def generate_visualization(self, report: Dict[str, Any]) -> None:
        """Generate optimization visualization"""
        
        try:
            # Extract data for plotting
            successful_results = [r for r in report['optimization_history'] if r['success']]
            
            if not successful_results:
                logger.warning("No successful results to visualize")
                return
            
            scores = [r['score'] for r in successful_results]
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Hyperparameter Optimization Results', fontsize=16)
            
            # 1. Score progression
            axes[0, 0].plot(scores, 'b-', alpha=0.7, label='Scores')
            running_best = []
            current_best = -np.inf
            for score in scores:
                if score > current_best:
                    current_best = score
                running_best.append(current_best)
            axes[0, 0].plot(running_best, 'r-', linewidth=2, label='Running Best')
            axes[0, 0].set_xlabel('Configuration Index')
            axes[0, 0].set_ylabel('Score (MRR)')
            axes[0, 0].set_title('Score Progression')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Score distribution
            axes[0, 1].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            axes[0, 1].axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.3f}')
            axes[0, 1].set_xlabel('Score (MRR)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Score Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Parameter importance
            param_importance = report['analysis'].get('parameter_importance', {})
            if param_importance:
                params = list(param_importance.keys())
                importance_values = list(param_importance.values())
                
                axes[1, 0].barh(params, importance_values, color='lightcoral')
                axes[1, 0].set_xlabel('Importance Score')
                axes[1, 0].set_title('Parameter Importance')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Top configurations comparison
            top_configs = report['analysis'].get('top_configurations', [])[:10]
            if top_configs:
                top_scores = [config['score'] for config in top_configs]
                ranks = [config['rank'] for config in top_configs]
                
                axes[1, 1].bar(ranks, top_scores, color='lightgreen', alpha=0.7)
                axes[1, 1].set_xlabel('Configuration Rank')
                axes[1, 1].set_ylabel('Score (MRR)')
                axes[1, 1].set_title('Top 10 Configurations')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f'optimization_visualization_{timestamp}.png'
            plot_filepath = self.results_dir / plot_filename
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {plot_filepath}")
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")

def load_test_queries() -> List[Dict[str, Any]]:
    """Load test queries for optimization"""
    
    # Generate synthetic test queries for optimization
    test_queries = []
    
    for i in range(50):  # Use smaller set for optimization
        query = {
            'query_id': f'opt_query_{i+1:03d}',
            'departure_city': f'City_{np.random.randint(1, 11)}',
            'arrival_city': f'City_{np.random.randint(1, 11)}',
            'preferences': {
                'safety_priority': np.random.uniform(0.3, 0.9),
                'cost_priority': np.random.uniform(0.2, 0.8),
                'time_priority': np.random.uniform(0.1, 0.7)
            },
            'ground_truth_ranking': [f'flight_{j+1:03d}' for j in range(10)],
            'relevance_scores': {
                f'flight_{j+1:03d}': np.random.uniform(0.1, 1.0) for j in range(10)
            }
        }
        test_queries.append(query)
    
    return test_queries

def main():
    """Main function"""
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(random_seed=42)
    
    # Load test queries
    test_queries = load_test_queries()
    
    # Run optimization
    report = optimizer.run_optimization(
        test_queries=test_queries,
        strategy='focused',  # Use focused search for better results
        max_configs=30
    )
    
    # Generate visualization
    optimizer.generate_visualization(report)
    
    print("Hyperparameter optimization completed!")
    print(f"Best score: {report['metadata']['best_score']:.4f}")
    print(f"Best configuration: {report['metadata']['best_config']}")
    
    return report

if __name__ == "__main__":
    main() 