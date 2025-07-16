#!/usr/bin/env python3
"""
Standard Evaluator - MAMA System Academic Experiments
Ensures all models use the same evaluation standards and metrics to avoid evaluation bias
"""

import json
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardEvaluator:
    """Standard Evaluator"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize standard evaluator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Academic definitions of evaluation metrics
        self.metrics_definitions = {
            'MRR': 'Mean Reciprocal Rank',
            'NDCG@5': 'Normalized Discounted Cumulative Gain at 5',
            'NDCG@10': 'Normalized Discounted Cumulative Gain at 10',
            'MAP': 'Mean Average Precision',
            'ART': 'Average Response Time',
            'Precision@1': 'Precision at 1',
            'Precision@5': 'Precision at 5',
            'Kendall_Tau': 'Kendall Tau correlation coefficient',
            'Spearman_Rho': 'Spearman rank correlation coefficient'
        }
        
        # Record all evaluation results
        self.evaluation_history = []
        
    def load_test_data(self, test_file: str) -> List[Dict[str, Any]]:
        """
        Load test data
        
        Args:
            test_file: Test data file path
            
        Returns:
            List of test queries
        """
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            logger.info(f"✅ Loaded test data: {len(test_data)} queries")
            return test_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            return []
    
    def evaluate_model(self, model: Any, test_data: List[Dict[str, Any]], 
                      model_name: str = "Unknown") -> Dict[str, Any]:
        """
        Evaluate single model performance
        
        Args:
            model: Model instance to evaluate
            test_data: Test data
            model_name: Model name
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"🔄 Starting evaluation for model: {model_name}")
        
        # Record start time
        evaluation_start_time = time.time()
        
        # Initialize result storage
        results = {
            'model_name': model_name,
            'evaluation_start_time': datetime.now().isoformat(),
            'total_queries': len(test_data),
            'successful_queries': 0,
            'failed_queries': 0,
            'query_results': [],
            'response_times': [],
            'rankings': [],
            'relevance_scores': [],
            'ground_truth_rankings': []
        }
        
        # Process test queries one by one
        for i, query_data in enumerate(test_data):
            try:
                # Record query start time
                query_start_time = time.time()
                
                # Call model to process query
                model_result = self._call_model_safely(model, query_data)
                
                # Record response time
                response_time = time.time() - query_start_time
                
                if model_result:
                    # Process model output
                    predicted_ranking = self._extract_ranking_from_result(model_result)
                    ground_truth_ranking = query_data['ground_truth_ranking']
                    relevance_scores = query_data['relevance_scores']
                    
                    # Store results
                    results['query_results'].append({
                        'query_id': query_data['query_id'],
                        'predicted_ranking': predicted_ranking,
                        'ground_truth_ranking': ground_truth_ranking,
                        'relevance_scores': relevance_scores,
                        'response_time': response_time,
                        'model_result': model_result
                    })
                    
                    results['response_times'].append(response_time)
                    results['rankings'].append(predicted_ranking)
                    results['relevance_scores'].append(relevance_scores)
                    results['ground_truth_rankings'].append(ground_truth_ranking)
                    results['successful_queries'] += 1
                    
                else:
                    results['failed_queries'] += 1
                    logger.warning(f"⚠️ Query {query_data['query_id']} processing failed")
                
                # Progress report
                if (i + 1) % 50 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(test_data)} queries")
                    
            except Exception as e:
                logger.error(f"❌ Error processing query {query_data.get('query_id', 'unknown')}: {e}")
                results['failed_queries'] += 1
        
        # Calculate all evaluation metrics
        if results['successful_queries'] > 0:
            metrics = self._calculate_comprehensive_metrics(results)
            results['metrics'] = metrics
        else:
            results['metrics'] = self._get_zero_metrics()
        
        # Record total evaluation time
        results['total_evaluation_time'] = time.time() - evaluation_start_time
        results['evaluation_end_time'] = datetime.now().isoformat()
        
        # Save evaluation history
        self.evaluation_history.append(results)
        
        logger.info(f"✅ Model {model_name} evaluation completed")
        logger.info(f"📊 Successful queries: {results['successful_queries']}/{results['total_queries']}")
        logger.info(f"⏱️  Total time: {results['total_evaluation_time']:.2f} seconds")
        
        return results
    
    def _call_model_safely(self, model: Any, query_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely call model, handle various exceptions"""
        try:
            # Choose calling method based on model type
            if hasattr(model, 'process_query'):
                return model.process_query(query_data)
            elif hasattr(model, 'predict'):
                return model.predict(query_data)
            elif hasattr(model, 'recommend'):
                return model.recommend(query_data)
            elif callable(model):
                return model(query_data)
            else:
                logger.error(f"❌ Unsupported model type: {type(model)}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Model call failed: {e}")
            return None
    
    def _extract_ranking_from_result(self, model_result: Dict[str, Any]) -> List[str]:
        """Extract ranking from model result"""
        if 'ranking' in model_result:
            return model_result['ranking']
        elif 'recommendations' in model_result:
            # Extract ranking from recommendations
            recommendations = model_result['recommendations']
            if isinstance(recommendations, list):
                return [rec.get('flight_id', f"flight_{i:03d}") for i, rec in enumerate(recommendations)]
        elif 'predicted_ranking' in model_result:
            return model_result['predicted_ranking']
        else:
            # Default ranking
            return [f"flight_{i:03d}" for i in range(1, 11)]
    
    def _calculate_comprehensive_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # 1. Mean Reciprocal Rank (MRR)
        metrics['MRR'] = self._calculate_mrr(
            results['rankings'], 
            results['ground_truth_rankings']
        )
        
        # 2. NDCG@5 and NDCG@10
        metrics['NDCG@5'] = self._calculate_ndcg(
            results['rankings'], 
            results['relevance_scores'], 
            k=5
        )
        
        metrics['NDCG@10'] = self._calculate_ndcg(
            results['rankings'], 
            results['relevance_scores'], 
            k=10
        )
        
        # 3. Mean Average Precision (MAP)
        metrics['MAP'] = self._calculate_map(
            results['rankings'], 
            results['relevance_scores']
        )
        
        # 4. Average Response Time (ART)
        metrics['ART'] = np.mean(results['response_times'])
        
        # 5. Precision@1 and Precision@5
        metrics['Precision@1'] = self._calculate_precision_at_k(
            results['rankings'], 
            results['ground_truth_rankings'], 
            k=1
        )
        
        metrics['Precision@5'] = self._calculate_precision_at_k(
            results['rankings'], 
            results['ground_truth_rankings'], 
            k=5
        )
        
        # 6. Rank Correlation
        metrics['Kendall_Tau'] = self._calculate_kendall_tau(
            results['rankings'], 
            results['ground_truth_rankings']
        )
        
        metrics['Spearman_Rho'] = self._calculate_spearman_rho(
            results['rankings'], 
            results['ground_truth_rankings']
        )
        
        # 7. System performance metrics
        metrics['Success_Rate'] = results['successful_queries'] / results['total_queries']
        metrics['Average_Response_Time'] = np.mean(results['response_times'])
        metrics['Response_Time_Std'] = np.std(results['response_times'])
        
        return metrics
    
    def _calculate_mrr(self, predicted_rankings: List[List[str]], 
                      ground_truth_rankings: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        MRR = 1/|Q| × Σ(1/rank_i)
        """
        reciprocal_ranks = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not gt_ranking:
                continue
                
            # Find position of first relevant item
            relevant_item = gt_ranking[0]  # Most relevant item
            try:
                rank = pred_ranking.index(relevant_item) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _calculate_ndcg(self, predicted_rankings: List[List[str]], 
                       relevance_scores_list: List[Dict[str, float]], 
                       k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k)
        """
        ndcg_scores = []
        
        for pred_ranking, relevance_scores in zip(predicted_rankings, relevance_scores_list):
            if not pred_ranking or not relevance_scores:
                continue
            
            # Build true relevance and predicted relevance
            y_true = []
            y_score = []
            
            for item in pred_ranking[:k]:
                relevance = relevance_scores.get(item, 0.0)
                y_true.append(relevance)
                y_score.append(1.0)  # Simplified prediction score
            
            if len(y_true) > 0:
                try:
                    # Use sklearn's ndcg_score
                    ndcg = ndcg_score([y_true], [y_score], k=k)
                    ndcg_scores.append(ndcg)
                except:
                    # Manual NDCG calculation
                    dcg = self._calculate_dcg(y_true, k)
                    ideal_relevance = sorted(y_true, reverse=True)
                    idcg = self._calculate_dcg(ideal_relevance, k)
                    ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_dcg(self, relevance_scores: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain (DCG)"""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    def _calculate_map(self, predicted_rankings: List[List[str]], 
                      relevance_scores_list: List[Dict[str, float]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        ap_scores = []
        
        for pred_ranking, relevance_scores in zip(predicted_rankings, relevance_scores_list):
            if not pred_ranking or not relevance_scores:
                continue
            
            # Calculate Average Precision
            relevant_items = []
            precision_at_k = []
            
            for i, item in enumerate(pred_ranking):
                if relevance_scores.get(item, 0.0) > 0.5:  # Relevance threshold
                    relevant_items.append(i + 1)
                    precision_at_k.append(len(relevant_items) / (i + 1))
            
            if relevant_items:
                ap = np.mean(precision_at_k)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _calculate_precision_at_k(self, predicted_rankings: List[List[str]], 
                                 ground_truth_rankings: List[List[str]], 
                                 k: int) -> float:
        """Calculate P@k precision"""
        precisions = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not pred_ranking or not gt_ranking:
                continue
            
            # Calculate how many of top k predictions are relevant
            relevant_set = set(gt_ranking[:k])
            predicted_set = set(pred_ranking[:k])
            
            intersection = relevant_set.intersection(predicted_set)
            precision = len(intersection) / k if k > 0 else 0.0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_kendall_tau(self, predicted_rankings: List[List[str]], 
                              ground_truth_rankings: List[List[str]]) -> float:
        """Calculate Kendall Tau correlation coefficient"""
        correlations = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not pred_ranking or not gt_ranking:
                continue
            
            # Create ranking mappings
            pred_ranks = {item: i for i, item in enumerate(pred_ranking)}
            gt_ranks = {item: i for i, item in enumerate(gt_ranking)}
            
            # Find common items
            common_items = set(pred_ranks.keys()).intersection(set(gt_ranks.keys()))
            
            if len(common_items) > 1:
                pred_vals = [pred_ranks[item] for item in common_items]
                gt_vals = [gt_ranks[item] for item in common_items]
                
                try:
                    tau, _ = kendalltau(pred_vals, gt_vals)
                    if not np.isnan(tau):
                        correlations.append(tau)
                except:
                    pass
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_spearman_rho(self, predicted_rankings: List[List[str]], 
                               ground_truth_rankings: List[List[str]]) -> float:
        """Calculate Spearman correlation coefficient"""
        correlations = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not pred_ranking or not gt_ranking:
                continue
            
            # Create ranking mappings
            pred_ranks = {item: i for i, item in enumerate(pred_ranking)}
            gt_ranks = {item: i for i, item in enumerate(gt_ranking)}
            
            # Find common items
            common_items = set(pred_ranks.keys()).intersection(set(gt_ranks.keys()))
            
            if len(common_items) > 1:
                pred_vals = [pred_ranks[item] for item in common_items]
                gt_vals = [gt_ranks[item] for item in common_items]
                
                try:
                    rho, _ = spearmanr(pred_vals, gt_vals)
                    if not np.isnan(rho):
                        correlations.append(rho)
                except:
                    pass
        
        return np.mean(correlations) if correlations else 0.0
    
    def _get_zero_metrics(self) -> Dict[str, Any]:
        """Return zero metrics (when evaluation fails)"""
        return {
            'MRR': 0.0,
            'NDCG@5': 0.0,
            'NDCG@10': 0.0,
            'MAP': 0.0,
            'ART': 0.0,
            'Precision@1': 0.0,
            'Precision@5': 0.0,
            'Kendall_Tau': 0.0,
            'Spearman_Rho': 0.0,
            'Success_Rate': 0.0,
            'Average_Response_Time': 0.0,
            'Response_Time_Std': 0.0
        }
    
    def generate_comparison_report(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison report for multiple models"""
        if not results_list:
            return {}
        
        # Aggregate metrics for all models
        comparison_data = {}
        
        for result in results_list:
            model_name = result['model_name']
            metrics = result['metrics']
            
            comparison_data[model_name] = {
                'MRR': metrics['MRR'],
                'NDCG@5': metrics['NDCG@5'],
                'NDCG@10': metrics['NDCG@10'],
                'MAP': metrics['MAP'],
                'ART': metrics['ART'],
                'Precision@1': metrics['Precision@1'],
                'Precision@5': metrics['Precision@5'],
                'Success_Rate': metrics['Success_Rate'],
                'Kendall_Tau': metrics['Kendall_Tau'],
                'Spearman_Rho': metrics['Spearman_Rho']
            }
        
        # Find best models
        best_models = {}
        for metric in ['MRR', 'NDCG@5', 'MAP', 'Precision@1']:
            best_model = max(comparison_data.keys(), 
                           key=lambda x: comparison_data[x][metric])
            best_models[metric] = best_model
        
        # Generate report
        report = {
            'comparison_data': comparison_data,
            'best_models': best_models,
            'total_models': len(results_list),
            'evaluation_date': datetime.now().isoformat(),
            'metrics_definitions': self.metrics_definitions
        }
        
        return report
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Evaluation results saved to: {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def evaluate_single_agent_output(self, agent_output: Dict[str, Any], 
                                     ground_truth: Dict[str, Any], 
                                     agent_type: str) -> float:
        """
        Evaluate single agent output for real accuracy score
        
        Args:
            agent_output: Agent output
            ground_truth: Corresponding ground truth data
            agent_type: Agent type (for selecting evaluation strategy)
        
        Returns:
            Real accuracy score (0.0 - 1.0)
        """
        try:
            # Use different evaluation strategies based on agent type
            if 'safety' in agent_type.lower():
                return self._evaluate_safety_agent(agent_output, ground_truth)
            elif 'economic' in agent_type.lower():
                return self._evaluate_economic_agent(agent_output, ground_truth)
            elif 'weather' in agent_type.lower():
                return self._evaluate_weather_agent(agent_output, ground_truth)
            elif 'flight' in agent_type.lower():
                return self._evaluate_flight_agent(agent_output, ground_truth)
            else:
                # Generic evaluation method
                return self._evaluate_generic_agent(agent_output, ground_truth)
                
        except Exception as e:
            logger.error(f"Single agent evaluation failed {agent_type}: {e}")
            return 0.0
    
    def _evaluate_safety_agent(self, agent_output: Dict[str, Any], 
                              ground_truth: Dict[str, Any]) -> float:
        """Evaluate safety assessment agent"""
        try:
            result = agent_output.get('result', {})
            gt_safety = ground_truth.get('safety_score', 0.0)
            
            if isinstance(result, dict):
                # Extract safety score
                predicted_safety = result.get('overall_safety_score', 
                                            result.get('safety_score', 
                                                     result.get('score', 0.5)))
            else:
                predicted_safety = 0.5
            
            # Calculate accuracy based on closeness to ground truth
            error = abs(predicted_safety - gt_safety)
            accuracy = max(0.0, 1.0 - error)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Safety agent evaluation failed: {e}")
            return 0.0
    
    def _evaluate_economic_agent(self, agent_output: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> float:
        """Evaluate economic agent"""
        try:
            result = agent_output.get('result', {})
            gt_cost = ground_truth.get('economic_score', 0.0)
            
            if isinstance(result, dict):
                # Extract economic score
                predicted_cost = result.get('total_cost_per_flight', 
                                          result.get('cost_score', 
                                                   result.get('economic_score', 
                                                            result.get('score', 0.5))))
            else:
                predicted_cost = 0.5
            
            # Normalization
            if gt_cost > 0:
                error = abs(predicted_cost - gt_cost) / max(gt_cost, predicted_cost)
                accuracy = max(0.0, 1.0 - error)
            else:
                accuracy = 0.5
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Economic agent evaluation failed: {e}")
            return 0.0
    
    def _evaluate_weather_agent(self, agent_output: Dict[str, Any], 
                               ground_truth: Dict[str, Any]) -> float:
        """Evaluate weather agent"""
        try:
            result = agent_output.get('result', {})
            gt_weather = ground_truth.get('weather_score', 0.0)
            
            if isinstance(result, dict):
                predicted_weather = result.get('safety_score', 
                                             result.get('weather_score', 
                                                      result.get('score', 0.5)))
            else:
                predicted_weather = 0.5
            
            error = abs(predicted_weather - gt_weather)
            accuracy = max(0.0, 1.0 - error)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Weather agent evaluation failed: {e}")
            return 0.0
    
    def _evaluate_flight_agent(self, agent_output: Dict[str, Any], 
                              ground_truth: Dict[str, Any]) -> float:
        """Evaluate flight information agent"""
        try:
            result = agent_output.get('result', {})
            
            # Check if flight information was successfully retrieved
            if isinstance(result, dict) and 'flight_list' in result:
                flight_list = result['flight_list']
                if flight_list and len(flight_list) > 0:
                    # Evaluate based on number of flights retrieved
                    expected_count = ground_truth.get('expected_flight_count', 5)
                    actual_count = len(flight_list)
                    
                    # Calculate coverage
                    coverage = min(1.0, actual_count / expected_count)
                    return coverage
                else:
                    return 0.0
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Flight agent evaluation failed: {e}")
            return 0.0
    
    def _evaluate_generic_agent(self, agent_output: Dict[str, Any], 
                               ground_truth: Dict[str, Any]) -> float:
        """Generic agent evaluation"""
        try:
            # Evaluate based on output completeness and quality
            result = agent_output.get('result', {})
            success = agent_output.get('success', True)
            confidence = agent_output.get('confidence', 0.5)
            
            if not success:
                return 0.0
            
            # If there are specific score fields
            if isinstance(result, dict):
                score = result.get('score', result.get('confidence', confidence))
                return min(1.0, max(0.0, score))
            else:
                return confidence
                
        except Exception as e:
            logger.warning(f"Generic agent evaluation failed: {e}")
            return 0.0
    
    def print_metrics_summary(self, results: Dict[str, Any]):
        """Print evaluation metrics summary"""
        model_name = results['model_name']
        metrics = results['metrics']
        
        print(f"\n📊 Model {model_name} Evaluation Results:")
        print("=" * 60)
        print(f"📈 MRR (Mean Reciprocal Rank): {metrics['MRR']:.4f}")
        print(f"📈 NDCG@5: {metrics['NDCG@5']:.4f}")
        print(f"📈 NDCG@10: {metrics['NDCG@10']:.4f}")
        print(f"📈 MAP (Mean Average Precision): {metrics['MAP']:.4f}")
        print(f"📈 Precision@1: {metrics['Precision@1']:.4f}")
        print(f"📈 Precision@5: {metrics['Precision@5']:.4f}")
        print(f"⏱️  ART (Average Response Time): {metrics['ART']:.4f}s")
        print(f"✅ Success Rate: {metrics['Success_Rate']:.4f}")
        print(f"🔗 Kendall Tau: {metrics['Kendall_Tau']:.4f}")
        print(f"🔗 Spearman Rho: {metrics['Spearman_Rho']:.4f}")
        print("=" * 60) 