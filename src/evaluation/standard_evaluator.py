#!/usr/bin/env python3
"""
标准化评估器 - MAMA 系统学术实验
确保所有模型使用相同的评估标准和指标，避免评估偏差
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
    """标准化评估器"""
    
    def __init__(self, random_seed: int = 42):
        """
        初始化标准化评估器
        
        Args:
            random_seed: 随机种子确保可复现性
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # 评估指标的学术定义
        self.metrics_definitions = {
            'MRR': 'Mean Reciprocal Rank - 平均倒数排名',
            'NDCG@5': 'Normalized Discounted Cumulative Gain at 5',
            'NDCG@10': 'Normalized Discounted Cumulative Gain at 10',
            'MAP': 'Mean Average Precision - 平均精确度',
            'ART': 'Average Response Time - 平均响应时间',
            'Precision@1': 'Precision at 1 - 第一位精确度',
            'Precision@5': 'Precision at 5 - 前五位精确度',
            'Kendall_Tau': 'Kendall Tau correlation coefficient',
            'Spearman_Rho': 'Spearman rank correlation coefficient'
        }
        
        # 记录所有评估结果
        self.evaluation_history = []
        
    def load_test_data(self, test_file: str) -> List[Dict[str, Any]]:
        """
        加载测试数据
        
        Args:
            test_file: 测试数据文件路径
            
        Returns:
            测试查询列表
        """
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            logger.info(f"✅ 加载测试数据: {len(test_data)} 条查询")
            return test_data
            
        except Exception as e:
            logger.error(f"❌ 加载测试数据失败: {e}")
            return []
    
    def evaluate_model(self, model: Any, test_data: List[Dict[str, Any]], 
                      model_name: str = "Unknown") -> Dict[str, Any]:
        """
        评估单个模型的性能
        
        Args:
            model: 待评估的模型实例
            test_data: 测试数据
            model_name: 模型名称
            
        Returns:
            完整的评估结果
        """
        logger.info(f"🔄 开始评估模型: {model_name}")
        
        # 记录开始时间
        evaluation_start_time = time.time()
        
        # 初始化结果存储
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
        
        # 逐个处理测试查询
        for i, query_data in enumerate(test_data):
            try:
                # 记录单个查询的开始时间
                query_start_time = time.time()
                
                # 调用模型处理查询
                model_result = self._call_model_safely(model, query_data)
                
                # 记录响应时间
                response_time = time.time() - query_start_time
                
                if model_result:
                    # 处理模型输出
                    predicted_ranking = self._extract_ranking_from_result(model_result)
                    ground_truth_ranking = query_data['ground_truth_ranking']
                    relevance_scores = query_data['relevance_scores']
                    
                    # 存储结果
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
                    logger.warning(f"⚠️ 查询 {query_data['query_id']} 处理失败")
                
                # 进度报告
                if (i + 1) % 50 == 0:
                    logger.info(f"📊 已处理 {i + 1}/{len(test_data)} 条查询")
                    
            except Exception as e:
                logger.error(f"❌ 处理查询 {query_data.get('query_id', 'unknown')} 时出错: {e}")
                results['failed_queries'] += 1
        
        # 计算所有评估指标
        if results['successful_queries'] > 0:
            metrics = self._calculate_comprehensive_metrics(results)
            results['metrics'] = metrics
        else:
            results['metrics'] = self._get_zero_metrics()
        
        # 记录总评估时间
        results['total_evaluation_time'] = time.time() - evaluation_start_time
        results['evaluation_end_time'] = datetime.now().isoformat()
        
        # 保存评估历史
        self.evaluation_history.append(results)
        
        logger.info(f"✅ 模型 {model_name} 评估完成")
        logger.info(f"📊 成功查询: {results['successful_queries']}/{results['total_queries']}")
        logger.info(f"⏱️  总用时: {results['total_evaluation_time']:.2f} 秒")
        
        return results
    
    def _call_model_safely(self, model: Any, query_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """安全调用模型，处理各种异常情况"""
        try:
            # 根据模型类型选择调用方式
            if hasattr(model, 'process_query'):
                return model.process_query(query_data)
            elif hasattr(model, 'predict'):
                return model.predict(query_data)
            elif hasattr(model, 'recommend'):
                return model.recommend(query_data)
            elif callable(model):
                return model(query_data)
            else:
                logger.error(f"❌ 模型类型不支持: {type(model)}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 模型调用失败: {e}")
            return None
    
    def _extract_ranking_from_result(self, model_result: Dict[str, Any]) -> List[str]:
        """从模型结果中提取排名"""
        if 'ranking' in model_result:
            return model_result['ranking']
        elif 'recommendations' in model_result:
            # 从推荐结果中提取排名
            recommendations = model_result['recommendations']
            if isinstance(recommendations, list):
                return [rec.get('flight_id', f"flight_{i:03d}") for i, rec in enumerate(recommendations)]
        elif 'predicted_ranking' in model_result:
            return model_result['predicted_ranking']
        else:
            # 默认排名
            return [f"flight_{i:03d}" for i in range(1, 11)]
    
    def _calculate_comprehensive_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合评估指标"""
        metrics = {}
        
        # 1. Mean Reciprocal Rank (MRR)
        metrics['MRR'] = self._calculate_mrr(
            results['rankings'], 
            results['ground_truth_rankings']
        )
        
        # 2. NDCG@5 和 NDCG@10
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
        
        # 5. Precision@1 和 Precision@5
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
        
        # 7. 系统性能指标
        metrics['Success_Rate'] = results['successful_queries'] / results['total_queries']
        metrics['Average_Response_Time'] = np.mean(results['response_times'])
        metrics['Response_Time_Std'] = np.std(results['response_times'])
        
        return metrics
    
    def _calculate_mrr(self, predicted_rankings: List[List[str]], 
                      ground_truth_rankings: List[List[str]]) -> float:
        """
        计算平均倒数排名 (MRR)
        MRR = 1/|Q| × Σ(1/rank_i)
        """
        reciprocal_ranks = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not gt_ranking:
                continue
                
            # 找到第一个相关项目的位置
            relevant_item = gt_ranking[0]  # 最相关的项目
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
        计算归一化折扣累积增益 (NDCG@k)
        """
        ndcg_scores = []
        
        for pred_ranking, relevance_scores in zip(predicted_rankings, relevance_scores_list):
            if not pred_ranking or not relevance_scores:
                continue
            
            # 构建真实相关性和预测相关性
            y_true = []
            y_score = []
            
            for item in pred_ranking[:k]:
                relevance = relevance_scores.get(item, 0.0)
                y_true.append(relevance)
                y_score.append(1.0)  # 简化的预测分数
            
            if len(y_true) > 0:
                try:
                    # 使用sklearn的ndcg_score
                    ndcg = ndcg_score([y_true], [y_score], k=k)
                    ndcg_scores.append(ndcg)
                except:
                    # 手动计算NDCG
                    dcg = self._calculate_dcg(y_true, k)
                    ideal_relevance = sorted(y_true, reverse=True)
                    idcg = self._calculate_dcg(ideal_relevance, k)
                    ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_dcg(self, relevance_scores: List[float], k: int) -> float:
        """计算折扣累积增益 (DCG)"""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    def _calculate_map(self, predicted_rankings: List[List[str]], 
                      relevance_scores_list: List[Dict[str, float]]) -> float:
        """计算平均精确度 (MAP)"""
        ap_scores = []
        
        for pred_ranking, relevance_scores in zip(predicted_rankings, relevance_scores_list):
            if not pred_ranking or not relevance_scores:
                continue
            
            # 计算Average Precision
            relevant_items = []
            precision_at_k = []
            
            for i, item in enumerate(pred_ranking):
                if relevance_scores.get(item, 0.0) > 0.5:  # 相关阈值
                    relevant_items.append(i + 1)
                    precision_at_k.append(len(relevant_items) / (i + 1))
            
            if relevant_items:
                ap = np.mean(precision_at_k)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _calculate_precision_at_k(self, predicted_rankings: List[List[str]], 
                                 ground_truth_rankings: List[List[str]], 
                                 k: int) -> float:
        """计算P@k精确度"""
        precisions = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not pred_ranking or not gt_ranking:
                continue
            
            # 计算前k个预测中有多少是相关的
            relevant_set = set(gt_ranking[:k])
            predicted_set = set(pred_ranking[:k])
            
            intersection = relevant_set.intersection(predicted_set)
            precision = len(intersection) / k if k > 0 else 0.0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_kendall_tau(self, predicted_rankings: List[List[str]], 
                              ground_truth_rankings: List[List[str]]) -> float:
        """计算Kendall Tau相关系数"""
        correlations = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not pred_ranking or not gt_ranking:
                continue
            
            # 创建排名映射
            pred_ranks = {item: i for i, item in enumerate(pred_ranking)}
            gt_ranks = {item: i for i, item in enumerate(gt_ranking)}
            
            # 找到共同项目
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
        """计算Spearman相关系数"""
        correlations = []
        
        for pred_ranking, gt_ranking in zip(predicted_rankings, ground_truth_rankings):
            if not pred_ranking or not gt_ranking:
                continue
            
            # 创建排名映射
            pred_ranks = {item: i for i, item in enumerate(pred_ranking)}
            gt_ranks = {item: i for i, item in enumerate(gt_ranking)}
            
            # 找到共同项目
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
        """返回零值指标（当评估失败时）"""
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
        """生成多个模型的对比报告"""
        if not results_list:
            return {}
        
        # 汇总所有模型的指标
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
        
        # 找出最佳模型
        best_models = {}
        for metric in ['MRR', 'NDCG@5', 'MAP', 'Precision@1']:
            best_model = max(comparison_data.keys(), 
                           key=lambda x: comparison_data[x][metric])
            best_models[metric] = best_model
        
        # 生成报告
        report = {
            'comparison_data': comparison_data,
            'best_models': best_models,
            'total_models': len(results_list),
            'evaluation_date': datetime.now().isoformat(),
            'metrics_definitions': self.metrics_definitions
        }
        
        return report
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存评估结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 评估结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
    
    def evaluate_single_agent_output(self, agent_output: Dict[str, Any], 
                                     ground_truth: Dict[str, Any], 
                                     agent_type: str) -> float:
        """
        评估单个智能体输出的真实准确性分数
        
        Args:
            agent_output: 智能体的输出
            ground_truth: 对应的ground truth数据
            agent_type: 智能体类型（用于选择评估策略）
        
        Returns:
            真实的准确性分数 (0.0 - 1.0)
        """
        try:
            # 根据智能体类型采用不同的评估策略
            if 'safety' in agent_type.lower():
                return self._evaluate_safety_agent(agent_output, ground_truth)
            elif 'economic' in agent_type.lower():
                return self._evaluate_economic_agent(agent_output, ground_truth)
            elif 'weather' in agent_type.lower():
                return self._evaluate_weather_agent(agent_output, ground_truth)
            elif 'flight' in agent_type.lower():
                return self._evaluate_flight_agent(agent_output, ground_truth)
            else:
                # 通用评估方法
                return self._evaluate_generic_agent(agent_output, ground_truth)
                
        except Exception as e:
            logger.error(f"单智能体评估失败 {agent_type}: {e}")
            return 0.0
    
    def _evaluate_safety_agent(self, agent_output: Dict[str, Any], 
                              ground_truth: Dict[str, Any]) -> float:
        """评估安全评估智能体"""
        try:
            result = agent_output.get('result', {})
            gt_safety = ground_truth.get('safety_score', 0.0)
            
            if isinstance(result, dict):
                # 提取安全分数
                predicted_safety = result.get('overall_safety_score', 
                                            result.get('safety_score', 
                                                     result.get('score', 0.5)))
            else:
                predicted_safety = 0.5
            
            # 计算准确性：基于与ground truth的接近程度
            error = abs(predicted_safety - gt_safety)
            accuracy = max(0.0, 1.0 - error)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"安全智能体评估失败: {e}")
            return 0.0
    
    def _evaluate_economic_agent(self, agent_output: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> float:
        """评估经济智能体"""
        try:
            result = agent_output.get('result', {})
            gt_cost = ground_truth.get('economic_score', 0.0)
            
            if isinstance(result, dict):
                # 提取经济分数
                predicted_cost = result.get('total_cost_per_flight', 
                                          result.get('cost_score', 
                                                   result.get('economic_score', 
                                                            result.get('score', 0.5))))
            else:
                predicted_cost = 0.5
            
            # 标准化处理
            if gt_cost > 0:
                error = abs(predicted_cost - gt_cost) / max(gt_cost, predicted_cost)
                accuracy = max(0.0, 1.0 - error)
            else:
                accuracy = 0.5
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"经济智能体评估失败: {e}")
            return 0.0
    
    def _evaluate_weather_agent(self, agent_output: Dict[str, Any], 
                               ground_truth: Dict[str, Any]) -> float:
        """评估天气智能体"""
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
            logger.warning(f"天气智能体评估失败: {e}")
            return 0.0
    
    def _evaluate_flight_agent(self, agent_output: Dict[str, Any], 
                              ground_truth: Dict[str, Any]) -> float:
        """评估航班信息智能体"""
        try:
            result = agent_output.get('result', {})
            
            # 检查是否成功获取航班信息
            if isinstance(result, dict) and 'flight_list' in result:
                flight_list = result['flight_list']
                if flight_list and len(flight_list) > 0:
                    # 基于获取到的航班数量评估
                    expected_count = ground_truth.get('expected_flight_count', 5)
                    actual_count = len(flight_list)
                    
                    # 计算覆盖率
                    coverage = min(1.0, actual_count / expected_count)
                    return coverage
                else:
                    return 0.0
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"航班智能体评估失败: {e}")
            return 0.0
    
    def _evaluate_generic_agent(self, agent_output: Dict[str, Any], 
                               ground_truth: Dict[str, Any]) -> float:
        """通用智能体评估"""
        try:
            # 基于输出的完整性和质量评估
            result = agent_output.get('result', {})
            success = agent_output.get('success', True)
            confidence = agent_output.get('confidence', 0.5)
            
            if not success:
                return 0.0
            
            # 如果有特定的分数字段
            if isinstance(result, dict):
                score = result.get('score', result.get('confidence', confidence))
                return min(1.0, max(0.0, score))
            else:
                return confidence
                
        except Exception as e:
            logger.warning(f"通用智能体评估失败: {e}")
            return 0.0
    
    def print_metrics_summary(self, results: Dict[str, Any]):
        """打印评估指标摘要"""
        model_name = results['model_name']
        metrics = results['metrics']
        
        print(f"\n📊 模型 {model_name} 评估结果:")
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