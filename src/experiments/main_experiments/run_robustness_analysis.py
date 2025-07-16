#!/usr/bin/env python3
"""
Ground Truth鲁棒性敏感性分析实验 - 真实模型版本
验证MAMA框架的性能优势对Ground Truth生成器中的过滤参数变化不敏感

使用真实的MAMA系统模型：
- MAMA (Full) - 完整的多智能体系统
- MAMA (No Trust) - 无信任机制版本 
- Single Agent - 单智能体基线
- Traditional Ranking - 传统排名基线

实验设计：
1. 定义三种Ground Truth生成模式：Normal, Loose, Strict
2. 对每种模式重新生成Ground Truth
3. 在150个查询的测试集上重新评估真实模型
4. 计算MAMA相对于Single Agent的优势
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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入MAMA系统的真实模型
from models.mama_full import MAMAFull
from models.base_model import ModelConfig
from main import MAMAFlightAssistant, QueryProcessingConfig
from models.traditional_ranking import generate_decision_tree_ground_truth

# 设置随机种子确保可复现性
np.random.seed(42)

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealMAMARobustnessExperiment:
    """使用真实MAMA模型的Ground Truth鲁棒性敏感性分析实验"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # 定义三种过滤模式的参数
        self.filter_modes = {
            'Normal': {
                'safety_threshold': 0.4,
                'budget_multiplier': 1.0,
                'description': '基准模式 - 论文既定参数'
            },
            'Loose': {
                'safety_threshold': 0.3,
                'budget_multiplier': 1.5,
                'description': '宽松模式 - 更多候选航班进入排序阶段'
            },
            'Strict': {
                'safety_threshold': 0.5,
                'budget_multiplier': 0.8,
                'description': '严格模式 - 更少候选航班，排序问题更简单'
            }
        }
        
        # 初始化真实模型实例
        self.models = self._initialize_real_models()
        
    def _initialize_real_models(self) -> Dict[str, Any]:
        """初始化真实的MAMA系统模型"""
        logger.info("🔄 初始化真实MAMA系统模型...")
        
        models = {}
        
        try:
            # 1. MAMA Full System
            mama_config = ModelConfig(
                alpha=0.7,  # SBERT权重
                beta=0.2,   # 信任权重  
                gamma=0.1,  # 历史表现权重
                max_agents=3,
                trust_threshold=0.5
            )
            models['MAMA_Full'] = MAMAFull(config=mama_config)
            logger.info("✅ MAMA Full System 初始化完成")
            
            # 2. MAMA No Trust System (修改配置禁用信任)
            no_trust_config = ModelConfig(
                alpha=0.8,  # 增加SBERT权重
                beta=0.0,   # 禁用信任权重
                gamma=0.2,  # 增加历史表现权重
                max_agents=3,
                trust_threshold=0.0
            )
            models['MAMA_NoTrust'] = MAMAFull(config=no_trust_config)
            models['MAMA_NoTrust'].trust_enabled = False  # 显式禁用信任
            logger.info("✅ MAMA No Trust System 初始化完成")
            
            # 3. Single Agent System (限制为单个智能体)
            single_config = ModelConfig(
                alpha=1.0,  # 只使用语义相似度
                beta=0.0,   # 无信任机制
                gamma=0.0,  # 无历史表现
                max_agents=1,  # 限制为单个智能体
                trust_threshold=0.0
            )
            models['SingleAgent'] = MAMAFull(config=single_config)
            models['SingleAgent'].trust_enabled = False
            models['SingleAgent'].historical_enabled = False
            models['SingleAgent'].marl_enabled = False
            logger.info("✅ Single Agent System 初始化完成")
            
            # 4. Traditional Ranking System (使用基础配置)
            traditional_config = ModelConfig(
                alpha=0.0,  # 无语义相似度
                beta=0.0,   # 无信任机制
                gamma=0.0,  # 无历史表现
                max_agents=1,
                trust_threshold=0.0
            )
            models['Traditional'] = MAMAFull(config=traditional_config)
            models['Traditional'].sbert_enabled = False
            models['Traditional'].trust_enabled = False
            models['Traditional'].historical_enabled = False
            models['Traditional'].marl_enabled = False
            logger.info("✅ Traditional Ranking System 初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
        
        logger.info(f"✅ 成功初始化 {len(models)} 个真实模型")
        return models
    
    def generate_modified_ground_truth(self, flight_options: List[Dict[str, Any]], 
                                     user_preferences: Dict[str, str],
                                     mode: str) -> List[str]:
        """
        基于不同过滤模式生成Ground Truth排名（修改版本）
        
        Args:
            flight_options: 包含10个候选航班对象的列表
            user_preferences: 用户偏好字典
            mode: 过滤模式 ('Normal', 'Loose', 'Strict')
            
        Returns:
            排序后的航班ID列表，作为Ground Truth
        """
        mode_params = self.filter_modes[mode]
        safety_threshold = mode_params['safety_threshold']
        budget_multiplier = mode_params['budget_multiplier']
        
        # 临时修改Ground Truth生成函数的参数
        # 这里我们需要创建一个修改版本的函数
        modified_flight_options = []
        
        for flight in flight_options:
            # 确保所有航班都有必需的字段
            modified_flight = {
                'flight_id': flight.get('flight_id', f"flight_{len(modified_flight_options)+1:03d}"),
                'safety_score': flight.get('safety_score', np.random.uniform(0.2, 0.95)),
                'price': flight.get('price', np.random.uniform(300, 1200)),
                'duration': flight.get('duration', np.random.uniform(2.0, 8.0)),
                'availability': flight.get('availability', True)
            }
            
            # 应用修改后的过滤条件
            # 安全分过滤
            if modified_flight['safety_score'] <= safety_threshold:
                continue
                
            # 预算过滤（应用倍数修正）
            budget = user_preferences.get('budget', 'medium')
            price = modified_flight['price']
            
            if budget == 'low' and price >= (500 * budget_multiplier):
                continue
            elif budget == 'medium' and price >= (1000 * budget_multiplier):
                continue
                
            modified_flight_options.append(modified_flight)
        
        # 如果过滤后航班太少，放宽条件
        if len(modified_flight_options) < 3:
            logger.warning(f"模式{mode}: 过滤后航班过少，使用原始航班列表")
            modified_flight_options = flight_options
        
        # 使用原始Ground Truth生成函数进行排序
        try:
            # 这里我们模拟原始函数的排序逻辑
            priority = user_preferences.get('priority', 'safety')
            
            if priority == 'safety':
                modified_flight_options.sort(key=lambda x: x.get('safety_score', 0.5), reverse=True)
            elif priority == 'cost':
                modified_flight_options.sort(key=lambda x: x.get('price', 1000), reverse=False)
            elif priority == 'time':
                modified_flight_options.sort(key=lambda x: x.get('duration', 5.0), reverse=False)
            else:
                modified_flight_options.sort(key=lambda x: x.get('safety_score', 0.5), reverse=True)
            
            # 生成最终排名
            ground_truth_ranking = [flight['flight_id'] for flight in modified_flight_options]
            
            # 如果不足10个，用原始航班填充
            all_flight_ids = [f.get('flight_id', f"flight_{i:03d}") for i, f in enumerate(flight_options)]
            for flight_id in all_flight_ids:
                if flight_id not in ground_truth_ranking:
                    ground_truth_ranking.append(flight_id)
            
            return ground_truth_ranking[:10]
            
        except Exception as e:
            logger.error(f"Ground Truth生成失败: {e}")
            # 返回默认排序
            return [f.get('flight_id', f"flight_{i:03d}") for i, f in enumerate(flight_options[:10])]
    
    def load_test_set(self) -> List[Dict[str, Any]]:
        """加载或生成150个测试查询"""
        
        # 尝试从现有数据集加载
        dataset_path = Path('data/standard_dataset.json')
        
        if dataset_path.exists():
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                
                if 'test' in dataset and len(dataset['test']) >= 150:
                    test_queries = dataset['test'][:150]
                    logger.info(f"✅ 从现有数据集加载了150个测试查询")
                    return test_queries
                    
            except Exception as e:
                logger.warning(f"加载现有数据集失败: {e}")
        
        # 生成新的150个测试查询
        logger.info("📊 生成150个新的测试查询...")
        test_queries = []
        
        # 城市对列表
        cities = [
            'Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu',
            'Hangzhou', 'Nanjing', 'Wuhan', 'Chongqing', 'Tianjin',
            'Shenyang', 'Dalian', 'Harbin', 'Changchun', 'Jinan',
            'Qingdao', 'Zhengzhou', 'Taiyuan', 'Shijiazhuang', 'Hohhot'
        ]
        
        # 偏好组合
        priorities = ['safety', 'cost', 'time', 'comfort']
        budgets = ['low', 'medium', 'high']
        
        for i in range(150):
            # 随机选择城市对
            departure = np.random.choice(cities)
            destination = np.random.choice([c for c in cities if c != departure])
            
            # 随机选择偏好
            priority = np.random.choice(priorities)
            budget = np.random.choice(budgets)
            
            # 生成10个航班选项
            flight_options = []
            for j in range(10):
                flight_options.append({
                    'flight_id': f"flight_{j+1:03d}",
                    'safety_score': np.random.uniform(0.2, 0.95),
                    'price': np.random.uniform(300, 1200),
                    'duration': np.random.uniform(2.0, 8.0),
                    'availability': True,
                    'airline': np.random.choice(['CA', 'MU', 'CZ', '3U', 'HU']),
                    'aircraft_type': np.random.choice(['Boeing 737', 'Airbus A320', 'Boeing 777'])
                })
            
            # 生成查询
            query = {
                'query_id': f'robustness_query_{i+1:03d}',
                'query_text': f"Find flights from {departure} to {destination}",
                'departure': departure,
                'destination': destination,
                'preferences': {
                    'priority': priority,
                    'budget': budget
                },
                'flight_options': flight_options,
                'metadata': {
                    'query_complexity': np.random.uniform(0.3, 0.9),
                    'route_popularity': np.random.uniform(0.1, 1.0)
                }
            }
            
            test_queries.append(query)
        
        logger.info(f"✅ 生成了{len(test_queries)}个测试查询")
        return test_queries
    
    def evaluate_model_on_query(self, model: Any, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用真实模型评估单个查询
        
        Args:
            model: 真实的模型实例
            query: 查询数据
            
        Returns:
            评估结果
        """
        try:
            start_time = time.time()
            
            # 调用真实模型的process_query方法
            result = model.process_query(query)
            
            processing_time = time.time() - start_time
            
            # 提取模型预测的排名
            predicted_ranking = result.get('ranking', [])
            
            # 获取Ground Truth排名
            ground_truth = query.get('ground_truth_ranking', [])
            
            # 计算MRR
            mrr = self.calculate_mrr(predicted_ranking, ground_truth)
            
            # 计算NDCG@5
            ndcg = self.calculate_ndcg_5(predicted_ranking, ground_truth)
            
            return {
                'query_id': query['query_id'],
                'success': result.get('success', True),
                'MRR': float(mrr),
                'NDCG@5': float(ndcg),
                'processing_time': float(processing_time),
                'predicted_ranking': predicted_ranking,
                'ground_truth': ground_truth,
                'model_result': result
            }
            
        except Exception as e:
            logger.error(f"模型评估失败 {query['query_id']}: {e}")
            return {
                'query_id': query['query_id'],
                'success': False,
                'MRR': 0.0,
                'NDCG@5': 0.0,
                'processing_time': 30.0,  # 超时
                'error': str(e)
            }
    
    def calculate_mrr(self, predicted_ranking: List[str], ground_truth: List[str]) -> float:
        """计算Mean Reciprocal Rank (MRR)"""
        if not predicted_ranking or not ground_truth:
            return 0.0
        
        # 找到第一个正确预测的位置
        for i, predicted_flight in enumerate(predicted_ranking):
            if predicted_flight in ground_truth[:3]:  # 考虑前3个为相关
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_ndcg_5(self, predicted_ranking: List[str], ground_truth: List[str]) -> float:
        """计算NDCG@5"""
        if not predicted_ranking or not ground_truth:
            return 0.0
        
        try:
            # 构建相关性分数
            relevance_scores = []
            for flight in predicted_ranking[:5]:
                if flight in ground_truth[:1]:  # 最相关
                    relevance_scores.append(3)
                elif flight in ground_truth[:3]:  # 相关
                    relevance_scores.append(2)
                elif flight in ground_truth[:5]:  # 部分相关
                    relevance_scores.append(1)
                else:
                    relevance_scores.append(0)
            
            # 理想排序的相关性分数
            ideal_scores = [3, 2, 2, 1, 1]  # 假设理想情况
            
            # 使用sklearn计算NDCG
            ndcg = ndcg_score([ideal_scores], [relevance_scores], k=5)
            return float(ndcg)
            
        except Exception as e:
            logger.debug(f"NDCG计算失败: {e}")
            return 0.0
    
    def calculate_model_performance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算模型的整体性能指标"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'avg_mrr': 0.0,
                'avg_ndcg': 0.0,
                'success_rate': 0.0,
                'avg_processing_time': 30.0
            }
        
        return {
            'avg_mrr': float(np.mean([r['MRR'] for r in successful_results])),
            'avg_ndcg': float(np.mean([r['NDCG@5'] for r in successful_results])),
            'success_rate': float(len(successful_results) / len(results)),
            'avg_processing_time': float(np.mean([r['processing_time'] for r in successful_results]))
        }
    
    def run_robustness_analysis(self) -> Dict[str, Any]:
        """运行完整的鲁棒性分析实验"""
        print("🚀 Ground Truth鲁棒性敏感性分析实验 - 真实模型版本")
        print("=" * 70)
        print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载测试集
        test_queries = self.load_test_set()
        
        # 存储所有结果
        all_mode_results = {}
        
        # 对每种过滤模式运行实验
        for mode_name, mode_params in self.filter_modes.items():
            print(f"\n🔬 运行过滤模式: {mode_name}")
            print(f"   安全阈值: {mode_params['safety_threshold']}")
            print(f"   预算倍数: {mode_params['budget_multiplier']}x")
            print(f"   描述: {mode_params['description']}")
            
            # 重新生成Ground Truth（对于该模式）
            print(f"   📊 重新生成Ground Truth...")
            for query in test_queries:
                query['ground_truth_ranking'] = self.generate_modified_ground_truth(
                    query['flight_options'],
                    query['preferences'],
                    mode_name
                )
            
            # 运行所有四个真实模型
            print(f"   🔬 在150个查询上评估所有真实模型...")
            
            mode_model_results = {}
            
            for model_name, model in self.models.items():
                print(f"     🤖 评估 {model_name} 模型...")
                
                model_results = []
                for i, query in enumerate(test_queries):
                    if (i + 1) % 30 == 0:
                        print(f"       进度: {i+1}/150")
                    
                    result = self.evaluate_model_on_query(model, query)
                    model_results.append(result)
                
                # 计算性能指标
                performance = self.calculate_model_performance(model_results)
                mode_model_results[model_name] = {
                    'results': model_results,
                    'performance': performance
                }
                
                print(f"     ✅ {model_name}: MRR={performance['avg_mrr']:.3f}, "
                      f"NDCG={performance['avg_ndcg']:.3f}, "
                      f"成功率={performance['success_rate']:.1%}")
            
            # 计算MAMA相对优势
            mama_full_mrr = mode_model_results['MAMA_Full']['performance']['avg_mrr']
            single_agent_mrr = mode_model_results['SingleAgent']['performance']['avg_mrr']
            
            if single_agent_mrr > 0:
                relative_advantage = ((mama_full_mrr - single_agent_mrr) / single_agent_mrr) * 100
            else:
                relative_advantage = 0.0
            
            # 存储模式结果
            all_mode_results[mode_name] = {
                'mode_params': mode_params,
                'model_results': mode_model_results,
                'mama_full_mrr': mama_full_mrr,
                'single_agent_mrr': single_agent_mrr,
                'relative_advantage': relative_advantage
            }
            
            print(f"   📈 MAMA相对优势: {relative_advantage:.1f}%")
        
        # 生成最终结果表格
        markdown_table = self.generate_results_table(all_mode_results)
        
        # 计算鲁棒性指标
        advantages = [result['relative_advantage'] for result in all_mode_results.values()]
        robustness_score = 1.0 - (np.std(advantages) / np.mean(advantages)) if np.mean(advantages) > 0 else 0.0
        
        # 准备实验数据
        experiment_data = {
            'metadata': {
                'experiment_name': 'Ground Truth Robustness Analysis - Real Models',
                'timestamp': self.timestamp,
                'test_set_size': len(test_queries),
                'models_tested': list(self.models.keys()),
                'filter_modes': list(self.filter_modes.keys()),
                'random_seed': 42
            },
            'mode_results': all_mode_results,
            'robustness_metrics': {
                'advantages_range': [float(min(advantages)), float(max(advantages))],
                'advantages_std': float(np.std(advantages)),
                'advantages_mean': float(np.mean(advantages)),
                'robustness_score': float(robustness_score),
                'coefficient_of_variation': float(np.std(advantages) / np.mean(advantages)) if np.mean(advantages) > 0 else 0.0
            },
            'conclusions': {
                'robust_performance': robustness_score > 0.9,
                'consistent_advantage': all(adv > 0 for adv in advantages),
                'max_advantage_variation': float(max(advantages) - min(advantages))
            }
        }
        
        # 保存到文件
        output_file = self.results_dir / f'real_robustness_analysis_{self.timestamp}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: {output_file}")
        print(f"\n📋 鲁棒性分析结果表格:")
        print(markdown_table)
        print(f"\n🎯 鲁棒性评估:")
        print(f"   变异系数: {experiment_data['robustness_metrics']['coefficient_of_variation']:.3f}")
        print(f"   鲁棒性分数: {robustness_score:.3f}")
        print(f"   结论: {'高度鲁棒' if robustness_score > 0.9 else '中等鲁棒' if robustness_score > 0.7 else '需要改进'}")
        
        return experiment_data
    
    def generate_results_table(self, all_mode_results: Dict[str, Any]) -> str:
        """生成Markdown格式的结果表格"""
        table_lines = [
            "| Filter Mode | Safety Threshold | Budget Multiplier | MAMA (Full) MRR | Single Agent MRR | MAMA's Relative Advantage (%) |",
            "| --- | --- | --- | --- | --- | --- |"
        ]
        
        # 按指定顺序显示结果
        mode_order = ['Loose', 'Normal', 'Strict']
        
        for mode_name in mode_order:
            if mode_name not in all_mode_results:
                continue
                
            mode_data = all_mode_results[mode_name]
            mode_params = mode_data['mode_params']
            
            # 格式化行
            if mode_name == 'Normal':
                # 基准模式用粗体
                mode_display = f"**{mode_name} (Baseline)**"
                safety_display = f"**{mode_params['safety_threshold']}**"
                budget_display = f"**{mode_params['budget_multiplier']}x**"
                mama_mrr_display = f"**{mode_data['mama_full_mrr']:.3f}**"
                single_mrr_display = f"**{mode_data['single_agent_mrr']:.3f}**"
                advantage_display = f"**{mode_data['relative_advantage']:.1f}%**"
            else:
                mode_display = mode_name
                safety_display = str(mode_params['safety_threshold'])
                budget_display = f"{mode_params['budget_multiplier']}x"
                mama_mrr_display = f"{mode_data['mama_full_mrr']:.3f}"
                single_mrr_display = f"{mode_data['single_agent_mrr']:.3f}"
                advantage_display = f"{mode_data['relative_advantage']:.1f}%"
            
            table_line = f"| {mode_display} | {safety_display} | {budget_display} | {mama_mrr_display} | {single_mrr_display} | {advantage_display} |"
            table_lines.append(table_line)
        
        return "\n".join(table_lines)


# ============================================================================
# 主执行函数
# ============================================================================

def main():
    """主函数 - 运行真实模型的鲁棒性分析实验"""
    
    print("🎓 MAMA Framework Ground Truth Robustness Analysis")
    print("使用真实模型进行敏感性分析")
    print("=" * 70)
    
    try:
        # 创建实验实例
        experiment = RealMAMARobustnessExperiment()
        
        # 运行完整实验
        results = experiment.run_robustness_analysis()
        
        print("\n✅ 真实模型鲁棒性分析实验完成！")
        print(f"📊 实验数据已保存，可用于论文附录")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 实验执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 