#!/usr/bin/env python3
"""
Traditional Ranking System - 传统排名系统基线模型
使用传统的信息检索方法（BM25, TF-IDF, 规则基排名）
不使用现代AI技术，代表传统的非学习方法
"""

import numpy as np
import math
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter
from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class TraditionalRanker:
    """
    传统排名器 - 基于BM25和规则的航班排名系统
    
    这个类实现了一个传统的信息检索系统，使用BM25算法进行文本匹配，
    结合规则基的多因素加权排名。不使用任何现代AI技术。
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        初始化传统排名器
        
        Args:
            k1: BM25参数，控制词频饱和度 (默认1.2)
            b: BM25参数，控制文档长度归一化 (默认0.75)
        """
        # BM25参数
        self.bm25_k1 = k1
        self.bm25_b = b
        
        # 排名规则权重
        self.ranking_weights = {
            'price': 0.30,           # 价格因素权重
            'duration': 0.20,        # 飞行时长权重
            'departure_time': 0.15,  # 起飞时间权重
            'airline_rating': 0.20,  # 航空公司评级权重
            'availability': 0.15     # 座位可用性权重
        }
        
        # 初始化航班数据库和索引
        self.flight_database = self._initialize_flight_database()
        self.inverted_index = self._build_inverted_index()
        self.document_frequencies = self._calculate_document_frequencies()
        
        logger.info(f"✅ 传统排名器初始化完成 - BM25(k1={k1}, b={b})")
    
    def rank(self, user_query: str) -> List[Dict[str, Any]]:
        """
        对用户查询进行排名，返回排好序的航班列表
        
        Args:
            user_query: 用户查询字符串
            
        Returns:
            排序后的航班列表，每个元素包含flight_id和score
        """
        start_time = time.time()
        
        # 第1步：BM25文本相似度计算
        bm25_scores = self._calculate_bm25_scores(user_query)
        
        # 第2步：规则基多因素评分
        rule_scores = self._calculate_rule_based_scores()
        
        # 第3步：综合评分 (50% BM25 + 50% 规则基)
        final_scores = []
        for flight_id in self.flight_database.keys():
            bm25_score = bm25_scores.get(flight_id, 0.0)
            rule_score = rule_scores.get(flight_id, 0.0)
            
            # 综合评分
            final_score = 0.5 * bm25_score + 0.5 * rule_score
            
            final_scores.append({
                'flight_id': flight_id,
                'score': final_score,
                'bm25_component': bm25_score,
                'rule_component': rule_score
            })
        
        # 第4步：按分数排序
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        
        processing_time = time.time() - start_time
        logger.debug(f"传统排名完成，处理时间: {processing_time:.3f}s")
        
        return final_scores
    
    def _initialize_flight_database(self) -> Dict[str, Dict[str, Any]]:
        """
        初始化模拟的航班数据库
        
        Returns:
            航班数据库字典
        """
        flight_db = {}
        
        for i in range(1, 11):  # 10个航班
            flight_id = f"flight_{i:03d}"
            flight_db[flight_id] = {
                'flight_id': flight_id,
                'price': np.random.uniform(300, 1200),
                'duration': np.random.uniform(2.0, 8.0),  # 小时
                'departure_time': np.random.choice(['morning', 'afternoon', 'evening']),
                'airline_rating': np.random.uniform(0.6, 0.95),
                'availability': np.random.choice([True, False], p=[0.8, 0.2]),
                'description': f"Flight {flight_id} from Beijing to Shanghai via {np.random.choice(['direct', 'one-stop', 'two-stop'])} route"
            }
        
        return flight_db
    
    def _build_inverted_index(self) -> Dict[str, Set[str]]:
        """
        构建倒排索引
        
        Returns:
            倒排索引字典 {term: {flight_ids}}
        """
        index = {}
        
        for flight_id, flight_data in self.flight_database.items():
            # 提取文档词汇
            doc_text = flight_data['description'].lower()
            terms = doc_text.split()
            
            for term in terms:
                if term not in index:
                    index[term] = set()
                index[term].add(flight_id)
        
        return index
    
    def _calculate_document_frequencies(self) -> Dict[str, int]:
        """
        计算文档频率
        
        Returns:
            文档频率字典 {term: frequency}
        """
        doc_freq = {}
        total_docs = len(self.flight_database)
        
        for term, doc_set in self.inverted_index.items():
            doc_freq[term] = len(doc_set)
        
        return doc_freq
    
    def _calculate_bm25_scores(self, query: str) -> Dict[str, float]:
        """
        计算BM25相似度分数
        
        Args:
            query: 查询字符串
            
        Returns:
            BM25分数字典 {flight_id: score}
        """
        query_terms = query.lower().split()
        scores = {}
        
        # 计算平均文档长度
        total_length = sum(len(flight['description'].split()) 
                          for flight in self.flight_database.values())
        avg_doc_length = total_length / len(self.flight_database)
        
        for flight_id, flight_data in self.flight_database.items():
            doc_terms = flight_data['description'].lower().split()
            doc_length = len(doc_terms)
            term_freq = Counter(doc_terms)
            
            bm25_score = 0.0
            
            for term in query_terms:
                if term in term_freq:
                    # TF组件
                    tf = term_freq[term]
                    tf_component = (tf * (self.bm25_k1 + 1)) / (tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (doc_length / avg_doc_length)))
                    
                    # IDF组件
                    df = self.document_frequencies.get(term, 0)
                    if df > 0:
                        idf = math.log((len(self.flight_database) - df + 0.5) / (df + 0.5))
                        bm25_score += idf * tf_component
            
            scores[flight_id] = bm25_score
        
        # 归一化到[0,1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def _calculate_rule_based_scores(self) -> Dict[str, float]:
        """
        计算规则基评分
        
        Returns:
            规则基分数字典 {flight_id: score}
        """
        scores = {}
        
        # 获取各因素的数值范围用于归一化
        prices = [flight['price'] for flight in self.flight_database.values()]
        durations = [flight['duration'] for flight in self.flight_database.values()]
        ratings = [flight['airline_rating'] for flight in self.flight_database.values()]
        
        price_range = (min(prices), max(prices))
        duration_range = (min(durations), max(durations))
        
        for flight_id, flight_data in self.flight_database.items():
            # 价格评分 (越低越好，归一化到[0,1])
            price_score = 1.0 - (flight_data['price'] - price_range[0]) / (price_range[1] - price_range[0])
            
            # 时长评分 (越短越好，归一化到[0,1])
            duration_score = 1.0 - (flight_data['duration'] - duration_range[0]) / (duration_range[1] - duration_range[0])
            
            # 起飞时间评分 (偏好上午航班)
            time_scores = {'morning': 1.0, 'afternoon': 0.7, 'evening': 0.5}
            departure_score = time_scores.get(flight_data['departure_time'], 0.5)
            
            # 航空公司评级评分 (已经是[0,1]范围)
            airline_score = flight_data['airline_rating']
            
            # 可用性评分
            availability_score = 1.0 if flight_data['availability'] else 0.0
            
            # 加权综合评分
            final_score = (
                self.ranking_weights['price'] * price_score +
                self.ranking_weights['duration'] * duration_score +
                self.ranking_weights['departure_time'] * departure_score +
                self.ranking_weights['airline_rating'] * airline_score +
                self.ranking_weights['availability'] * availability_score
            )
            
            scores[flight_id] = final_score
        
        return scores

def generate_decision_tree_ground_truth(flight_options: List[Dict[str, Any]], 
                                      user_preferences: Dict[str, str]) -> List[str]:
    """
    基于硬规则决策树生成Ground Truth排名
    
    这个函数使用严格的决策树逻辑，完全不同于MAMA系统的加权模型，
    确保Ground Truth的生成与模型决策逻辑完全解耦。
    
    Args:
        flight_options: 包含10个候选航班对象的列表
        user_preferences: 用户偏好字典，例如 {'priority': 'safety', 'budget': 'medium'}
        
    Returns:
        排序后的航班ID列表，作为Ground Truth
    """
    
    # 第1步：硬性过滤 (Deal-breaker Filters)
    filtered_flights = []
    
    for flight in flight_options:
        # 安全分必须 > 0.4
        safety_score = flight.get('safety_score', np.random.uniform(0.3, 0.95))
        if safety_score <= 0.4:
            continue
        
        # 座位可用性必须为True
        if not flight.get('availability', True):
            continue
        
        # 预算约束
        price = flight.get('price', np.random.uniform(300, 1200))
        budget = user_preferences.get('budget', 'medium')
        
        if budget == 'low' and price >= 500:
            continue
        elif budget == 'medium' and price >= 1000:
            continue
        # high budget无价格限制
        
        # 通过筛选的航班
        filtered_flights.append({
            'flight_id': flight.get('flight_id', f"flight_{len(filtered_flights)+1:03d}"),
            'safety_score': safety_score,
            'price': price,
            'duration': flight.get('duration', np.random.uniform(2.0, 8.0)),
            'original_data': flight
        })
    
    # 如果过滤后航班太少，放宽条件
    if len(filtered_flights) < 3:
        logger.warning("硬性过滤后航班过少，放宽安全分要求")
        filtered_flights = []
        for flight in flight_options:
            safety_score = flight.get('safety_score', np.random.uniform(0.3, 0.95))
            if safety_score > 0.3 and flight.get('availability', True):
                filtered_flights.append({
                    'flight_id': flight.get('flight_id', f"flight_{len(filtered_flights)+1:03d}"),
                    'safety_score': safety_score,
                    'price': flight.get('price', np.random.uniform(300, 1200)),
                    'duration': flight.get('duration', np.random.uniform(2.0, 8.0)),
                    'original_data': flight
                })
    
    # 第2步：优先级排序 (Primary Sorting)
    priority = user_preferences.get('priority', 'safety')
    
    if priority == 'safety':
        # 按安全分从高到低排序
        filtered_flights.sort(key=lambda x: x['safety_score'], reverse=True)
    elif priority == 'cost':
        # 按价格从低到高排序
        filtered_flights.sort(key=lambda x: x['price'], reverse=False)
    elif priority == 'time':
        # 按总飞行时长从低到高排序
        filtered_flights.sort(key=lambda x: x['duration'], reverse=False)
    else:
        # 默认按安全分排序
        filtered_flights.sort(key=lambda x: x['safety_score'], reverse=True)
    
    # 第3.1步：处理平局 (Tie-Breaking)
    # 对于相同优先级指标的航班，用价格作为第二排序标准
    if priority == 'safety':
        # 按安全分分组，组内按价格排序
        filtered_flights.sort(key=lambda x: (-x['safety_score'], x['price']))
    elif priority == 'cost':
        # 按价格分组，组内按安全分排序
        filtered_flights.sort(key=lambda x: (x['price'], -x['safety_score']))
    elif priority == 'time':
        # 按时长分组，组内按价格排序
        filtered_flights.sort(key=lambda x: (x['duration'], x['price']))
    
    # 第3.2步：处理再次平局 (Final Tie-Breaking)
    # 最终用飞行时长作为决胜标准
    if priority == 'safety':
        filtered_flights.sort(key=lambda x: (-x['safety_score'], x['price'], x['duration']))
    elif priority == 'cost':
        filtered_flights.sort(key=lambda x: (x['price'], -x['safety_score'], x['duration']))
    elif priority == 'time':
        filtered_flights.sort(key=lambda x: (x['duration'], x['price']))
    
    # 第4步：生成最终排名
    ground_truth_ranking = [flight['flight_id'] for flight in filtered_flights]
    
    # 如果排名不足10个，用剩余航班填充
    all_flight_ids = [f.get('flight_id', f"flight_{i:03d}") for i, f in enumerate(flight_options)]
    for flight_id in all_flight_ids:
        if flight_id not in ground_truth_ranking:
            ground_truth_ranking.append(flight_id)
    
    logger.debug(f"决策树Ground Truth生成: 优先级={priority}, 筛选后={len(filtered_flights)}个航班")
    
    return ground_truth_ranking[:10]  # 返回前10个

class TraditionalRanking(BaseModel):
    """传统排名系统基线模型"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化传统排名系统
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.model_description = "传统排名系统 - BM25 + TF-IDF + 规则基排名"
        
        # 初始化传统排名器
        self.ranker = TraditionalRanker()
    
    def _initialize_model(self):
        """初始化传统排名系统"""
        # 禁用所有现代AI功能
        self.sbert_enabled = False
        self.trust_enabled = False
        self.historical_enabled = False
        self.marl_enabled = False
        
        logger.info("✅ 传统排名系统初始化完成 - 现代AI功能已禁用")
    
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """传统系统不使用智能体选择，返回空列表"""
        return []
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """使用传统排名器处理查询"""
        query_text = query_data.get('query_text', '')
        
        # 使用传统排名器
        ranking_results = self.ranker.rank(query_text)
        
        return {
            'traditional_ranker': {
                'success': True,
                'recommendations': ranking_results,
                'processing_time': 0.1,  # 传统方法通常较快
                'method': 'BM25 + Rule-based ranking'
            }
        }
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """整合传统排名结果"""
        ranker_result = agent_results['traditional_ranker']
        recommendations = ranker_result['recommendations']
        
        # 提取排名
        final_ranking = [rec['flight_id'] for rec in recommendations]
        
        return {
            'query_id': query_data.get('query_id', 'unknown'),
            'success': ranker_result['success'],
            'ranking': final_ranking,
            'recommendations': recommendations,
            'system_confidence': 0.8,  # 传统方法置信度固定
            'model_name': self.model_name,
            'processing_summary': {
                'total_time': ranker_result['processing_time'],
                'method': ranker_result['method'],
                'architecture_type': 'traditional_ir'
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'model_type': 'Traditional_Ranking_Baseline',
            'description': self.model_description,
            'methodology': {
                'text_retrieval': 'BM25 (Best Matching 25)',
                'ranking_algorithm': 'Rule-based scoring',
                'feature_engineering': 'Manual feature definition',
                'learning_method': 'none'
            },
            'features': {
                'modern_ai': False,
                'machine_learning': False,
                'deep_learning': False,
                'semantic_understanding': False,
                'trust_mechanism': False,
                'agent_collaboration': False
            },
            'algorithms': {
                'bm25': {
                    'k1': self.ranker.bm25_k1,
                    'b': self.ranker.bm25_b,
                    'purpose': 'Text similarity calculation'
                },
                'rule_based_ranking': {
                    'weights': self.ranker.ranking_weights,
                    'purpose': 'Multi-factor ranking'
                }
            },
            'limitations': {
                'no_learning': 'Cannot improve from user feedback',
                'static_rules': 'Fixed ranking rules',
                'no_semantic_understanding': 'Limited to keyword matching',
                'no_personalization': 'Same ranking for all users'
            },
            'academic_purpose': 'Baseline representing traditional IR methods',
            'expected_performance': 'Moderate but consistent performance'
        }
    
    def get_flight_by_id(self, flight_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取航班信息"""
        return self.ranker.flight_database.get(flight_id)
    
    def get_ranking_explanation(self, flight_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取排名解释"""
        flight_info = self.get_flight_by_id(flight_id)
        if not flight_info:
            return {}
        
        return {
            'flight_id': flight_id,
            'ranking_method': 'Traditional BM25 + Rule-based',
            'factors': {
                'text_similarity': 'BM25 score based on query terms',
                'price': f"Price: {flight_info['price']:.2f}",
                'duration': f"Duration: {flight_info['duration']:.2f} hours",
                'departure_time': f"Departure: {flight_info['departure_time']}",
                'airline_rating': f"Rating: {flight_info['airline_rating']:.2f}",
                'availability': f"Seats: {'Available' if flight_info['availability'] else 'Not Available'}"
            },
            'algorithm_details': {
                'bm25_parameters': {'k1': self.ranker.bm25_k1, 'b': self.ranker.bm25_b},
                'combination_method': 'Weighted sum of BM25 and rule-based scores',
                'personalization': 'none'
            }
        } 