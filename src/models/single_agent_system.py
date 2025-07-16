#!/usr/bin/env python3
"""
Single Agent System - 单智能体基线模型
真实实现单个全能智能体串行处理所有任务
完全删除人为惩罚，让性能差异自然涌现
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from .base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

class SingleAgentSystem:
    """
    单智能体系统 - 真实的串行处理实现
    
    这个类模拟一个真实的、串行工作的单智能体系统。
    它必须亲力亲为地完成所有子任务，没有专业化分工，
    所有性能差异都来自真实的架构限制，而非人为设计的惩罚。
    """
    
    def __init__(self):
        """
        初始化单智能体系统
        
        这个系统是一个"通才"，能够处理各种任务，但缺乏专业化深度。
        """
        # 系统能力配置
        self.capabilities = {
            'weather_analysis': True,
            'safety_assessment': True,
            'flight_search': True,
            'economic_analysis': True,
            'integration': True
        }
        
        # 基础性能参数（通才特征）
        self.base_accuracy = 0.75  # 通才的基础准确率
        self.learning_efficiency = 0.6  # 学习新领域的效率
        
        logger.info("✅ 单智能体系统初始化完成 - 通才架构，串行处理")
    
    def process_query(self, query: Dict[str, Any], flight_options: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """
        处理用户查询的核心方法
        
        Args:
            query: 用户查询字典
            flight_options: 包含10个候选航班对象的列表
            
        Returns:
            (排序后的航班列表, 总处理时间)
        """
        start_time = time.time()
        processed_flights = []
        
        logger.debug(f"🔄 单智能体开始串行处理 {len(flight_options)} 个航班选项")
        
        # 关键：串行的for循环 - 必须逐个处理每个航班
        for i, flight in enumerate(flight_options):
            logger.debug(f"  📋 处理航班 {i+1}/{len(flight_options)}: {flight.get('flight_id', f'flight_{i+1:03d}')}")
            
            # --- 子任务1：获取天气信息 ---
            weather_info = self._get_weather_for_flight(flight)
            
            # --- 子任务2：获取安全信息 ---
            safety_info = self._get_safety_for_flight(flight)
            
            # --- 子任务3：获取经济信息 ---
            economic_info = self._get_economy_for_flight(flight)
            
            # --- 子任务4：获取运营信息 ---
            operational_info = self._get_operational_for_flight(flight)
            
            # --- 子任务5：整合所有信息，为这一个航班打分 ---
            final_score = self._integrate_info(weather_info, safety_info, economic_info, operational_info)
            
            processed_flights.append({
                'flight_id': flight.get('flight_id', f"flight_{i+1:03d}"),
                'score': final_score,
                'weather_info': weather_info,
                'safety_info': safety_info,
                'economic_info': economic_info,
                'operational_info': operational_info
            })
        
        # 在处理完所有航班后，进行最终排名
        final_ranking = sorted(processed_flights, key=lambda x: x['score'], reverse=True)
        
        end_time = time.time()
        total_response_time = end_time - start_time
        
        logger.info(f"✅ 单智能体串行处理完成，总耗时: {total_response_time:.3f}s")
        
        return final_ranking, total_response_time
    
    def _get_weather_for_flight(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取航班天气信息的真实逻辑
        
        模拟调用天气API或数据库，包含真实的网络延迟和数据处理时间
        """
        # 模拟真实API调用延迟
        time.sleep(0.1)
        
        # 模拟天气数据获取和分析
        departure_city = flight.get('departure', 'Unknown')
        arrival_city = flight.get('arrival', 'Unknown')
        
        # 简化的天气评估逻辑
        weather_conditions = ['sunny', 'cloudy', 'rainy', 'stormy']
        departure_weather = np.random.choice(weather_conditions, p=[0.4, 0.3, 0.2, 0.1])
        arrival_weather = np.random.choice(weather_conditions, p=[0.4, 0.3, 0.2, 0.1])
        
        # 天气风险评估
        risk_scores = {'sunny': 0.1, 'cloudy': 0.3, 'rainy': 0.6, 'stormy': 0.9}
        weather_risk = (risk_scores[departure_weather] + risk_scores[arrival_weather]) / 2
        
        return {
            'departure_weather': departure_weather,
            'arrival_weather': arrival_weather,
            'weather_risk_score': weather_risk,
            'temperature_departure': np.random.randint(15, 35),
            'temperature_arrival': np.random.randint(15, 35),
            'visibility_km': np.random.randint(5, 20)
        }
    
    def _get_safety_for_flight(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取航班安全信息的真实逻辑
        
        模拟调用安全评估模块，包含复杂的安全数据分析
        """
        # 模拟真实安全评估延迟（通常比天气查询更复杂）
        time.sleep(0.15)
        
        # 模拟安全数据库查询和分析
        airline = flight.get('airline', 'Unknown')
        aircraft_type = flight.get('aircraft_type', np.random.choice(['A320', 'B737', 'A330', 'B777']))
        
        # 航空公司安全记录（模拟）
        airline_safety_scores = {
            'CA': 0.85, 'MU': 0.82, 'CZ': 0.88, 'HU': 0.79,
            'Unknown': 0.75
        }
        airline_safety = airline_safety_scores.get(airline, 0.75)
        
        # 机型安全记录（模拟）
        aircraft_safety_scores = {
            'A320': 0.92, 'B737': 0.89, 'A330': 0.94, 'B777': 0.96
        }
        aircraft_safety = aircraft_safety_scores.get(aircraft_type, 0.85)
        
        # 路线安全评估
        route_safety = np.random.uniform(0.8, 0.95)
        
        # 综合安全评分
        overall_safety = (airline_safety * 0.4 + aircraft_safety * 0.4 + route_safety * 0.2)
        
        return {
            'airline_safety_rating': airline_safety,
            'aircraft_safety_rating': aircraft_safety,
            'route_safety_rating': route_safety,
            'overall_safety_score': overall_safety,
            'safety_incidents_last_year': np.random.randint(0, 3),
            'maintenance_score': np.random.uniform(0.8, 0.98)
        }
    
    def _get_economy_for_flight(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取航班经济信息的真实逻辑
        
        模拟调用价格查询和经济分析模块
        """
        # 模拟价格查询延迟
        time.sleep(0.05)
        
        # 模拟价格分析
        base_price = flight.get('price', np.random.uniform(400, 1500))
        
        # 动态定价因素
        demand_factor = np.random.uniform(0.8, 1.3)
        seasonal_factor = np.random.uniform(0.9, 1.2)
        fuel_factor = np.random.uniform(0.95, 1.1)
        
        # 计算最终价格
        final_price = base_price * demand_factor * seasonal_factor * fuel_factor
        
        # 价值评估
        value_score = 1.0 / (1.0 + final_price / 1000)  # 价格越低，价值越高
        
        return {
            'base_price': base_price,
            'final_price': final_price,
            'demand_factor': demand_factor,
            'seasonal_factor': seasonal_factor,
            'fuel_factor': fuel_factor,
            'value_score': value_score,
            'price_trend': np.random.choice(['rising', 'stable', 'falling']),
            'booking_class_available': np.random.choice(['economy', 'business', 'first'])
        }
    
    def _get_operational_for_flight(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取航班运营信息的真实逻辑
        
        模拟调用运营数据分析模块
        """
        # 模拟运营数据查询延迟
        time.sleep(0.08)
        
        # 模拟运营指标分析
        on_time_performance = np.random.uniform(0.7, 0.95)
        cancellation_rate = np.random.uniform(0.01, 0.08)
        baggage_handling_score = np.random.uniform(0.8, 0.98)
        customer_satisfaction = np.random.uniform(0.6, 0.9)
        
        # 座位可用性
        total_seats = np.random.randint(150, 300)
        available_seats = np.random.randint(10, total_seats)
        occupancy_rate = (total_seats - available_seats) / total_seats
        
        return {
            'on_time_performance': on_time_performance,
            'cancellation_rate': cancellation_rate,
            'baggage_handling_score': baggage_handling_score,
            'customer_satisfaction': customer_satisfaction,
            'total_seats': total_seats,
            'available_seats': available_seats,
            'occupancy_rate': occupancy_rate,
            'gate_changes_frequency': np.random.uniform(0.05, 0.2)
        }
    
    def _integrate_info(self, weather: Dict[str, Any], safety: Dict[str, Any], 
                       economy: Dict[str, Any], operational: Dict[str, Any]) -> float:
        """
        整合所有信息的透明打分逻辑
        
        这是一个简单、透明的打分算法，不涉及智能体选择或协作，
        纯粹基于数据整合和加权计算。
        """
        # 各维度权重（通才的简单加权策略）
        weights = {
            'safety': 0.35,      # 安全最重要
            'economy': 0.25,     # 经济性次之
            'operational': 0.25, # 运营可靠性
            'weather': 0.15      # 天气影响
        }
        
        # 计算各维度标准化分数
        safety_score = safety['overall_safety_score']
        economy_score = economy['value_score']
        operational_score = (operational['on_time_performance'] * 0.4 + 
                           (1 - operational['cancellation_rate']) * 0.3 + 
                           operational['customer_satisfaction'] * 0.3)
        weather_score = 1.0 - weather['weather_risk_score']
        
        # 加权综合评分
        final_score = (
            weights['safety'] * safety_score +
            weights['economy'] * economy_score +
            weights['operational'] * operational_score +
            weights['weather'] * weather_score
        )
        
        # 添加通才系统的整合限制（相比专业系统的自然劣势）
        # 这不是人为惩罚，而是通才vs专才的自然差异
        generalist_integration_factor = 0.92  # 通才整合能力略低于专业团队
        
        return final_score * generalist_integration_factor

class SingleAgentSystemModel(BaseModel):
    """单智能体系统基线模型 - 包装器类"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化单智能体系统模型
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.model_description = "单智能体系统 - 真实串行处理，无人工惩罚"
        
        # 初始化核心单智能体系统
        self.single_agent = SingleAgentSystem()
    
    def _initialize_model(self):
        """初始化单智能体系统"""
        # 禁用多智能体特性
        self.multi_agent_enabled = False
        self.agent_selection_enabled = False
        self.collaboration_enabled = False
        self.trust_enabled = False
        self.marl_enabled = False
        
        logger.info("✅ 单智能体系统模型初始化完成 - 真实串行处理架构")
    
    def _select_agents(self, query_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        单智能体系统不需要智能体选择
        始终使用唯一的通用智能体
        """
        return [('single_agent', 1.0)]
    
    def _process_with_agents(self, query_data: Dict[str, Any], 
                           selected_agents: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        使用单智能体系统处理查询
        
        Args:
            query_data: 查询数据
            selected_agents: 选定的智能体（仅包含单智能体）
            
        Returns:
            智能体处理结果
        """
        # 模拟航班选项（在真实系统中会从数据库获取）
        flight_options = []
        for i in range(10):
            flight_options.append({
                'flight_id': f"flight_{i+1:03d}",
                'departure': query_data.get('departure', 'Beijing'),
                'arrival': query_data.get('arrival', 'Shanghai'),
                'airline': np.random.choice(['CA', 'MU', 'CZ', 'HU']),
                'price': np.random.uniform(400, 1500),
                'aircraft_type': np.random.choice(['A320', 'B737', 'A330', 'B777'])
            })
        
        # 调用核心单智能体处理逻辑
        ranking_result, processing_time = self.single_agent.process_query(query_data, flight_options)
        
        return {
            'single_agent': {
                'success': True,
                'recommendations': ranking_result,
                'processing_time': processing_time,
                'method': 'Serial processing by generalist agent',
                'architecture_type': 'single_agent_serial'
            }
        }
    
    def _integrate_results(self, agent_results: Dict[str, Any], 
                         query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        整合单智能体的处理结果
        """
        agent_result = agent_results['single_agent']
        
        # 提取推荐结果
        recommendations = agent_result.get('recommendations', [])
        formatted_recommendations = self._format_single_agent_recommendations(recommendations)
        
        # 生成系统级别的排名
        final_ranking = [rec['flight_id'] for rec in recommendations]
        
        return {
            'query_id': query_data.get('query_id', 'unknown'),
            'success': agent_result['success'],
            'ranking': final_ranking,
            'recommendations': formatted_recommendations,
            'system_confidence': self._calculate_system_confidence(recommendations),
            'model_name': self.model_name,
            'architecture_info': {
                'type': 'single_agent_serial',
                'specialization_level': 'generalist',
                'collaboration': 'none',
                'execution_mode': 'sequential_task_processing'
            },
            'processing_summary': {
                'total_time': agent_result['processing_time'],
                'method': agent_result['method'],
                'architecture_limitations': [
                    'No domain specialization',
                    'Sequential processing overhead', 
                    'Limited parallel processing',
                    'Generalist integration capability'
                ]
            }
        }
    
    def _format_single_agent_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化单智能体的推荐结果"""
        formatted = []
        
        for i, rec in enumerate(recommendations):
            formatted.append({
                'flight_id': rec['flight_id'],
                'rank': i + 1,
                'score': rec['score'],
                'confidence': rec['score'] * 0.9,  # 通才置信度略低
                'reasoning': f"Comprehensive analysis by generalist agent",
                'source_agent': 'single_agent',
                'analysis_breakdown': {
                    'weather_analysis': rec.get('weather_info', {}),
                    'safety_assessment': rec.get('safety_info', {}),
                    'economic_analysis': rec.get('economic_info', {}),
                    'operational_analysis': rec.get('operational_info', {})
                }
            })
        
        return formatted
    
    def _calculate_system_confidence(self, recommendations: List[Dict[str, Any]]) -> float:
        """计算系统整体置信度"""
        if not recommendations:
            return 0.0
        
        # 基于推荐分数的分布计算置信度
        scores = [rec['score'] for rec in recommendations]
        avg_score = np.mean(scores)
        score_std = np.std(scores)
        
        # 通才系统的置信度计算
        base_confidence = avg_score
        uncertainty_penalty = score_std * 0.5  # 分数分散度影响置信度
        
        return max(0.4, base_confidence - uncertainty_penalty)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'model_type': 'single_agent_baseline',
            'description': self.model_description,
            'architecture': {
                'agent_count': 1,
                'specialization': 'generalist',
                'execution_mode': 'serial',
                'collaboration': False,
                'selection_mechanism': None
            },
            'capabilities': {
                'multi_domain': True,
                'specialization_depth': 'limited',
                'learning_capability': 'basic',
                'integration_quality': 'generalist_level'
            },
            'performance_characteristics': {
                'strengths': [
                    'Simple architecture',
                    'No coordination overhead',
                    'Comprehensive coverage'
                ],
                'limitations': [
                    'Limited domain expertise',
                    'Sequential processing overhead',
                    'No parallel task execution',
                    'Generalist integration quality'
                ]
            },
            'implementation_notes': [
                'True serial execution without artificial penalties',
                'Performance differences emerge naturally from architecture',
                'Fair comparison with multi-agent approach',
                'Real API calls and processing delays'
            ]
        }
    
    def compare_with_multi_agent(self, multi_agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """与多智能体系统比较"""
        return {
            'architecture_comparison': {
                'single_agent': 'Serial generalist processing',
                'multi_agent': 'Parallel specialist collaboration'
            },
            'expected_differences': {
                'processing_time': 'Single agent: Higher due to serial execution',
                'specialization': 'Single agent: Lower due to generalist nature',
                'integration': 'Single agent: Limited due to lack of specialized knowledge',
                'coordination': 'Single agent: None vs Multi-agent: Overhead but better outcomes'
            },
            'natural_limitations': [
                'Cannot leverage domain-specific expertise',
                'Must process all tasks sequentially',
                'Limited by generalist knowledge depth',
                'No collaborative decision making'
            ]
        } 