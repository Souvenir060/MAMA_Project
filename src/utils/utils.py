#!/usr/bin/env python3
"""
MAMA Framework - Utility Functions
包含鲁棒的API调用函数和重试机制
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import aiohttp
import json

# 设置日志
logger = logging.getLogger(__name__)

def make_robust_api_call(
    api_function: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0,
    **kwargs
) -> Dict[str, Any]:
    """
    鲁棒的API调用函数，包含重试机制和错误处理
    
    Args:
        api_function: 要调用的API函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        backoff_factor: 退避因子（每次重试延迟倍数）
        timeout: 超时时间（秒）
        **kwargs: 传递给API函数的参数
    
    Returns:
        API响应数据，失败时返回错误信息
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # 计算当前延迟时间
            if attempt > 0:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"API调用重试 {attempt}/{max_retries}，延迟 {delay:.2f}秒...")
                time.sleep(delay)
            
            # 调用API函数
            logger.debug(f"正在调用API函数: {api_function.__name__}")
            
            # 设置超时
            if 'timeout' not in kwargs:
                kwargs['timeout'] = timeout
            
            result = api_function(**kwargs)
            
            # 检查结果是否有效
            if result is not None and not (isinstance(result, dict) and result.get('error')):
                logger.info(f"API调用成功: {api_function.__name__}")
                return result
            else:
                logger.warning(f"API调用返回空结果或错误: {api_function.__name__}")
                if attempt < max_retries:
                    continue
                else:
                    return {"error": "API调用返回空结果", "success": False}
                    
        except (RequestException, ConnectionError, Timeout) as e:
            last_exception = e
            logger.warning(f"API调用网络错误 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        except Exception as e:
            last_exception = e
            logger.error(f"API调用未知错误 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        # 如果是最后一次尝试，记录失败
        if attempt == max_retries:
            logger.error(f"API调用最终失败: {api_function.__name__}, 错误: {last_exception}")
    
    # 返回失败结果
    return {
        "error": f"API调用失败，已重试{max_retries}次",
        "last_exception": str(last_exception),
        "success": False
    }


async def make_robust_async_api_call(
    api_function: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0,
    **kwargs
) -> Dict[str, Any]:
    """
    异步版本的鲁棒API调用函数
    
    Args:
        api_function: 要调用的异步API函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        backoff_factor: 退避因子
        timeout: 超时时间（秒）
        **kwargs: 传递给API函数的参数
    
    Returns:
        API响应数据，失败时返回错误信息
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # 计算当前延迟时间
            if attempt > 0:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"异步API调用重试 {attempt}/{max_retries}，延迟 {delay:.2f}秒...")
                await asyncio.sleep(delay)
            
            # 调用异步API函数
            logger.debug(f"正在调用异步API函数: {api_function.__name__}")
            
            # 设置超时
            if 'timeout' not in kwargs:
                kwargs['timeout'] = timeout
            
            result = await api_function(**kwargs)
            
            # 检查结果是否有效
            if result is not None and not (isinstance(result, dict) and result.get('error')):
                logger.info(f"异步API调用成功: {api_function.__name__}")
                return result
            else:
                logger.warning(f"异步API调用返回空结果或错误: {api_function.__name__}")
                if attempt < max_retries:
                    continue
                else:
                    return {"error": "异步API调用返回空结果", "success": False}
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            logger.warning(f"异步API调用网络错误 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        except Exception as e:
            last_exception = e
            logger.error(f"异步API调用未知错误 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        # 如果是最后一次尝试，记录失败
        if attempt == max_retries:
            logger.error(f"异步API调用最终失败: {api_function.__name__}, 错误: {last_exception}")
    
    # 返回失败结果
    return {
        "error": f"异步API调用失败，已重试{max_retries}次",
        "last_exception": str(last_exception),
        "success": False
    }


def make_robust_http_request(
    url: str,
    method: str = 'GET',
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    鲁棒的HTTP请求函数
    
    Args:
        url: 请求URL
        method: HTTP方法 (GET, POST, PUT, DELETE)
        headers: 请求头
        params: URL参数
        data: 表单数据
        json_data: JSON数据
        max_retries: 最大重试次数
        base_delay: 基础延迟时间
        backoff_factor: 退避因子
        timeout: 超时时间
    
    Returns:
        响应数据
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # 计算当前延迟时间
            if attempt > 0:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"HTTP请求重试 {attempt}/{max_retries}，延迟 {delay:.2f}秒...")
                time.sleep(delay)
            
            logger.debug(f"正在发送HTTP请求: {method} {url}")
            
            # 发送HTTP请求
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=timeout
            )
            
            # 检查响应状态
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"HTTP请求成功: {method} {url}")
                    return {"success": True, "data": result, "status_code": response.status_code}
                except json.JSONDecodeError:
                    return {"success": True, "data": response.text, "status_code": response.status_code}
            else:
                logger.warning(f"HTTP请求返回错误状态码: {response.status_code}")
                if attempt < max_retries:
                    continue
                else:
                    return {
                        "error": f"HTTP请求失败，状态码: {response.status_code}",
                        "status_code": response.status_code,
                        "success": False
                    }
                    
        except (RequestException, ConnectionError, Timeout) as e:
            last_exception = e
            logger.warning(f"HTTP请求网络错误 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        except Exception as e:
            last_exception = e
            logger.error(f"HTTP请求未知错误 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        # 如果是最后一次尝试，记录失败
        if attempt == max_retries:
            logger.error(f"HTTP请求最终失败: {method} {url}, 错误: {last_exception}")
    
    # 返回失败结果
    return {
        "error": f"HTTP请求失败，已重试{max_retries}次",
        "last_exception": str(last_exception),
        "success": False
    }


# API密钥配置
API_KEYS = {
    "WEATHER_API_KEY": "498ae38fb9831291de1d0432ea2fdf07",
    "FLIGHT_API_KEY": "10fa3e346f7606412b86a9cbde4b00a4",
    "OPENAI_API_KEY": "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",
    "DEEPSEEK_API_KEY": "sk-488fe6d732734e3097631f0ba4eafd29"
}

def get_api_key(key_name: str) -> Optional[str]:
    """
    获取API密钥
    
    Args:
        key_name: 密钥名称
    
    Returns:
        API密钥或None
    """
    import os
    
    # 首先尝试从环境变量获取
    env_key = os.getenv(key_name)
    if env_key:
        return env_key
    
    # 然后从内置配置获取
    return API_KEYS.get(key_name)


def validate_api_response(response: Dict[str, Any]) -> bool:
    """
    验证API响应是否有效
    
    Args:
        response: API响应数据
    
    Returns:
        是否有效
    """
    
    if not response:
        return False
    
    if isinstance(response, dict):
        # 检查是否有错误标识
        if response.get('error') or response.get('success') is False:
            return False
        
        # 检查是否有实际数据
        if 'data' in response and response['data']:
            return True
        elif len(response) > 1:  # 有多个字段，假设包含有效数据
            return True
    
    return False 