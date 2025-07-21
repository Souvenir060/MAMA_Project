#!/usr/bin/env python3
"""
MAMA Framework - Utility Functions
Contains robust API call functions and retry mechanisms
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import aiohttp
import json

# Set up logging
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
    Robust API call function with retry mechanism and error handling
    
    Args:
        api_function: API function to call
        max_retries: Maximum number of retries
        base_delay: Base delay time (seconds)
        backoff_factor: Backoff factor (delay multiplier for each retry)
        timeout: Timeout (seconds)
        **kwargs: Parameters to pass to the API function
    
    Returns:
        API response data, or error information on failure
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Calculate current delay time
            if attempt > 0:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"API call retry {attempt}/{max_retries}, delaying {delay:.2f} seconds...")
                time.sleep(delay)
            
            # Call API function
            logger.debug(f"Calling API function: {api_function.__name__}")
            
            # Set timeout
            if 'timeout' not in kwargs:
                kwargs['timeout'] = timeout
            
            result = api_function(**kwargs)
            
            # Check if result is valid
            if result is not None and not (isinstance(result, dict) and result.get('error')):
                logger.info(f"API call successful: {api_function.__name__}")
                return result
            else:
                logger.warning(f"API call returned empty result or error: {api_function.__name__}")
                if attempt < max_retries:
                    continue
                else:
                    return {"error": "API call returned empty result", "success": False}
                    
        except (RequestException, ConnectionError, Timeout) as e:
            last_exception = e
            logger.warning(f"API call network error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        except Exception as e:
            last_exception = e
            logger.error(f"API call unknown error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        # If this is the last attempt, record the failure
        if attempt == max_retries:
            logger.error(f"API call ultimately failed: {api_function.__name__}, error: {last_exception}")
    
    # Return failure result
    return {
        "error": f"API call failed after {max_retries} retries",
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
    Asynchronous version of the robust API call function
    
    Args:
        api_function: Async API function to call
        max_retries: Maximum number of retries
        base_delay: Base delay time (seconds)
        backoff_factor: Backoff factor
        timeout: Timeout (seconds)
        **kwargs: Parameters to pass to the API function
    
    Returns:
        API response data, or error information on failure
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Calculate current delay time
            if attempt > 0:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"Async API call retry {attempt}/{max_retries}, delaying {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            
            # Call async API function
            logger.debug(f"Calling async API function: {api_function.__name__}")
            
            # Set timeout
            if 'timeout' not in kwargs:
                kwargs['timeout'] = timeout
            
            result = await api_function(**kwargs)
            
            # Check if result is valid
            if result is not None and not (isinstance(result, dict) and result.get('error')):
                logger.info(f"Async API call successful: {api_function.__name__}")
                return result
            else:
                logger.warning(f"Async API call returned empty result or error: {api_function.__name__}")
                if attempt < max_retries:
                    continue
                else:
                    return {"error": "Async API call returned empty result", "success": False}
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            logger.warning(f"Async API call network error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        except Exception as e:
            last_exception = e
            logger.error(f"Async API call unknown error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        # If this is the last attempt, record the failure
        if attempt == max_retries:
            logger.error(f"Async API call ultimately failed: {api_function.__name__}, error: {last_exception}")
    
    # Return failure result
    return {
        "error": f"Async API call failed after {max_retries} retries",
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
    Robust HTTP request function
    
    Args:
        url: Request URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Request headers
        params: URL parameters
        data: Form data
        json_data: JSON data
        max_retries: Maximum number of retries
        base_delay: Base delay time
        backoff_factor: Backoff factor
        timeout: Timeout
    
    Returns:
        Response data
    """
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Calculate current delay time
            if attempt > 0:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.info(f"HTTP request retry {attempt}/{max_retries}, delaying {delay:.2f} seconds...")
                time.sleep(delay)
            
            logger.debug(f"Sending HTTP request: {method} {url}")
            
            # Send HTTP request
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=timeout
            )
            
            # Check response status
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"HTTP request successful: {method} {url}")
                    return {"success": True, "data": result, "status_code": response.status_code}
                except json.JSONDecodeError:
                    return {"success": True, "data": response.text, "status_code": response.status_code}
            else:
                logger.warning(f"HTTP request returned error status code: {response.status_code}")
                if attempt < max_retries:
                    continue
                else:
                    return {
                        "error": f"HTTP request failed, status code: {response.status_code}",
                        "status_code": response.status_code,
                        "success": False
                    }
                    
        except (RequestException, ConnectionError, Timeout) as e:
            last_exception = e
            logger.warning(f"HTTP request network error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        except Exception as e:
            last_exception = e
            logger.error(f"HTTP request unknown error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            
        # If this is the last attempt, record the failure
        if attempt == max_retries:
            logger.error(f"HTTP request ultimately failed: {method} {url}, error: {last_exception}")
    
    # Return failure result
    return {
        "error": f"HTTP request failed after {max_retries} retries",
        "last_exception": str(last_exception),
        "success": False
    }


# API key configuration
API_KEYS = {
    "WEATHER_API_KEY": "498ae38fb9831291de1d0432ea2fdf07",
    "FLIGHT_API_KEY": "10fa3e346f7606412b86a9cbde4b00a4",
    "OPENAI_API_KEY": "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",
    "DEEPSEEK_API_KEY": "sk-488fe6d732734e3097631f0ba4eafd29"
}

def get_api_key(key_name: str) -> Optional[str]:
    """
    Get API key
    
    Args:
        key_name: Key name
    
    Returns:
        API key or None
    """
    import os
    
    # First try to get from environment variables
    env_key = os.getenv(key_name)
    if env_key:
        return env_key
    
    # Then get from built-in configuration
    return API_KEYS.get(key_name)


def validate_api_response(response: Dict[str, Any]) -> bool:
    """
    Validate if API response is valid
    
    Args:
        response: API response data
    
    Returns:
        Whether it's valid
    """
    
    if not response:
        return False
    
    if isinstance(response, dict):
        # Check if there's an error indicator
        if response.get('error') or response.get('success') is False:
            return False
        
        # Check if there's actual data
        if 'data' in response and response['data']:
            return True
        elif len(response) > 1:  # Has multiple fields, assume it contains valid data
            return True
    
    return False 