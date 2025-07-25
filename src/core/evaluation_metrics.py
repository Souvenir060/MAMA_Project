# DVC_exp/core/evaluation_metrics.py

import numpy as np
from typing import List, Dict, Any

def calculate_mrr(results: List[Dict[str, Any]]) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR).

    Args:
        results (list): A list of result dictionaries. Each dict must contain:
                        'recommendations' (list of recommended item IDs) and
                        'ground_truth_id' (the ID of the relevant item).

    Returns:
        float: The MRR score.
    """
    reciprocal_ranks = []
    for res in results:
        try:
            rank = res['recommendations'].index(res['ground_truth_id']) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # The ground truth item was not in the recommendations
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def calculate_ndcg_at_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at k.
    This function is simplified to use a binary relevance model based on ground_truth_id.

    Args:
        results (list): A list of result dictionaries. Each dict must contain:
                        'recommendations' (list of recommended item IDs) and
                        'ground_truth_id' (the ID of the relevant item).
        k (int): The cutoff for the ranking.

    Returns:
        float: The NDCG@k score.
    """
    ndcg_scores = []
    for res in results:
        # Create a binary relevance list for the top-k recommendations
        relevance_list = []
        for i, rec_id in enumerate(res['recommendations'][:k]):
            if rec_id == res['ground_truth_id']:
                relevance_list.append(1.0)  # Relevant item
            else:
                relevance_list.append(0.0)  # Non-relevant item
        
        # Calculate DCG for the recommendations
        dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_list)])
        
        # For binary relevance with one relevant item, IDCG is just 1.0 (the ideal ranking)
        idcg = 1.0
        
        if idcg == 0:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / idcg)
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def calculate_ndcg(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        results (list): A list of result dictionaries. Each dict must contain:
                        'recommendations' (list of recommended item IDs) and
                        'relevance_scores' (a dict mapping item IDs to their relevance score).
        k (int): The cutoff for the ranking.

    Returns:
        float: The NDCG@k score.
    """
    ndcg_scores = []
    for res in results:
        # Create a relevance list for the top-k recommendations
        relevance_list = [res['relevance_scores'].get(rec_id, 0) for rec_id in res['recommendations'][:k]]
        
        # Calculate DCG for the recommendations
        dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_list)])
        
        # Calculate Ideal DCG (IDCG)
        ideal_relevance = sorted(res['relevance_scores'].values(), reverse=True)
        idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k])])
        
        if idcg == 0:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / idcg)
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def calculate_art(results: List[Dict[str, Any]]) -> float:
    """
    Calculates the Average Response Time (ART).

    Args:
        results (list): A list of result dictionaries. Each dict must contain:
                        'response_time' (float, in seconds).

    Returns:
        float: The ART score.
    """
    response_times = [res['response_time'] for res in results if 'response_time' in res]
    return np.mean(response_times) if response_times else 0.0