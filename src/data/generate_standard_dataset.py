#!/usr/bin/env python3
"""
Standard Dataset Generator
"""

import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_standard_dataset():
    """Generate standardized flight query dataset"""
    print("Generating standardized flight query dataset...")
    
    """Initialize dataset generator"""
    # City data
    cities = [
        "New York", "London", "Tokyo", "Paris", "Sydney", "Los Angeles", "Berlin",
        "Toronto", "Singapore", "Dubai", "Beijing", "Madrid", "Rome", "Amsterdam",
        "Hong Kong", "Istanbul", "Bangkok", "Mumbai", "Seoul", "San Francisco"
        ]
        
    # Preference settings
    preference_settings = [
        {"safety_weight": 0.5, "price_weight": 0.3, "time_weight": 0.2, "weather_weight": 0.0},
        {"safety_weight": 0.7, "price_weight": 0.2, "time_weight": 0.1, "weather_weight": 0.0},
        {"safety_weight": 0.3, "price_weight": 0.6, "time_weight": 0.1, "weather_weight": 0.0},
        {"safety_weight": 0.2, "price_weight": 0.2, "time_weight": 0.2, "weather_weight": 0.4},
        {"safety_weight": 0.25, "price_weight": 0.25, "time_weight": 0.25, "weather_weight": 0.25},
        {"safety_weight": 0.8, "price_weight": 0.1, "time_weight": 0.1, "weather_weight": 0.0},
        {"safety_weight": 0.1, "price_weight": 0.8, "time_weight": 0.1, "weather_weight": 0.0},
        {"safety_weight": 0.1, "price_weight": 0.1, "time_weight": 0.8, "weather_weight": 0.0},
        {"safety_weight": 0.1, "price_weight": 0.1, "time_weight": 0.1, "weather_weight": 0.7}
        ]
        
    # Query templates (user query patterns)
    templates = [
        "I need a flight from {origin} to {destination} on {date}",
        "Looking for {origin} to {destination} flights on {date}",
        "Find me flights between {origin} and {destination} for {date}",
        "What are the best flight options from {origin} to {destination} on {date}?",
        "Compare flights from {origin} to {destination} on {date}",
        "I want to fly from {origin} to {destination} on {date}",
        "Search for flights: {origin} to {destination}, {date}",
        "Need to book {origin} to {destination} on {date}",
        "What flights are available from {origin} to {destination} on {date}?",
        "Help me find a flight from {origin} to {destination} on {date}"
    ]
    
    # Relevance labels based on flight selection criteria)
    relevance_standards = [
        {
            "name": "safety_first",
            "description": "Safety is the primary concern, followed by price",
            "weights": {"safety": 0.7, "price": 0.2, "convenience": 0.1},
            "threshold": {"safety": 0.8}
        },
        {
            "name": "balanced",
            "description": "Balanced consideration of all factors",
            "weights": {"safety": 0.33, "price": 0.33, "convenience": 0.33},
            "threshold": {}
        },
        {
            "name": "budget",
            "description": "Price is the primary concern",
            "weights": {"safety": 0.2, "price": 0.7, "convenience": 0.1},
            "threshold": {"price": 0.7}
        },
        {
            "name": "convenience",
            "description": "Flight timing and duration are primary concerns",
            "weights": {"safety": 0.2, "price": 0.2, "convenience": 0.6},
            "threshold": {"convenience": 0.7}
        }
    ]
    
    # Generate datasets
    train_queries = _generate_query_set(700, cities, templates, preference_settings, relevance_standards)
    validation_queries = _generate_query_set(150, cities, templates, preference_settings, relevance_standards)
    test_queries = _generate_query_set(150, cities, templates, preference_settings, relevance_standards)
        
    # Save datasets
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / 'train_queries.json', 'w') as f:
        json.dump(train_queries, f, indent=2)
    
    with open(data_dir / 'validation_queries.json', 'w') as f:
        json.dump(validation_queries, f, indent=2)
    
    with open(data_dir / 'test_queries.json', 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    # Combined standard dataset
    standard_dataset = {
        "train": train_queries,
        "validation": validation_queries,
        "test": test_queries,
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_size": len(train_queries),
            "validation_size": len(validation_queries),
            "test_size": len(test_queries),
            "cities": cities,
            "relevance_standards": relevance_standards
        }
    }
    
    with open(data_dir / 'standard_dataset.json', 'w') as f:
        json.dump(standard_dataset, f, indent=2)
    
    print(f"✅ Standard dataset generated successfully:")
    print(f"  - Train queries: {len(train_queries)}")
    print(f"  - Validation queries: {len(validation_queries)}")
    print(f"  - Test queries: {len(test_queries)}")
    print(f"  - Saved to: {data_dir}/standard_dataset.json")
        
    return standard_dataset

def _generate_candidate_flights(origin, destination, query_id):
    """Generate candidate flight options for a query"""
    # Generate 10 candidate flights
    flights = []
    airlines = ["AA", "UA", "DL", "BA", "LH", "AF", "KL", "QF", "SQ", "TK"]
    
    for i in range(10):
        flight_id = f"{query_id}_flight_{i+1:02d}"
        
        # Generate flight attributes
        safety_score = max(0.1, min(1.0, np.random.normal(0.8, 0.15)))
        price_score = max(0.1, min(1.0, np.random.normal(0.6, 0.2)))
        convenience_score = max(0.1, min(1.0, np.random.normal(0.7, 0.15)))
        weather_score = max(0.1, min(1.0, np.random.normal(0.75, 0.1)))
        
        flight = {
            "flight_id": flight_id,
            "airline": random.choice(airlines),
            "origin": origin,
            "destination": destination,
            "safety_score": round(safety_score, 3),
            "price_score": round(price_score, 3),
            "convenience_score": round(convenience_score, 3),
            "weather_score": round(weather_score, 3)
        }
        flights.append(flight)
    
    return flights

def _calculate_ground_truth_ranking(flights, relevance_standard):
    """Calculate ground truth ranking based on lexicographic preferences (论文方法)"""
    
    # 论文中的Lexicographic Preference Ordering model (非补偿性)
    # Primary -> Secondary -> Tertiary 层次排序
    if relevance_standard == "safety_first":
        # Primary: safety, Secondary: price, Tertiary: convenience
        sorted_flights = sorted(flights, key=lambda f: (
            -f["safety_score"],      # Higher safety first
            -f["price_score"],       # Then higher price score (lower actual price)
            -f["convenience_score"]  # Then higher convenience
        ))
    elif relevance_standard == "budget":
        # Primary: price, Secondary: safety, Tertiary: convenience
        sorted_flights = sorted(flights, key=lambda f: (
            -f["price_score"],       # Higher price score first (lower actual price)
            -f["safety_score"],      # Then higher safety
            -f["convenience_score"]  # Then higher convenience
        ))
    elif relevance_standard == "convenience":
        # Primary: convenience, Secondary: safety, Tertiary: price
        sorted_flights = sorted(flights, key=lambda f: (
            -f["convenience_score"], # Higher convenience first
            -f["safety_score"],      # Then higher safety
            -f["price_score"]        # Then higher price score
        ))
    else:  # balanced
        # Use weighted sum for balanced approach (论文中的实现)
        for flight in flights:
            flight["combined_score"] = (
                0.33 * flight["safety_score"] +
                0.33 * flight["price_score"] +
                0.33 * flight["convenience_score"]
            )
        sorted_flights = sorted(flights, key=lambda f: -f["combined_score"])
    
    return [flight["flight_id"] for flight in sorted_flights]

def _generate_query_set(num_queries, cities, templates, preference_settings, relevance_standards):
    """Generate a set of flight queries"""
    queries = []
    
    for i in range(num_queries):
        # Select random cities (ensuring origin != destination)
        origin, destination = random.sample(cities, 2)
        
        # Generate random date (within next 90 days)
        days_ahead = random.randint(7, 90)
        date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Select random template and fill in
        template = random.choice(templates)
        query_text = template.format(
            origin=origin,
            destination=destination,
            date=date
        )
        
        # Select random preference
        preference = random.choice(preference_settings)
        
        # Select random ground truth standard
        relevance_standard = random.choice(relevance_standards)
        
        # Generate query ID
        query_id = f"std_query_{i+1:03d}"
        
        # Generate candidate flights
        candidate_flights = _generate_candidate_flights(origin, destination, query_id)
        
        # Calculate ground truth ranking using lexicographic preferences
        ground_truth_ranking = _calculate_ground_truth_ranking(candidate_flights, relevance_standard["name"])
        
        # Generate relevance scores for each flight
        relevance_scores = {}
        for j, flight in enumerate(candidate_flights):
            # Calculate relevance based on position in ground truth ranking
            position = ground_truth_ranking.index(flight["flight_id"])
            # Higher relevance for better position (position 0 = relevance 1.0)
            relevance = max(0.1, 1.0 - (position * 0.1))
            relevance_scores[flight["flight_id"]] = round(relevance, 3)
        
        # Generate query object
        query = {
            "query_id": query_id,
            "query_text": query_text,
            "parameters": {
                "origin": origin,
                "destination": destination,
                "date": date
            },
            "user_preferences": preference,
            "relevance_standard": relevance_standard["name"],
            "candidate_flights": candidate_flights,
            "ground_truth_ranking": ground_truth_ranking,
            "relevance_scores": relevance_scores,
            "metadata": {
                "query_complexity": round(random.uniform(0.3, 0.9), 2),
                "route_popularity": round(random.uniform(0.1, 1.0), 2)
            }
        }
        
        queries.append(query)
    
    return queries

if __name__ == "__main__":
    generate_standard_dataset() 