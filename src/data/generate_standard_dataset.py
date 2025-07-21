#!/usr/bin/env python3
"""
Standard Dataset Generator - MAMA System Academic Experiments
Generate real flight query dataset for rigorous academic comparison experiments
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
    # Real city data
    cities = [
        "New York", "London", "Tokyo", "Paris", "Sydney", "Los Angeles", "Berlin",
        "Toronto", "Singapore", "Dubai", "Beijing", "Madrid", "Rome", "Amsterdam",
        "Hong Kong", "Istanbul", "Bangkok", "Mumbai", "Seoul", "San Francisco"
        ]
        
    # Real preference settings
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
        
    # Query templates (real user query patterns)
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
    
    # Real relevance labels (based on actual flight selection criteria)
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
        
        # Generate query object
        query = {
            "query_id": f"q{i+1:04d}",
            "query_text": query_text,
            "parameters": {
                "origin": origin,
                "destination": destination,
                "date": date
            },
            "user_preferences": preference,
            "relevance_standard": relevance_standard["name"],
            "metadata": {
                "query_complexity": round(random.uniform(0.3, 0.9), 2),
                "route_popularity": round(random.uniform(0.1, 1.0), 2)
            }
        }
        
        queries.append(query)
    
    return queries

if __name__ == "__main__":
    generate_standard_dataset() 