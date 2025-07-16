#!/usr/bin/env python3
"""
Standardized Dataset Generator - MAMA System Academic Experiments
Generate real flight query datasets for rigorous academic comparison experiments
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class StandardDatasetGenerator:
    """Generate standardized flight query datasets"""
    
    def __init__(self):
        """Initialize dataset generator"""
        # Real city data
        self.cities = [
            'Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu',
            'Hangzhou', 'Nanjing', 'Wuhan', 'Xi\'an', 'Chongqing',
            'Tianjin', 'Shenyang', 'Dalian', 'Changsha', 'Zhengzhou',
            'Jinan', 'Harbin', 'Changchun', 'Taiyuan', 'Kunming',
            'Urumqi', 'Lhasa', 'Haikou', 'Sanya', 'Xiamen'
        ]
        
        # Real preference settings
        self.preferences = [
            {'budget': 'low', 'priority': 'cost', 'flexibility': 'high'},
            {'budget': 'medium', 'priority': 'time', 'flexibility': 'medium'},
            {'budget': 'high', 'priority': 'safety', 'flexibility': 'low'},
            {'budget': 'medium', 'priority': 'comfort', 'flexibility': 'medium'},
            {'budget': 'low', 'priority': 'flexibility', 'flexibility': 'high'},
            {'budget': 'high', 'priority': 'direct_flight', 'flexibility': 'low'},
            {'budget': 'medium', 'priority': 'airline_preference', 'flexibility': 'medium'},
            {'budget': 'low', 'priority': 'off_peak', 'flexibility': 'high'}
        ]
        
        # Query templates (real user query patterns)
        self.query_templates = [
            "Find flights from {departure} to {destination} on {date}",
            "Search for {budget} budget flights from {departure} to {destination} on {date}",
            "Looking for {priority} priority flights from {departure} to {destination} on {date}",
            "Need safe and reliable flights from {departure} to {destination} on {date}",
            "Find the best value flights from {departure} to {destination} on {date}",
            "Book flights from {departure} to {destination} on {date} with {flexibility} flexibility",
            "Urgent flight booking from {departure} to {destination} on {date}",
            "Family vacation flights from {departure} to {destination} on {date}",
            "Business trip flights from {departure} to {destination} on {date}",
            "Budget-friendly flights from {departure} to {destination} on {date}"
        ]
        
        # Airlines and realistic flight data
        self.airlines = [
            'Air China', 'China Eastern', 'China Southern', 'Hainan Airlines',
            'Shenzhen Airlines', 'Sichuan Airlines', 'Xiamen Airlines',
            'Spring Airlines', 'Juneyao Airlines', 'Lucky Air'
        ]
        
        # Aircraft types
        self.aircraft_types = [
            'Boeing 737', 'Boeing 777', 'Boeing 787', 'Airbus A320',
            'Airbus A330', 'Airbus A350', 'Embraer E190', 'Comac C919'
        ]
    
    def generate_date_range(self, start_days: int = 1, end_days: int = 90) -> List[str]:
        """Generate realistic date range for flight booking"""
        dates = []
        base_date = datetime.now()
        
        for i in range(start_days, end_days + 1):
            date = base_date + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
        
        return dates
    
    def generate_flight_candidates(self, departure: str, destination: str, 
                                 date: str, num_candidates: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic flight candidates for a route"""
        candidates = []
        
        for i in range(num_candidates):
            # Generate realistic flight times
            departure_hour = random.randint(6, 22)
            departure_minute = random.choice([0, 15, 30, 45])
            departure_time = f"{departure_hour:02d}:{departure_minute:02d}"
            
            # Flight duration based on route (simplified)
            base_duration = random.uniform(1.5, 8.0)  # 1.5 to 8 hours
            arrival_hour = (departure_hour + int(base_duration)) % 24
            arrival_minute = (departure_minute + int((base_duration % 1) * 60)) % 60
            arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"
            
            # Realistic pricing
            base_price = random.uniform(300, 2000)
            
            # Safety score (based on airline and aircraft)
            safety_score = random.uniform(0.7, 0.95)
            
            # On-time performance
            on_time_rate = random.uniform(0.65, 0.92)
            
            candidate = {
                'flight_id': f"FL{i+1:03d}_{departure[:2]}{destination[:2]}_{date.replace('-', '')}",
                'airline': random.choice(self.airlines),
                'aircraft_type': random.choice(self.aircraft_types),
                'departure_city': departure,
                'destination_city': destination,
                'departure_time': departure_time,
                'arrival_time': arrival_time,
                'duration_hours': round(base_duration, 2),
                'price_cny': round(base_price, 2),
                'safety_score': round(safety_score, 3),
                'on_time_rate': round(on_time_rate, 3),
                'available_seats': random.randint(5, 180),
                'booking_class': random.choice(['Economy', 'Business', 'First']),
                'meal_service': random.choice([True, False]),
                'wifi_available': random.choice([True, False]),
                'entertainment_system': random.choice([True, False]),
                'baggage_allowance_kg': random.choice([20, 23, 30]),
                'cancellation_policy': random.choice(['flexible', 'standard', 'strict']),
                'date': date
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def generate_ground_truth_ranking(self, candidates: List[Dict[str, Any]], 
                                    preferences: Dict[str, str]) -> List[str]:
        """Generate ground truth ranking based on preferences"""
        
        # Score each candidate based on preferences
        scored_candidates = []
        
        for candidate in candidates:
            score = 0.0
            
            # Priority-based scoring
            priority = preferences.get('priority', 'safety')
            
            if priority == 'cost':
                # Lower price is better
                max_price = max(c['price_cny'] for c in candidates)
                min_price = min(c['price_cny'] for c in candidates)
                if max_price > min_price:
                    score += 0.4 * (1 - (candidate['price_cny'] - min_price) / (max_price - min_price))
            
            elif priority == 'safety':
                score += 0.4 * candidate['safety_score']
            
            elif priority == 'time':
                # Shorter duration is better
                max_duration = max(c['duration_hours'] for c in candidates)
                min_duration = min(c['duration_hours'] for c in candidates)
                if max_duration > min_duration:
                    score += 0.4 * (1 - (candidate['duration_hours'] - min_duration) / (max_duration - min_duration))
            
            elif priority == 'comfort':
                comfort_score = 0.0
                if candidate['booking_class'] == 'First':
                    comfort_score += 0.5
                elif candidate['booking_class'] == 'Business':
                    comfort_score += 0.3
                
                if candidate['meal_service']:
                    comfort_score += 0.1
                if candidate['wifi_available']:
                    comfort_score += 0.1
                if candidate['entertainment_system']:
                    comfort_score += 0.1
                
                score += 0.4 * comfort_score
            
            # Budget constraint
            budget = preferences.get('budget', 'medium')
            if budget == 'low' and candidate['price_cny'] <= 800:
                score += 0.2
            elif budget == 'medium' and 800 < candidate['price_cny'] <= 1500:
                score += 0.2
            elif budget == 'high' and candidate['price_cny'] > 1500:
                score += 0.2
            
            # On-time performance
            score += 0.2 * candidate['on_time_rate']
            
            # Availability
            if candidate['available_seats'] > 20:
                score += 0.1
            
            # Flexibility
            flexibility = preferences.get('flexibility', 'medium')
            if flexibility == 'high' and candidate['cancellation_policy'] == 'flexible':
                score += 0.1
            
            scored_candidates.append((candidate['flight_id'], score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked flight IDs
        return [flight_id for flight_id, _ in scored_candidates]
    
    def generate_standard_dataset(self, num_queries: int = 200) -> Dict[str, Any]:
        """Generate standardized dataset with specified number of queries"""
        
        print(f"🔧 Generating standardized dataset with {num_queries} queries...")
        
        dataset = {
            'metadata': {
                'generator': 'StandardDatasetGenerator',
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'num_queries': num_queries,
                'random_seed': 42,
                'description': 'Standardized flight query dataset for MAMA system academic experiments'
            },
            'queries': []
        }
        
        # Generate date range
        available_dates = self.generate_date_range(1, 90)
        
        for i in range(num_queries):
            # Select random cities (ensure different departure and destination)
        departure = random.choice(self.cities)
            destination = random.choice([city for city in self.cities if city != departure])
        
            # Select random date
            date = random.choice(available_dates)
        
            # Select random preferences
            preferences = random.choice(self.preferences).copy()
        
            # Generate query text
        template = random.choice(self.query_templates)
        query_text = template.format(
            departure=departure,
            destination=destination,
                date=date,
                budget=preferences.get('budget', 'medium'),
                priority=preferences.get('priority', 'safety'),
                flexibility=preferences.get('flexibility', 'medium')
        )
        
            # Generate flight candidates
            candidates = self.generate_flight_candidates(departure, destination, date)
        
            # Generate ground truth ranking
            ground_truth = self.generate_ground_truth_ranking(candidates, preferences)
        
            # Create query entry
            query_entry = {
                'query_id': f"std_query_{i+1:03d}",
            'query_text': query_text,
                'departure_city': departure,
                'destination_city': destination,
                'travel_date': date,
            'preferences': preferences,
                'flight_candidates': candidates,
                'ground_truth_ranking': ground_truth,
            'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'template_used': template,
                    'num_candidates': len(candidates)
                }
            }
            
            dataset['queries'].append(query_entry)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{num_queries} queries...")
        
        print(f"✅ Dataset generation completed: {num_queries} queries")
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save dataset to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"standard_dataset_{timestamp}.json"
            
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Dataset saved to: {filepath}")
        return filepath
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from JSON file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"📂 Dataset loaded from: {filepath}")
        print(f"  Queries: {len(dataset['queries'])}")
        print(f"  Generated: {dataset['metadata']['timestamp']}")
        
        return dataset
    
    def validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset integrity and quality"""
        
        print("🔍 Validating dataset...")
        
        validation_report = {
            'total_queries': len(dataset['queries']),
            'validation_errors': [],
            'statistics': {
                'unique_routes': set(),
                'date_range': {'min': None, 'max': None},
                'preference_distribution': {},
                'city_distribution': {},
                'average_candidates_per_query': 0
            }
        }
        
        total_candidates = 0
        
        for i, query in enumerate(dataset['queries']):
            # Check required fields
            required_fields = ['query_id', 'query_text', 'departure_city', 'destination_city', 
                             'travel_date', 'preferences', 'flight_candidates', 'ground_truth_ranking']
            
            for field in required_fields:
                if field not in query:
                    validation_report['validation_errors'].append(f"Query {i+1}: Missing field '{field}'")
            
            # Collect statistics
            route = f"{query['departure_city']}-{query['destination_city']}"
            validation_report['statistics']['unique_routes'].add(route)
            
            # Date range
            date = query['travel_date']
            if validation_report['statistics']['date_range']['min'] is None or date < validation_report['statistics']['date_range']['min']:
                validation_report['statistics']['date_range']['min'] = date
            if validation_report['statistics']['date_range']['max'] is None or date > validation_report['statistics']['date_range']['max']:
                validation_report['statistics']['date_range']['max'] = date
            
            # Preference distribution
            priority = query['preferences'].get('priority', 'unknown')
            validation_report['statistics']['preference_distribution'][priority] = validation_report['statistics']['preference_distribution'].get(priority, 0) + 1
            
            # City distribution
            for city in [query['departure_city'], query['destination_city']]:
                validation_report['statistics']['city_distribution'][city] = validation_report['statistics']['city_distribution'].get(city, 0) + 1
            
            # Candidate count
            total_candidates += len(query['flight_candidates'])
            
            # Validate ground truth
            if len(query['ground_truth_ranking']) != len(query['flight_candidates']):
                validation_report['validation_errors'].append(f"Query {i+1}: Ground truth length mismatch")
        
        # Calculate averages
        validation_report['statistics']['unique_routes'] = len(validation_report['statistics']['unique_routes'])
        validation_report['statistics']['average_candidates_per_query'] = total_candidates / len(dataset['queries'])
        
        # Print validation results
        print(f"✅ Validation completed:")
        print(f"  Total queries: {validation_report['total_queries']}")
        print(f"  Validation errors: {len(validation_report['validation_errors'])}")
        print(f"  Unique routes: {validation_report['statistics']['unique_routes']}")
        print(f"  Date range: {validation_report['statistics']['date_range']['min']} to {validation_report['statistics']['date_range']['max']}")
        print(f"  Average candidates per query: {validation_report['statistics']['average_candidates_per_query']:.1f}")
        
        if validation_report['validation_errors']:
            print("⚠️  Validation errors found:")
            for error in validation_report['validation_errors'][:10]:  # Show first 10 errors
                print(f"    {error}")
        
        return validation_report

def main():
    """Main function to generate and save standard dataset"""
    
    # Create generator
    generator = StandardDatasetGenerator()
    
    # Generate dataset
    dataset = generator.generate_standard_dataset(num_queries=200)
    
    # Save dataset
    filepath = generator.save_dataset(dataset)
    
    # Validate dataset
    validation_report = generator.validate_dataset(dataset)
    
    print(f"\n🎉 Standard dataset generation completed!")
    print(f"📁 Dataset file: {filepath}")
    print(f"📊 Validation status: {'✅ PASSED' if len(validation_report['validation_errors']) == 0 else '⚠️ ISSUES FOUND'}")
    
    return dataset, filepath

if __name__ == "__main__":
    main() 