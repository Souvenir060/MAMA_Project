#!/usr/bin/env python3
"""
CSV Flight Data Processor
Handles reading and processing of flight data from CSV files
"""

import os
import csv
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FlightRecord:
    """Standardized flight record structure"""
    
    def __init__(self, 
                 flight_id: str,
                 airline: str,
                 flight_number: str,
                 origin: str,
                 destination: str,
                 departure_time: str,
                 arrival_time: str,
                 duration: int,
                 aircraft_type: str,
                 price: float,
                 seats_available: int,
                 cabin_class: str,
                 **kwargs):
        """Initialize flight record with required fields"""
        self.flight_id = flight_id
        self.airline = airline
        self.flight_number = flight_number
        self.origin = origin
        self.destination = destination
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.duration = duration  # Minutes
        self.aircraft_type = aircraft_type
        self.price = price
        self.seats_available = seats_available
        self.cabin_class = cabin_class
        
        # Store additional data
        self.additional_data = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'flight_id': self.flight_id,
            'airline': self.airline,
            'flight_number': self.flight_number,
            'origin': self.origin,
            'destination': self.destination,
            'departure_time': self.departure_time,
            'arrival_time': self.arrival_time,
            'duration': self.duration,
            'aircraft_type': self.aircraft_type,
            'price': self.price,
            'seats_available': self.seats_available,
            'cabin_class': self.cabin_class,
            **self.additional_data
        }
    
    def __str__(self) -> str:
        return (f"{self.airline} {self.flight_number}: {self.origin} → {self.destination} "
                f"({self.departure_time} - {self.arrival_time}), ${self.price:.2f}")

class CSVFlightDataProcessor:
    """Processor for flight data from CSV files"""
    
    def __init__(self, csv_file_path: str):
        """
        Initialize CSV flight data processor
        
        Args:
            csv_file_path: Path to the CSV file containing flight data
        """
        self.csv_file_path = csv_file_path
        self.records: List[FlightRecord] = []
        self.airports: Set[str] = set()
        self.airlines: Set[str] = set()
        
        # Indexed records for faster lookups
        self.by_origin: Dict[str, List[FlightRecord]] = {}
        self.by_destination: Dict[str, List[FlightRecord]] = {}
        self.by_route: Dict[str, List[FlightRecord]] = {}
        self.by_airline: Dict[str, List[FlightRecord]] = {}
        self.by_flight_id: Dict[str, FlightRecord] = {}
        
        logger.info(f"🛫 Initializing CSV flight data processor: {csv_file_path}")
        self.load_data()
    
    def load_data(self):
        """Load CSV data"""
        try:
            if not os.path.exists(self.csv_file_path):
                raise FileNotFoundError(f"CSV file does not exist: {self.csv_file_path}")
            
            logger.info(f"📖 Loading CSV data: {self.csv_file_path}")
            df = pd.read_csv(self.csv_file_path)
            
            # Data quality check
            total_records = len(df)
            df = self.validate_data(df)
            valid_records = len(df)
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            # Build indices
            self.build_indices(df)
            
            logger.info(f"✅ CSV data loading complete: {total_records} records, {valid_records} valid records")
            logger.info(f"📊 Data statistics: {len(self.airports)} airports, {len(self.airlines)} airlines")
            
        except Exception as e:
            logger.error(f"❌ CSV data loading failed: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality"""
        logger.info("🔍 Performing data quality validation...")
        
        # Remove records with missing essential fields
        required_fields = ['flight_id', 'airline', 'flight_number', 'origin', 
                          'destination', 'departure_time', 'arrival_time', 
                          'price', 'aircraft_type']
        
        initial_count = len(df)
        
        for field in required_fields:
            if field in df.columns:
                df = df[df[field].notna()]
        
        # Validate price > 0
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        
        # Validate origin != destination
        if 'origin' in df.columns and 'destination' in df.columns:
            df = df[df['origin'] != df['destination']]
        
        final_count = len(df)
        removed = initial_count - final_count
        
        logger.info(f"🧹 Removed {removed} invalid records ({removed/initial_count:.1%})")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for better performance"""
        # Ensure all flight IDs are strings
        df['flight_id'] = df['flight_id'].astype(str)
        
        # Convert airports to uppercase
        if 'origin' in df.columns:
            df['origin'] = df['origin'].str.upper()
        
        if 'destination' in df.columns:
            df['destination'] = df['destination'].str.upper()
        
        # Parse timestamps if needed
        # (assuming they're already in proper format for this example)
        
        # Convert data types
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        if 'seats_available' in df.columns:
            df['seats_available'] = pd.to_numeric(df['seats_available'], errors='coerce').fillna(0).astype(int)
        
        return df
    
    def build_indices(self, df: pd.DataFrame):
        """Build indices for faster lookups"""
        # Reset all indices
        self.records = []
        self.airports = set()
        self.airlines = set()
        self.by_origin = {}
        self.by_destination = {}
        self.by_route = {}
        self.by_airline = {}
        self.by_flight_id = {}
        
        for _, row in df.iterrows():
            try:
                # Create FlightRecord object
                flight_record = FlightRecord(
                    flight_id=str(row.get('flight_id', '')),
                    airline=str(row.get('airline', '')),
                    flight_number=str(row.get('flight_number', '')),
                    origin=str(row.get('origin', '')),
                    destination=str(row.get('destination', '')),
                    departure_time=str(row.get('departure_time', '')),
                    arrival_time=str(row.get('arrival_time', '')),
                    duration=int(row.get('duration', 0)),
                    aircraft_type=str(row.get('aircraft_type', '')),
                    price=float(row.get('price', 0.0)),
                    seats_available=int(row.get('seats_available', 0)),
                    cabin_class=str(row.get('cabin_class', 'ECONOMY')),
                    # Additional fields
                    stops=int(row.get('stops', 0)),
                    has_wifi=bool(row.get('has_wifi', False)),
                    has_entertainment=bool(row.get('has_entertainment', False)),
                    refundable=bool(row.get('refundable', False)),
                    baggage_allowance=int(row.get('baggage_allowance', 0))
                )
        
                # Store in main records list
                self.records.append(flight_record)
                
                # Update sets
                self.airports.add(flight_record.origin)
                self.airports.add(flight_record.destination)
                self.airlines.add(flight_record.airline)
                
                # Index by origin
                if flight_record.origin not in self.by_origin:
                    self.by_origin[flight_record.origin] = []
                self.by_origin[flight_record.origin].append(flight_record)
        
                # Index by destination
                if flight_record.destination not in self.by_destination:
                    self.by_destination[flight_record.destination] = []
                self.by_destination[flight_record.destination].append(flight_record)
                
                # Index by route (origin-destination pair)
                route_key = f"{flight_record.origin}-{flight_record.destination}"
                if route_key not in self.by_route:
                    self.by_route[route_key] = []
                self.by_route[route_key].append(flight_record)
                
                # Index by airline
                if flight_record.airline not in self.by_airline:
                    self.by_airline[flight_record.airline] = []
                self.by_airline[flight_record.airline].append(flight_record)
                
                # Index by flight ID
                self.by_flight_id[flight_record.flight_id] = flight_record
                
            except Exception as e:
                logger.warning(f"Failed to process flight record: {e}")
    
    def get_flights(self, origin: Optional[str] = None, 
                   destination: Optional[str] = None, 
                   airline: Optional[str] = None,
                   date: Optional[str] = None,
                   max_results: int = 10) -> List[FlightRecord]:
        """
        Query flights based on criteria
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            airline: Airline code
            date: Flight date
            max_results: Maximum number of results to return
            
        Returns:
            List of flight records matching the criteria
        """
        filtered_flights = self.records
        
        # Filter by origin
        if origin:
            origin = origin.upper()
            if origin in self.by_origin:
                filtered_flights = [f for f in filtered_flights if f.origin == origin]
            else:
                logger.warning(f"No flights found from origin: {origin}")
                return []
        
        # Filter by destination
        if destination:
        destination = destination.upper()
            if destination in self.by_destination:
                filtered_flights = [f for f in filtered_flights if f.destination == destination]
            else:
                logger.warning(f"No flights found to destination: {destination}")
                return []
        
        # Filter by route
        if origin and destination:
            route_key = f"{origin}-{destination}"
            if route_key in self.by_route:
                filtered_flights = self.by_route[route_key]
            else:
                logger.warning(f"No flights found for route: {route_key}")
                return []
        
        # Filter by airline
        if airline:
            if airline in self.by_airline:
                filtered_flights = [f for f in filtered_flights if f.airline == airline]
            else:
                logger.warning(f"No flights found for airline: {airline}")
                return []
        
        # Filter by date (approximation since our data might not have real dates)
        if date:
            # In a real implementation, we'd do proper date filtering
            # For now, we just simulate date filtering by taking a subset of flights
            seed = int(datetime.strptime(date, "%Y-%m-%d").timestamp())
            np.random.seed(seed)
            indices = np.random.choice(len(filtered_flights), 
                                      min(max_results * 3, len(filtered_flights)), 
                                      replace=False)
            filtered_flights = [filtered_flights[i] for i in indices]
        
        # Sort by price
        filtered_flights.sort(key=lambda x: x.price)
        
        # Return limited results
        return filtered_flights[:max_results]
    
    def get_flight_by_id(self, flight_id: str) -> Optional[FlightRecord]:
        """Get flight by ID"""
        return self.by_flight_id.get(flight_id)
    
    def get_all_routes(self) -> List[Tuple[str, str]]:
        """Get all available routes"""
        routes = []
        for route_key in self.by_route.keys():
            origin, destination = route_key.split('-')
            routes.append((origin, destination))
        return routes
    
    def get_total_flights(self) -> int:
        """Get total number of flights"""
        return len(self.records) 