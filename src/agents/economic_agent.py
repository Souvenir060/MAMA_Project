# MAMA_exp/agents/economic_agent.py

"""
Economic Agent: Calculates the total cost of a flight, including fare and additional costs.
"""
import logging
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent
from config import LLM_CONFIG

def get_city_accommodation_cost(city_code: str, city_name: str = "") -> float:
    """
    Calculate realistic accommodation costs based on city economic level and market data
    
    Args:
        city_code: IATA airport code
        city_name: City name for backup lookup
        
    Returns:
        Estimated overnight accommodation cost in USD
    """
    # Normalize inputs
    city_code = city_code.upper().strip() if city_code else ""
    city_name = city_name.lower().strip() if city_name else ""
    
    # Tier 1: Premium global cities (high cost)
    premium_cities = {
        "JFK": 280, "LGA": 280, "EWR": 280,  # New York
        "LHR": 250, "LGW": 220, "STN": 180,  # London
        "NRT": 220, "HND": 240,              # Tokyo
        "CDG": 230, "ORY": 200,              # Paris
        "ZUR": 260, "GVA": 240,              # Switzerland
        "SIN": 180,                          # Singapore
        "HKG": 200,                          # Hong Kong
        "SFO": 280, "LAX": 240,              # California
        "DXB": 200,                          # Dubai
    }
    
    # Tier 2: Major business cities (medium-high cost)
    major_cities = {
        "PEK": 120, "PKX": 120,              # Beijing
        "PVG": 140, "SHA": 130,              # Shanghai
        "CAN": 100,                          # Guangzhou
        "SZX": 110,                          # Shenzhen
        "CTU": 80,                           # Chengdu
        "XIY": 70,                           # Xi'an
        "ICN": 130,                          # Seoul
        "BOM": 90, "DEL": 85,                # India
        "FRA": 180, "MUC": 170,              # Germany
        "AMS": 160, "BRU": 140,              # Netherlands/Belgium
        "ARN": 180, "CPH": 190,              # Scandinavia
        "YYZ": 160, "YVR": 150,              # Canada
        "SYD": 180, "MEL": 170,              # Australia
    }
    
    # Tier 3: Regional cities (medium cost)
    regional_cities = {
        "KMG": 60,   # Kunming
        "URC": 50,   # Urumqi
        "XMN": 70,   # Xiamen
        "CKG": 65,   # Chongqing
        "WUH": 65,   # Wuhan
        "NKG": 70,   # Nanjing
        "HGH": 75,   # Hangzhou
        "TSN": 65,   # Tianjin
        "DLC": 60,   # Dalian
        "SHE": 55,   # Shenyang
    }
    
    # Check by airport code first
    if city_code in premium_cities:
        return premium_cities[city_code]
    elif city_code in major_cities:
        return major_cities[city_code]
    elif city_code in regional_cities:
        return regional_cities[city_code]
    
    # Check by city name if code lookup fails
    city_name_mappings = {
        "new york": 280, "london": 250, "tokyo": 230, "paris": 230,
        "singapore": 180, "hong kong": 200, "dubai": 200,
        "beijing": 120, "shanghai": 140, "guangzhou": 100, "shenzhen": 110,
        "seoul": 130, "mumbai": 90, "delhi": 85,
        "frankfurt": 180, "munich": 170, "amsterdam": 160,
        "zurich": 260, "geneva": 240,
        "toronto": 160, "vancouver": 150,
        "sydney": 180, "melbourne": 170
    }
    
    for city, cost in city_name_mappings.items():
        if city in city_name:
            return cost
    
    # Default for unknown cities (conservative estimate)
    return 100

class EconomicAgent(BaseAgent):
    """
    Agent responsible for calculating flight costs and performing economic analysis.
    
    Uses academic economic models for price optimization and cost prediction.
    """
    
    def __init__(self, 
                 name: str = None,
                 role: str = "economic_analysis", 
                 **kwargs):
        """Initialize economic agent with academic capabilities"""
        super().__init__(name=name, role=role, **kwargs)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("✅ Economic Agent initialized with academic capabilities")

    def process_task(self, flight_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the total cost for a single flight.

        Args:
            flight_details (Dict[str, Any]): A dictionary representing a single flight,
                                             containing price and arrival information.

        Returns:
            Dict[str, Any]: A dictionary with the detailed cost breakdown.
        """
        try:
            base_price = flight_details['price']['value']
            arrival_time_str = flight_details['arrival']['value']['scheduledTime']
            destination_iata = flight_details['arrival']['value']['iataCode']
            
            arrival_time = datetime.fromisoformat(arrival_time_str.replace('Z', '+00:00'))
            arrival_hour = arrival_time.hour
            
            hidden_cost = 0
            cost_reason = "none"

            # Check for inconvenient arrival times (11 PM to 5 AM)
            if arrival_hour >= 23 or arrival_hour < 5:
                hidden_cost = get_city_accommodation_cost(destination_iata, destination_iata)
                cost_reason = f"Estimated hotel cost for late arrival at {destination_iata}."
                logging.info(f"Applying hidden cost of {hidden_cost} for late arrival.")

            total_cost = base_price + hidden_cost
            
            return {
                "base_price": float(base_price),
                "hidden_cost": float(hidden_cost),
                "cost_reason": cost_reason,
                "total_cost": float(total_cost)
            }
        except KeyError as e:
            logging.error(f"EconomicAgent missing expected key in flight_details: {e}")
            return {"error": f"Incomplete flight data provided. Missing key: {e}"}

def create_economic_agent():
    """
    Creates and returns an EconomicAgent instance.
    
    Returns:
        EconomicAgent: Configured economic agent instance
    """
    return EconomicAgent(role="economic_analysis")

def calculate_total_cost_tool(flight_data_with_safety: str) -> str:
    """
    Tool function to calculate total flight costs
    
    Args:
        flight_data_with_safety: JSON string of flight data with safety assessment
        
    Returns:
        JSON string containing cost analysis results
    """
    import json
    import logging
    
    logging.info("EconomicAgent: Starting total cost calculation")
    
    try:
        # Parse input data
        if not flight_data_with_safety.strip():
            return json.dumps({
                "status": "error",
                "message": "No flight data provided"
            })
        
        flight_data = json.loads(flight_data_with_safety)
        
        if "flights" not in flight_data or not flight_data["flights"]:
            return json.dumps({
                "status": "error",
                "message": "No flight information found in flight data"
            })
        
        flights = flight_data["flights"]
        processed_flights = []
        
        # Calculate cost for each flight
        for flight in flights:
            try:
                # Calculate base price
                base_price = flight.get('price', {}).get('value', 0)
                if isinstance(base_price, str):
                    # Handle price strings, remove currency symbols
                    base_price = float(''.join(filter(str.isdigit, base_price)) or 0)
                
                # Calculate hidden costs
                hidden_cost = 0
                cost_reason = "No additional fees"
                
                # Check if arrival time is inconvenient
                arrival_time = flight.get('arrival_time', '')
                destination = flight.get('destination', '')
                
                if arrival_time:
                    try:
                        # Extract hour
                        if ':' in arrival_time:
                            hour = int(arrival_time.split(':')[0])
                            # Late night arrival requires hotel cost
                            if hour >= 23 or hour < 5:
                                hidden_cost = get_city_accommodation_cost(destination, destination)
                                cost_reason = f"Hotel accommodation needed for late arrival in {destination}"
                    except:
                        pass
                
                # Check for multiple transfers
                stops = flight.get('stops', 0)
                if stops > 1:
                    # Multiple transfers add meal and waiting costs
                    transfer_cost = stops * 30  # 30 CNY per transfer
                    hidden_cost += transfer_cost
                    if cost_reason == "No additional fees":
                        cost_reason = f"Meal costs for {stops} transfers"
                    else:
                        cost_reason += f", meal costs for {stops} transfers"
                
                # Check flight duration
                duration = flight.get('duration_minutes', 0)
                if duration > 8 * 60:  # Over 8 hours
                    long_flight_cost = 50  # Long flight additional fee
                    hidden_cost += long_flight_cost
                    if cost_reason == "No additional fees":
                        cost_reason = "Long-haul flight additional service fees"
                    else:
                        cost_reason += ", long-haul flight additional service fees"
                
                total_cost = base_price + hidden_cost
                
                # Build cost analysis
                cost_analysis = {
                    "base_price": round(base_price, 2),
                    "hidden_costs": round(hidden_cost, 2),
                    "cost_breakdown": {
                        "ticket_price": round(base_price, 2),
                        "additional_costs": round(hidden_cost, 2),
                        "cost_reason": cost_reason
                    },
                    "total_cost": round(total_cost, 2),
                    "currency": "CNY"
                }
                
                # Add cost analysis to flight data
                flight_with_cost = {
                    **flight,
                    "cost_analysis": cost_analysis
                }
                
                processed_flights.append(flight_with_cost)
                
            except Exception as e:
                logging.error(f"Error processing flight cost: {e}")
                # If individual flight processing fails, use default values
                flight_with_cost = {
                    **flight,
                    "cost_analysis": {
                        "base_price": 0,
                        "hidden_costs": 0,
                        "total_cost": 0,
                        "error": f"Cost calculation failed: {str(e)}"
                    }
                }
                processed_flights.append(flight_with_cost)
        
        return json.dumps({
            "status": "success",
            "flights": processed_flights,
            "cost_summary": {
                "total_flights_analyzed": len(processed_flights),
                "average_cost": round(sum(f.get('cost_analysis', {}).get('total_cost', 0) 
                                       for f in processed_flights) / len(processed_flights), 2) if processed_flights else 0
            }
        })
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Flight data format error: {str(e)}"
        })
    except Exception as e:
        logging.error(f"Cost calculation tool error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error occurred during cost calculation: {str(e)}"
        })
