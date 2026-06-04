import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_days=180, start_date='2025-10-01'):
    """
    Generates synthetic passenger demand data for different areas.
    """
    base_locations = {
        'Mansoura': 70,
        'Talkha': 50,
        'Meet Ghamr': 55,
        'Dekernes': 80,
        'Sherbin': 60
    }
    
    routes_demand = {}
    for loc1, d1 in base_locations.items():
        for loc2, d2 in base_locations.items():
            if loc1 != loc2:
                route_name = f"{loc1} - {loc2}"
                routes_demand[route_name] = (d1 + d2) // 2
                
    areas = list(routes_demand.keys())
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_dt + timedelta(days=i) for i in range(num_days)]
    
    data = []
    
    for date in dates:
        # Determine if it's a weekend (Friday=4, Saturday=5 in Python if Monday is 0)
        # In Egypt, weekend is usually Friday and Saturday
        is_weekend = 1 if date.weekday() in [4, 5] else 0
        
        for hour in range(24):
            for area in areas:
                # Base demand depends on route
                base_demand = routes_demand[area]
                
                # Demand multiplier based on hour
                hour_multiplier = 1.0
                if 7 <= hour <= 10:  # Morning rush
                    hour_multiplier = 2.5
                elif 14 <= hour <= 17:  # Afternoon rush
                    hour_multiplier = 2.0
                elif 0 <= hour <= 5:  # Late night
                    hour_multiplier = 0.2
                
                # Weekend effect
                weekend_multiplier = 0.7 if is_weekend else 1.0
                if is_weekend and (18 <= hour <= 23): # High demand on weekend evenings
                    weekend_multiplier = 1.5
                
                # Add some random noise
                noise = np.random.normal(0, 0.1 * base_demand)
                
                # Calculate final passengers
                passengers = int(base_demand * hour_multiplier * weekend_multiplier + noise)
                passengers = max(0, passengers) # Cannot be negative
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Time': f"{hour:02d}:00",
                    'Area': area,
                    'Passengers': passengers
                })
                
    df = pd.DataFrame(data)
    df.to_csv('transport_data.csv', index=False)
    print(f"Generated transport_data.csv with {len(df)} records.")

if __name__ == '__main__':
    generate_synthetic_data()
