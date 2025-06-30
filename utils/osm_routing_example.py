"""
Example: Fetch realistic road network and travel times for Mumbai using OpenStreetMap and OpenRouteService (free API)
"""
import openrouteservice
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st
# 1. Register at https://openrouteservice.org/sign-up/ and get your API key
ORS_API_KEY = os.getenv("ORS_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = openrouteservice.Client(key=ORS_API_KEY)

# 2. Load your orders.csv
orders = pd.read_csv('data/orders.csv')

# 3. Example: Use central warehouse as origin, or kirana/zone coordinates
central_warehouse = (19.205, 72.85)  # Example: Mumbai central point

orders['ors_distance_km'] = None
orders['ors_eta_min'] = None

for idx, row in orders.iterrows():
    dest = (row['latitude'], row['longitude'])
    try:
        route = client.directions(
            coordinates=[central_warehouse, dest],
            profile='driving-car',
            format='geojson',
            validate=False
        )
        summary = route['features'][0]['properties']['summary']
        orders.at[idx, 'ors_distance_km'] = summary['distance'] / 1000  # meters to km
        orders.at[idx, 'ors_eta_min'] = summary['duration'] / 60  # seconds to min
    except Exception as e:
        print(f"Error for order {row['order_id']}: {e}")

orders.to_csv('data/orders_with_ors.csv', index=False)
print('Updated orders_with_ors.csv with OpenRouteService data!')
