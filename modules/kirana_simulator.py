"""
Kirana Micro-Warehouse Simulator module
- Simulate delivery from kirana stores vs. central warehouse
- Compare KPIs (ETA, cost, distance)
- Provide data for map/chart visualization
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from utils.osm_routing import enrich_orders_with_ors, CENTRAL_WAREHOUSE_COORDS
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("kirana_simulator")

def simulate_kirana_vs_central(orders, kiranas, zones, model='distance', co2_per_km=0.12, use_osm=False):
    logger.info(f"Simulating kirana vs central for {len(orders)} orders, model={model}, use_osm={use_osm}")
    """
    Assign each order to either the nearest kirana (within service radius) or central warehouse (if no kirana can serve).
    Calculate ETA, cost, distance, and CO2 for each assignment.
    Return DataFrame with: order_id, assigned_to, eta, cost, distance, co2_kg, model_type
    """
    # Enrich orders with OSM/ORS data if use_osm is True
    if use_osm:
        orders = enrich_orders_with_ors(orders, origin_coords=CENTRAL_WAREHOUSE_COORDS)
        
    # Central warehouse assumed at mean of all zone coordinates
    central_lat = zones['latitude'].mean() if 'latitude' in zones else orders['latitude'].mean()
    central_lon = zones['longitude'].mean() if 'longitude' in zones else orders['longitude'].mean()
    results = []
    traffic_map = {'Low': 1.0, 'Medium': 1.2, 'High': 1.4}
    for _, order in orders.iterrows():
        order_loc = (order['latitude'], order['longitude'])
        # Find eligible kiranas (within service radius)
        eligible = []
        for _, k in kiranas.iterrows():
            k_loc = (k['lat'], k['lon'])
            dist = geodesic(order_loc, k_loc).km
            if dist <= float(k['service_radius_km']):
                eligible.append((k['kirana_id'], dist, k_loc))
        if eligible:
            # Assign to nearest kirana
            assigned_kirana = min(eligible, key=lambda x: x[1])
            assigned_to = assigned_kirana[0]
            delivery_dist = assigned_kirana[1]
            model_type = 'kirana'
        else:
            # Assign to central warehouse
            assigned_to = 'central_warehouse'
            delivery_dist = geodesic(order_loc, (central_lat, central_lon)).km
            model_type = 'central'
        # Simulate ETA and cost (simple model: time = dist/speed * traffic, cost = base + per_km)
        speed = order['rider_speed'] if 'rider_speed' in order else 20
        try:
            raw_traffic = order['zone_traffic']
        except KeyError:
            raw_traffic = 1.0
        if isinstance(raw_traffic, str):
            traffic = traffic_map.get(raw_traffic.strip(), 1.0)
        elif pd.isnull(raw_traffic):
            traffic = 1.0
        else:
            traffic = float(raw_traffic)
        eta = round((delivery_dist / speed) * 60 * traffic, 1)
        cost = round(20 + 8 * delivery_dist, 2)
        co2_kg = round(delivery_dist * co2_per_km, 3)
        results.append({
            'order_id': order['order_id'],
            'assigned_to': assigned_to,
            'eta': eta,
            'cost': cost,
            'distance': round(delivery_dist, 2),
            'co2_kg': co2_kg,
            'model_type': model_type
        })
        logger.info(f"Order {order['order_id']}: assigned_to={assigned_to}, eta={eta}, cost={cost}, distance={delivery_dist}, co2_kg={co2_kg}, model_type={model_type}")
    logger.info("Simulation complete.")
    return pd.DataFrame(results)
