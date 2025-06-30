"""
Smart batching logic for orders based on proximity, weight, and SLA.
"""
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from utils.osm_routing import enrich_orders_with_ors
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("order_batching")

def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great circle distance between two points on the earth (km)
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def batch_orders(orders, riders, zones, max_weight=15, max_batch_size=4, use_osm=False):
    logger.info(f"Batching {len(orders)} orders with max_weight={max_weight}, max_batch_size={max_batch_size}, use_osm={use_osm}")
    """
    Simulate batching orders for cost vs. time optimization.
    - Proximity: Haversine distance
    - Weight constraint: max_weight per batch (kg)
    - ETA vs cost: fewer batches = lower cost, but larger batches may increase ETA
    Returns: list of dicts with order_ids, expected ETA, and cost for each batch
    """
    # Merge orders with zone coordinates
    orders = orders.merge(zones[['zone_name', 'latitude', 'longitude']], left_on='zone', right_on='zone_name', how='left')
    unbatched = orders.copy()
    batches = []
    batch_id = 0
    # Determine correct latitude/longitude columns
    lat_col = 'latitude_x' if 'latitude_x' in unbatched.columns else 'latitude'
    lon_col = 'longitude_x' if 'longitude_x' in unbatched.columns else 'longitude'
    if use_osm:
        # Always pass correct lat/lon columns to enrich_orders_with_ors
        orders = enrich_orders_with_ors(orders, lat_col=lat_col, lon_col=lon_col)
    while not unbatched.empty:
        # Start with the heaviest order
        seed = unbatched.sort_values('weight_kg', ascending=False).iloc[0]
        batch_orders = [seed['order_id']]
        batch_weight = seed['weight_kg']
        seed_lat, seed_lon = seed[lat_col], seed[lon_col]
        candidates = unbatched[unbatched['order_id'] != seed['order_id']]
        # Find closest orders by Haversine, add if weight allows
        for _, row in candidates.iterrows():
            if len(batch_orders) >= max_batch_size:
                break
            dist = haversine(seed_lat, seed_lon, row[lat_col], row[lon_col])
            if batch_weight + row['weight_kg'] <= max_weight and dist < 2.5:  # 2.5km proximity
                batch_orders.append(row['order_id'])
                batch_weight += row['weight_kg']
        # Estimate ETA: base + 2min per stop, +1min per extra km
        base_eta = 10  # min
        extra_km = 0
        if len(batch_orders) > 1:
            coords = unbatched[unbatched['order_id'].isin(batch_orders)][[lat_col, lon_col]].values
            for i in range(1, len(coords)):
                extra_km += haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        eta = base_eta + 2 * len(batch_orders) + int(extra_km)
        # Cost: base 10 + 2 per order (simulate cost tradeoff)
        cost = 10 + 2 * len(batch_orders)
        batches.append({
            'batch_id': batch_id,
            'order_ids': batch_orders,
            'total_weight': batch_weight,
            'expected_eta_min': eta,
            'expected_cost': cost
        })
        logger.info(f"Created batch {batch_id}: orders={batch_orders}, total_weight={batch_weight}")
        # Remove batched orders
        unbatched = unbatched[~unbatched['order_id'].isin(batch_orders)]
        batch_id += 1
    logger.info(f"Batching complete. Total batches: {len(batches)}")
    return pd.DataFrame(batches)
