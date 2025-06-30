"""
OSM/ORS Routing Utilities for CityScale
- Fetches realistic road network distances and travel times using OpenRouteService (ORS) API
- Used for analytics in ETA prediction, batching, and simulation modules
"""
import openrouteservice
import pandas as pd
import os
import time
import logging

# Set your ORS API key here or via environment variable
ORS_API_KEY = os.environ.get('ORS_API_KEY', '5b3ce3597851110001cf6248bfd5d0fc674642d6a4960b9c6d864829')
client = openrouteservice.Client(key=ORS_API_KEY)

# Mumbai central warehouse (example)
CENTRAL_WAREHOUSE_COORDS = (19.205, 72.85)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("osm_routing")

def fetch_ors_route(origin, dest, profile='driving-motorcycle', retries=2, sleep=1):
    """Fetch route distance (km) and ETA (min) from ORS API using motorcycle profile."""
    logger.info(f"Requesting ORS route: origin={origin}, dest={dest}, profile={profile}")
    for attempt in range(retries):
        try:
            route = client.directions(
                coordinates=[origin, dest],
                profile=profile,
                format='geojson',
                validate=False
            )
            summary = route['features'][0]['properties']['summary']
            logger.info(f"ORS route success: distance={summary['distance']/1000:.2f} km, eta={summary['duration']/60:.2f} min")
            return summary['distance'] / 1000, summary['duration'] / 60  # km, min
        except Exception as e:
            logger.warning(f"ORS route failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(sleep)
    logger.error(f"ORS route failed after {retries} attempts: origin={origin}, dest={dest}")
    return None, None

def enrich_orders_with_ors(orders, origin_coords=CENTRAL_WAREHOUSE_COORDS, lat_col='latitude', lon_col='longitude'):
    """Add OSM/ORS route distance and ETA columns to orders DataFrame using motorcycle profile."""
    logger.info(f"Enriching {len(orders)} orders with OSM/ORS route data...")
    orders = orders.copy()
    ors_distances = []
    ors_etas = []
    for idx, row in orders.iterrows():
        dest = (row[lat_col], row[lon_col])
        dist, eta = fetch_ors_route(origin_coords, dest, profile='driving-motorcycle')
        logger.info(f"Order {row.get('order_id', idx)}: ORS distance={dist}, ORS eta={eta}")
        ors_distances.append(dist)
        ors_etas.append(eta)
    orders['ors_distance_km'] = ors_distances
    orders['ors_eta_min'] = ors_etas
    logger.info("OSM/ORS enrichment complete.")
    return orders
