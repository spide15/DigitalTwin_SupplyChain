"""
Inventory/Restocking Analysis module (optional)
- Track SKU depletion per kirana
- Raise alerts and provide heatmap data
"""

import pandas as pd
import numpy as np

def analyze_inventory(kirana_inventory, threshold=10):
    """
    Analyze kirana inventory, flag low stock, and prepare for heatmap.
    kirana_inventory: DataFrame with columns kirana_id, name, lat, lon, sku, stock
    Returns: DataFrame with kirana_id, name, lat, lon, sku, stock, alert (bool)
    """
    df = kirana_inventory.copy()
    df['alert'] = df['stock'] < threshold
    return df

def simulate_kirana_inventory(kiranas, skus=['atta','rice','oil','milk','snacks']):
    """
    Simulate inventory for each kirana and SKU.
    Returns DataFrame: kirana_id, name, lat, lon, sku, stock
    """
    rows = []
    for _, k in kiranas.iterrows():
        for sku in skus:
            stock = np.random.randint(0, 50)
            rows.append({
                'kirana_id': k['kirana_id'],
                'name': k['name'],
                'lat': k['lat'],
                'lon': k['lon'],
                'sku': sku,
                'stock': stock
            })
    return pd.DataFrame(rows)
