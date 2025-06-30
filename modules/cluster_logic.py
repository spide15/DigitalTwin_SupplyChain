"""
Customer segmentation using KMeans.
"""
import pandas as pd
from sklearn.cluster import KMeans

def cluster_customers(orders):
    """Cluster customers using KMeans."""
    features = orders[["order_frequency", "time_pref", "urgency"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    orders["cluster"] = clusters
    return orders[["customer_id", "cluster"]], None
