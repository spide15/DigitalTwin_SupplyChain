"""
ML model for ETA prediction using Random Forest/XGBoost.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
from utils.osm_routing import enrich_orders_with_ors
import logging

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'eta_model.pkl')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("eta_predictor")


def train_model(orders, model_type='random_forest', model_path=MODEL_PATH, use_osm=False):
    logger.info(f"Training ETA model: {model_type}, orders={len(orders)}, use_osm={use_osm}")
    """
    Train ETA prediction model and save to disk.
    model_type: 'random_forest' or 'xgboost'
    model_path: path to save the trained model
    Returns: trained model, test R^2 score
    """
    if use_osm:
        orders = enrich_orders_with_ors(orders)
    # Map zone_traffic to numeric if needed
    traffic_map = {'Low': 1.0, 'Medium': 1.2, 'High': 1.4}
    orders = orders.copy()
    if orders['zone_traffic'].dtype == object:
        orders['zone_traffic'] = orders['zone_traffic'].map(traffic_map)
    X = orders[["distance_km", "zone_traffic", "order_type", "rider_speed"]]
    # Encode categorical order_type if needed
    if X["order_type"].dtype == object:
        X = X.copy()
        X["order_type"] = X["order_type"].astype('category').cat.codes
    y = orders["actual_eta_min"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'xgboost' and xgb_available:
        model = XGBRegressor(n_estimators=50, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    test_score = model.score(X_test, y_test)
    logger.info(f"Model trained. Test R^2 score: {test_score:.3f}")
    return model, test_score


def predict_eta(orders, model_path=MODEL_PATH, use_osm=False):
    logger.info(f"Predicting ETA for {len(orders)} orders, use_osm={use_osm}")
    """
    Predict ETAs for orders using the trained model.
    model_path: path to load the trained model
    Returns: DataFrame with order_id and predicted_eta_min
    """
    if use_osm:
        orders = enrich_orders_with_ors(orders)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not trained. Please run train_model() first.")
    model = joblib.load(model_path)
    # Map zone_traffic to numeric if needed
    traffic_map = {'Low': 1.0, 'Medium': 1.2, 'High': 1.4}
    orders = orders.copy()
    # Robustly handle zone_traffic mapping for any string values
    if orders['zone_traffic'].dtype == object or orders['zone_traffic'].apply(lambda x: isinstance(x, str)).any():
        orders['zone_traffic'] = orders['zone_traffic'].map(traffic_map).fillna(1.0)
    X = orders[["distance_km", "zone_traffic", "order_type", "rider_speed"]]
    if X["order_type"].dtype == object or X["order_type"].apply(lambda x: isinstance(x, str)).any():
        X = X.copy()
        X["order_type"] = X["order_type"].astype('category').cat.codes
    preds = model.predict(X)
    orders["predicted_eta_min"] = preds
    logger.info("ETA prediction complete.")
    return orders[["order_id", "predicted_eta_min"]]
