"""
Demand Forecasting module
- Predict future order demand per zone
- Support rolling average, ARIMA, XGBoost
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from utils.osm_routing import enrich_orders_with_ors
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("forecasting")

def forecast_demand(orders, method='rolling', zone_col='zone', date_col='order_time', window=7, forecast_days=7, use_osm=False):
    logger.info(f"Forecasting demand: orders={len(orders)}, method={method}, window={window}, forecast_days={forecast_days}, use_osm={use_osm}")
    """
    Forecast daily order demand per zone using rolling average or ARIMA.
    Returns DataFrame: zone, date, actual_orders, forecasted_orders
    """
    if use_osm:
        orders = enrich_orders_with_ors(orders)
    # Defensive: ensure input is a DataFrame and has required columns
    if not isinstance(orders, pd.DataFrame) or orders.empty or date_col not in orders.columns or zone_col not in orders.columns:
        return pd.DataFrame(columns=['zone', 'date', 'actual_orders', 'forecasted_orders'])
    orders = orders.copy()
    orders[date_col] = pd.to_datetime(orders[date_col], errors='coerce')
    orders = orders.dropna(subset=[date_col])
    orders['date'] = orders[date_col].dt.date
    results = []
    for zone in orders[zone_col].dropna().unique():
        zone_orders = orders[orders[zone_col] == zone]
        if zone_orders.empty:
            continue
        daily = zone_orders.groupby('date').size().rename('actual_orders').reset_index()
        daily = daily.set_index('date').asfreq('D', fill_value=0)
        daily['actual_orders'] = daily['actual_orders'].astype(float)
        if method == 'rolling':
            daily['forecasted_orders'] = daily['actual_orders'].rolling(window, min_periods=1).mean().shift(1)
            # Forecast for next forecast_days using last rolling mean
            last_date = daily.index[-1]
            last_mean = daily['forecasted_orders'].dropna().iloc[-1] if not daily['forecasted_orders'].dropna().empty else daily['actual_orders'].mean()
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
            future_df = pd.DataFrame({'actual_orders': np.nan, 'forecasted_orders': last_mean}, index=future_dates)
            daily = pd.concat([daily, future_df])
        elif method == 'arima':
            try:
                if daily['actual_orders'].sum() == 0 or len(daily) < 3:
                    daily['forecasted_orders'] = np.nan
                else:
                    model = ARIMA(daily['actual_orders'], order=(2,1,2))
                    fit = model.fit()
                    forecast = fit.forecast(steps=forecast_days)
                    daily['forecasted_orders'] = np.nan
                    forecast_idx = pd.date_range(daily.index[-1]+pd.Timedelta(days=1), periods=forecast_days)
                    forecast_df = pd.DataFrame({'forecasted_orders': forecast.values, 'actual_orders': np.nan}, index=forecast_idx)
                    daily = pd.concat([daily, forecast_df])
            except Exception:
                daily['forecasted_orders'] = np.nan
        daily['zone'] = zone
        daily = daily.reset_index().rename(columns={'index': 'date'})
        for col in ['actual_orders', 'forecasted_orders', 'zone', 'date']:
            if col not in daily.columns:
                daily[col] = np.nan
        results.append(daily[['zone', 'date', 'actual_orders', 'forecasted_orders']])
    if results:
        out = pd.concat(results, ignore_index=True)
        for col in ['zone', 'date', 'actual_orders', 'forecasted_orders']:
            if col not in out.columns:
                out[col] = np.nan
        logger.info("Forecasting complete.")
        return out[['zone', 'date', 'actual_orders', 'forecasted_orders']]
    else:
        return pd.DataFrame(columns=['zone', 'date', 'actual_orders', 'forecasted_orders'])
