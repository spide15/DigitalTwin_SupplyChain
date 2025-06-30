"""
Streamlit dashboard entry point for CityScale Digital Twin ETA Prediction System.
"""

import streamlit as st
from modules import eta_predictor, cluster_logic, order_batching
import pandas as pd
import os
import plotly.express as px
from utils.osm_routing import enrich_orders_with_ors
from dotenv import load_dotenv


st.set_page_config(page_title="CityScale ETA Dashboard", layout="wide")

# Load sample data
import numpy as np
from datetime import datetime, timedelta

def generate_demo_orders(num_days=30, min_orders_per_day=20):
    np.random.seed(42)
    # Use real zone centroids and sample only valid coordinates from zones.csv
    zones_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'zones.csv'))
    zone_coords = {row['zone_name']: (row['latitude'], row['longitude']) for _, row in zones_df.iterrows()}
    order_types = ['food', 'grocery', 'pharmacy']
    traffic_levels = ['Low', 'Medium', 'High']
    now = datetime.now()
    records = []
    order_id = 10000
    for day in range(num_days):
        date = (now - timedelta(days=num_days - day - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(min_orders_per_day):
            zone = np.random.choice(list(zone_coords.keys()))
            # Generate points within 500m of the zone centroid, clipped to valid lat/lon
            base_lat, base_lon = zone_coords[zone]
            # Offset within ~500m using realistic deltas
            dlat = np.random.uniform(-0.003, 0.003)
            dlon = np.random.uniform(-0.003, 0.003)
            lat = round(base_lat + dlat, 6)
            lon = round(base_lon + dlon, 6)
            # Ensure lat/lon are within Mumbai/Navi Mumbai bounding box
            lat = min(max(lat, 18.9), 19.25)
            lon = min(max(lon, 72.85), 73.13)
            distance_km = np.round(np.random.uniform(1.0, 4.0), 2)
            zone_traffic = np.random.choice(traffic_levels, p=[0.3, 0.5, 0.2])
            order_type = np.random.choice(order_types)
            rider_speed = int(np.random.normal(25, 2))
            actual_eta_min = int((distance_km / max(rider_speed, 10)) * 60 * np.random.uniform(1.1, 1.4))
            delay_reason = ''
            order_frequency = np.random.randint(2, 6)
            time_pref = np.random.randint(8, 15)
            urgency = np.random.randint(1, 3)
            weight_kg = np.round(np.random.uniform(1.0, 3.5), 1)
            order_time = (date + timedelta(minutes=np.random.randint(0, 1440))).strftime('%Y-%m-%dT%H:%M:%S')
            customer_id = np.random.randint(200, 400)
            rider_id = np.random.randint(201, 221)
            records.append([
                order_id, customer_id, zone, lat, lon, distance_km, zone_traffic, order_type, rider_speed,
                actual_eta_min, delay_reason, order_frequency, time_pref, urgency, weight_kg, order_time, rider_id
            ])
            order_id += 1
    columns = ['order_id','customer_id','zone','latitude','longitude','distance_km','zone_traffic','order_type','rider_speed','actual_eta_min','delay_reason','order_frequency','time_pref','urgency','weight_kg','order_time','rider_id']
    return pd.DataFrame(records, columns=columns)

# Load sample data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
riders = pd.read_csv(os.path.join(data_dir, 'riders.csv'))
zones = pd.read_csv(os.path.join(data_dir, 'zones.csv'))

# --- Top Navigation Bar ---
menu_options = [
    "Home",
    "ETA Prediction",
    # "Delay Explanation",  # Removed
    "Order Batching",
    "Customer Clustering",
    "Kirana vs Central Delivery Simulator",
    "Forecasting",
    "Inventory Heatmap",
    "Congestion Heatmap"
]
from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    /* Responsive and aesthetic top menu bar with border and improved responsiveness */
    .block-container {padding-top: 2.5rem !important;}
    .css-18e3th9 {padding-top: 0rem !important;}
    .top-menu-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        background: linear-gradient(90deg, #e3f0ff 0%, #f0f6ff 100%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        padding: 0.5em 0 0.5em 0;
        margin-bottom: 1.2em;
        border-radius: 0 0 12px 12px;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        border: 2.5px solid #0d6efd;
        border-top: none;
        border-left: none;
        border-right: none;
        box-shadow: 0 4px 16px rgba(13,110,253,0.10);
        overflow-x: auto;
        white-space: normal;
        min-width: 0;
    }
    .top-menu-bar::-webkit-scrollbar {height: 6px; background: #e3f0ff;}
    .top-menu-bar::-webkit-scrollbar-thumb {background: #b6d8ff; border-radius: 6px;}
    @media (max-width: 900px) {
        .block-container {padding-top: 1.2rem !important;}
        .top-menu-bar {padding: 0.2em 0; font-size: 15px;}
    }
    @media (max-width: 600px) {
        .block-container {padding-top: 0.5rem !important;}
        .top-menu-bar {
            padding: 0.1em 0;
            font-size: 13px;
            flex-wrap: nowrap;
            overflow-x: auto;
            white-space: nowrap;
        }
    }
    </style>
    """, unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="top-menu-bar"></div>', unsafe_allow_html=True)
    selected_menu = option_menu(
        menu_title=None,
        options=menu_options,
        orientation="horizontal",
        icons=["house", "clock", "question-circle", "layers", "people", "truck", "graph-up", "grid-3x3-gap", "exclamation-triangle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "transparent", "border": "none", "box-shadow": "none"},
            "icon": {"color": "#0d6efd", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "margin":"0 8px", "color": "#333", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#b6d8ff", "color": "#0d6efd", "font-weight": "bold"},
        }
    )

# --- Sidebar Filters Only ---
st.sidebar.header("Filters")
st.markdown("""
<style>
/* Aesthetic sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e3f0ff 0%, #f0f6ff 100%) !important;
    border-radius: 0 18px 18px 0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.07);
    padding-top: 2.5em !important;
    padding-bottom: 2.5em !important;
    min-width: 320px !important;
    max-width: 400px !important;
    min-height: 100vh !important;
    height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
[data-testid="stSidebar"] .stHeader {
    font-size: 1.2em !important;
    font-weight: bold;
    color: #0d6efd;
    margin-bottom: 1em;
    white-space: normal !important;
    word-break: break-word !important;
}
[data-testid="stSidebar"] .stMultiSelect, [data-testid="stSidebar"] .stSelectbox {
    background: #fff !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 1.2em !important;
    padding: 0.5em 0.7em !important;
    min-width: 90% !important;
    max-width: 98% !important;
    white-space: normal !important;
    word-break: break-word !important;
}
[data-testid="stSidebar"] label {
    color: #0d6efd !important;
    font-weight: 500;
    white-space: normal !important;
    word-break: break-word !important;
}
</style>
""", unsafe_allow_html=True)
zone_options = zones['zone_name'].unique()
selected_zones = st.sidebar.multiselect("Select Zone(s)", zone_options, default=list(zone_options))
filtered_orders = orders[orders['zone'].isin(selected_zones)] if selected_zones else orders

# --- Chatbot in Sidebar (bottom, OpenAI or OpenRouter, API key in code) ---
# Ensure chatbot always uses the same filtered data as the main modules (including demo data and sidebar filters)
# filtered_orders, riders, and zones are always up-to-date and passed to the chatbot below
PROJECT_CONTEXT = '''\
CityScale Digital Twin is a Streamlit dashboard for simulating and analyzing last-mile delivery operations in Thane/Navi Mumbai. It features robust ETA prediction (Random Forest/XGBoost), OSM/ORS routing, realistic demo data, COO-level analytics, and a fully functional, project-aware chatbot. The dashboard supports:
- ETA prediction and delay analysis
- Order batching and customer clustering
- Kirana vs central delivery simulation
- Demand forecasting and inventory heatmaps
- Congestion heatmaps and OSM analytics
- Responsive, modern UI/UX with onboarding and sidebar filters
- All analytics and operations are based on real or demo data (orders, riders, zones, kirana)
- The chatbot can answer any question about the dashboard, analytics, or operations, and always has this project context

EXAMPLES (Q&A):
Q: How many orders are there currently?
A: There are {n} orders currently in the system.

Q: Are there any deliveries from Nerul?
A: Based on the current data, there are no deliveries from Nerul.

Q: What is the average ETA for Thane West?
A: The average ETA for Thane West is {x} minutes.

Q: What causes most delivery delays?
A: Most delays are due to high traffic and order batching.

Q: How do I generate demo data?
A: Go to the Home page and click "Generate Demo Data" to create realistic sample orders for testing.

Q: Can you explain the difference between Kirana and Central delivery?
A: Kirana delivery fulfills orders from local stores, while Central delivery uses a main warehouse. Kirana is often faster and more eco-friendly for local orders.

Q: How do I use the forecasting module?
A: Select a zone and forecast method on the Forecasting page, then click "Run Forecast" to see demand predictions.
'''
# --- LLM PROVIDER CONFIG ---
# Only OpenRouter (Mistral) is supported
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")  # <-- Set your OpenRouter API key here
OPENROUTER_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
from modules import delay_agent_openrouter as delay_agent_mod

st.sidebar.markdown("---")
st.sidebar.markdown("<b>CityScale Chatbot</b>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='font-size:0.95em;color:#888;'>Ask about delivery delays, operations, or analytics.</span>", unsafe_allow_html=True)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
user_input = st.sidebar.text_input("Type your message...", key="chatbot_input_sidebar")
if st.sidebar.button("Send", key="chatbot_send_sidebar") and user_input:
    try:
        response = delay_agent_mod.explain_delays(
            filtered_orders, riders, zones,
            query=user_input,
            llm_type="openrouter",
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL,
            project_context=PROJECT_CONTEXT
        )
        st.session_state['chat_history'].append((user_input, response))
    except Exception as e:
        st.session_state['chat_history'].append((user_input, f"Error: {str(e)}"))
# Display chat history (last 8)
for q, a in st.session_state['chat_history'][-8:]:
    st.sidebar.markdown(f"<b>You:</b> {q}", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Bot:</b> {a}", unsafe_allow_html=True)

# --- Page Routing ---
page = selected_menu

# Home Page
if page == "Home":
    st.title("CityScale Digital Twin for Predictive Quick Commerce")
    st.markdown("""
    <style>
    .welcome-block {background-color: #f0f6ff; border-radius: 8px; padding: 1.5em; margin-bottom: 1.5em;}
    </style>
    <div class="welcome-block">
    <h2>Welcome to the CityScale Digital Twin Demo</h2>
    <p>This platform simulates and analyzes last-mile delivery operations for quick commerce in Thane/Navi Mumbai.</p>
    </div>
    """, unsafe_allow_html=True)
    if 'home_step' not in st.session_state:
        st.session_state['home_step'] = 0
    if st.session_state['home_step'] == 0:
        if st.button("Start"):
            st.session_state['home_step'] = 1
    elif st.session_state['home_step'] == 1:
        st.markdown("""
        <div class="welcome-block">
        <b>How to use this project:</b>
        <ol>
          <li><b>Step 1:</b> Review the project overview and instructions below.</li>
          <li><b>Step 2:</b> Click <b>Next</b> to begin exploring the data and modules.</li>
        </ol>
        <b>Tip:</b> Reload the app to reset demo data and return to your original dataset.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Next"):
            st.session_state['home_step'] = 2
    elif st.session_state['home_step'] == 2:
        # Only show demo data button if not already active
        if 'orders' in st.session_state:
            st.info("Demo data is active. Reload the app to reset to original data.")
        else:
            if st.button("Generate Demo Data (30 days, 20+ orders/day)"):
                st.session_state['orders'] = generate_demo_orders()
                st.success("Demo data generated! All modules will now use this data until you reload the app.")
        orders_to_use = st.session_state['orders'] if 'orders' in st.session_state else orders
        filtered_orders = orders_to_use[orders_to_use['zone'].isin(selected_zones)] if selected_zones else orders_to_use
        # OSM/ORS enrichment button
        if st.button("Enrich with OSM/ORS Route Data"):
            with st.spinner("Fetching OSM/ORS route distances and ETAs (may take time)..."):
                enriched = enrich_orders_with_ors(filtered_orders)
                st.session_state['orders'] = enriched
                st.success("Orders enriched with OSM/ORS route data! Columns 'ors_distance_km' and 'ors_eta_min' added.")
        with st.container():
            st.header("Active Deliveries (Sample)")
            st.dataframe(filtered_orders.head(20), use_container_width=True)
            

# ETA Prediction Page
elif page == "ETA Prediction":
    st.header("Train ETA Prediction Model")
    model_type = st.selectbox("Select Model Type", ["random_forest", "xgboost"] if eta_predictor.xgb_available else ["random_forest"], key="model_type")
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, test_score = eta_predictor.train_model(filtered_orders, model_type=model_type)
        st.success(f"Model trained! Test R² score: {test_score:.3f}")
        with st.expander("What does R² mean?", expanded=True):
            st.markdown("""
            **R² (R-squared)** is a statistical measure of how well the model's predictions match the actual data. 
            - **R² = 1.0**: Perfect prediction.
            - **R² = 0.0**: Model predicts no better than the mean of the data.
            - **R² < 0.0**: Model performs worse than simply predicting the mean (poor fit).

            **Why can R² be negative?**
            - A negative R² means the model's predictions are worse than just using the average value for all predictions.
            - This can happen if the model is not capturing the relationship in the data, if the features are not informative, if there is too little data, or if the model type is not suitable for the problem.
            - Try improving data quality, adding better features, or using a different model.
            """)
    st.header("ETA Prediction")
    if st.button("Predict ETAs"):
        try:
            etas = eta_predictor.predict_eta(filtered_orders)
            st.write(etas)
        except FileNotFoundError as e:
            st.error(str(e))

# Order Batching Page
elif page == "Order Batching":
    st.header("Smart Order Batching")
    use_osm = st.checkbox("Use OSM/ORS Route Data for Distance/ETA (slower, more realistic)", value=False)
    if st.button("Batch Orders"):
        batches = order_batching.batch_orders(filtered_orders, riders, zones, use_osm=use_osm)
        st.write("### Order Batches")
        for _, row in batches.iterrows():
            st.markdown(f"#### Batch {row['batch_id']}")
            st.markdown(f"- **Total Weight:** {row['total_weight']} kg  ")
            st.markdown(f"- **Expected ETA:** {row['expected_eta_min']} min  ")
            st.markdown(f"- **Expected Cost:** ₹{row['expected_cost']}")
            # Show order details in a table
            order_ids = row['order_ids'] if isinstance(row['order_ids'], list) else [row['order_ids']]
            batch_orders_df = filtered_orders[filtered_orders['order_id'].isin(order_ids)][['order_id', 'customer_id', 'weight_kg', 'urgency', 'order_time']]
            batch_orders_df = batch_orders_df.rename(columns={
                'order_id': 'Order ID',
                'customer_id': 'Customer ID',
                'weight_kg': 'Weight (kg)',
                'urgency': 'Urgency',
                'order_time': 'Order Time'
            })
            st.table(batch_orders_df)
        # --- Batch Orders Map Visualization ---
        batch_orders = batches.explode('order_ids')
        batch_orders = batch_orders.rename(columns={'order_ids': 'order_id'})
        batch_plot_df = batch_orders.merge(filtered_orders, on='order_id', how='left')
        if not batch_plot_df.empty and 'latitude' in batch_plot_df.columns and 'longitude' in batch_plot_df.columns:
            fig = px.scatter_mapbox(
                batch_plot_df,
                lat="latitude",
                lon="longitude",
                color="batch_id",
                hover_data=["order_id", "total_weight", "expected_eta_min", "expected_cost"],
                zoom=11,
                height=400,
                title="Order Batches on Map"
            )
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

# Customer Clustering Page
elif page == "Customer Clustering":
    st.header("Customer Clustering")
    st.markdown("""
    Group customers based on their order frequency and time preferences. This helps identify patterns and target customer segments more effectively.
    """)
    if st.button("Cluster Customers"):
        clusters, _ = cluster_logic.cluster_customers(filtered_orders)
        st.success(f"Clustering complete! Found {clusters['cluster'].nunique()} clusters.")
        # Show summary table: number of customers per cluster
        cluster_counts = clusters['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Number of Customers']
        st.markdown("**Customers per Cluster:**")
        st.table(cluster_counts)
        # For each cluster, show a table of customers
        for cluster_id in sorted(clusters['cluster'].unique()):
            st.markdown(f"### Cluster {cluster_id}")
            cluster_customers = clusters[clusters['cluster'] == cluster_id]
            merged = filtered_orders.merge(cluster_customers, on="customer_id")
            show_cols = ['customer_id', 'order_frequency', 'time_pref', 'urgency']
            show_cols = [col for col in show_cols if col in merged.columns]
            st.dataframe(merged[show_cols].drop_duplicates().rename(columns={
                'customer_id': 'Customer ID',
                'order_frequency': 'Order Frequency',
                'time_pref': 'Time Preference',
                'urgency': 'Urgency'
            }), use_container_width=True)
        # Interactive scatter plot
        plot_df = filtered_orders.merge(clusters, on="customer_id")
        cluster_col = 'cluster'
        if 'cluster_x' in plot_df.columns:
            cluster_col = 'cluster_x'
        elif 'cluster_y' in plot_df.columns:
            cluster_col = 'cluster_y'
        fig = px.scatter(
            plot_df,
            x="order_frequency",
            y="time_pref",
            color=cluster_col,
            hover_data=["customer_id", "urgency"],
            title="Customer Clusters (by order frequency & time preference)"
        )
        st.plotly_chart(fig, use_container_width=True)

# Kirana vs Central Delivery Simulator Page
elif page == "Kirana vs Central Delivery Simulator":
    st.header("Kirana vs Central Delivery Simulator: What-If Analysis")
    st.markdown("""
    What if we fulfill orders from nearby kirana stores instead of a central warehouse?\
    Compare time, cost, and CO₂ emissions for both models.\
    """)
    kirana_path = os.path.join(data_dir, 'kirana.csv')
    if not os.path.exists(kirana_path):
        st.error("Missing kirana.csv in data folder.")
    else:
        kiranas = pd.read_csv(kirana_path)
        from modules import kirana_simulator
        if st.button("Run What-If Simulation"):
            sim_df = kirana_simulator.simulate_kirana_vs_central(filtered_orders, kiranas, zones)
            st.success(f"Simulation complete! {sim_df['model_type'].value_counts().to_dict()}")
            # KPI summary
            kpi = sim_df.groupby('model_type').agg({'eta':'mean','cost':'mean','distance':'sum','co2_kg':'sum','order_id':'count'}).rename(columns={'order_id':'orders'}).reset_index()
            kpi['eta'] = kpi['eta'].round(1)
            kpi['cost'] = kpi['cost'].round(2)
            kpi['distance'] = kpi['distance'].round(2)
            kpi['co2_kg'] = kpi['co2_kg'].round(2)
            st.markdown("**KPI Comparison (Total/Avg):**")
            st.dataframe(kpi.style.format({'eta':'{:.1f} min','cost':'₹{:.2f}','distance':'{:.2f} km','co2_kg':'{:.2f} kg'}), use_container_width=True)
            # What-if savings (always show)
            k = kpi.set_index('model_type')
            if 'central' in k.index and 'kirana' in k.index:
                time_saving = k.loc['central','eta'] - k.loc['kirana','eta']
                co2_saving = k.loc['central','co2_kg'] - k.loc['kirana','co2_kg']
                cost_saving = k.loc['central','cost'] - k.loc['kirana','cost']
                st.markdown(f"""
                ### What-If Savings (Kirana vs Central)
                - **Avg. Time Saved per Order:** `{time_saving:.1f} min`
                - **Total CO₂ Saved:** `{co2_saving:.2f} kg`
                - **Avg. Cost Saved per Order:** `₹{cost_saving:.2f}`
                """)
            else:
                st.warning("Not enough data for both models to compute savings. Try with more diverse orders or adjust kirana service radius.")
            # Map of assignments
            merged = filtered_orders.merge(sim_df[['order_id','assigned_to','model_type']], on='order_id')
            fig = px.scatter_mapbox(
                merged,
                lat="latitude",
                lon="longitude",
                color="model_type",
                hover_data=["order_id", "assigned_to", "zone", "distance_km"],
                zoom=11,
                height=400,
                title="Order Assignments: Kirana vs Central"
            )
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

# Forecasting Page
elif page == "Forecasting":
    st.header("Demand Forecasting")
    st.markdown("""
    Predict future order demand per zone using time series models (rolling average, ARIMA).\
    Use forecasts to simulate restocking needs for kiranas.
    """)
    from modules import forecasting
    zone_sel = st.selectbox("Select Zone", options=filtered_orders['zone'].unique())
    method = st.selectbox("Forecast Method", options=["rolling", "arima"], format_func=lambda x: "Rolling Average" if x=="rolling" else "ARIMA")
    window = st.slider("Rolling Window (days)", 3, 14, 7) if method=="rolling" else None
    forecast_days = st.slider("Forecast Days Ahead", 3, 14, 7)
    if st.button("Run Forecast"):
        fc_df = forecasting.forecast_demand(filtered_orders[filtered_orders['zone']==zone_sel], method=method, window=window or 7, forecast_days=forecast_days)
        st.success(f"Forecast complete for {zone_sel}!")
        # Show table
        st.dataframe(fc_df[['date','actual_orders','forecasted_orders']].tail(14), use_container_width=True)
        # Plot
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc_df['date'], y=fc_df['actual_orders'], mode='lines+markers', name='Actual Orders'))
        fig.add_trace(go.Scatter(x=fc_df['date'], y=fc_df['forecasted_orders'], mode='lines+markers', name='Forecasted Orders'))
        fig.update_layout(title=f"Order Demand Forecast for {zone_sel}", xaxis_title="Date", yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)

# Inventory Heatmap Page
elif page == "Inventory Heatmap":
    st.header("Inventory/Restocking Heatmap")
    st.markdown("""
    Track SKU depletion per kirana and raise alerts when below threshold.\
    Visualize inventory status on a map.
    """)
    kirana_path = os.path.join(data_dir, 'kirana.csv')
    if not os.path.exists(kirana_path):
        st.error("Missing kirana.csv in data folder.")
    else:
        kiranas = pd.read_csv(kirana_path)
        from modules import inventory
        skus = ['atta','rice','oil','milk','snacks']
        selected_sku = st.selectbox("Select SKU", skus)
        threshold = st.slider("Low Stock Threshold", 1, 20, 10)
        inv_df = inventory.simulate_kirana_inventory(kiranas, skus)
        alert_df = inventory.analyze_inventory(inv_df[inv_df['sku']==selected_sku], threshold=threshold)
        st.markdown(f"**Kirana Inventory for {selected_sku}:**")
        st.dataframe(alert_df[['kirana_id','name','stock','alert']], use_container_width=True)
        # Map visualization
        import plotly.express as px
        fig = px.scatter_mapbox(
            alert_df,
            lat="lat",
            lon="lon",
            color="alert",
            size="stock",
            hover_data=["name", "stock"],
            zoom=11,
            height=400,
            title=f"Kirana Inventory Heatmap for {selected_sku}"
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# Congestion Heatmap Page
elif page == "Congestion Heatmap":
    st.header("Congestion Heatmap (Simulated)")
    st.map(filtered_orders[['latitude', 'longitude']])

# --- Responsive UI CSS ---
st.markdown("""
    <style>
    @media (max-width: 900px) {
        .block-container {padding-left: 0.5em !important; padding-right: 0.5em !important;}
        .stButton button, .stSelectbox, .stMultiSelect, .stSlider, .stTextInput, .stDataFrame, .stTable, .stPlotlyChart {
            width: 100% !important;
            min-width: 0 !important;
        }
        .stColumn {flex: 1 1 100% !important; max-width: 100% !important;}
    }
    @media (max-width: 600px) {
        .block-container {padding-left: 0.2em !important; padding-right: 0.2em !important;}
        .stButton button, .stSelectbox, .stMultiSelect, .stSlider, .stTextInput, .stDataFrame, .stTable, .stPlotlyChart {
            font-size: 15px !important;
        }
        h1, h2, h3, h4 {font-size: 1.1em !important;}
    }
    </style>
    """, unsafe_allow_html=True)
