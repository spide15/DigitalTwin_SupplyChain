# CityScale Digital Twin for Predictive Quick Commerce

## Problem Statement
Last-mile delivery in dense, semi-urban areas like Mumbai suburbs faces challenges of high traffic, long delivery times, and increased operational costs. Traditional central-warehouse models often lead to inefficient routes, higher CO₂ emissions, and poor customer experience. There is a need for a data-driven, AI-powered system to simulate, compare, and optimize delivery strategies, including leveraging local kirana stores as micro-fulfillment centers.

## Solution Approach
CityScale is a modular digital twin and analytics dashboard that:
- Simulates and compares central warehouse vs. kirana-based fulfillment for each order using real geospatial and demand data.
- Predicts delivery ETAs, costs, and CO₂ emissions for both models.
- Provides a "what-if" scenario tool to quantify time, cost, and environmental savings from using local kiranas.
- Includes demand forecasting, inventory heatmaps, and delay explanation using ML/LLM.
- Visualizes all results on interactive maps and charts for actionable insights.
- Includes an example script (`utils/osm_routing_example.py`) for fetching realistic road network distances and travel times using OpenStreetMap data via the OpenRouteService API. This is for data enrichment and not required for core dashboard features.

## Further Space for Improvement
- Integrate  weather APIs for dynamic ETA and route optimization.
- Add live inventory sync with kirana POS systems.
- Incorporate rider assignment and shift optimization.
- Model customer satisfaction and churn risk.
- Support multi-city and multi-modal (bike, EV, drone) delivery scenarios.
- Add more granular CO₂ and cost modeling (vehicle type, fuel, etc).

## Overview
A modular, real-time ETA prediction and simulation system for urban logistics in Mumbai zones. Features:
- Multi-zone traffic-aware simulation
- Real-time ETA prediction (Random Forest/XGBoost)
- AI chatbot ,which can handle user query and resolve it
- Smart order batching with geospatial visualization
- Customer clustering with interactive Plotly visualization
- Streamlit dashboard with sidebar navigation and device-responsive UI
- Interactive map and cluster plots for easy analysis
- What-if scenario analysis for kirana vs central fulfillment (time, cost, CO₂)

## Features
- **Home:** View active deliveries and filter by zone.
- **ETA Prediction:** Train and use ETA models (Random Forest/XGBoost) on filtered data.
- **AI chatbot:** Ask questions about project,delivery,etc.
- **Order Batching:** Run smart batching optimizer and visualize batches on a map.
- **Customer Clustering:** Cluster customers and visualize clusters interactively.
- **Kirana vs Central Delivery Simulator:** What-if analysis of local vs central fulfillment, with savings KPIs.
- **Forecasting:** Predict future order demand per zone.
- **Inventory Heatmap:** Visualize kirana stock and restocking needs.
- **Congestion Heatmap:** View simulated congestion using a map-based heatmap.
- **Sidebar Navigation:** Easily switch between features/pages.

## File Structure
- `app.py`: Streamlit dashboard entry point (with navigation bar)
- `data/`: CSVs (orders.csv, zones.csv, kirana.csv, etc.)
- `modules/`: Feature modules (ETA, delay agent, clustering, batching, kirana simulator, forecasting, inventory)
- `utils/`: Helper functions (geo, traffic)
- `requirements.txt`: Python dependencies

## Quick Start
1. Install requirements: `pip install -r requirements.txt`
2. Run dashboard: `streamlit run app.py`
3. Use the sidebar to navigate between features and analyze deliveries.

## Onboarding & Demo Data
The dashboard now features an improved onboarding experience:
- **Stepper-style Home Page:** When you launch the app, you'll see a prominent welcome block with project overview and instructions. Click **Next** to reveal the main dashboard, data, and modules.
- **Real time Data Generation:** On the Home tab, use the **Generate  Data** button to instantly create 30 days of realistic order data (20+ orders per day) for demo and testing. Demo data is only loaded after you click the button, and persists during your session.
- **User Experience:** The onboarding flow is designed for quick, intuitive exploration—ideal for new users and COO-level demos. Helpful tips and instructions are always visible on the Home page.

## Sample Data
Place sample CSVs in the `data/` folder: `orders.csv`, `riders.csv`, `zones.csv`, `kirana.csv`. Or use the demo data generator for instant test data.


## Deployment
Deploy on Streamlit Community Cloud or Render.
