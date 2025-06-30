from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import os
import re

def intent_from_query(query):
    """
    Returns (intent, params) tuple based on the query string.
    """
    q = query.strip().lower()
    # Average ETA intent
    m_eta = re.match(r"what is the average eta for ([\w\s-]+)\??", q)
    if m_eta:
        return ("average_eta", {"zone": m_eta.group(1).strip().title()})
    # Total orders intent
    if any(kw in q for kw in ["how many orders", "total number of orders", "number of orders"]):
        return ("order_count", {})
    # Deliveries from zone intent
    m_deliv = re.match(r"are there any deliveries from ([\w\s-]+)\??", q)
    if m_deliv:
        return ("deliveries_from_zone", {"zone": m_deliv.group(1).strip().title()})
    # List zones intent
    if any(kw in q for kw in ["list all zones", "what zones are available", "which zones are there"]):
        return ("list_zones", {})
    # Add more intents as needed
    return (None, {})

def explain_delays(orders, riders, zones, query=None, llm_type='openrouter', openrouter_api_key=None, openrouter_model=None, project_context=None):
    """
    Explain delays in natural language using OpenRouter (Mistral) via ChatOpenAI.
    Returns: LLM-generated explanation string or data-aware answer
    """
    # --- INTENT-BASED ROUTING ---
    if query:
        intent, params = intent_from_query(query)
        if intent == "average_eta":
            zone = params["zone"]
            df = orders[orders['zone'].str.lower() == zone.lower()]
            if not df.empty and 'actual_eta_min' in df.columns:
                avg_eta = df['actual_eta_min'].mean()
                return f"The average ETA for {zone} is {avg_eta:.1f} minutes based on current data."
            else:
                return f"No orders found for {zone} in the current data."
        elif intent == "order_count":
            return f"There are {len(orders)} orders currently in the system."
        elif intent == "deliveries_from_zone":
            zone = params["zone"]
            df = orders[orders['zone'].str.lower() == zone.lower()]
            if not df.empty:
                return f"Yes, there are {len(df)} deliveries from {zone} in the current data."
            else:
                return f"No deliveries from {zone} found in the current data."
        elif intent == "list_zones":
            zone_list = ', '.join(sorted(orders['zone'].unique()))
            return f"Available zones: {zone_list}."
    # --- LLM fallback ---
    # Prepare context from recent orders and zone traffic
    recent_orders = orders.sort_values('order_time', ascending=False).head(10)
    traffic_info = zones[['zone_name', 'zone_traffic']].to_dict('records')
    context = f"Recent Orders:\n{recent_orders[['order_id','rider_id','zone','order_time','delay_reason']].to_string(index=False)}\n"
    context += f"\nZone Traffic:\n" + ", ".join([f"{z['zone_name']}: {z['zone_traffic']}" for z in traffic_info])
    if project_context:
        context = f"PROJECT CONTEXT:\n{project_context}\n\n" + context
    if not query:
        query = "Explain the main reasons for recent delivery delays."
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a delivery operations expert. Given the following context and question, provide a clear, human-readable explanation for delivery delays.

Context:
{context}
Question:
{question}

Explanation:
"""
    )
    llm = ChatOpenAI(
        temperature=0.2,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=openrouter_model,
        max_tokens=256
    )
    # New chaining style: prompt | llm, then .invoke({...})
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": query})
    # Robust extraction for all result types
    if hasattr(result, 'content'):
        return str(result.content)
    if isinstance(result, dict) and 'content' in result:
        return str(result['content'])
    if isinstance(result, str):
        return result
    return str(result)
