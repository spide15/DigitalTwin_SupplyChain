"""
LangChain agent logic for delay explanation using LLMs.
"""
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import os

raise ImportError("This module is deprecated. Use delay_agent_openrouter.py for all chatbot/LLM logic.")

def explain_delays(orders, riders, zones, query=None, llm_type='openai', openai_api_key=None, llama_model_path=None, project_context=None):
    """
    Explain delays in natural language using LangChain with LLaMA 3 or OpenAI.
    query: e.g. 'Why was Rider-102 late today?'
    llm_type: 'openai' or 'llama'
    openai_api_key: required if using OpenAI
    llama_model_path: required if using LLaMA 3
    project_context: (optional) string to prepend as system prompt/project context
    Returns: LLM-generated explanation string
    """
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
    if llm_type == 'llama' and llama_available and llama_model_path:
        llm = LlamaCpp(model_path=llama_model_path, temperature=0.2, max_tokens=256)
    else:
        llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key, max_tokens=256)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"context": context, "question": query})
    return result
