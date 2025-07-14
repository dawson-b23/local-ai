from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from . import model
from .press20_agent import press20_agent
from .rag_agent import rag_agent
from .calculator_agent import calculator_agent
from langfuse import observe, get_client
import httpx
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str

master_agent = Agent(
    model,
    system_prompt="""
    You are an assistant for H&H. Current date: {}.
    - Route document or general queries to the General_RAG agent.
    - For vague queries (e.g., "what can you do"), respond: 'I can help with document searches or general queries. What would you like to do?'
    - If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query.'
    """.format(datetime.now().strftime("%Y-%m-%d")),

    #system_prompt=f"""
    #You are an orchestrator agent for H&H. Current date: {datetime.now().strftime("%Y-%m-%d")}.
    #- If the query contains 'shot_num', 'overallpassfail', or 'actnozzletemp', route to the Press20 agent. Do NOT use this otherwise.
    #- If the query contains mathematical operators ('+', '-', '*', '/'), route to the Calculator agent.
    #- If the query is about documents or general information, route to the General_RAG agent.
    #- If the query is vague (e.g., contains 'hello', 'what can you do', 'hi', or is empty), do not call any tools and return: 'I can help with Press20 data queries, document searches, or numerical calculations. What would you like to do?'
    #- If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query.'
    #- Do not include '[FINAL ANSWER]' or '[FIXED]' in responses.
    #- For vague queries, do not call any tools; return the response directly.
    #""",
    deps_type=Deps,
    retries=2
)

@master_agent.tool
@observe()
async def general_rag(ctx: RunContext[Deps], query: str) -> str:
    try:
        result = await rag_agent.run(query, deps=ctx.deps)
        return result.data if result.data else "No response from RAG agent"
    except Exception as e:
        logger.error(f"Error in general_rag: {str(e)}", exc_info=True)
        return f"Error in RAG query: {str(e)}"
'''
@master_agent.tool
@observe()
async def press20(ctx: RunContext[Deps], query: str) -> str:
    #langfuse.update_current_trace(metadata={"query": query})
    result = await press20_agent.run(query, deps=ctx.deps)
    #langfuse.update_current_trace(metadata={"status": "success", "result": result})
    return result

@master_agent.tool
@observe()
async def general_rag(ctx: RunContext[Deps], query: str) -> str:
    #langfuse.update_current_trace(metadata={"query": query})
    result = await rag_agent.run(query, deps=ctx.deps)
    #langfuse.update_current_trace(metadata={"status": "success", "result": result})
    return result

@master_agent.tool
@observe()
async def calculator(ctx: RunContext[Deps], expression: str) -> str:
    #langfuse.update_current_trace(metadata={"expression": expression})
    result = await calculator_agent.run(expression, deps=ctx.deps)
    #langfuse.update_current_trace(metadata={"status": "success", "result": result})
    return result
'''

async def create_master_agent():
    return master_agent
