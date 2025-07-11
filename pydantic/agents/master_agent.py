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
    system_prompt=f"""
    You are an orchestrator agent for H&H. Current date: {datetime.now().strftime("%Y-%m-%d")}.
    - If the query contains 'shot_num', 'overallpassfail', or 'actnozzletemp', route to the Press20 agent.
    - If the query is a numerical expression (e.g., contains '+', '-', '*', '/'), route to the Calculator agent.
    - If the query is about documents or general information, route to the General_RAG agent.
    - For vague queries like 'hello', 'what can you do', or 'hi', respond with: 'I can help with Press20 data queries, document searches, or numerical calculations. What would you like to do?'
    - If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query.'
    - Do not include '[FINAL ANSWER]' or '[FIXED]' in responses.
    """,
    deps_type=Deps,
    retries=2
)

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

def create_master_agent():
    return master_agent
