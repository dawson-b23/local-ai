from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from .rag_agent import rag_agent
from .calculator_agent import calculator_agent
from .press20_agent import press20_agent
from supabase import create_client, Client
import httpx
from datetime import datetime
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)

model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str
    supabase: 'supabase.Client'

master_agent = Agent(
    model,
    system_prompt="""
# Overview
You are an orchestrator agent for H&H Molds Inc. Your sole task is to either return a predefined response for specific queries or route queries to the appropriate tool and return ONLY the tool's output in markdown format. Do not generate answers, summarize, or add any text beyond the predefined response or tool output.

## Predefined Responses
Return the exact response below for these queries (case-insensitive, ignore extra whitespace):
- Query: "who are you"
  - Response: I am the H&H AI Assistant, built to help with injection molding queries for H&H Molds Inc. I can answer questions about Press20 data, perform calculations, and retrieve information from documents.
- Query: "what can you do"
  - Response: I can:
    - Answer questions about press20_data (e.g., "press20_data list failed shots").
    - Perform numerical calculations (e.g., "2 + 2").
    - Retrieve and summarize document information (e.g., "summarize the scope meeting").
    Ask away or check the docs at https://dawson-b23.github.io/HHDocs/!
- Query: "help"
  - Response: Need help? I can:
    - Query press20_data (e.g., "press20_data list failed shots").
    - Calculate numerical expressions (e.g., "5 * 3").
    - Search documents (e.g., "summarize the scope meeting").
    Contact Dawson at intern@hhmoldsinc.com or 832-977-3004 for issues. Docs: https://dawson-b23.github.io/HHDocs/.

## Tools
- **general_rag**: Use for queries not matching predefined responses, queries that dont contain the exact match to 'press20_data', and not calculations.
- **press20**: Use for queries only containing "press20_data" (case-sensitive, must be exact match) and no others.
- **calculator**: Use for queries with numerical operators (+, -, *, /).

## Rules
1. Check if the query exactly matches (case-insensitive) "who are you", "what can you do", or "help". If so, return the predefined response and do NOT call any tools.
2. If the query contains 'press20_data' (case-sensitive), call `press20` tool, never call this tool otherwise.
3. If the query contains numerical operators (+, -, *, /), call `calculator` tool.
4. For all other queries, call `general_rag` tool.
5. Return ONLY the predefined response or the tool’s output in markdown format.
6. If the tool returns an error or no data, return the error message as-is.
7. Do NOT generate additional text, explanations, or tool call structures (e.g., JSON).
8. Do NOT mention tools used unless explicitly asked.
9. Do NOT call the press20 tool unless the query contains an exact match of 'press20_data'

## Examples
- Query: "who are you"
  - Action: Return predefined response
  - Output: I am the H&H AI Assistant, built to help with injection molding queries for H&H Molds Inc. I can answer questions about press20_data, perform calculations, and retrieve information from documents.
- Query: "What Can You Do"
  - Action: Return predefined response
  - Output: I can:
    - Answer questions about press20_data (e.g., "list failed shots from press20_data").
    - Perform numerical calculations (e.g., "2 + 2").
    - Retrieve and summarize document information (e.g., "summarize the scope meeting").
    Ask away or check the docs at https://dawson-b23.github.io/HHDocs/!
- Query: "List FAILED shots from press20_data"
  - Action: Call `press20` tool
  - Output: [Raw output from press20 tool]
- Query: "What is 10 + 10"
  - Action: Call `calculator` tool
  - Output: 20
- Query: "Summarize the scope meeting"
  - Action: Call `general_rag` tool
  - Output: [Raw output from general_rag tool]

## Final Reminder
- Current date: {}
- Return ONLY the predefined response or tool output in markdown format.
- Never return JSON structures or explanations unless explicitly requested.
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    deps_type=Deps,
    retries=2
)

@master_agent.tool
@observe()
async def general_rag(ctx: RunContext[Deps], query: str) -> str:
    if not query.strip():
        logger.error("Empty query received in general_rag")
        return "Error: Empty query provided."
    try:
        result = await rag_agent.run(query, deps=ctx.deps)
        return result
    except Exception as e:
        logger.error(f"Error in general_rag: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"
"""
@master_agent.tool
@observe()
async def general_rag(ctx: RunContext[Deps], query: str) -> str:
    try:
        result = await rag_agent.run(query, deps=ctx.deps)
        if not result:
            return "No response from RAG agent."
        return result
    except Exception as e:
        return f"Error in General RAG: {str(e)}"
"""

@master_agent.tool
@observe()
async def calculator(ctx: RunContext[Deps], expression: str) -> str:
    if not expression.strip():
        logger.error("Empty expression received in calculator")
        return "Error: Empty expression provided."
    try:
        result = await calculator_agent.run(expression, deps=ctx.deps)
        return result
    except Exception as e:
        logger.error(f"Error in calculator: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

"""
@master_agent.tool
@observe()
async def calculator(ctx: RunContext[Deps], expression: str) -> str:
    try:
        result = await calculator_agent.run(expression, deps=ctx.deps)
        return result
    except Exception as e: 
        return f"Error in calling calculator_agent tool: {str(e)}"
"""

@master_agent.tool
@observe()
async def press20(ctx: RunContext[Deps], query: str) -> str:
    if not query.strip():
        logger.error("Empty query received in press20")
        return "Error: Empty query provided."
    try:
        result = await press20_agent.run(query, deps=ctx.deps)
        return result
    except Exception as e:
        logger.error(f"Error in press20: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

"""
@master_agent.tool
@observe()
async def press20(ctx: RunContext[Deps], query: str) -> str:
    try:
        result = await press20_agent.run(query, deps=ctx.deps)
        #print(f"Press20 result: {vars(result)}")  # Debug print
        if not result:
            return "No response from Press20 agent."
        return result
    except Exception as e:
        return f"Error in Press20: {str(e)}"

async def create_master_agent():
    return master_agent
"""

""" backup plan to re route manually
import re

async def preprocess_query(query: str) -> str:
    # Check for calculation (contains operators)
    if re.search(r'[\+\-\*/]', query):
        return "calculator"
    # Check for Press20
    if "press" in query.lower():
        return "press20"
    # Default to general RAG
    return "general_rag"

@master_agent.tool
@observe()
async def route_query(ctx: RunContext[Deps], query: str)  -> str:
    tool_name = await preprocess_query(query)
    if tool_name == "calculator":
        return await calculator(ctx, query)
    elif tool_name == "press20":
        return await press20(ctx, query)
    else:
        return await general_rag(ctx, query)
"""
