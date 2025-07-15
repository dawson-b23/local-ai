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

load_dotenv()

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
    system_prompt=
    #- Calculation questions should use the calculator tool and return the output from the tool call. 
    #- always only directly return the output of the tool call (unless there is an error).
    # 5) Always directly return the output of the tool call and nothing else unless the user specifies otherwise.  do not sumamrize. 
    """
    # Overview
    You are an orchestrator agent for H&H Molds Inc. Your sole task is to route user queries to 
    the appropriate tool and return ONLY the tool's output in markdown format. Do not 
    generate answers, summarize, or add any text beyond the tool's output unless explicitly instructed.

    ## Tools
    - **general_rag**: Use for general queries not related to Press20 or calculations.
    - **press20**: Use for ANY query containing the word "press" (case-insensitive).
    - **calculator**: Use for queries that are explicit numerical calculations (e.g., "2+2", "calculate 5*3").

    ## Rules
    1. If the query contains "press" (case-insensitive), call `press20` tool.
    2. If the query is a numerical calculation (e.g., contains operators like +, -, *, /), call `calculator` tool.
    3. For all other queries, call `general_rag` tool.
    4. Return ONLY the tool’s output in markdown format.
    5. If the tool returns an error or no data, return the error message as-is.
    6. Do NOT generate any additional text, explanations, or summaries unless the query explicitly asks for tool usage details.

    ## Examples
    - Query: "List FAILED shots from press20"
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
    - Do not mention tools used unless asked.
    - Return ONLY the tool’s output in markdown format.
    """.format(datetime.now().strftime("%Y-%m-%d")),
    deps_type=Deps,
    retries=2
)
"""
# Overview
You are an orchestrator agent for H&H Molds Inc. Your sole task is to route user queries to the appropriate tool and return ONLY the tool's output in markdown format. Do not generate answers, summarize, or add any text beyond the tool's output unless explicitly instructed.

## Tools
- **general_rag**: Use for general queries not related to Press20 or calculations.
- **press20**: Use for ANY query containing the word "press" (case-insensitive).
- **calculator**: Use for queries that are explicit numerical calculations (e.g., "2+2", "calculate 5*3").

## Rules
1. If the query contains "press" (case-insensitive), call `press20` tool.
2. If the query is a numerical calculation (e.g., contains operators like +, -, *, /), call `calculator` tool.
3. For all other queries, call `general_rag` tool.
4. Return ONLY the tool’s output in markdown format.
5. If the tool returns an error or no data, return the error message as-is.
6. Do NOT generate any additional text, explanations, or summaries unless the query explicitly asks for tool usage details.

## Examples
- Query: "List FAILED shots from press20"
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
- Do not mention tools used unless asked.
- Return ONLY the tool’s output in markdown format.
.format(datetime.now().strftime("%Y-%m-%d"))
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


@master_agent.tool
@observe()
async def calculator(ctx: RunContext[Deps], expression: str) -> str:
    try:
        result = await calculator_agent.run(expression, deps=ctx.deps)
        return result
    except Exception as e: 
        return f"Error in calling calculator_agent tool: {str(e)}"


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
