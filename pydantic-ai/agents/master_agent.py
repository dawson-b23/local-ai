from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from .rag_agent import rag_agent
from .calculator_agent import calculator_agent
from .press20_agent import run as press20_run
from .analysis_agent import analysis_agent
from .web_search_agent import web_search_agent  
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx
from datetime import datetime
import logging

load_dotenv()

client = AsyncOpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), api_key="ollama")
model = OpenAIModel(model_name=os.getenv("LLM_MODEL", "deepseek-coder:16b"), provider=OpenAIProvider(base_url=os.getenv("OLLAMA_URL") + "/v1"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

cot = False # enable thinking/chain of thought.

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase: 'supabase.Client'

master_agent = Agent(
    model,
    system_prompt=f"""
Current date: {datetime.now().strftime("%Y-%m-%d")}.
Orchestrator for H&H injection molding. FIRST check for exact matches (ignore case/whitespace) and return predefined DIRECTLY without tools or thinking:
- "who are you": I am H&H AI Assistant for molding queries (docs, Press20, calculations, trends, defects).
- "what can you do": - Docs summaries\n- Press20 data (e.g., failed shots)\n- Calculations\n- Trends analysis\n- Defect fixes\nDocs: https://dawson-b23.github.io/HHDocs/
- "help": Try asking 'what can you do' or click the docs link in the sidebar. Contact: intern@hhmoldsinc.com | 832-977-3004
Example:
- Query: help
  Action: Return "Try asking 'what can you do' or click the docs link in the sidebar. Contact: intern@hhmoldsinc.com | 832-977-3004" directly.
If no exact match, route:
- Starts with "websearch.": Route to web_search
- Starts with "press20_data": Route to press20 
- Numbers/operators: Route to calculator
- Else: Route to rag
After routing, ALWAYS use 'think' for chain-of-thought reasoning to verify decision and plan. Do not ever return chain of though/think unless COT is true.
Return ONLY predefined or agent's markdown output. No extras.""",
    deps_type=Deps,
    retries=3
)
#- 'trend', 'analyze', 'pattern': Route to analysis
#- 'defect', 'fix', 'tweak': Route to analysis

@master_agent.tool
@observe()
async def route_to_rag(ctx: RunContext[Deps], query: str) -> str:
    return await rag_agent.run(query, deps=ctx.deps)

@master_agent.tool
@observe()
async def route_to_calculator(ctx: RunContext[Deps], query: str) -> str:
    return await calculator_agent.run(query, deps=ctx.deps)

@master_agent.tool
@observe()
async def route_to_press20(ctx: RunContext[Deps], query: str) -> str:
    # Strip "press20_data" prefix if present
    cleaned_query = query.replace("press20_data", "").strip()
    return await press20_run(cleaned_query, deps=ctx.deps)

#@master_agent.tool
#@observe()
#async def route_to_analysis(ctx: RunContext[Deps], query: str) -> str:
#    return await analysis_agent.run(query, deps=ctx.deps)

@master_agent.tool
@observe()
async def route_to_web_search(ctx: RunContext[Deps], query: str) -> str:
    # Strip "websearch." prefix if present
    cleaned_query = query.replace("websearch.", "").strip()
    return await web_search_agent.run(cleaned_query, deps=ctx.deps)

@master_agent.tool
@observe()
async def think(ctx: RunContext[Deps], thought: str) -> str:
    # Internal: No output, just for agent reasoning
    if cot is True:
        return f"- **Reasoning:** {thought}"
    else:
        return ""
