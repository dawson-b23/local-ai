from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx
#from .press20_agent import query_press20_data
import logging

load_dotenv()

client = AsyncOpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), api_key="ollama")
model = OpenAIModel(model_name=os.getenv("LLM_MODEL", "deepseek-coder:16b"), provider=OpenAIProvider(base_url=os.getenv("OLLAMA_URL") + "/v1"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase: 'supabase.Client'

analysis_agent = Agent(
    model,
    system_prompt="""
Analyze trends/defects. Use tools, return ONLY markdown bullets. Fetch data via query_press20_data if needed.
Knowledge base (expandable):
- short shot: Increase injection pressure/temperature.
- flash: Decrease pressure/check mold.
Examples:
- "trends in ActNozzleTemp": - **Trend:** Increased 5% over shots.
- "fix short shot": - **Suggestion:** Increase injection pressure/temperature.
""",
    deps_type=Deps,
    retries=3
)

@analysis_agent.tool
@observe()
async def analyze_trends(ctx: RunContext[Deps], query: str) -> str:
    sql = "SELECT shot_num, ActNozzleTemp FROM press20_data ORDER BY shot_num LIMIT 100"  # Dynamic SQL generation
    data = await query_press20_data(ctx, sql)
    analysis_prompt = f"Summarize trends in markdown bullets: {data}"
    response = await client.chat.completions.create(model=os.getenv("LLM_MODEL"), messages=[{"role": "system", "content": analysis_prompt}])
    return response.choices[0].message.content

@analysis_agent.tool
@observe()
async def suggest_fix(ctx: RunContext[Deps], defect: str) -> str:
    knowledge = {
        "short shot": "- **Suggestion:** Increase injection pressure or temperature.",
        "flash": "- **Suggestion:** Decrease pressure or check mold alignment.",
        "warp": "- **Suggestion:** Adjust cooling time or material.",
        # Expand based on domain knowledge/docs
    }
    return knowledge.get(defect.lower(), "- **No suggestion found.**")

@analysis_agent.tool
@observe()
async def think(ctx: RunContext[Deps], thought: str) -> str:
    # Internal chain-of-thought: Use for reasoning, verification, planning. No visible output.
    return ""
