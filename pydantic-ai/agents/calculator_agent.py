from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import numexpr
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

client = AsyncOpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), api_key="ollama")
model = OpenAIModel(model_name=os.getenv("LLM_MODEL", "deepseek-coder:16b"), provider=OpenAIProvider(base_url=os.getenv("OLLAMA_URL") + "/v1"))

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase: 'supabase.Client'

calculator_agent = Agent(
    model,
    system_prompt="""
You are a calculator for numerical expressions. Use 'calculate' tool and return ONLY markdown result/error. No explanations.
Examples:
- "2 + 2" -> **Result:** 4
- "1/0" -> **Error:** Division by zero
""",
    deps_type=Deps,
    retries=3
)

@calculator_agent.tool
@observe()
async def calculate(ctx: RunContext[Deps], expression: str) -> str:
    if not expression.strip():
        return "**Error:** Empty expression."
    try:
        result = numexpr.evaluate(expression.strip())
        return f"**Result:** {result}"
    except Exception as e:
        return f"**Error:** {str(e)}"
