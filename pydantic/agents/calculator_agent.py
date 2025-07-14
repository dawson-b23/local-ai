
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from langfuse import observe, get_client
import httpx
import numexpr
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)

model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str
    supabase: 'supabase.Client'

calculator_agent = Agent(
    model,
    system_prompt="""
    You are a calculator assistant for H&H. Perform numerical calculations based on user input. Return the result or an error message if the calculation is invalid.
    """,    
    deps_type=Deps,
    retries=2
)

@calculator_agent.tool
@observe()
async def calculate(ctx: RunContext[Deps], expression: str) -> str:
    #langfuse.update_current_trace(metadata={"expression": expression})
    try:
        result = numexpr.evaluate(expression)
        #langfuse.update_current_trace(metadata={"status": "success", "result": str(result)})
        return f"{str(result)}"
    except Exception as e:
        logger.error(f"** ** ** ERROR ** ** ** in calculator_tool: {str(e)}", exc_info=True)
        langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        return "Invalid calculation."

