
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
You are a calculator assistant for H&H Molds Inc. Your sole task is to perform numerical calculations based on user input using the `calculate` tool. Follow these rules:
1. Only process valid numerical expressions (e.g., "2+2", "5*3/2").
2. Use the `calculate` tool to evaluate the expression.
3. Return ONLY the result of the calculation or an error message if invalid.
4. Do NOT generate explanations or additional text.
5. Format all outputs (results or errors) in markdown.
6. If the input is empty or invalid, return an error message.

## Examples
- Input: "2 + 2"
  - Output: 4
- Input: "1/0"
  - Output: Error: Division by zero
- Input: ""
  - Output: Error: Empty expression provided
""",
    deps_type=Deps,
    retries=2
)

@calculator_agent.tool
@observe()
async def calculate(ctx: RunContext[Deps], expression: str) -> str:
    if not expression.strip():
        logger.error("Empty expression received in calculate")
        return "Error: Empty expression provided."
    try:
        result = numexpr.evaluate(expression.strip())
        return str(result)
    except Exception as e:
        logger.error(f"Error in calculate: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

"""
@calculator_agent.tool
@observe()
async def calculate(ctx: RunContext[Deps], expression: str) -> str:
    #langfuse.update_current_trace(metadata={"expression": expression})
    try:
        result = numexpr.evaluate(expression)
        #langfuse.update_current_trace(metadata={"status": "success", "result": str(result)})
        return str(result)
    except Exception as e:
        logger.error(f"** ** ** ERROR ** ** ** in calculator_tool: {str(e)}", exc_info=True)
        langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        return "Invalid calculation."
"""
