from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
import sys
sys.path.append("..")
from . import model
from langfuse import observe, get_client
import httpx
from database import query_press20_data
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str

press20_agent = Agent(
    model,
    system_prompt="""
    You are a Press20 data assistant for H&H. Query the press20_data table for fields like shot_num, overallpassfail, actnozzletemp, etc., based on user input. Provide accurate data in a structured format. Return the query results or 'No data found in press20_data.' if no data is found. Do not include '[FINAL ANSWER]' or '[FIXED]' in responses. Ensure all queries use 'press20_data.overallpassfail' for the overallpassfail field.
    """,
    deps_type=Deps,
    retries=2
)

@press20_agent.tool
async def press20_query(ctx: RunContext[Deps], query: str) -> str:
    langfuse.update_current_trace(metadata={"tool": "press20_query", "query": query})
    try:
        if not query:
            langfuse.update_current_trace(metadata={"tool": "press20_query", "status": "success", "result": "empty query"})
            return "No data found in press20_data."
        modified_query = query.replace("overallpassfail", "press20_data.overallpassfail")
        data = await query_press20_data(modified_query)
        if not data:
            langfuse.update_current_trace(metadata={"tool": "press20_query", "status": "success", "result": "no data"})
            return "No data found in press20_data."
        result = "\n".join([str(row) for row in data])
        langfuse.update_current_trace(metadata={"tool": "press20_query", "status": "success", "result": result})
        return result
    except Exception as e:
        logger.error(f"Error in press20_query: {str(e)}", exc_info=True)
        langfuse.update_current_trace(metadata={"tool": "press20_query", "status": "error", "error": str(e)})
        return f"Error querying press20_data: {str(e)}"
