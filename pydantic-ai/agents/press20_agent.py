from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx
import asyncio
from supabase import Client
from datetime import datetime
import logging
from database import supabase  # Import global supabase
from .sql_agent import sql_agent

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase: 'supabase.Client'

@observe()
async def run(query: str, deps) -> str:
    # Directly call sql_agent to get SQL
    sql_result = await sql_agent.run(query, deps=deps)
    sql_query = sql_result.output if hasattr(sql_result, 'output') else str(sql_result) if sql_result else ""

    if not sql_query.strip():
        return "- **Error:** No SQL generated or empty query."

    # Execute SQL if exists
    try:
        response = await asyncio.to_thread(supabase.rpc("query_press20_data", {"query_text": sql_query}).execute)
        if not response.data:
            return "- **No data.**"
        content = []
        for row in response.data:
            row_data = row['result']
            formatted = "\n".join([f"- **{key}**: {value}" for key, value in row_data.items() if value is not None])
            content.append(formatted)
        return "\n\n".join(content) if content else "- **No data.**"
    except Exception as e:
        return f"- **Error executing SQL:** {str(e)}"
