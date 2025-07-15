from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import httpx
import os
from dotenv import load_dotenv
import json
import asyncio
from langfuse import observe, get_client
from supabase import create_client, Client
from datetime import datetime

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)

model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
langfuse = get_client()


# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str
    supabase: 'supabase.Client'

press20_agent = Agent(
    model,
    system_prompt="""You are a Press20 data query assistant for H&H. Current date: {}.
You help answer questions from data in the press20_data table using the query_press20_data tool. Follow these rules:
1. Use query_press20_data to run full SQL queries (e.g., SELECT, WHERE, GROUP BY) on press20_data.
2. For specific shots (e.g., 'shot_num 123'), filter and order by shot_num.
3. For pass/fail queries (e.g., 'failed shots' or 'out of spec shots'), filter by overallpassfail, bottompassfail, or toppassfail depending on user input.
4. For aggregations (e.g., 'average ActNozzleTemp'), use SQL functions like AVG, MIN, MAX, or COUNT.
5. Never include dataset_id or id in queries.
6. Run the SQL query and return the result. If the query fails, return the query, error, and'No data found in press20_data.'
7. Do not fabricate answers if no data is found.
8. Format results as markdown for user readability.
9. All column names are all lowercase, keep that in mind when querying.
11. When running any queries that involve the shot_num field, ie "list shot_num for FAIL in press20_data" make sure to always use SELECT DISTINCT and order by shot_num
""".format(datetime.now().strftime("%Y-%m-%d")),
    deps_type=Deps,
    retries=3
)

@press20_agent.tool
@observe()
async def query_press20_data(ctx: RunContext[Deps], sql_query: str) -> str:
    try:
        print(f"Executing SQL query: {sql_query}")  # Debug print
        response = await asyncio.to_thread(
            supabase.rpc(
                "query_press20_data",
                {"query_text": sql_query}
            ).execute
        )
        if not response.data:
            return "No data found in press20_data."
        result_lines = []
        for row in response.data:
            # Extract JSON content from the 'result' column
            json_data = row['result']
            formatted_row = ", ".join([f"{key}: {value}" for key, value in json_data.items() if value is not None])
            result_lines.append(formatted_row)
        return "\n".join(result_lines) if result_lines else "No data found in press20_data."
    except Exception as e:
        print(f"Error in query_press20_data: {str(e)}")  # Debug print
        return f"Error querying press20_data: {str(e)}"

"""
async def query_press20_data(ctx: RunContext[Deps], sql_query: str) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.rpc(
                "query_press20_data",
                {"query_text": sql_query}
            ).execute
        )
        if not response.data:
            return "No data found in press20_data."
        result_lines = []
        for row in response.data:
            formatted_row = ", ".join([f"{key}: {value}" for key, value in row.items() if value is not None])
            result_lines.append(formatted_row)
        return "\n".join(result_lines) if result_lines else "No data found in press20_data."
    except Exception as e:
        print(f"Error in query_press20_data: {str(e)}")  # Debug print
        return f"Error querying press20_data: {str(e)}"
"""
