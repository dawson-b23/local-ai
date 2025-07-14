from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
import httpx
import os
from dotenv import load_dotenv
import json
import asyncio
from langfuse import observe, get_client

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)
model = OpenAIProvider(base_url=os.getenv("OLLAMA_URL")).create_model("llama3.1:8b")
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
1. Use query_press20_data to run SQL queries on press20_data for all queries.
2. For specific shots (e.g., 'shot_num 123'), filter by shot_num and dataset_id (from document_metadata).
3. For pass/fail queries (e.g., 'failed shots'), filter by overallPassFail, bottomPassFail, or topPassFail.
4. For aggregations (e.g., 'average ActNozzleTemp'), use SQL functions like AVG, MIN, MAX, or COUNT.
5. Always include dataset_id in queries (e.g., '/data/shared/press20_file.csv') unless unspecified.
6. Run the SQL query and return the result. If the query fails, return the error or 'No data found in press20_data.'
7. Do not fabricate answers if no data is found.
8. Format results as markdown for user readability.
""".format(datetime.now().strftime("%Y-%m-%d")),
    deps_type=Deps,
    retries=3
)

@press20_agent.tool
@observe()
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
        return f"Error querying press20_data: {str(e)}"
