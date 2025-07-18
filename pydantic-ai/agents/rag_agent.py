from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx
from database import setup_vector_store, supabase
import asyncio
from supabase import Client
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
    supabase: Client

rag_agent = Agent(
    model,
    system_prompt="""
Handle document queries (non-Press20). Use tools, return ONLY markdown (bullets/tables). Start with rag_search unless SQL needed.
- Summaries: rag_search
- List docs: list_documents
- File text: get_file_contents
- Tabular: query_document_rows (SQL)
If no answer: - **No answer found.**
""",
    deps_type=Deps,
    retries=3
)

@rag_agent.tool
@observe()
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    if not query.strip():
        return "- **Error:** Empty query."
    vector_store = await setup_vector_store()  # Uses global inside
    results = await vector_store.as_retriever().ainvoke(query)
    if not results:
        return "- **No results.**"
    return "\n\n".join([f"- **Document:** {doc.page_content}" for doc in results])

@rag_agent.tool
@observe()
async def list_documents(ctx: RunContext[Deps]) -> str:
    response = await asyncio.to_thread(ctx.deps.supabase.table("document_metadata").select("*").execute)
    if not response.data:
        return "- **No documents.**"
    return "\n".join([f"- {row['title']} (ID: {row['id']}, Schema: {row.get('data_data_schema', 'N/A')})" for row in response.data])

@rag_agent.tool
@observe()
async def get_file_contents(ctx: RunContext[Deps], file_id: str) -> str:
    if not file_id.strip():
        return "- **Error:** Empty file ID."
    response = await asyncio.to_thread(ctx.deps.supabase.rpc("get_file_contents", {"file_id": file_id}).execute)
    if not response.data:
        return f"- **No content for ID:** {file_id}"
    return f"- **Content:** {response.data[0].get('document_text', 'None')}"

@rag_agent.tool
@observe()
async def query_document_rows(ctx: RunContext[Deps], sql_query: str) -> str:
    if not sql_query.strip():
        return "- **Error:** Empty SQL."
    response = await asyncio.to_thread(ctx.deps.supabase.rpc("query_document_rows", {"query_text": sql_query}).execute)
    if not response.data:
        return "- **No data.**"
    table = "| " + " | ".join(response.data[0].keys()) + " |\n| " + " --- |" * len(response.data[0]) + "\n"
    for row in response.data:
        table += "| " + " | ".join(str(v) for v in row.values()) + " |\n"
    return table
