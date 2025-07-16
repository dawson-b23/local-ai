from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import httpx
from database import setup_vector_store
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import json
import asyncio
from langfuse import get_client, observe
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)
#model = OpenAIModel("llama3.1:8b", openai_client=client)
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


rag_agent = Agent(
    model,
    system_prompt="""
You are a document query assistant for H&H Molds Inc. Your task is to answer queries using documents or tabular data, returning results in strict markdown format (e.g., bullet points, headers). Use the provided tools based on the query type. Follow these rules:
1. For text-based queries (e.g., "summarize", "what is"), use `rag_search` to retrieve relevant documents and return content as markdown.
2. For queries about available documents (e.g., "what documents are available"), use `list_documents` and return a markdown list.
3. For specific file text (e.g., "get text from file_id xyz"), use `get_file_contents` and return content as markdown.
4. For tabular data queries (e.g., "average value from dataset xyz"), use `query_document_rows` with a SQL query and return results as markdown.
5. Return ONLY the tool’s output in markdown format, using bullet points or tables for lists.
6. If no data is found, return a markdown-formatted error message (e.g., "- No relevant documents found").
7. Do NOT fabricate answers or add explanations.
8. If the query is empty, return "- Error: Empty query provided."

## Examples
- Query: "Summarize the scope meeting"
  - Action: Use `rag_search`
  - Output: - Summary: [Document content]
- Query: "What documents are available"
  - Action: Use `list_documents`
  - Output: 
    - id: file1, title: Scope Meeting
    - id: file2, title: Sales Data
- Query: "Get text from file_id xyz"
  - Action: Use `get_file_contents`
  - Output: - File content: [Text]
- Query: "Average value from dataset xyz"
  - Action: Use `query_document_rows`
  - Output: - Average: 123.4
""",
    deps_type=Deps,
    retries=3
)

@rag_agent.tool
@observe()
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    if not query.strip():
        logger.error("Empty query received in rag_search")
        return "Error: Empty query provided."
    try:
        vector_store = await setup_vector_store()
        results = await vector_store.as_retriever().ainvoke(query)
        if not results:
            return "No relevant documents found."
        return "\n".join([f"- {str(doc.page_content)}" for doc in results])
    except Exception as e:
        logger.error(f"Error in rag_search: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@rag_agent.tool
@observe()
async def list_documents(ctx: RunContext[Deps]) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.table("document_metadata").select("*").execute
        )
        if not response.output:
            return "No documents found in document_metadata."
        return "\n".join([f"- id: {row['id']}, title: {row['title']}" for row in response.output])
    except Exception as e:
        logger.error(f"Error in list_documents: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@rag_agent.tool
@observe()
async def get_file_contents(ctx: RunContext[Deps], file_id: str) -> str:
    if not file_id.strip():
        logger.error("Empty file_id received in get_file_contents")
        return "Error: Empty file_id provided."
    try:
        response = await asyncio.to_thread(
            supabase.rpc("get_file_contents", {"file_id": file_id}).execute
        )
        if not response.output:
            return f"No text found for file_id: {file_id}"
        return f"- File content: {response.output[0].get('document_text', 'No text content')}"
    except Exception as e:
        logger.error(f"Error in get_file_contents: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@rag_agent.tool
@observe()
async def query_document_rows(ctx: RunContext[Deps], sql_query: str) -> str:
    if not sql_query.strip():
        logger.error("Empty SQL query received in query_document_rows")
        return "Error: Empty SQL query provided."
    try:
        response = await asyncio.to_thread(
            supabase.rpc("query_document_rows", {"query_text": sql_query}).execute
        )
        if not response.output:
            return "No data found in document_rows."
        return "\n".join([f"- {json.dumps(row)}" for row in response.output])
    except Exception as e:
        logger.error(f"Error in query_document_rows: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

"""

@rag_agent.tool
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    try:
        vector_store = await setup_vector_store()
        results = await vector_store.as_retriever().ainvoke(query)
        if not results:
            return "No relevant documents found."
        return "\n".join([str(doc.page_content) for doc in results])
    except Exception as e:
        return f"Error in RAG search: {str(e)}"


@rag_agent.tool
@observe()
async def list_documents(ctx: RunContext[Deps]) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.table("document_metadata").select("*").execute
        )
        if not response.output:
            return "No documents found in document_metadata."
        return json.dumps(response.output, indent=2)
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@rag_agent.tool
@observe()
async def get_file_contents(ctx: RunContext[Deps], file_id: str) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.rpc(
                "get_file_contents",
                {"file_id": file_id}
            ).execute
        )
        if not response.output:
            return f"No text found for file_id: {file_id}"
        return response.output[0].get("document_text", "No text content")
    except Exception as e:
        return f"Error getting file contents: {str(e)}"


@rag_agent.tool
@observe()
async def query_document_rows(ctx: RunContext[Deps], sql_query: str) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.rpc(
                "query_document_rows",
                {"query_text": sql_query}
            ).execute
        )
        if not response.output:
            return "No data found in document_rows."
        return json.dumps(response.output, indent=2)
    except Exception as e:
        return f"Error querying document rows: {str(e)}"
    """
