from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import httpx
from database import setup_vector_store
import os
from dotenv import load_dotenv
import json

load_dotenv()


client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)
#model = OpenAIModel("llama3.1:8b", openai_client=client)
model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str
    supabase: 'supabase.Client'

rag_agent = Agent(
    model,
    system_prompt="""You are a document query assistant for H&H. Use the provided tools to search and retrieve information from documents or tabular data. Start with a RAG search for text-based queries. For tabular data queries (e.g., sums, averages), use the Query Document Rows tool. Use List Documents to check available documents and Get File Contents to extract text for a specific file_id. If no relevant data is found, return 'No relevant data found.'""",
    #system_prompt="""
    #You are a document query assistant for H&H. Search and retrieve relevant information 
    #from documents or chat history based on user input. Use the vector store for document 
    #searches and query the documents, document_rows, or chat_history tables as needed. 
    #Return the retrieved content or an error message if no relevant data is found. 
    #""",
    #system_prompt="""You are a document query assistant for H&H. Search and retrieve relevant 
    #information from documents using the vector store. If no relevant data is found, return 
    #'No relevant documents found.'""",
    deps_type=Deps,
    retries=2
)

@rag_agent.tool
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    try:
        vector_store = await setup_vector_store()
        results = await vector_store.as_retriever().ainvoke(query)
        if not results:
            return "No relevant documents found."
        return "\n".join([str(doc.page_content) for doc in results])
    except Exception as e:
        return f"Error searching documents: {str(e)}"


@rag_agent.tool
async def list_documents(ctx: RunContext[Deps]) -> str:
    try:
        response = await asyncio.to_thread(
            ctx.deps.supabase.table("document_metadata").select("*").execute
        )
        if not response.data:
            return "No documents found in document_metadata."
        return json.dumps(response.data, indent=2)
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@rag_agent.tool
async def get_file_contents(ctx: RunContext[Deps], file_id: str) -> str:
    try:
        response = await asyncio.to_thread(
            ctx.deps.supabase.rpc(
                "get_file_contents",
                {"file_id": file_id}
            ).execute
        )
        if not response.data:
            return f"No text found for file_id: {file_id}"
        return response.data[0].get("document_text", "No text content")
    except Exception as e:
        return f"Error getting file contents: {str(e)}"


@rag_agent.tool
async def query_document_rows(ctx: RunContext[Deps], sql_query: str) -> str:
    try:
        response = await asyncio.to_thread(
            ctx.deps.supabase.rpc(
                "query_document_rows",
                {"query_text": sql_query}
            ).execute
        )
        if not response.data:
            return "No data found in document_rows."
        return json.dumps(response.data, indent=2)
    except Exception as e:
        return f"Error querying document rows: {str(e)}"
