from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
import sys
sys.path.append("..")
from . import model
from langfuse import observe, get_client
import httpx
from database import setup_vector_store, query_document_rows, get_chat_history, query_documents
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

rag_agent = Agent(
    model,
    system_prompt="""
    You are a document query assistant for H&H. Search and retrieve relevant information from documents or chat history based on user input. Use the vector store for document searches and query the documents, document_rows, or chat_history tables as needed. Return the retrieved content or an error message if no relevant data is found. Do not include '[FINAL ANSWER]' or '[FIXED]' in responses.
    """,
    deps_type=Deps,
    retries=2
)

@rag_agent.tool
@observe()
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    try:
        vector_store = await setup_vector_store()
        results = await vector_store.as_retriever().ainvoke(query)
        if not results:
            return "No relevant documents found."
        return "\n".join([str(doc.page_content) for doc in results])
    except Exception as e:
        logger.error(f"Error in rag_search: {str(e)}", exc_info=True)
        return "Error searching documents."

@rag_agent.tool
@observe()
async def document_rows_query(ctx: RunContext[Deps], query: str) -> str:
    try:
        data = await query_document_rows(query)
        if not data:
            return "No data found in document_rows."
        return "\n".join([str(row) for row in data])
    except Exception as e:
        logger.error(f"Error in document_rows_query: {str(e)}", exc_info=True)
        return "Error querying document_rows."

@rag_agent.tool
@observe()
async def chat_history_query(ctx: RunContext[Deps], session_id: str) -> str:
    try:
        history = await get_chat_history(session_id)
        if not history:
            return "No chat history found."
        return "\n".join([str(msg) for msg in history])
    except Exception as e:
        logger.error(f"Error in chat_history_query: {str(e)}", exc_info=True)
        return "Error querying chat history."

@rag_agent.tool
@observe()
async def documents_query(ctx: RunContext[Deps], query: str) -> str:
    try:
        data = await query_documents(query)
        if not data:
            return "No data found in documents table."
        return "\n".join([str(row) for row in data])
    except Exception as e:
        logger.error(f"Error in documents_query: {str(e)}", exc_info=True)
        return "Error querying documents table."
