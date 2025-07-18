# database.py (Async wrappers, added vector setup with schema awareness)
from supabase import Client, create_client
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import uuid
import asyncio
import os
from dotenv import load_dotenv
from langfuse import observe
import logging
from langfuse import get_client, observe
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

langfuse = get_client()

SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@observe()
async def setup_vector_store():
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=os.getenv("OLLAMA_URL"))
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        return vector_store
    except Exception as e:
        print('\n---------- error in vector store -------')
        return None

@observe()
async def query_document_rows(query: str):
    try:
        response = await asyncio.to_thread(
            supabase.rpc("query_document_rows", {"query_text": query}).execute
        )
        return response.output
    except Exception as e:
        logger.error(f"Error in query_document_rows: {str(e)}", exc_info=True)
        return []

@observe()
async def query_documents(query: str):
    try:
        response = await asyncio.to_thread(
            supabase.table("documents").select("*").text_search("content", query).execute
        )
        return response.output
    except Exception as e:
        logger.error(f"Error in query_documents: {str(e)}", exc_info=True)
        return []

@observe()
async def save_chat_message(user_id: str, session_id: str, message: dict):
    try:
        timestamp = datetime.now().isoformat()
        data = {
            "userid": user_id,
            "sessionid": session_id,
            "message": message,
            "timestamp": timestamp  # Add timestamp to each message
        }
        await asyncio.to_thread(supabase.table("chat_history").insert(data).execute)
    except Exception as e:
        logger.error(f"Error saving chat message: {str(e)}", exc_info=True)
        pass

@observe()
async def get_chat_history(session_id: str):
    try:
        response = await asyncio.to_thread(
            supabase.table("chat_history").select("message").eq("sessionid", session_id).order("id").execute
        )
        return [row["message"] for row in response.output]
    except Exception:
        print('\n---------- error in get chat history -------')
        return []

@observe()
async def fetch_sessions(user_id: str):
    try:
        response = await asyncio.to_thread(
            supabase.table("chat_sessions").select("sessionid, title, timestamp").eq("userid", user_id).order("timestamp", desc=True).execute
        )
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error fetching sessions: {str(e)}")
        return []

@observe()
async def create_session(session_id: str, user_id: str, title: str = "New Chat"):
    timestamp = datetime.now().isoformat()
    data = {
        "sessionid": session_id,
        "userid": user_id,
        "title": title,
        "timestamp": timestamp
    }
    try:
        await asyncio.to_thread(supabase.table("chat_sessions").insert(data).execute)
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")

@observe()
async def update_session_title(session_id: str, title: str):
    try:
        await asyncio.to_thread(supabase.table("chat_sessions").update({"title": title}).eq("sessionid", session_id).execute)
    except Exception as e:
        logger.error(f"Error updating session title: {str(e)}")
