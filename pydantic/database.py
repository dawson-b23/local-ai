from supabase import create_client, Client
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import asyncio
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

langfuse = get_client()

SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
        return None  # Fallback to None to prevent crashes

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

#@observe()
async def save_chat_message(user_id: str, session_id: str, message: dict):
    try:
        data = {
            "userid": user_id,
            "sessionid": session_id,
            "message": message
        }
        await asyncio.to_thread(supabase.table("chat_history").insert(data).execute)
    except Exception as e:
        logger.error(f"Error saving chat message: {str(e)}", exc_info=True)
        pass  # Silently fail to avoid blocking the app

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
async def fetch_sessions():
    try:
        response = await asyncio.to_thread(supabase.table("chat_history").select("sessionid", distinct=True).execute)
        return [row["sessionid"] for row in response.output]
    except Exception:
        print('\n---------- error in fetch sessions -------')
