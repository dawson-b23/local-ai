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
async def create_tables():
    #langfuse.update_current_trace(metadata={"action": "create_tables"})
    #langfuse.update_current_trace(metadata={"status": "skipped", "reason": "Handled by Supabase schema"})
    pass

@observe()
async def insert_metadata(metadata):
    #langfuse.update_current_trace(metadata={"metadata_id": metadata.id})
    try:
        data = {
            "id": metadata.id,
            "title": metadata.title,
            "data_schema": metadata.data_schema,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None
        }
        await asyncio.to_thread(supabase.table("document_metadata").upsert(data).execute)
        #langfuse.update_current_trace(metadata={"status": "success", "metadata_id": metadata.id})
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in insert metadata: {str(e)}", exc_info=True)

@observe()
async def insert_press20_data(data):
    #langfuse.update_current_trace(metadata={"shot_num": data.dict().get("shot_num")})
    try:
        data_dict = data.dict(exclude_unset=True)
        await asyncio.to_thread(supabase.table("press20_data").insert(data_dict).execute)
        #langfuse.update_current_trace(metadata={"status": "success", "shot_num": data_dict.get("shot_num")})
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in insert press20 data: {str(e)}", exc_info=True)

@observe()
async def insert_document_row(dataset_id: str, row_data: dict):
    #langfuse.update_current_trace(metadata={"dataset_id": dataset_id})
    try:
        data = {"dataset_id": dataset_id, "row_data": row_data}
        await asyncio.to_thread(supabase.table("document_rows").insert(data).execute)
        #langfuse.update_current_trace(metadata={"status": "success", "dataset_id": dataset_id})
    except Exception as e:
        logger.error(f"** ** ** ERROR ** ** ** in insert document row: {str(e)}", exc_info=True)
        ########langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})

@observe()
async def query_press20_data(query: str):
    try:
        if not query:
            langfuse.update_current_trace(metadata={"function": "query_press20_data", "status": "success", "result": "empty query"})
            return []
        modified_query = query.replace("overallpassfail", "press20_data.overallpassfail")
        response = await asyncio.to_thread(
            supabase.rpc("query_press20_data", {"query_text": modified_query}).execute
        )
        return response.output
    except Exception as e:
        logger.error(f"Error in query_press20_data: {str(e)}", exc_info=True)
        return []

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
        logger.error(f"Error in setup_vector_store: {str(e)}", exc_info=True)
        raise

@observe()
async def save_chat_message(user_id: str, session_id: str, message: dict):
    try:
        data = {
            "userid": user_id,
            "sessionid": session_id,
            "message": message
        }
        await asyncio.to_thread(supabase.table("chat_history").insert(data).execute)
    except Exception as e:
        logger.error(f"Error in save_chat_message: {str(e)}", exc_info=True)


@observe()
async def get_chat_history(session_id: str):
    try:
        response = await asyncio.to_thread(
            supabase.table("chat_history").select("message").eq("sessionid", session_id).order("id").execute
        )
        return [row["message"] for row in response.output]
    except Exception as e:
        logger.error(f"Error in get_chat_history: {str(e)}", exc_info=True)
        return []
