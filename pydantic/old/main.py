from fastapi import FastAPI, HTTPException
from models import QueryInput
from agents.master_agent import create_master_agent, Deps
from ingestion import process_file
from datetime import datetime
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx
import logging
import json
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="H&H AI Assistant API")
langfuse = get_client()

class QueryInput(BaseModel):
    chatInput: str
    sessionId: str

@observe()
async def handle_query(query: QueryInput):
    langfuse.update_current_trace(metadata={"query": query.chatInput, "session": query.sessionId})
    async with httpx.AsyncClient() as client:
        try:
            agent = create_master_agent()
            deps = Deps(client=client, supabase_key=os.getenv("SUPABASE_KEY"))
            result = await agent.run(query.chatInput, deps=deps)
            response_data = str(result.data) if result.data else "No response from model"
            langfuse.update_current_trace(metadata={"status": "success", "response": response_data})
            return {"response": response_data}
        except Exception as e:
            logger.error(f"Error in handle_query: {str(e)}", exc_info=True)
            langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query at {datetime.now()}: {str(e)}"
            )

app.add_api_route("/rag-docs", handle_query, methods=["POST"])
