from fastapi import FastAPI
from models import QueryInput
from agents.master_agent import master_agent, Deps
import httpx
import os
from dotenv import load_dotenv
from supabase import create_client
from langfuse import observe, get_client
import logging

load_dotenv()

app = FastAPI(title="H&H AI Assistant API")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.post("/query")
@observe()
async def handle_query(query: QueryInput):
    if not query.chatInput.strip():
        return "- **Error:** Empty query provided."
    async with httpx.AsyncClient() as client:
        try:
            deps = Deps(client=client, supabase=supabase)
            result = await master_agent.run(query.chatInput, deps=deps)
            return result if result else "- **No response from agent.**"
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"- **Error:** {str(e)}"

@app.get("/health")
async def health_check():
    return {"status": "Backend running"}
