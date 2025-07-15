from fastapi import FastAPI
from models import QueryInput
from agents.master_agent import master_agent, Deps
import httpx
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langfuse import observe, get_client
from typing import Optional
import logging

load_dotenv()

app = FastAPI(title="H&H AI Assistant API")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

langfuse = get_client()

#### DEBUG #####
DEBUG = True

async def handle_predefined_query(query: str) -> Optional[str]:
    query_normalized = query.strip().lower()
    predefined_responses = {
        "who are you": "I am the H&H AI Assistant, built to help with injection molding queries for H&H Molds Inc. I can answer questions about Press20 data, perform calculations, and retrieve information from documents.",
        "what can you do": """I can:
- Answer questions about Press20 data (e.g., "list failed shots from press20").
- Perform numerical calculations (e.g., "2 + 2").
- Retrieve and summarize document information (e.g., "summarize the scope meeting").
Ask away or check the docs at https://dawson-b23.github.io/HHDocs/!""",
        "help": """Need help? I can:
- Query Press20 data (e.g., "list failed shots from press20").
- Calculate numerical expressions (e.g., "5 * 3").
- Search documents (e.g., "summarize the scope meeting").
Contact Dawson at intern@hhmoldsinc.com or 832-977-3004 for issues. Docs: https://dawson-b23.github.io/HHDocs/."""
    }
    if query_normalized in predefined_responses:
        logger.debug(f"Matched predefined query: {query_normalized}")
        return predefined_responses[query_normalized]
    logger.debug(f"No predefined response matched for query: {query_normalized}")
    return None

@app.post("/rag-docs")
@observe()
async def handle_query(query: QueryInput):
    if not query.chatInput.strip():
        logger.error("Empty query received in FastAPI")
        return "Error: Empty query provided."
    predefined_response = await handle_predefined_query(query.chatInput)
    if predefined_response:
        logger.debug(f"Returning predefined response for query: {query.chatInput}")
        return predefined_response
    async with httpx.AsyncClient(timeout=float(os.getenv("HTTP_TIMEOUT", 30.0))) as client:
        try:
            deps = Deps(client=client, supabase_key=SUPABASE_KEY, supabase=supabase)
            result = await master_agent.run(query.chatInput, deps=deps)
            response_text = result.output if hasattr(result, "output") and result.output else "No response from model"
            if os.getenv("DEBUG", "False").lower() == "true":
                logger.debug(f"FastAPI response: {response_text}")
            if response_text.startswith("{") and response_text.endswith("}"):
                logger.warning(f"Unexpected JSON response from master_agent: {response_text}")
                return "Error: Invalid response format from agent."
            return response_text
        except httpx.TimeoutException:
            logger.error("Timeout in FastAPI query", exc_info=True)
            return "Error: Request timed out."
        except Exception as e:
            logger.error(f"Error in FastAPI: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

@app.get("/health")
async def health_check():
    return {"status": "Backend is running"}
