from fastapi import FastAPI
from models import QueryInput
from agents.master_agent import master_agent, Deps
import httpx
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langfuse import observe, get_client

load_dotenv()

app = FastAPI(title="H&H AI Assistant API")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

langfuse = get_client()

#### DEBUG #####
DEBUG = True

@app.post("/rag-docs")
@observe()
async def handle_query(query: QueryInput):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            deps = Deps(client=client, supabase_key=SUPABASE_KEY, supabase=supabase)
            result = await master_agent.run(query.chatInput, deps=deps)
            return result.output if result.output else "No response from model"
            if DEBUG is True:
                print(f"FastAPI response: {response_text}")  # Debug print
            #return {"response": result.output if result.output else "No response from model"}
        except Exception as e:
            langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
            error_msg = f"Error in FastAPI: {str(e)}"
            print(error_msg)
            return error_msg
            #return {"response": f"Error processing query: {str(e)}. Please check if the backend is running."}

@app.get("/health")
async def health_check():
    return {"status": "Backend is running"}
