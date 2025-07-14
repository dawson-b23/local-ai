from agents.master_agent import create_master_agent, Deps
from supabase import create_client
import httpx
import asyncio
from supabase import create_client

SUPABASE_URL="http://localhost:8000"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIiwiaWF0IjoxNzUwMzE2NDAwLCJleHAiOjE5MDgwODI4MDB9.4HKpi7J5zX_eavvlWmvbE0U54v66nldDcS3Kqt87NUI"
OLLAMA_URL="http://localhost:11434"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

async def test():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    async with httpx.AsyncClient() as client:
        deps = Deps(client=client, supabase_key=SUPABASE_KEY, supabase=supabase)
        agent = await create_master_agent()
        result = await agent.run("please generate a detailed summary from the scope meeting", deps=deps)
        print("\n \n \n ******* RESULTS ******* \n")
        print(result.output)

asyncio.run(test())

