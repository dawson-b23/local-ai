from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from .rag_agent import rag_agent
from .calculator_agent import calculator_agent
import httpx
from datetime import datetime
import os
from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)

model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str

master_agent = Agent(
    model,
    system_prompt="""
    You are an orchestrator agent for H&H. Current date: {}.
    - Use the Think tool to verify your steps before and after selecting a tool.
    - If the query contains mathematical operators ('+', '-', '*', '/') or a question regarding math, route to the Calculator agent.
    - If the query is about documents or general information, route to the General_RAG agent.
    - If the query is vague (e.g., contains 'hello', 'what can you do', 'hi', or is empty), do not call any tools and return: 'I can help with Press20 data queries, document searches, (web search) or numerical calculations. What would you like to do?'
    - If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query. Contact Dawson for further support.'
    - For vague queries, do not call any tools; return the response directly.
    """.format(datetime.now().strftime("%Y-%m-%d")),
    #TODO: add this  - If the query contains 'press20', 'overallpassfail', or 'press20_data', route to the Press20 agent. Do NOT use this otherwise.

#    system_prompt="""You are an assistant for H&H. Current date: {}.
#- Route document or general queries to the General RAG tool.
#- For vague queries (e.g., 'what can you do'), respond: 'I can help with document searches or general queries. What would you like to do?'
#- If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query.'
#""".format(datetime.now().strftime("%Y-%m-%d")),
    deps_type=Deps,
    retries=2
)

@master_agent.tool
@observe()
async def general_rag(ctx: RunContext[Deps], query: str) -> str:
    try:
        result = await rag_agent.run(query, deps=ctx.deps)
        #return result.data if result.data else "No response from RAG agent"
    except Exception as e:
        return f"Error in calling rag_agent tool: {str(e)}"

@master_agent.tool
@observe()
async def calculator(ctx: RunContext[Deps], expression: str) -> str:
    try:
        result = await calculator_agent.run(expression, deps=ctx.deps)
    except Exception as e: 
        return f"Error in calling calculator_agent tool: {str(e)}"

@master_agent.tool
@observe()
async def think(ctx: RunContext[Deps], query: str, selected_tool: str) -> str:
    try:
        # Simulate thinking by prompting the model to reflect on the chosen tool and query
        think_prompt = f"""You selected the tool '{selected_tool}' for the query: '{query}'.
        Reflect on whether this is the correct tool. If not, suggest an alternative approach.
        Return a concise explanation of your reasoning."""
        response = await client.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "You are a reflective assistant. Analyze the tool selection and provide reasoning."},
                {"role": "user", "content": think_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in think tool: {str(e)}"
