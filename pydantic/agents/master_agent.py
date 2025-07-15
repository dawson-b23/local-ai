from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from .rag_agent import rag_agent
from .calculator_agent import calculator_agent
from .press20_agent import press20_agent
from supabase import create_client, Client
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
    supabase: 'supabase.Client'

master_agent = Agent(
    model,
    system_prompt=
    #- Calculation questions should use the calculator tool and return the output from the tool call. 
    #- always only directly return the output of the tool call (unless there is an error).
    # 5) Always directly return the output of the tool call and nothing else unless the user specifies otherwise.  do not sumamrize. 
    """
    # Overview 
    You are an orchestrator agent for the injection molding company H&H Molds Inc. Your only 
    job is to send the users query to the correct tool and return the output up the chain from 
    the running the called tool, and nothing else. Do not write summaries or do anything else but call tools. you just need to call 
    the correct tool, and use its response. 

    ## Tools
    - General RAG: Use this tool for general RAG look ups (for anything not related to press20). if the query does not include press20, this is the first call. 
    - Press20: Use this tool for ANY questions or queries related to press20. if the word 'press' is included somewhere in the query use this tool, otherwise do not. 
    - Calculator: Use this tool if a user asks a question that is a simple calculation. always only Return output from tool call.

    ## Rules
    - Press20 related questions (queries containing 'press') MUST only use the Press20 tool. Anything else should attempt to use General RAG first or other tools.
    - Do not tell the user what tools you used to perform actions unless they ask
    - Output must be in markdown format

    ## Instructions
    1) Call the neccessary tools based on the query
    2) Calculation questions should use the calculator tool and return the result.
    3) When you call a tool and get an none-error response, only return the result of that tool to the user.

    ## Examples
    1) 
    - Input: Can you list the FAIL shots from press20
      - Action: Use press20_agent to query the database, since 'press' is in the query.
    - Output: call the press20_agent 
    2) 
    - Input: What is 10+10
      - Action: Use the calculator tool to answer, since this is an equation
    - Output: call the calculator_agent. 
    3)
    - Input: Can you summarize the scope meeting for tech troubleshooting
      - Action: Use the General_RAG tool to answer, since 'press' is not in the query.
    - Output: call general rag agent.

    ## Final Reminders
    - Here is the current date/time: {}
    - If no a not good answer (or error) is returned, tell the user to contact Dawson Burgess and write the current time down, and offer to search the web. 
    - Do not specify what tools were used unless asked by the user.
    - Output Answers MUST BE IN MARKDOWN FORMAT
    - Remember, you are only tasked with routing queries, not answering questions. Do not state the tool used or a provide a summary.
    """.format(datetime.now().strftime("%Y-%m-%d")),
    deps_type=Deps,
    retries=2
)

@master_agent.tool
@observe()
async def general_rag(ctx: RunContext[Deps], query: str) -> str:
    try:
        result = await rag_agent.run(query, deps=ctx.deps)
        if not result.output:
            return "No response from RAG agent."
        return result.output
    except Exception as e:
        return f"Error in General RAG: {str(e)}"


@master_agent.tool
@observe()
async def calculator(ctx: RunContext[Deps], expression: str) -> str:
    try:
        result = await calculator_agent.run(expression, deps=ctx.deps)
        #think_result = await think(ctx, expression, "calculator")
        #return f"Result: {result}\n\nThink verification: {think_result}"
        return f"Result: {result}"
    except Exception as e: 
        return f"Error in calling calculator_agent tool: {str(e)}"


@master_agent.tool
@observe()
async def press20(ctx: RunContext[Deps], query: str) -> str:
    try:
        result = await press20_agent.run(query, deps=ctx.deps)
        #print(f"Press20 result: {vars(result)}")  # Debug print
        if not result.data:
            return "No response from Press20 agent."
        return result.data
    except Exception as e:
        return f"Error in Press20: {str(e)}"

async def create_master_agent():
    return master_agent
