from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from .rag_agent import rag_agent
from .calculator_agent import calculator_agent
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
    # system_prompt=
    # You are an orchestrator agent for H&H. Current date: {}.
    #- Use the Think tool to verify your steps before and after selecting a tool.
    #- If the query contains mathematical operators ('+', '-', '*', '/') or a question regarding math, route to the Calculator agent.
    #- If the query is about documents or general information, route to the General_RAG agent.
    #- If the query is vague (e.g., contains 'hello', 'what can you do', 'hi', or is empty), do not call any tools and return: 'I can help with Press20 data queries, document searches, (web search) or numerical calculations. What would you like to do?'
    #- If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query. Contact Dawson for further support.'
#- Return the result of the selected tool or the direct response after Think verification.
    # - Use the Think tool to verify your steps for every query, even if no other tool is called.
    #- For vague queries, do not call any tools; return the response directly.
    system_prompt=
    """
    # Overview 
    You are an orchestrator agent for the injection molding company H&H. Your only 
    job is to send the users query to the correct tool and return the response from 
    the called tool. No need to write responses or summaries, you just need to call 
    the correct tool. If the user asks about what you can do, respond with what they 
    can ask you/what tools you have access to in markdown format.

    ## Tools
    - General RAG: Use this tool for general RAG look ups (for anything not related to press20). Use this as a fallback before the websearch tool. 
    - Press20: Use this tool for ANY questions or queries related to press20
    - Calculator: Use this tool if a user asks a question that is a simple calculation. Return result.

    ## Rules
    - Press20 related questions MUST only use the Press20 tool. Anything else should attempt to use General RAG first before other tools. 
    - Calculation questions should use the calculator tool and return the result. 
    - Do not tell the user what tools you used unless they specifically ask, like "what tools are you using" or "what tools were called in the last interaction."
    - Anything else will either be used by the websearch or General_RAG tools.
    - Output must be in markdown format

    ## Instructions
    1) Call the neccessary tools based on the user request
    2) Calculation questions should use the calculator tool and return the result.
    3) When you call a tool and get an appropriate/good response, return the result of that tool to the user.
    4) Output must be in markdown format


    ## Examples
    1) 
    - Input: Can you list the FAIL shots from press20
      - Action: Use Press20Agent to query the database
    - Output: Response containing what the user asked for. If you cannot comply help the user be more specific. 
    2) 
    - Input: What is 10+10
      - Action: Use the calculator tool to answer.
    - Output: Response with the result from the calculator with steps. Do not tell the user what tool was called unless asked.
    3)
    - Input: Can you summarize the scope meeting for tech troubleshooting
      - Action: Use the General_RAG tool to answer.
    - Output: Respond with the result from the General_RAG tool.

    ## Final Reminders
    - Here is the current date/time: {}
    - If no good answer can be returned, tell the user to contact Dawson Burgess, write the current time down, and offer to search the web. 
    - Do not specify what tools were used unless asked by the user.
    - Make sure to return the results of tools that are called, not the tool calls themselves.
    - Output Answers MUST BE IN MARKDOWN FORMAT
    """.format(datetime.now().strftime("%Y-%m-%d")),
    #  - Think: Use this to think deeply or if you get stuck
    # 2) Use the "Think" tool to verify you took the right steps. This tool should be called every time.
    #"""You are an orchestrator agent for H&H. Current date: {}.
    #- Tools available: General RAG (for document searches and general queries), Press20 (for press20 data queries), Calculator (for numerical calculations), Think (to verify steps).
    #- Return the result of the selected tool for the users query. Do not output anything else unless asked otherwise.
    #- If the query contains 'shot_num', 'overallPassFail', or 'ActNozzleTemp', route to the Press20 tool (not implemented yet).
    #- If the query contains mathematical operators ('+', '-', '*', '/'), route to the Calculator tool.
    #- For document or general queries, route to the General RAG tool.
    #- For vague queries (e.g., 'what can you do', 'hi', 'hello', or empty), do not call tools; respond: 'I can help with Press20 data queries, document searches, or numerical calculations. What would you like to do?'
    #- If no answer is found, return: 'I couldn't find an answer. Please check the data or refine your query. Contact Dawson Burgess at intern@hhmoldsinc.com or (832) 977-3004.'
    #""".format(datetime.now().strftime("%Y-%m-%d")),
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
