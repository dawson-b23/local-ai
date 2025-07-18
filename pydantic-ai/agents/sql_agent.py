# agents/sql_agent.py (Converted to Pydantic Agent, no tools, direct LLM for NL-to-SQL)
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import httpx

load_dotenv()

client = AsyncOpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), api_key="ollama")
model = OpenAIModel(model_name="llama3.1:8b", provider=OpenAIProvider(base_url=os.getenv("OLLAMA_URL") + "/v1"))

langfuse = get_client()

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase: 'supabase.Client'

sql_agent = Agent(
    model,
    system_prompt='''
You are a SQL expert converting natural language to SQL for press20_data table.
Table schema:
class Press20Data(BaseModel):
        id: Optional[int] = None
        shot_num: Optional[int] = None
        overallpassfail: Optional[str] = None
        dataset_id: Optional[str] = None
        cameratimestamp: Optional[str] = None
        bottompassfail: Optional[str] = None
        toppassfail: Optional[str] = None
        bottomanomalylevel: Optional[float] = None
        topanomalylevel: Optional[float] = None
        machinetimestamp: Optional[str] = None
        actcycletime: Optional[int] = None
        actclpclstime: Optional[int] = None
        actcoolingtime: Optional[int] = None
        actcurrentservodrive_disp_1: Optional[int] = None
        actcurrentservodrive_disp_2: Optional[int] = None
        actcurrentservodrive_1: Optional[int] = None
        actcurrentservodrive_2: Optional[int] = None
        actcushionposition: Optional[int] = None
        actfeedtemp: Optional[int] = None
        actfill: Optional[int] = None
        actfilltime_0: Optional[int] = None
        actfilltime_1: Optional[int] = None
        actfilltime_2: Optional[int] = None
        actinjectionpos: Optional[int] = None
        actinjfillspd: Optional[int] = None
        actcalejtfwdspd: Optional[int] = None
        actcalejtretspd: Optional[int] = None
        actinjfwdstagepos_0: Optional[int] = None
        actinjfwdstagepos_1: Optional[int] = None
        actinjfwdstagepos_2: Optional[int] = None
        inj_act_prs_0: Optional[int] = None
        inj_act_prs_1: Optional[int] = None
        inj_act_prs_2: Optional[int] = None
        actinjfwdstageprs_0: Optional[int] = None
        actinjfwdstageprs_1: Optional[int] = None
        actinjfwdstageprs_2: Optional[int] = None
        actinjfwdstagetime_0: Optional[int] = None
        actinjfwdstagetime_1: Optional[int] = None
        actinjfwdstagetime_2: Optional[int] = None
        actmotorrpmservodrive_0: Optional[int] = None
        actmotorrpmservodrive_1: Optional[int] = None
        actnozzlecurrent: Optional[int] = None
        actnozzlepidper: Optional[int] = None
        actnozzletemp: Optional[int] = None
        actoiltemp: Optional[int] = None
        actprocofmaxinjprs: Optional[int] = None
        actprocofmaxinjprspos: Optional[int] = None
        actrearsuckbackspd: Optional[int] = None
        actrearsuckbacktime: Optional[int] = None
        actrefilltime: Optional[int] = None
        actsysprsservodrive_0: Optional[int] = None
        actsysprsservodrive_1: Optional[int] = None
        acttempservodrive_0: Optional[int] = None
        acttempservodrive_1: Optional[int] = None
        acttempservomotor_0: Optional[int] = None
        acttempservomotor_1: Optional[int] = None
        actccprs: Optional[int] = None
        actzone1temp: Optional[int] = None
        actzone2temp: Optional[int] = None
        actzone3temp: Optional[int] = None
        actzone4temp: Optional[int] = None
        actzone5temp: Optional[int] = None
        actzone6temp: Optional[int] = None
        prvactinj1plasttime: Optional[int] = None
        backprs_value: Optional[int] = None
        actprocmonmininjpos: Optional[int] = None
        flow_value: Optional[int] = None
All columns are lowercase. Use DISTINCT and ORDER BY shot_num for shot lists.
Generate ONLY the SQL query. No explanations. Nothing extra. The table name is ALWAYS press20_data. Use the table layout to answer questions
Always return a valid SQL query. NEVER make anything up.
Examples:
- "how many failed shots": SELECT * shot_num FROM press20_data WHERE overallPassFail = 'FAIL'
- "average ActNozzleTemp over shot_num 100 to 200": SELECT AVG(ActNozzleTemp) AS average FROM press20_data WHERE shot_num BETWEEN 100 AND 200
    ''',
    deps_type=Deps,
    retries=3
)
