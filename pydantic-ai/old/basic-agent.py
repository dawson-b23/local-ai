import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# logging
from langfuse import observe, get_client

import httpx
from httpx import AsyncClient
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, RunContext, ModelRetry
import logging

load_dotenv()
llm = os.getenv('LLM_MODEL') 

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)
#model = OpenAIModel("llama3.1:8b", openai_client=client)
model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

@dataclass
class Deps:
    client: AsyncClient

basic_agent = Agent(
    model,
    system_prompt=f'You are a helpful model that answers questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
    deps_type=Deps,
    retries=2
)

async def main():
    async with AsyncClient() as client:
        deps = Deps(client=client)

        result = await basic_agent.run(
            'What can you do?', deps=deps
        )
        
        #debug(result)
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())
