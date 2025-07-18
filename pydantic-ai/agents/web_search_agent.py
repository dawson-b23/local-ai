# agents/web_search_agent.py (New agent for web search/crawl)
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from langfuse import observe
import httpx
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.content_filter_strategy import LLMContentFilter
import asyncio

load_dotenv()

client = AsyncOpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), api_key="ollama")
model = OpenAIModel(model_name=os.getenv("LLM_MODEL", "llama3.1:8b"), provider=OpenAIProvider(base_url=os.getenv("OLLAMA_URL") + "/v1"))

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase: 'supabase.Client'

web_search_agent = Agent(
    model,
    system_prompt="""
You are a web search assistant for H&H. Perform search if query exists. Use 'web_search' tool to search, crawl, filter, return markdown summary.
Return ONLY markdown bullets/tables with key findings. If no results: - **No results found.**
""",
    deps_type=Deps,
    retries=3
)

@web_search_agent.tool
@observe()
async def web_search(ctx: RunContext[Deps], query: str) -> str:
    if not query.strip():
        return "- **Error:** Empty query."
    try:
        # Aggregate links
        results = DDGS().text(query, max_results=5)
        webpages_list = [item['href'] for item in results]

        # Crawl and filter
        browser_conf = BrowserConfig(
            browser_type="chromium",
            headless=True,
            text_mode=True,
            light_mode=True,
            verbose=True,
        )

        run_conf = CrawlerRunConfig(
            magic=True,  # Simplifies a lot of interaction
            remove_overlay_elements=True,
            exclude_external_links=True,
        )

        ollama_config = LLMConfig(
            provider="ollama/llama3.1:8b",
            base_url="http://localhost:11434",  # Correct endpoint for local Ollama
            api_token="no-token",  # No token needed for Ollama
        )

        filter = LLMContentFilter(
            llm_config=ollama_config,
            ignore_cache = True,
            instruction=f"""
            Extract the main content related to {query} while preserving its original wording and substance completely. Your task is to:

            1. Maintain the exact language and terminology used in the main content
            2. Keep all technical explanations, examples, and educational content intact
            3. Preserve the original flow and structure of the core content
            4. Remove only clearly irrelevant elements like:
            - Navigation menus
            - Advertisement sections
            - Cookie notices
            - Footers with site information
            - Sidebars with external links
            - Any UI elements

            The goal is to create a clean markdown version that reads exactly like the original article, 
            keeping all valuable content but free from distracting elements. Imagine you're creating 
            a perfect reading experience where nothing valuable is lost, but all noise is removed.
            """,
            verbose=True
        )
        summaries = []
        async with AsyncWebCrawler(config = browser_conf) as crawler:
            for link in webpages_list:
                result = await crawler.arun(
                    url = link, 
                    config = run_conf
                )
                html = result.cleaned_html
                filtered_content = filter.filter_content(html)
                summaries.append(filtered_content)

        if summaries:
            combined_summaries = "\n\n---\n\n".join([f"Summary from page {i+1}:\n{s}" for i, s in enumerate(summaries)])
            
            aggregate_instruction = f"""
            Aggregate these individual webpage summaries about "{query}" into one comprehensive, cohesive summary. 
            Synthesize key information, remove redundancies, and ensure the final output is concise yet complete. 
            Structure it with bullet points or paragraphs for readability.
            """
            
            agg_filter = LLMContentFilter(
                llm_config=ollama_config,
                ignore_cache=True,
                instruction=aggregate_instruction,
                verbose=True
            )
            
            final_summary = agg_filter.filter_content(combined_summaries)

        return "\n".join(final_summary) if final_summary else "- **No results.**"
    except Exception as e:
        return f"- **Error in web search:** {str(e)}"
