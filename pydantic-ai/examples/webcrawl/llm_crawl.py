import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig, LLMContentFilter, DefaultMarkdownGenerator
from crawl4ai import JsonCssExtractionStrategy

async def main():
    # 1) Browser config: headless, bigger viewport, no proxy
    browser_conf = BrowserConfig(
        browser_type="chromium",
        headless=True,
        text_mode=True,
        light_mode=True,
    )

    # 3) Example LLM content filtering

    llama_config = LLMConfig(
        provider="ollama/llama3.1:8b", 
        api_token = "http://localhost:11434"
    )

    # Initialize LLM filter with specific instruction
    filter = LLMContentFilter(
        llm_config=llama_config,  # or your preferred provider
        instruction="""
        Focus on extracting the relevant content to python.
        Include:
        - Key concepts and explanations
        - Important examples
        - Essential technical details
        Exclude:
        - Navigation elements
        - Sidebars
        - Footer content
        Format the output as clean markdown with proper code blocks and headers.
        """,
        chunk_token_threshold=500,  # Adjust based on your needs
        verbose=True
    )

    # 4) Crawler run config: skip cache, use extraction
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # 4) Execute the crawl
        result = await crawler.arun(url="https://docs.python.org/3/tutorial/interpreter.html", config=run_conf)

        if result.success:
            print("Extracted content:", result.markdown)
        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())

