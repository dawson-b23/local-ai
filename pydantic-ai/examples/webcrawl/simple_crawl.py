import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com")
        print(result.markdown[:300])  # Print first 300 chars
        print("\n Raw markdown \n") 
        print(result.markdown.raw_markdown) # Raw markdown from cleaned html
        print("\n fit markdown \n") 
        print(result.markdown.fit_markdown) # Most relevant content in markdown

if __name__ == "__main__":
    asyncio.run(main())
