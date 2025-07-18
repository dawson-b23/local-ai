import asyncio
from crawl4ai import AsyncWebCrawler, AdaptiveCrawler

async def adaptive_example():
    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler)

        # Start adaptive crawling
        result = await adaptive.digest(
            start_url="https://docs.python.org/3/",
            query="async context managers"
        )

        # View results
        adaptive.print_stats()
        print(f"Crawled {len(result.crawled_urls)} pages")
        print(f"Achieved {adaptive.confidence:.0%} confidence")

if __name__ == "__main__":
    asyncio.run(adaptive_example())
