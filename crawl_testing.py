import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://docs.polygonscan.com/api-endpoints/accounts#get-pol-balance-for-multiple-addresses-in-a-single-call",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())