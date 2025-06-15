import asyncio
from duckduckgo_search import DDGS
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel

NUM_SEARCH_RESULTS = 1      # Number of search results to return for each sub prompt


class SubQuery(BaseModel):
    sub_prompt: str

class WebSearchTool:

    async def web_search(self, query: SubQuery) -> str:
        print(f"[TOOL] Searching subprompt: {query.sub_prompt}")

        # DUCKDUCKGO SEARCH
        # urls of sources
        try:
            search_results = DDGS().text(f"{query.sub_prompt}", max_results=NUM_SEARCH_RESULTS)
        except Exception as e:
            return f"[!] Error - could not search: {e}"
        
        if search_results:
            print(f"Search results: {search_results}")

        # CONFIG FOR CRAWLER
        # pruning filter
        prune_filter = PruningContentFilter(
            # Lower → more content retained, higher → more content pruned
            threshold=0.48,           
            # dynamic - adjusts according to type, text/link density, eyc
            threshold_type="dynamic",     
        )

        # markdown generator
        md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

        # give it to the config
        config = CrawlerRunConfig(
            markdown_generator=md_generator
        )

        # RUN CRAWLER
        # run crawler for each result
        try:
            cleaned_results = []

            for result in search_results:
                title = result["title"] if result["title"] else "No Title Found"
                url = result["href"]
                
                async with AsyncWebCrawler() as crawler:
                    crawl_result = await crawler.arun(
                        url=f"{url}",
                        config=config
                    )
                    print(f"Crawled {result["href"]}")

                if not crawl_result:
                    return f"Crawl failed for {result["href"]}"

                page = f"Title: {title}\nResearch: {crawl_result.markdown.fit_markdown}\nURL: {url}"
                cleaned_results.append(page)

            
            print(f"[TOOL] Research complete for subprompt {query.sub_prompt}")
            combined_results = "\n---\n".join(cleaned_results)
            return combined_results

        except Exception as e:
            return f"[!] Crawl4AI error: {e}"