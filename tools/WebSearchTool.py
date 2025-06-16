import re
from utils.logger import tool_logger
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel


class SubQuery(BaseModel):
    sub_prompt: str

class WebSearchTool:

    async def web_search(self, query: SubQuery) -> str:
        tool_logger.info(f"Searching subprompt: {query.sub_prompt}")

        # DUCKDUCKGO SEARCH
        # urls of sources
        try:
            search_results = DDGS().text(f"{query.sub_prompt}", max_results=2)
        except Exception as e:
            return f"[!] Error - could not search: {e}"
        
        if search_results:
            print(f"    Search results:")
            for result in search_results:
                print(f"    {result["title"]}\n    URL: {result["href"]}\n")

        # CONFIG FOR CRAWLER
        # pruning filter
        prune_filter = PruningContentFilter(
            # Lower → more content retained, higher → more content pruned
            threshold=0.38,           
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
            cleaned_results = [f"Rsearch Question/Topic: {query.sub_prompt}"]

            for result in search_results:
                title = result["title"] if result["title"] else "No Title Found"
                url = result["href"]
                
                async with AsyncWebCrawler() as crawler:
                    crawl_result = await crawler.arun(
                        url=f"{url}",
                        config=config
                    )
                    tool_logger.info(f"Crawled {result["href"]}")

                if not crawl_result:
                    return f"Crawl failed for {result["href"]}"



                page_markdown = crawl_result.markdown.fit_markdown
                if not page_markdown:
                    cleaned_markdown = "Empty Result: may need another tool to crawl.\nIgnore this source."
                else:
                    cleaned_markdown = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1', page_markdown)

                page = f"\nTitle: {title}\nContent: {cleaned_markdown}\nURL: {url}"
                cleaned_results.append(page)


            combined_results = "\n---\n".join(cleaned_results)
            tool_logger.info(f"[✔] Research complete for subprompt {query.sub_prompt}")
            return combined_results

        except Exception as e:
            return f"[!] Crawl4AI error: {e}"