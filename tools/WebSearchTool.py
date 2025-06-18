import re
from pydantic import BaseModel
from duckduckgo_search import DDGS
from typing import List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from utils.logger import tool_logger
from models.ResearchModel import SourceSection, ResearchSection


class SubQuery(BaseModel):
    sub_prompt: str

class WebSearchTool:
    def ddg_browser(self, query: SubQuery) -> List:
        try:
            search_results = DDGS().text(f"{query.sub_prompt}", max_results=2)
        except Exception as e:
            tool_logger.error(f"[!] DuckDuckGo search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")

        
        if search_results:
            print(f"    Search results:")
            for result in search_results:
                print(f"    {result["title"]}\n    URL: {result["href"]}\n")
        
        return search_results

    async def crawl_sites(self, list_of_sites: List, query: SubQuery):
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
            cleaned_results = []

            for result in list_of_sites:
                title = result["title"] if result["title"] else "No Title Found"
                url = result["href"]
                
                async with AsyncWebCrawler() as crawler:
                    crawl_result = await crawler.arun(
                        url=f"{url}",
                        config=config
                    )
                    tool_logger.info(f"Crawled {result["href"]}")

                if not crawl_result:
                    content = f"Crawl failed for {result["href"]}"

                page_markdown = crawl_result.markdown.fit_markdown
                if not page_markdown:
                    content = "Empty Result: may need another tool to crawl.\nIgnore this source."
                else:
                    # clean out all the links within a source's content
                    content = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1', page_markdown)

                source = SourceSection(
                    title=title,
                    content=content,
                    url=url
                )
                cleaned_results.append(source)

            research_subsection = ResearchSection(
                subquestion=f"Rsearch Question/Topic: {query.sub_prompt}",
                sources=cleaned_results   
            )

            tool_logger.info(f"[✔] Research complete for subprompt {query.sub_prompt}")
            return research_subsection

        except Exception as e:
            return f"[!] Crawl4AI error: {e}"
        
    async def web_search(self, query: SubQuery) -> SourceSection:
        tool_logger.info(f"Searching subprompt: {query.sub_prompt}")

        # DUCKDUCKGO SEARCH
        # urls of sources
        search_results = self.ddg_browser(query)

        if search_results:
            research = await self.crawl_sites(search_results, query)

            return research
        return None
        