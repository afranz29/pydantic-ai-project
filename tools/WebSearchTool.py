import re
from urllib.parse import urlparse

from pydantic import BaseModel
from duckduckgo_search import DDGS
from googlesearch import search
from typing import List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from utils.logger import tool_logger
from models.ResearchModel import SourceSection, ResearchSection, NormalizedSearchResult
from models.QueryModels import SubQuery
from config.config import app_settings



class WebSearchTool:

    def filter_results(self, norm_results: List[NormalizedSearchResult]) -> List:
        # Filters out searched results based on the blacklist in config.py
        # Will not be used if there are NOT more than 3 results
        if app_settings.WEB_SEARCH_TOOL.BLACKLIST_ON and app_settings.WEB_SEARCH_TOOL.NUM_SEARCH_RESULTS > 3:
            tool_logger.info(f"Filter Setting is ON.\nFiltering results based on {str(app_settings.WEB_SEARCH_TOOL.DOMAIN_BLACKLIST)}")
            filtered_results = [
                    source for source in norm_results
                    if urlparse(source.url).netloc not in app_settings.WEB_SEARCH_TOOL.DOMAIN_BLACKLIST
                ]

            return filtered_results
        else:
            tool_logger.info("Filter Setting is OFF. Will NOT filter results")
            return norm_results


    def print_search_results(self, final_results: List[NormalizedSearchResult]):
        print(f"    Search results:")
        for result in final_results:
            print(f"    Title: {result.title}\n    URL: {result.url}\n")


    def ddg_browser(self, query: SubQuery) -> List[NormalizedSearchResult]:
        # rate limited
        tool_logger.info("Using DuckDuckGo search")
        try:
            ddg_search_results = DDGS().text(
                f"{query.sub_prompt}",
                max_results=app_settings.WEB_SEARCH_TOOL.NUM_SEARCH_RESULTS
                )
            
            norm_results = [
                    NormalizedSearchResult(title=source["title"], url=source["href"])
                    for source in ddg_search_results
                ]
            
            final_results = self.filter_results(norm_results)
            
            print(final_results)

            return final_results
        except Exception as e:
            tool_logger.error(f"[!] DuckDuckGo search failed: {e}")
            raise


    def google_browser(self, query: SubQuery) -> List[NormalizedSearchResult]:
        tool_logger.info("Using Google search")
        
        try:
            google_search_results = list(search(
                f"{query.sub_prompt}",
                num_results=app_settings.WEB_SEARCH_TOOL.NUM_SEARCH_RESULTS,
                sleep_interval=3,
                advanced=True
                ))

            norm_results = [
                    NormalizedSearchResult(title=source.title, url=source.url)
                    for source in google_search_results
                ]
            
            final_results = self.filter_results(norm_results)
            
            print(final_results)

            return final_results
        except Exception as e:
            tool_logger.error(f"[!] Google search failed: {e}")
            raise


    async def crawl_sites(self, norm_list_results: List[NormalizedSearchResult], query: SubQuery):
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
        cleaned_results = []

        for result in norm_list_results:    
            try:
                title = result.title
                url = result.url
                
                async with AsyncWebCrawler() as crawler:
                    crawl_result = await crawler.arun(
                        url=f"{url}",
                        config=config
                    )
                
                tool_logger.info(f"Crawled {result.url}")

                if not (crawl_result and crawl_result.markdown and crawl_result.markdown.fit_markdown):
                    tool_logger.warning(f"Crawl for {url} resulted in empty content. Skipping.")
                    continue

                page_markdown = crawl_result.markdown.fit_markdown

                # clean out all the links within a source's content
                content = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1', page_markdown)

                source = SourceSection(
                    title=title,
                    content=content,
                    url=url
                )

                cleaned_results.append(source)
            
            except Exception as e:
                tool_logger.error(f"Failed to crawl or process {result.url}: {e}. Skipping source.")
                continue

        if cleaned_results:
            research_subsection = ResearchSection(
                subquestion=f"Rsearch Question/Topic: {query.sub_prompt}",
                sources=cleaned_results   
            )

            tool_logger.info(f"[✔] Research complete for subprompt {query.sub_prompt}")
            return research_subsection
        else:
            tool_logger.warning(f"All sources failed to crawl for subprompt: {query.sub_prompt}")
            return None

            
    async def web_search(self, query: SubQuery) -> SourceSection:
        tool_logger.info(f"Searching subprompt: {query.sub_prompt}")

        # uses app_settings to determine which browswer to use
        match app_settings.WEB_SEARCH_TOOL.WEB_BROWSER.lower():
            case "duckduckgo":
                try:
                    search_results = self.ddg_browser(query)
                except Exception as e:
                    tool_logger.warning(f"DuckDuckGo search failed: {e}. Attempting fallback with Google.")
                    try:
                        search_results = self.google_browser(query)
                    except Exception as fallback_e:
                        tool_logger.error(f"Fallback search with Google also failed: {fallback_e}")
            
            case "google":
                search_results = self.google_browser(query)
            
            case _:
                raise RuntimeError(f"{app_settings.WEB_SEARCH_TOOL.WEB_BROWSER} is an invalid browswer.")

        if search_results:

            research = await self.crawl_sites(search_results, query)

            return research
        return None
        