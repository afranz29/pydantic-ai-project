import os
from pydantic import BaseModel
from typing import List, Dict
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import  Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from firecrawl import FirecrawlApp, ScrapeOptions


class SubQuery(BaseModel):
    sub_prompt: str

class ResearchAgent:
    def __init__(self, firecrawl_client: FirecrawlApp):
        self.firecrawl_client = firecrawl_client

        model = OpenAIModel(
            model_name='qwen3:8b',
            provider=OpenAIProvider(base_url=f'http://{os.getenv('OLLAMA_HOST')}/v1')
        )

        self.agent = Agent(
            model=model,
            system_prompt="""
            You are a research assistant with access to a tool to search the web called `internet_research`. Your task is to research the user's query and provide comprehensive, factual information.

            TASK:
            1. Based on the user's query, generate a list of 3-5 logical sub-questions.
            3. For each question, call the `internet_research` tool like this:
                internet_research(sub_prompt="...")
            You MUST actually invoke the tool using this format. Do not write a research plan — call the tool and use its output. You MUST call the tool for EACH subtopic.
            5. After gathering research for all questions, consolidate the findings into a single, cohesive document, with clear headings for each sub topic.
            6. You MUST include the urls of sources at the end of each section.
            
            Please conduct thorough research and provide your findings.
            """
        )

        @self.agent.tool
        def internet_research(ctx: RunContext, query: SubQuery) -> str:
            return self.run_firecrawl(query)


    def run_firecrawl(self, query: SubQuery) -> str:
        try:
            
            results = self.firecrawl_client.search(
                query.sub_prompt,
                scrape_options=ScrapeOptions(formats=["markdown", "links"]),
                limit=5
            )
            if not results or not results.data:
                return "[!] No search results found."

            combined = []
            for result in results.data[:2]:
                section = f"Title: {result['title']}\nContent: {result['markdown']}\nURL: {result['url']}"
                combined.append(section)

            return "\n\n---\n\n".join(combined)
        except Exception as e:
            return f"[!] Firecrawl error: {e}"

    async def run(self, user_prompt: str):
        print(f"[TOOL] Searching subprompt: {user_prompt}")
        return await self.agent.run(user_prompt)
