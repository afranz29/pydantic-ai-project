import os
from typing import List

from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from tools.WebSearchTool import WebSearchTool, SubQuery
from models.ResearchModel import ResearchSection, StructuredResearchOutput
from utils.logger import agent_logger, tool_logger

class ResearchAgent:
    def __init__(self):
        self.s_tool = WebSearchTool()
        self.logged_outputs: List[ResearchSection] = []

        # Wrap the tool function to capture outputs
        async def wrapped_web_search(query: SubQuery):
            try:
                result = await self.s_tool.web_search(query)
                self.logged_outputs.append(result)
                return result
            except Exception as e:
                tool_logger.warning(f"Search failed for subquery: {query.sub_prompt} - {e}")
                return None


        model = OpenAIModel(
            model_name='llama3.1:8b',
            provider=OpenAIProvider(base_url=f'http://{os.getenv("OLLAMA_HOST")}/v1')
        )

        web_search_tool = Tool(
            function=wrapped_web_search,
            name="web_search"
        )

        settings = ModelSettings(parallel_tool_calls=True)

        self.agent = Agent(
            model=model,
            model_settings=settings,
            system_prompt=f"""
            You are the top researcher with access to a tool to search the web called `web_search`. Your task is to research the user's query and provide comprehensive, factual information.

            TASK:
            1. First, based on the user's query, generate exactly **5** logical sub-questions.
            2. Next, for each question, use the search tool like this:
                web_search(sub_prompt="...")
                YOU MUST INVOKE THE TOOL FOR EACH QUESTION.

            IMPORTANT RULES:
            - You MUST call `web_search` five times — once for each sub-question.
            - You MUST make all five tool calls in the same step.
            - Do NOT describe what you are doing. Only output tool calls.
            - Do NOT write a plan or a summary.
            - After calling the tool five times, STOP.

            After your tool calls, stop.
            """,
            tools=[web_search_tool]
        )

    async def run(self, user_prompt: str):
        agent_logger.info("Research agent called")
        self.logged_outputs.clear()  # Reset for each run

        await self.agent.run(user_prompt)
        
        if not self.logged_outputs:
            raise RuntimeError("All subqueries failed. No research could be gathered.")

        return StructuredResearchOutput(
            query=user_prompt,
            sections=self.logged_outputs
        )
