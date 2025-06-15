import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import  Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools.WebSearchTool import WebSearchTool

class ResearchAgent:
    def __init__(self):
        self.s_tool = WebSearchTool()

        model = OpenAIModel(
            model_name='qwen3:14b',
            provider=OpenAIProvider(base_url=f'http://{os.getenv('OLLAMA_HOST')}/v1')
        )

        print(model)

        internet_search_tool = Tool(
            function=self.s_tool.web_search,
            name="internet_search"
            )

        self.agent = Agent(
            model=model,
            system_prompt="""
            You are a research assistant with access to a tool to search the web called `internet_research`. Your task is to research the user's query and provide comprehensive, factual information.

            TASK:
            1. First, based on the user's query, you MUST generate a list of 3-5 logical sub-questions.
            2. Next, for each question, use the search tool the like this:
                wen_search(sub_prompt="...")
                You MUST actually invoke the tool for EACH question. 
            3. Combine the results into a single document with clear section headers.
            4. You MUST include the urls of sources at the end of each section.
            
            Do not write a research plan — call the tool and use its output. Only use the tool for your sub questions. 
            
            Please conduct thorough research and provide your findings.
            """,
            tools=[internet_search_tool]
        )

    async def run(self, user_prompt: str):
        print("[AGENT] Research agent called")
        return await self.agent.run(user_prompt)
