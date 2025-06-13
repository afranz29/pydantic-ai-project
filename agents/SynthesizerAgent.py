import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class SynthesizerAgent:
    def __init__(self):
        model = OpenAIModel(
            model_name='qwen3:8b',
            provider=OpenAIProvider(base_url=f'http://{os.getenv('OLLAMA_HOST')}/v1')
        )

        self.agent = Agent(
            model=model,
            system_prompt="""
            You are a professional report writer. Your task is to transform the provided
            research notes into a research report.

            Task:
            1. Write a research report that is well-structured, eloquent, and easy-to-read 
            2. Use clear headings and bullet points. 
            3. Rely solely on the provided context.
            4. At the end of the report, you MUST write all used source urls as references.
            """
        )

    async def run(self, research_prompt: str):
        return await self.agent.run(research_prompt)
