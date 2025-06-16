import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class SynthesizerAgent:
    def __init__(self):
        model = OpenAIModel(
            model_name='qwen3:14b',
            provider=OpenAIProvider(base_url=f'http://{os.getenv('OLLAMA_HOST')}/v1')
        )

        self.agent = Agent(
            model=model,
            system_prompt="""
            You are a professional report writer. Your task is to transform the provided
            research notes into a research report.

            TASK:
            1. Carefully read and analyze the provided research context.
            2. Write a research report based on the context that is well-structured and easy-to-follow.
            3. Use clear headings for each section.
            4. USE PARAGRAPHS.
            5. Integrate and qoute relevant information from the research context smoothly into your writing.

            REPORT RULES:
            - The report MUST be at least 800 words.
            - You MUST give the report a title.
            - Do NOT summarize — compose a full, stand-alone report based entirely on the given content.
            - You MUST qoute the context if you copy from it.
            - You MUST include a section at the end of the report. It MUST be titled "References" and list all source URLs from the context.
                For example:

            References:
                1. https://example.com/article1
                2. https://another-example.org/post

            STYLE:
            - Use formal academic language.
            - Do NOT bullet points and long lists. Use paragraphs.
            - Do NOT use bold, italic, underlining, or any other formatting for emphasis.
            
            IMPORTANT:
            - Rely ONLY on the provided context. Do NOT use any outside context.
            - Do NOT make up references, sources, or qoutes. 
            - Only include URLs that appear in the context.

            """
        )

    async def run(self, research_prompt: str):
        result = await self.agent.run(research_prompt)

        result.output = result.output.split("</think>")[-1].strip()

        return result
