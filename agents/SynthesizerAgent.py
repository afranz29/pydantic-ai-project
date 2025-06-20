import os

from utils.logger import agent_logger

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from models.ReportModel import Report

from config.config import app_settings


class SynthesizerAgent:
    def __init__(self):
        model = OpenAIModel(
            model_name='qwen3:14b',
            provider=OpenAIProvider(base_url=f'http://{os.getenv("OLLAMA_HOST")}/v1')
        )

        self.agent = Agent(
            model=model,
            system_prompt=f"""
            You are a professional report writer. Your task is to transform the provided
            research notes into a research report.

            **!! CRITICAL INSTRUCTIONS !!**
            1. Your output MUST be a valid JSON object that strictly follows the structure defined in the "REQUIRED OUTPUT FORMAT" section below. This is your most important task.
            2. You MUST populate the `references.sources` list with ALL URLs from the top-level `all_urls` field in the provided JSON research context. This list cannot be empty.

            WRITING TASK:
            1. Write a formal research report based on the provided context that is well-structured and easy-to-follow.
            2. Use clear headings for each section.
            3. USE PARAGRAPHS. 

            **REPORT CONTENT REQUIREMENTS**
            - The report must be comprehensive and well-detailed, based *only* on the provided context.
            - Word Count: The report's `sections.content` parts MUST add up to contain at least {app_settings.SYNTH_AGENT.WORD_COUNT_REQ} words.
            - Writing Style: Use formal academic language and write in paragraphs with NO bold, italic, underlining, or any other formatting for emphasis.

            REQUIRED OUTPUT FORMAT:
            - You MUST output a valid JSON object that strictly follows the required structure.
            - The JSON object must have top-level keys: "title", "abstract", "sections", and "references".
            - The "sections" key must be a list of objects, each with a "header" and "content" string.
            - The "references" key MUST be an object containing a "header" (which should be "References") and a "sources" key.
            - The "sources" key inside the "references" object MUST be a flat list of all the source URL strings extracted from the research context. Only include URLs that appear in the context.
            
            Here is an example of the expected output structure. Adhere to it strictly:
            {{
                "title": "Example Title",
                "abstract": "This is an example abstract.",
                "sections": [
                    {{
                        "header": "Section 1 Header",
                        "content": "Content of section 1..."
                    }},
                    {{
                        "header": "Section 2 Header",
                        "content": "Content of section 2..."
                    }}
                ],
                "references": {{
                    "header": "References",
                    "sources": [
                        "http://example.com/source1",
                        "http://example.com/source2"
                    ]
                }}
            }}
            """,
            output_type=Report
        )

    async def run(self, research_prompt: str):
        agent_logger.info("Synthesizer agent called")
        result = await self.agent.run(research_prompt)

        return result
