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
            2. You MUST populate the `references.sources` list with ALL URLs found in the research context. You can find the URLs in the provided JSON data under the path: `sections` -> `sources` -> `url`. This list cannot be empty.


            TASK:
            1. Write a formal research report based on the provided context that is well-structured and easy-to-follow.
            2. Use clear headings for each section.
            3. Integrate and qoute relevant information from the research context smoothly into your writing.
            4. USE PARAGRAPHS. 

            REPORT RULES AND STYLE:
            - The report MUST be at least {app_settings.SYNTH_AGENT.WORD_COUNT_REQ} words excluding sources.
            - You MUST give the report a title.
            - Do NOT summarize — compose a full, stand-alone report based entirely on the given content.
            - You MUST qoute the context if you copy from it.
            - Use formal academic language and write in paragraphs.
            - Do NOT use bold, italic, underlining, or any other formatting for emphasis.
            - Rely ONLY on the provided context. Do NOT use any outside context.
            - Do NOT make up references, sources, or qoutes. 

            REQUIRED OUTPUT FORMAT:
            - You MUST output a valid JSON object that strictly follows the required structure.
            - The JSON object must have top-level keys: "title", "abstract", "sections", and "references".
            - The "sections" key must be a list of objects, each with a "header" and "content" string.
            - The "references" key MUST be an object containing a "header" (which should be "References") and a "sources" key.
            - The "sources" key inside the "references" object MUST be a flat list of all the source URL strings extracted from the research context. Only include URLs that appear in the context.
            Here is an example of the expected output structure:
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
