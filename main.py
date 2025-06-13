import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import uvicorn

from firecrawl import FirecrawlApp
from agents.ResearchAgent import ResearchAgent
from agents.SynthesizerAgent import SynthesizerAgent


# Data models
class UserQuery(BaseModel):
    prompt: str

class Report(BaseModel):
    report: str

# App state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Starting app ---")
    load_dotenv()

    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise ValueError("FIRECRAWL_API_KEY not found in environment.")

    # Initialize agents
    firecrawl_client = FirecrawlApp(api_key=firecrawl_api_key)
    research_agent = ResearchAgent(firecrawl_client)
    synthesizer_agent = SynthesizerAgent()

    # Register to app state
    app_state["research_agent"] = research_agent
    app_state["synthesizer_agent"] = synthesizer_agent
    print("--- App live ---")

    yield

    print("--- Shutting down ---")
    app_state.clear()


app = FastAPI(
    title="Autonomous Report Builder",
    lifespan=lifespan
)

@app.post("/generate-report", response_model=Report)
async def generate_text(query: UserQuery):

    try:
        research_context = await app_state["research_agent"].run(query.prompt)

        print("Research successful. Writing report using the prompt: ")
        report_prompt = f"""
        Generate a professional report with references on the topic: '{query.prompt}'.
        Use the following research context as your sole source of information:
        ---
        {research_context.output}
        ---
        """

        final_report = await app_state["synthesizer_agent"].run(report_prompt)
        return Report(report=final_report.output)

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama communication error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
