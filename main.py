import requests
import uvicorn
import json

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from pydantic import BaseModel
from models.ResearchModel import StructuredResearchOutput
from models.ReportModel import Report

from agents.ResearchAgent import ResearchAgent
from agents.SynthesizerAgent import SynthesizerAgent

from utils.save_file import save_to_output_file
from utils.logger import logger


# Data models
class UserQuery(BaseModel):
    prompt: str

# App state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    

    # Initialize agents
    research_agent = ResearchAgent()
    synthesizer_agent = SynthesizerAgent()

    # Register to app state
    app_state["research_agent"] = research_agent
    app_state["synthesizer_agent"] = synthesizer_agent
    logger.info("App is live")

    yield

    print("--- Shutting down ---")
    app_state.clear()


app = FastAPI(
    title="Autonomous Report Builder",
    lifespan=lifespan
)
app.mount("/app", StaticFiles(directory="static", html=True), name="static")


@app.post("/generate-report", response_model=Report)
async def generate_text(query: UserQuery):

    try:
        logger.info(f"[REQUEST] Request recieved. User wants to research: {query.prompt}")
        
        # research agent
        user_prompt = f"""
            Research the topic {query.prompt} using the tools available to you, then combine all the research into one organized document with sections.
            You MUST include the source URLs at the end of each section.
        """

        research_context: StructuredResearchOutput = await app_state["research_agent"].run(user_prompt)

        research_json = research_context.model_dump_json(indent=2)

        # synthesizer (report writer) agent
        logger.info("Research complete.")
        report_prompt = f"""
        Generate a professional report with references on the topic: '{query.prompt}'.
        Use the following research context gathered by the Researcher as your sole source of information:
        ---
        JSON SCHEMA:
        - query: the original user prompt
        - sections: a list of subquestions, each with:
            - subquestion: a refined research question
            - sources: list of articles, each with title, content, and URL

        JSON CONTENT:
        {research_json}
        ---
        
        """
        save_to_output_file(report_prompt)

        final_report = await app_state["synthesizer_agent"].run(report_prompt)

        save_to_output_file(str(final_report.output))
        logger.info("Report done")

        return final_report.output

    except requests.RequestException as e:
        logger.error(f"Ollama communication error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ollama communication error: {str(e)}")

@app.get("/")
async def root():
    return RedirectResponse(url="/app")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
