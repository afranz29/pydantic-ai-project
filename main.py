import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import uvicorn

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

    # Initialize agents
    research_agent = ResearchAgent()
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
        print(f"[REQUEST] Request recieved. User wants to research: {query.prompt}")
        
        # research agent
        user_prompt = f"""
            Reasech the topic {query.prompt} using the tools available to you, then combine all the research into one organized document with sections.
            You MUST include the source URLs at the end of each section.
        """

        research_context = await app_state["research_agent"].run(user_prompt)


        # synthesizer (report writer) agent
        print("Research complete. Writing report using the prompt: ")
        report_prompt = f"""
        Generate a professional report with references on the topic: '{query.prompt}'.
        Use the following research context gathered by the Researcher as your sole source of information:
        ---
        {research_context.output}
        ---
        """
        print(f"{report_prompt}")

        print(f"[AGENT] Synthesizer Agent called. Agent writing report\n")
        final_report = await app_state["synthesizer_agent"].run(report_prompt)

        print(f"Final Report:\n {final_report.output}")

        return Report(report=final_report.output)

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama communication error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
