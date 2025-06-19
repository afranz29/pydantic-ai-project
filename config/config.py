# config.py

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from typing import List, Literal


class WebSearchToolSettings(BaseModel):
    # In .env file, put "WEB_SEARCH_TOOL__" before each variable
    # e.g. WEB_SEARCH_TOOL__WEB_BROWSER

    # WEB_BROWSER="google"
    # MUST be one of the Literal values
    # duckduckgo - will search duckduckgo and use google if fail
    # google - will only search google
    WEB_BROWSER: Literal["google", "duckduckgo"] = "google" 

    # NUM_SEARCH_RESULTS=3
    NUM_SEARCH_RESULTS: int = 4
    
    # WEB_SEARCH_TOOL__DOMAIN_BLACKLIST='["x.com", "twitter.com"]' (comma seperated json array string)
    DOMAIN_BLACKLIST: List[str] = []
    
    # BLACKLIST_ON=True
    # Whether to filter out the blacklist
    BLACKLIST_ON: bool = True


class ResearchAgentSettings(BaseModel):
    # Number of questions/topics the research agent has to generate
    NUM_SUB_QUESTIONS: str = "5"


class SynthAgentSettings(BaseModel):
    # The number of words the report should contain excluding sources
    WORD_COUNT_REQ: str = "800"


class Settings(BaseSettings):
    # FIX: Use Field(default_factory=...) for nested models
    # This is the key change to fix the startup crash.
    WEB_SEARCH_TOOL: WebSearchToolSettings = Field(default_factory=WebSearchToolSettings)
    RESEARCH_AGENT: ResearchAgentSettings = Field(default_factory=ResearchAgentSettings)
    SYNTH_AGENT: SynthAgentSettings = Field(default_factory=SynthAgentSettings)

    # App-level secrets (ensure these are in your .env file)
    OLLAMA_HOST: str
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # This tells pydantic-settings how to map nested env vars
        # e.g., WEB_SEARCH_TOOL__NUM_SEARCH_RESULTS=5
        env_nested_delimiter = '__'

# Create a single, importable instance of the settings
app_settings = Settings()