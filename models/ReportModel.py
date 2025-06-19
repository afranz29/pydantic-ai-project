import re
import json

from pydantic import BaseModel, Field, field_validator, AliasChoices
from typing import List

URL_PATTERN = r"^https?://"


class ReportSection(BaseModel):
    header: str
    content: str


class ReferencesSection(BaseModel):
    header: str = "References"
    sources: List[str] = Field(
        min_length=1,
        description="A list of all source URLs used in the research. Each entry must be a valid URL string"
        )
    
    @field_validator('sources')
    def validate_urls(cls, urls):
        for url in urls:
            if not re.match(URL_PATTERN, url):
                raise ValueError(f"'{url}' is not a valid URL.")
        return urls


class Report(BaseModel):
    title: str
    abstract: str
    sections: List[ReportSection] = Field(
        min_length=1    # must have at least 1 section
    )
    references: ReferencesSection = Field(
        validation_alias=AliasChoices("references", "References"),
        description="The final section of the report containing all source URLs. This field is REQUIRED"
    )

    @field_validator('sections', mode='before')
    def sections_must_be_list(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("sections is not a valid JSON string")
        return v
