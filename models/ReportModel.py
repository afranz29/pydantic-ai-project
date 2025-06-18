from pydantic import BaseModel
from typing import List

class ReportSection(BaseModel):
    header: str
    content: str

class Report(BaseModel):
    title: str
    abstract: str
    sections: List[ReportSection]
    sources: List[str] = []
