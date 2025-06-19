from pydantic import BaseModel


class UserQuery(BaseModel):
    prompt: str

class SubQuery(BaseModel):
    sub_prompt: str