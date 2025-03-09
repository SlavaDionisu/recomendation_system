from pydantic import BaseModel
from datetime import datetime
from typing import List

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True
