from pydantic import BaseModel,Field
from typing import Literal

class ManifestRecord(BaseModel):
    image_id: str
    path: str
    breed: str
    species: Literal["cat", "dog"]
    class_index: int = Field(ge=0, le=36)
    split: Literal["train", "val", "test"]
    corruption: str | None = None
    severity: int = Field(default=0, ge=0, le=3)
    width: int = Field(ge=32)
    height: int = Field(ge=32)