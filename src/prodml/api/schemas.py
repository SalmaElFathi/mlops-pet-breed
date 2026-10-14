from typing import Literal

from pydantic import BaseModel, Field


class BreedScore(BaseModel):
    breed: str
    confidence: float = Field(ge=0, le=1)


class PetBreedResponse(BaseModel):
    breed: str
    species: Literal["cat", "dog"]
    confidence: float = Field(ge=0, le=1)
    top_3: list[BreedScore] = Field(max_length=3)
    decision: Literal["confident", "uncertain"]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_version: str
    num_classes: int
