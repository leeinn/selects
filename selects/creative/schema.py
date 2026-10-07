"""Versioned, agent-facing creative reviews. Scores describe preview evidence."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
Decision = Literal["hero", "grade", "story", "maybe", "reject"]
DECISIONS = ("hero", "grade", "story", "maybe", "reject")
SELECTED = frozenset({"hero", "grade", "story"})
Score = Annotated[float, Field(ge=0, le=10, allow_inf_nan=False, strict=True)]
Text = Annotated[str, Field(min_length=1, max_length=4000)]
WEIGHTS = {
    "composition": 0.22,
    "light": 0.18,
    "moment": 0.15,
    "grading_potential": 0.20,
    "story_value": 0.10,
    "uniqueness": 0.15,
}


class CreativeReview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    photo_id: Annotated[int, Field(gt=0, strict=True)]
    filename: Text
    sha256: str | None = None
    technical_score: Score
    composition: Score
    light: Score
    moment: Score
    story_value: Score
    uniqueness: Score
    grading_potential: Score
    creative_score: Score | None = None
    decision: Decision
    reason: Text
    grade_direction: Annotated[str, Field(max_length=4000)]
    reviewer: Text = "agent"
    # Technical quality is a gate only for an observed unrecoverable failure.
    technical_failure: bool = False
    # Optional editing annotations support diversity without adding a model.
    scene_type: (
        Literal[
            "establishing",
            "landscape",
            "human",
            "portrait",
            "transition",
            "transport",
            "food",
            "detail",
            "weather",
            "other",
        ]
        | None
    ) = None
    location: str | None = None
    subject: str | None = None
    composition_key: str | None = None
    moment_exception: (
        Literal[
            "expression",
            "orientation",
            "environment_detail",
            "sequence",
            "composition",
        ]
        | None
    ) = None
    moment_exception_reason: str | None = None

    @model_validator(mode="after")
    def check_scores_and_gate(self):
        calculated = round(sum(getattr(self, key) * weight for key, weight in WEIGHTS.items()), 2)
        if self.creative_score is not None and abs(self.creative_score - calculated) > 0.011:
            raise ValueError(f"creative_score must match the weighted score ({calculated})")
        self.creative_score = calculated
        if self.technical_failure and self.decision in SELECTED:
            raise ValueError("an unrecoverable technical failure cannot be hero/grade/story")
        if self.decision in SELECTED and not self.grade_direction:
            raise ValueError("selected photos need a short grade_direction")
        if self.moment_exception and not self.moment_exception_reason:
            raise ValueError("moment_exception needs a visual justification")
        return self


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SCHEMA_VERSION
    library: str | None = None
    reviews: list[CreativeReview]
