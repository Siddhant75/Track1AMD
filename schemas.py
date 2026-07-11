"""
Pydantic I/O layer for reading /input/tasks.json and writing /output/results.json.

Strict schema validation prevents INVALID_RESULTS_SCHEMA disqualification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    """Schema for a single task in /input/tasks.json."""

    task_id: str = Field(..., description="Unique identifier for the task")
    prompt: str = Field(..., description="The natural-language prompt to process")
    category: Optional[str] = Field(
        None, description="Optional category hint from the evaluation harness"
    )


class TaskOutput(BaseModel):
    """Schema for a single result in /output/results.json."""

    task_id: str = Field(..., description="Must match the corresponding TaskInput.task_id")
    answer: str = Field(..., description="The agent's response to the prompt")


def read_tasks(path: str = "/input/tasks.json") -> List[TaskInput]:
    """Read and validate all tasks from the input JSON file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        pydantic.ValidationError: If any task fails schema validation.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both a raw list and a wrapped {"tasks": [...]} format
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]

    return [TaskInput(**item) for item in data]


def write_results(results: List[TaskOutput], path: str = "/output/results.json") -> None:
    """Validate and write all results to the output JSON file.

    Creates parent directories if they do not exist.
    """
    # Ensure the output directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Serialize through Pydantic to guarantee schema compliance
    serialized = [r.model_dump() for r in results]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
