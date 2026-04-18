"""Pydantic payload schemas — the public contract between training scripts
and the mlearnweb dashboard's HTTP API.

Training side POSTs ``TrainingRecordPayload`` to
``http://{mlearnweb_host}:8000/api/training-records`` at the start of a
rolling run, then POSTs ``RunMappingPayload`` per rolling segment as they
complete.

Bump ``SCHEMA_VERSION`` when making breaking changes; consumers reject
major-mismatch payloads at the router layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


SCHEMA_VERSION = 1


class TrainingRecordPayload(BaseModel):
    """Top-level training record created when a rolling pipeline starts.

    Mirrors (a subset of) the mlearnweb ``training_records`` ORM model.
    """

    schema_version: int = Field(default=SCHEMA_VERSION)
    name: str
    experiment_name: str
    status: str = Field(default="running")
    market: Optional[str] = None
    model_class: Optional[str] = None
    dataset_class: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    notes: Optional[str] = None


class TrainingRecordUpdatePayload(BaseModel):
    """Partial update when a rolling pipeline finishes (or fails)."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    status: Optional[str] = None
    finished_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None
    error: Optional[str] = None


class RunMappingPayload(BaseModel):
    """One rolling segment → MLflow run_id mapping."""

    schema_version: int = Field(default=SCHEMA_VERSION)
    training_record_id: int
    run_id: str
    segment_index: Optional[int] = None
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    valid_start: Optional[datetime] = None
    valid_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
