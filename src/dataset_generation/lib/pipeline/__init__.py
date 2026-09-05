"""Pipeline / Stage 抽象（Phase 4）。"""

from .segment import (
    EnsureFlightActiveStage,
    LogSegmentStage,
    SaveMetadataStage,
    SegmentPipeline,
)
from .session import SessionPipeline
from .sortie import CleanupStage, PrepareEnvStage, RunSegmentsStage, SortiePipeline
from .stages import Pipeline, PipelineCtx, Stage, StageResult

__all__ = [
    "Pipeline",
    "PipelineCtx",
    "Stage",
    "StageResult",
    "EnsureFlightActiveStage",
    "LogSegmentStage",
    "SaveMetadataStage",
    "SegmentPipeline",
    "PrepareEnvStage",
    "RunSegmentsStage",
    "CleanupStage",
    "SortiePipeline",
    "SessionPipeline",
]
