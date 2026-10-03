"""Portable cache record types used by topology-data fixtures and converters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    frame: int
    bbox: list[float]
    image_path: Path
    area: float
    occluded: bool = False


@dataclass
class WindowTrack:
    scene_id: str
    view_id: int
    track_id: int
    start: int
    end: int
    observations: list[Observation]
    tracking_quality: float = 1.0
    tracking_metadata: dict[str, float] | None = None

    @property
    def key(self) -> tuple[str, int, int, int]:
        return self.scene_id, self.view_id, self.start, self.track_id

    @property
    def frames(self) -> set[int]:
        return {observation.frame for observation in self.observations}
