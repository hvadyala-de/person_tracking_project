from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    frame: int
    timestamp: float

    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float


@dataclass
class Tracklet:
    camera_id: str
    local_track_id: int

    detections: List[Detection] = field(
        default_factory=list
    )


    def add_detection(
        self,
        detection: Detection,
    ) -> None:

        self.detections.append(
            detection
        )


    @property
    def num_detections(self) -> int:

        return len(
            self.detections
        )


    @property
    def start_frame(self) -> int:

        if not self.detections:
            return -1

        return self.detections[0].frame


    @property
    def end_frame(self) -> int:

        if not self.detections:
            return -1

        return self.detections[-1].frame


    @property
    def start_time(self) -> float:

        if not self.detections:
            return 0.0

        return self.detections[0].timestamp


    @property
    def end_time(self) -> float:

        if not self.detections:
            return 0.0

        return self.detections[-1].timestamp


    @property
    def duration(self) -> float:

        if not self.detections:
            return 0.0

        return (
            self.end_time
            - self.start_time
        )


    @property
    def mean_confidence(self) -> float:

        if not self.detections:
            return 0.0

        total = sum(
            detection.confidence
            for detection
            in self.detections
        )

        return (
            total
            / len(self.detections)
        )


    @property
    def max_frame_gap(self) -> int:

        if len(self.detections) < 2:
            return 0

        frames = [
            detection.frame
            for detection
            in self.detections
        ]

        gaps = [
            frames[i]
            - frames[i - 1]
            for i in range(
                1,
                len(frames),
            )
        ]

        return max(gaps)


    def sort_detections(self) -> None:

        self.detections.sort(
            key=lambda detection:
            detection.frame
        )


    def summary(self) -> str:

        return (
            f"{self.camera_id}:"
            f"{self.local_track_id} | "
            f"detections={self.num_detections} | "
            f"frames={self.start_frame}-"
            f"{self.end_frame} | "
            f"duration={self.duration:.2f}s | "
            f"confidence="
            f"{self.mean_confidence:.3f} | "
            f"max_gap="
            f"{self.max_frame_gap}"
            )
