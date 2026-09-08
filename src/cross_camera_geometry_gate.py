from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .terrace_geometry import (
        detection_ground_point,
        ground_distance,
        normalize_camera_id,
    )
except ImportError:
    from terrace_geometry import (
        detection_ground_point,
        ground_distance,
        normalize_camera_id,
    )


MIN_SHARED_FRAMES = 10

# Temporary Terrace analysis bands.
GEOMETRY_SUPPORT_MAX = 30.0
GEOMETRY_CONTRADICTION_MIN = 80.0


@dataclass
class GeometryEvidence:

    status: str

    shared_frames: int
    valid_frames: int

    median_distance: Optional[float]
    mean_distance: Optional[float]
    p90_distance: Optional[float]



def detections_by_frame(tracklet):

    return {
        detection.frame: detection
        for detection in tracklet.detections
    }



def evaluate_cross_camera_geometry(
    tracklet_a,
    tracklet_b,
    homographies,
):

    camera_a = normalize_camera_id(
        tracklet_a.camera_id
    )

    camera_b = normalize_camera_id(
        tracklet_b.camera_id
    )


    # This module is specifically for
    # cross-camera comparisons.
    if camera_a == camera_b:

        return GeometryEvidence(
            status="UNKNOWN",
            shared_frames=0,
            valid_frames=0,
            median_distance=None,
            mean_distance=None,
            p90_distance=None,
        )


    detections_a = detections_by_frame(
        tracklet_a
    )

    detections_b = detections_by_frame(
        tracklet_b
    )


    shared_frames = sorted(
        set(detections_a)
        &
        set(detections_b)
    )


    if len(shared_frames) < MIN_SHARED_FRAMES:

        return GeometryEvidence(
            status="UNKNOWN",
            shared_frames=len(shared_frames),
            valid_frames=0,
            median_distance=None,
            mean_distance=None,
            p90_distance=None,
        )


    distances = []


    for frame in shared_frames:

        point_a = detection_ground_point(
            detections_a[frame],
            camera_a,
            homographies,
        )

        point_b = detection_ground_point(
            detections_b[frame],
            camera_b,
            homographies,
        )

        distance = ground_distance(
            point_a,
            point_b,
        )

        if np.isfinite(distance):

            distances.append(
                distance
            )


    if len(distances) < MIN_SHARED_FRAMES:

        return GeometryEvidence(
            status="UNKNOWN",
            shared_frames=len(shared_frames),
            valid_frames=len(distances),
            median_distance=None,
            mean_distance=None,
            p90_distance=None,
        )


    distances = np.asarray(
        distances,
        dtype=np.float64,
    )


    median = float(
        np.median(distances)
    )

    mean = float(
        np.mean(distances)
    )

    p90 = float(
        np.percentile(
            distances,
            90,
        )
    )


    # --------------------------------------------------------
    # Conservative three-state geometry decision.
    # --------------------------------------------------------

    if median <= GEOMETRY_SUPPORT_MAX:

        status = "SUPPORTED"

    elif median >= GEOMETRY_CONTRADICTION_MIN:

        status = "CONTRADICTED"

    else:

        status = "UNKNOWN"


    return GeometryEvidence(
        status=status,
        shared_frames=len(shared_frames),
        valid_frames=len(distances),
        median_distance=median,
        mean_distance=mean,
        p90_distance=p90,
    )
