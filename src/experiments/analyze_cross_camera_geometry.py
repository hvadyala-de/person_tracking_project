from pathlib import Path

import numpy as np

from src.build_tracklets import load_tracklets
from src.terrace_geometry import (
    load_ground_homographies,
    detection_ground_point,
    ground_distance,
    normalize_camera_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
)


# ============================================================
# VERIFIED SAME-PERSON CROSS-CAMERA PAIRS
# ============================================================

SAME_PAIRS = [
    (112, 41),
    (651, 560),
    (399, 503),
    (701, 639),
    (707, 536),
    (492, 511),
    (209, 227),
    (1, 1),
    (255, 298),
]


# ============================================================
# KNOWN DIFFERENT / LOOK-ALIKE GROUPS
#
# These are especially useful because appearance ReID
# considered some of them very similar.
# ============================================================

DIFFERENT_PAIRS = [
    (701, 511),
    (492, 639),
    (399, 298),
    (255, 503),
]


def load_camera_tracklets(camera_id):

    path = (
        TRACK_DIR
        / f"tracks_c{camera_id}.csv"
    )

    tracklets = load_tracklets(
        path
    )

    if isinstance(tracklets, dict):
        values = tracklets.values()
    else:
        values = tracklets

    return {
        tracklet.local_track_id: tracklet
        for tracklet in values
    }


def detections_by_frame(tracklet):

    return {
        detection.frame: detection
        for detection in tracklet.detections
    }


def geometry_stats(
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

    detections_a = detections_by_frame(
        tracklet_a
    )

    detections_b = detections_by_frame(
        tracklet_b
    )

    shared_frames = sorted(
        set(detections_a)
        & set(detections_b)
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

    if not distances:

        return None

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    return {
        "shared": len(shared_frames),
        "valid": len(distances),
        "min": float(np.min(distances)),
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
        "p90": float(np.percentile(distances, 90)),
        "max": float(np.max(distances)),
    }


def print_pair(
    label,
    c0_id,
    c1_id,
    stats,
):

    if stats is None:

        print(
            f"{label:<10} "
            f"c0:{c0_id:<4} "
            f"c1:{c1_id:<4} | "
            f"NO SHARED FRAMES"
        )

        return

    print(
        f"{label:<10} "
        f"c0:{c0_id:<4} "
        f"c1:{c1_id:<4} | "
        f"shared={stats['shared']:<4} | "
        f"median={stats['median']:>8.2f} | "
        f"mean={stats['mean']:>8.2f} | "
        f"p90={stats['p90']:>8.2f} | "
        f"min={stats['min']:>8.2f} | "
        f"max={stats['max']:>8.2f}"
    )


def main():

    homographies = (
        load_ground_homographies()
    )

    c0 = load_camera_tracklets(0)
    c1 = load_camera_tracklets(1)


    print()
    print("VERIFIED SAME-PERSON GEOMETRY")
    print("=============================")

    same_medians = []

    for c0_id, c1_id in SAME_PAIRS:

        stats = geometry_stats(
            c0[c0_id],
            c1[c1_id],
            homographies,
        )

        print_pair(
            "SAME",
            c0_id,
            c1_id,
            stats,
        )

        if stats is not None:
            same_medians.append(
                stats["median"]
            )


    print()
    print("LOOK-ALIKE DIFFERENT-PERSON GEOMETRY")
    print("====================================")

    different_medians = []

    for c0_id, c1_id in DIFFERENT_PAIRS:

        stats = geometry_stats(
            c0[c0_id],
            c1[c1_id],
            homographies,
        )

        print_pair(
            "DIFFERENT",
            c0_id,
            c1_id,
            stats,
        )

        if stats is not None:
            different_medians.append(
                stats["median"]
            )


    print()
    print("SUMMARY")
    print("=======")

    if same_medians:

        print(
            "Same-person median-distance range:",
            f"{min(same_medians):.2f}",
            "to",
            f"{max(same_medians):.2f}",
        )

    if different_medians:

        print(
            "Different-person median-distance range:",
            f"{min(different_medians):.2f}",
            "to",
            f"{max(different_medians):.2f}",
        )


if __name__ == "__main__":
    main()
