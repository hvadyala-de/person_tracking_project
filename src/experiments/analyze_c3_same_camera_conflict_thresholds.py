from itertools import combinations

import numpy as np

from src.experiments.analyze_c3_geometry_candidates import (
    load_camera_tracklets,
)

from src.identity_compatibility import (
    MIN_SHARED_FRAMES,
    MIN_SEPARATED_FRAMES,
    MIN_SEPARATED_RATIO,
    bbox_diagonal,
    bbox_iou,
    center_distance,
    detections_by_frame,
    same_camera_conflict,
)


# ============================================================
# PURPOSE
#
# Audit a possible scale-aware extension of the existing
# same-camera conflict veto.
#
# This file does NOT modify production code.
# This file does NOT modify production thresholds.
# ============================================================


CAMERA_ID = 3

TARGET_TRACK_A = 12
TARGET_TRACK_B = 15


# ============================================================
# PROPOSED AUDIT-ONLY RULE
#
# Existing rule:
#
#   IoU <= 0.05
#   center distance >= max(45 px, 0.75 * bbox diagonal)
#
# Candidate rule being measured here:
#
#   IoU <= 0.10
#   and
#   either center or bottom-center distance
#       >= max(20 px, 0.25 * bbox diagonal)
#
# The pair must still satisfy the existing persistence gate:
#
#   >= 10 separated simultaneous frames
#   >= 80% of simultaneous frames separated
#
# These are NOT production thresholds.
# ============================================================

PROPOSED_MAX_IOU = 0.10

PROPOSED_MIN_ABSOLUTE_DISTANCE = 20.0

PROPOSED_MIN_RELATIVE_DISTANCE = 0.25


# ============================================================
# BOX
# ============================================================

def detection_box(
    detection,
):

    return (
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    )


# ============================================================
# BOTTOM CENTER
# ============================================================

def bottom_center(
    box,
):

    x1, y1, x2, y2 = box

    return np.asarray(
        [
            (
                x1
                + x2
            )
            / 2.0,
            y2,
        ],
        dtype=np.float64,
    )


# ============================================================
# BOTTOM-CENTER DISTANCE
# ============================================================

def bottom_center_distance(
    box_a,
    box_b,
):

    return float(
        np.linalg.norm(
            bottom_center(
                box_a
            )
            -
            bottom_center(
                box_b
            )
        )
    )


# ============================================================
# PROPOSED FRAME-LEVEL SEPARATION
# ============================================================

def proposed_spatially_separated(
    detection_a,
    detection_b,
):

    box_a = detection_box(
        detection_a
    )

    box_b = detection_box(
        detection_b
    )


    iou = bbox_iou(
        box_a,
        box_b,
    )


    if iou > PROPOSED_MAX_IOU:

        return False


    scale = max(
        bbox_diagonal(
            box_a
        ),
        bbox_diagonal(
            box_b
        ),
        1.0,
    )


    required_distance = max(
        PROPOSED_MIN_ABSOLUTE_DISTANCE,
        PROPOSED_MIN_RELATIVE_DISTANCE
        * scale,
    )


    center = center_distance(
        box_a,
        box_b,
    )


    bottom = bottom_center_distance(
        box_a,
        box_b,
    )


    return (
        center
        >= required_distance

        or

        bottom
        >= required_distance
    )


# ============================================================
# PAIR ANALYSIS
# ============================================================

def analyze_pair(
    tracklet_a,
    tracklet_b,
):

    frames_a = detections_by_frame(
        tracklet_a
    )

    frames_b = detections_by_frame(
        tracklet_b
    )


    shared_frames = sorted(
        set(
            frames_a
        )
        &
        set(
            frames_b
        )
    )


    if (
        len(
            shared_frames
        )
        < MIN_SHARED_FRAMES
    ):

        return None


    ious = []

    center_distances = []

    bottom_distances = []

    normalized_centers = []

    normalized_bottoms = []

    proposed_separated = 0


    for frame in shared_frames:

        detection_a = frames_a[
            frame
        ]

        detection_b = frames_b[
            frame
        ]


        box_a = detection_box(
            detection_a
        )

        box_b = detection_box(
            detection_b
        )


        iou = bbox_iou(
            box_a,
            box_b,
        )


        center = center_distance(
            box_a,
            box_b,
        )


        bottom = bottom_center_distance(
            box_a,
            box_b,
        )


        scale = max(
            bbox_diagonal(
                box_a
            ),
            bbox_diagonal(
                box_b
            ),
            1.0,
        )


        ious.append(
            iou
        )

        center_distances.append(
            center
        )

        bottom_distances.append(
            bottom
        )

        normalized_centers.append(
            center
            / scale
        )

        normalized_bottoms.append(
            bottom
            / scale
        )


        if proposed_spatially_separated(
            detection_a,
            detection_b,
        ):

            proposed_separated += 1


    proposed_ratio = (
        proposed_separated
        / len(
            shared_frames
        )
    )


    proposed_conflict = (
        proposed_separated
        >= MIN_SEPARATED_FRAMES

        and

        proposed_ratio
        >= MIN_SEPARATED_RATIO
    )


    current_conflict = same_camera_conflict(
        tracklet_a,
        tracklet_b,
    )


    return {
        "track_a":
            tracklet_a.local_track_id,

        "track_b":
            tracklet_b.local_track_id,

        "shared":
            len(
                shared_frames
            ),

        "first_shared":
            shared_frames[
                0
            ],

        "last_shared":
            shared_frames[
                -1
            ],

        "current_conflict":
            current_conflict,

        "proposed_conflict":
            proposed_conflict,

        "proposed_separated":
            proposed_separated,

        "proposed_ratio":
            proposed_ratio,

        "iou_median":
            float(
                np.median(
                    ious
                )
            ),

        "iou_max":
            float(
                np.max(
                    ious
                )
            ),

        "center_median":
            float(
                np.median(
                    center_distances
                )
            ),

        "bottom_median":
            float(
                np.median(
                    bottom_distances
                )
            ),

        "normalized_center_median":
            float(
                np.median(
                    normalized_centers
                )
            ),

        "normalized_bottom_median":
            float(
                np.median(
                    normalized_bottoms
                )
            ),
    }


# ============================================================
# PRINT ROW
# ============================================================

def print_row(
    row,
):

    print(
        f"c3:{row['track_a']}"
        f" <-> "
        f"c3:{row['track_b']}"
        f" | shared="
        f"{row['shared']}"
        f" | current="
        f"{row['current_conflict']}"
        f" | proposed="
        f"{row['proposed_conflict']}"
        f" | proposed_sep="
        f"{row['proposed_separated']}"
        f"/"
        f"{row['shared']}"
        f" ({row['proposed_ratio']:.3f})"
        f" | IoU med/max="
        f"{row['iou_median']:.3f}/"
        f"{row['iou_max']:.3f}"
        f" | center med="
        f"{row['center_median']:.2f}"
        f" ({row['normalized_center_median']:.3f})"
        f" | bottom med="
        f"{row['bottom_median']:.2f}"
        f" ({row['normalized_bottom_median']:.3f})"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c3 SAME-CAMERA CONFLICT THRESHOLD AUDIT"
    )

    print(
        "======================================="
    )


    print()
    print(
        "No production code is modified."
    )

    print(
        "No production thresholds are modified."
    )


    print()
    print(
        "EXISTING PERSISTENCE GATE"
    )

    print(
        "========================="
    )

    print(
        "Minimum shared frames:",
        MIN_SHARED_FRAMES,
    )

    print(
        "Minimum separated frames:",
        MIN_SEPARATED_FRAMES,
    )

    print(
        "Minimum separated ratio:",
        MIN_SEPARATED_RATIO,
    )


    print()
    print(
        "AUDIT-ONLY CANDIDATE FRAME RULE"
    )

    print(
        "==============================="
    )

    print(
        "Maximum IoU:",
        PROPOSED_MAX_IOU,
    )

    print(
        "Minimum absolute distance:",
        PROPOSED_MIN_ABSOLUTE_DISTANCE,
    )

    print(
        "Minimum relative distance:",
        PROPOSED_MIN_RELATIVE_DISTANCE,
    )


    tracklets = load_camera_tracklets(
        CAMERA_ID
    )


    rows = []


    for tracklet_a, tracklet_b in combinations(
        tracklets.values(),
        2,
    ):

        row = analyze_pair(
            tracklet_a,
            tracklet_b,
        )


        if row is not None:

            rows.append(
                row
            )


    rows.sort(
        key=lambda row: (
            row[
                "track_a"
            ],
            row[
                "track_b"
            ],
        )
    )


    print()
    print(
        "PAIR POPULATION"
    )

    print(
        "==============="
    )

    print(
        "Useful c3 tracklets:",
        len(
            tracklets
        ),
    )

    print(
        "Pairs with >=",
        MIN_SHARED_FRAMES,
        "actual simultaneous detections:",
        len(
            rows
        ),
    )


    current_conflicts = [
        row

        for row in rows

        if row[
            "current_conflict"
        ]
    ]


    proposed_conflicts = [
        row

        for row in rows

        if row[
            "proposed_conflict"
        ]
    ]


    newly_blocked = [
        row

        for row in rows

        if (
            not row[
                "current_conflict"
            ]

            and

            row[
                "proposed_conflict"
            ]
        )
    ]


    no_longer_blocked = [
        row

        for row in rows

        if (
            row[
                "current_conflict"
            ]

            and

            not row[
                "proposed_conflict"
            ]
        )
    ]


    print()
    print(
        "CONFLICT COUNTS"
    )

    print(
        "==============="
    )

    print(
        "Current production conflicts:",
        len(
            current_conflicts
        ),
    )

    print(
        "Candidate-rule conflicts:",
        len(
            proposed_conflicts
        ),
    )

    print(
        "Newly blocked pairs:",
        len(
            newly_blocked
        ),
    )

    print(
        "Current conflicts lost by candidate rule:",
        len(
            no_longer_blocked
        ),
    )


    # ========================================================
    # TARGET
    # ========================================================

    target = None


    for row in rows:

        pair = {
            row[
                "track_a"
            ],
            row[
                "track_b"
            ],
        }


        if pair == {
            TARGET_TRACK_A,
            TARGET_TRACK_B,
        }:

            target = row

            break


    print()
    print(
        "TARGET c3:12 <-> c3:15"
    )

    print(
        "======================="
    )


    if target is None:

        print(
            "TARGET NOT FOUND"
        )

    else:

        print_row(
            target
        )


    # ========================================================
    # NEWLY BLOCKED
    # ========================================================

    print()
    print(
        "NEWLY BLOCKED BY CANDIDATE RULE"
    )

    print(
        "==============================="
    )


    if not newly_blocked:

        print(
            "NONE"
        )

    else:

        ranked = sorted(
            newly_blocked,
            key=lambda row: (
                -row[
                    "proposed_ratio"
                ],

                -row[
                    "shared"
                ],

                row[
                    "iou_median"
                ],
            ),
        )


        for row in ranked:

            print_row(
                row
            )


    # ========================================================
    # CURRENT CONFLICTS LOST
    # ========================================================

    print()
    print(
        "CURRENT CONFLICTS LOST BY CANDIDATE RULE"
    )

    print(
        "========================================"
    )


    if not no_longer_blocked:

        print(
            "NONE"
        )

    else:

        for row in no_longer_blocked:

            print_row(
                row
            )


    print()
    print(
        "AUDIT SUMMARY"
    )

    print(
        "============="
    )


    if target is None:

        print(
            "FAIL: target c3:12/c3:15 was not evaluated."
        )

    elif target[
        "current_conflict"
    ]:

        print(
            "UNEXPECTED: current production already "
            "flags the target."
        )

    elif not target[
        "proposed_conflict"
    ]:

        print(
            "FAIL: candidate rule still does not "
            "block c3:12/c3:15."
        )

    else:

        print(
            "TARGET BLOCKED BY CANDIDATE RULE."
        )


    if no_longer_blocked:

        print(
            "WARNING: candidate rule loses existing "
            "production conflicts."
        )

    else:

        print(
            "Candidate rule preserves all current "
            "production conflicts."
        )


    print()
    print(
        "No production decision is made by this diagnostic."
    )


if __name__ == "__main__":
    main()
