from collections import Counter
from pathlib import Path

import cv2

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.analyze_c2_601_segment_gid_ranking import (
    build_segment_embedding,
    build_segment_tracklet,
    rank_segment,
)

from src.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.reid_extractor import (
    ReIDExtractor,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c2.avi"
)

MIN_GAP = 20

MIN_SIDE_DETECTIONS = 25


# ============================================================
# SEGMENT IDENTITY DECISION
#
# Diagnostic only.
#
# Positive identity evidence must be:
#
#   - not identity-vetoed
#   - >= 2 joint CORE ReID + geometry supports
#   - CORE gallery max >= 0.82
#   - CORE gallery top3 >= 0.80
#
# These are NOT production thresholds.
# ============================================================

MIN_JOINT_CORE_SUPPORTS = 2

MIN_CORE_MAX = 0.82

MIN_CORE_TOP3 = 0.80


# ============================================================
# FIND VIABLE INTERNAL GAP BOUNDARIES
# ============================================================

def find_gap_boundaries(
    tracklet,
):

    detections = sorted(
        tracklet.detections,
        key=lambda detection:
            int(
                detection.frame
            ),
    )

    rows = []

    for index in range(
        1,
        len(detections),
    ):

        before = detections[
            index - 1
        ]

        after = detections[
            index
        ]

        gap = (
            int(
                after.frame
            )
            - int(
                before.frame
            )
        )

        if gap < MIN_GAP:
            continue

        before_count = index

        after_count = (
            len(
                detections
            )
            - index
        )

        if (
            before_count
            < MIN_SIDE_DETECTIONS
            or after_count
            < MIN_SIDE_DETECTIONS
        ):

            continue

        before_cx = (
            before.x1
            + before.x2
        ) / 2.0

        before_cy = (
            before.y1
            + before.y2
        ) / 2.0

        after_cx = (
            after.x1
            + after.x2
        ) / 2.0

        after_cy = (
            after.y1
            + after.y2
        ) / 2.0

        center_jump = (
            (
                after_cx
                - before_cx
            ) ** 2
            +
            (
                after_cy
                - before_cy
            ) ** 2
        ) ** 0.5

        before_diag = (
            (
                before.x2
                - before.x1
            ) ** 2
            +
            (
                before.y2
                - before.y1
            ) ** 2
        ) ** 0.5

        after_diag = (
            (
                after.x2
                - after.x1
            ) ** 2
            +
            (
                after.y2
                - after.y1
            ) ** 2
        ) ** 0.5

        scale = max(
            before_diag,
            after_diag,
            1.0,
        )

        normalized_jump = (
            center_jump
            / scale
        )

        rows.append(
            {
                "index":
                    index,

                "before_frame":
                    int(
                        before.frame
                    ),

                "after_frame":
                    int(
                        after.frame
                    ),

                "gap":
                    gap,

                "before_count":
                    before_count,

                "after_count":
                    after_count,

                "center_jump":
                    center_jump,

                "normalized_jump":
                    normalized_jump,
            }
        )

    return rows


# ============================================================
# STRONG SEGMENT GID CANDIDATES
# ============================================================

def strong_gid_candidates(
    ranked_rows,
):

    candidates = []

    for row in ranked_rows:

        if row[
            "identity_veto"
        ]:

            continue

        if (
            row[
                "joint_support_count"
            ]
            < MIN_JOINT_CORE_SUPPORTS
        ):

            continue

        if (
            row[
                "core_max"
            ]
            < MIN_CORE_MAX
        ):

            continue

        if (
            row[
                "core_top3"
            ]
            < MIN_CORE_TOP3
        ):

            continue

        candidates.append(
            row
        )

    candidates.sort(
        key=lambda row: (
            row[
                "joint_support_count"
            ],

            row[
                "core_top3"
            ],

            row[
                "core_max"
            ],
        ),
        reverse=True,
    )

    return candidates


# ============================================================
# SUMMARIZE SEGMENT
# ============================================================

def summarize_segment(
    ranked_rows,
):

    strong = strong_gid_candidates(
        ranked_rows
    )

    if not strong:

        return {
            "status":
                "NONE",

            "gid":
                None,

            "row":
                None,

            "candidate_count":
                0,
        }

    if len(
        strong
    ) > 1:

        return {
            "status":
                "AMBIGUOUS",

            "gid":
                None,

            "row":
                strong[
                    0
                ],

            "candidate_count":
                len(
                    strong
                ),

            "candidate_gids":
                [
                    row[
                        "gid"
                    ]
                    for row
                    in strong
                ],
        }

    return {
        "status":
            "UNIQUE",

        "gid":
            strong[
                0
            ][
                "gid"
            ],

        "row":
            strong[
                0
            ],

        "candidate_count":
            1,
    }


# ============================================================
# CLASSIFY BEFORE / AFTER RELATION
# ============================================================

def classify_boundary(
    before_summary,
    after_summary,
):

    before_status = (
        before_summary[
            "status"
        ]
    )

    after_status = (
        after_summary[
            "status"
        ]
    )

    if (
        before_status
        == "AMBIGUOUS"
        or after_status
        == "AMBIGUOUS"
    ):

        return "AMBIGUOUS"

    before_gid = (
        before_summary[
            "gid"
        ]
    )

    after_gid = (
        after_summary[
            "gid"
        ]
    )

    if (
        before_gid is None
        and after_gid is None
    ):

        return "UNRESOLVED"

    if (
        before_gid is None
        and after_gid is not None
    ):

        return "GAINED_IDENTITY"

    if (
        before_gid is not None
        and after_gid is None
    ):

        return "LOST_IDENTITY"

    if (
        before_gid
        == after_gid
    ):

        return "SAME_IDENTITY"

    return "IDENTITY_SHIFT"


# ============================================================
# FORMAT SEGMENT RESULT
# ============================================================

def format_summary(
    summary,
):

    status = summary[
        "status"
    ]

    if status == "NONE":

        return "NONE"

    if status == "AMBIGUOUS":

        return (
            "AMBIGUOUS "
            + str(
                summary[
                    "candidate_gids"
                ]
            )
        )

    row = summary[
        "row"
    ]

    return (
        f"GID {summary['gid']}"
        f" | joint={row['joint_support_count']}"
        f" | max={row['core_max']:.4f}"
        f" | top3={row['core_top3']:.4f}"
        f" | geom={row['geometry_support_count']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2 GAP-BASED IDENTITY CHANGE ANALYSIS"
    )

    print(
        "====================================="
    )

    print()
    print(
        "Split proposal gates:"
    )

    print(
        "  internal gap >=",
        MIN_GAP,
    )

    print(
        "  detections on each side >=",
        MIN_SIDE_DETECTIONS,
    )

    print()
    print(
        "Strong segment identity gates:"
    )

    print(
        "  joint CORE supports >=",
        MIN_JOINT_CORE_SUPPORTS,
    )

    print(
        "  CORE gallery max >=",
        MIN_CORE_MAX,
    )

    print(
        "  CORE gallery top3 >=",
        MIN_CORE_TOP3,
    )

    print()
    print(
        "These are diagnostic thresholds only."
    )

    (
        manager,
        _,
        tracklet_lookup,
        _,
    ) = replay_final_production()

    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()

    tracklet_lookup.update(
        c2_lookup
    )

    extractor = ReIDExtractor(
        device="cuda"
    )

    video = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    results = []

    synthetic_id = 900000

    try:

        for (
            tracklet,
            _,
        ) in c2_rows:

            boundaries = (
                find_gap_boundaries(
                    tracklet
                )
            )

            for boundary in boundaries:

                synthetic_id += 1

                before_tracklet = (
                    build_segment_tracklet(
                        source_tracklet=
                            tracklet,

                        start_frame=
                            tracklet.start_frame,

                        end_frame=
                            boundary[
                                "before_frame"
                            ],

                        synthetic_track_id=
                            synthetic_id,
                    )
                )

                synthetic_id += 1

                after_tracklet = (
                    build_segment_tracklet(
                        source_tracklet=
                            tracklet,

                        start_frame=
                            boundary[
                                "after_frame"
                            ],

                        end_frame=
                            tracklet.end_frame,

                        synthetic_track_id=
                            synthetic_id,
                    )
                )

                (
                    before_embedding,
                    before_frames,
                ) = build_segment_embedding(
                    extractor=
                        extractor,

                    video=
                        video,

                    tracklet=
                        before_tracklet,
                )

                (
                    after_embedding,
                    after_frames,
                ) = build_segment_embedding(
                    extractor=
                        extractor,

                    video=
                        video,

                    tracklet=
                        after_tracklet,
                )

                before_ranked = rank_segment(
                    manager=
                        manager,

                    candidate_tracklet=
                        before_tracklet,

                    candidate_embedding=
                        before_embedding,

                    tracklet_lookup=
                        tracklet_lookup,
                )

                after_ranked = rank_segment(
                    manager=
                        manager,

                    candidate_tracklet=
                        after_tracklet,

                    candidate_embedding=
                        after_embedding,

                    tracklet_lookup=
                        tracklet_lookup,
                )

                before_summary = (
                    summarize_segment(
                        before_ranked
                    )
                )

                after_summary = (
                    summarize_segment(
                        after_ranked
                    )
                )

                classification = (
                    classify_boundary(
                        before_summary,
                        after_summary,
                    )
                )

                results.append(
                    {
                        "track_id":
                            tracklet.local_track_id,

                        "before_frame":
                            boundary[
                                "before_frame"
                            ],

                        "after_frame":
                            boundary[
                                "after_frame"
                            ],

                        "gap":
                            boundary[
                                "gap"
                            ],

                        "before_count":
                            boundary[
                                "before_count"
                            ],

                        "after_count":
                            boundary[
                                "after_count"
                            ],

                        "normalized_jump":
                            boundary[
                                "normalized_jump"
                            ],

                        "before_summary":
                            before_summary,

                        "after_summary":
                            after_summary,

                        "classification":
                            classification,

                        "before_embedding_frames":
                            before_frames,

                        "after_embedding_frames":
                            after_frames,
                    }
                )

    finally:

        video.release()

    # ========================================================
    # SORT MOST INTERESTING FIRST
    # ========================================================

    priority = {
        "IDENTITY_SHIFT":
            0,

        "GAINED_IDENTITY":
            1,

        "LOST_IDENTITY":
            2,

        "AMBIGUOUS":
            3,

        "SAME_IDENTITY":
            4,

        "UNRESOLVED":
            5,
    }

    results.sort(
        key=lambda row: (
            priority[
                row[
                    "classification"
                ]
            ],

            -row[
                "normalized_jump"
            ],

            -row[
                "gap"
            ],
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    counts = Counter(
        row[
            "classification"
        ]
        for row
        in results
    )

    print()
    print(
        "CLASSIFICATION COUNTS"
    )

    print(
        "====================="
    )

    for key in [
        "IDENTITY_SHIFT",
        "GAINED_IDENTITY",
        "LOST_IDENTITY",
        "AMBIGUOUS",
        "SAME_IDENTITY",
        "UNRESOLVED",
    ]:

        print(
            f"{key:<18}: "
            f"{counts.get(key, 0)}"
        )

    print()
    print(
        "BOUNDARY RESULTS"
    )

    print(
        "================"
    )

    for row in results:

        marker = ""

        if row[
            "track_id"
        ] == 601:

            marker = "  <-- c2:601"

        print()
        print(
            f"c2:{row['track_id']}"
            f" | "
            f"{row['before_frame']}"
            f"->{row['after_frame']}"
            f" | gap="
            f"{row['gap']}"
            f" | sides="
            f"{row['before_count']}/"
            f"{row['after_count']}"
            f" | norm="
            f"{row['normalized_jump']:.3f}"
            f" | "
            f"{row['classification']}"
            f"{marker}"
        )

        print(
            "  BEFORE:",
            format_summary(
                row[
                    "before_summary"
                ]
            ),
        )

        print(
            "  AFTER :",
            format_summary(
                row[
                    "after_summary"
                ]
            ),
        )

    # ========================================================
    # EXPLICIT c2:601 CHECK
    # ========================================================

    print()
    print(
        "c2:601 SANITY CHECK"
    )

    print(
        "==================="
    )

    target_rows = [
        row

        for row in results

        if (
            row[
                "track_id"
            ] == 601

            and

            row[
                "before_frame"
            ] == 4016

            and

            row[
                "after_frame"
            ] == 4049
        )
    ]

    if not target_rows:

        print(
            "FAIL: expected boundary not found"
        )

    else:

        row = target_rows[
            0
        ]

        print(
            "Classification:",
            row[
                "classification"
            ],
        )

        print(
            "Before:",
            format_summary(
                row[
                    "before_summary"
                ]
            ),
        )

        print(
            "After:",
            format_summary(
                row[
                    "after_summary"
                ]
            ),
        )

        passed = (
            row[
                "classification"
            ]
            == "GAINED_IDENTITY"

            and

            row[
                "before_summary"
            ][
                "gid"
            ]
            is None

            and

            row[
                "after_summary"
            ][
                "gid"
            ]
            == 73
        )

        print(
            "PASS"
            if passed
            else "REVIEW REQUIRED"
        )

    print()
    print(
        "IDENTITIES MODIFIED: NO"
    )

    print(
        "No c2 tracklets were permanently split."
    )

    print(
        "Diagnostic only."
    )


if __name__ == "__main__":
    main()
