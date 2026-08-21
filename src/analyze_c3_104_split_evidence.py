from pathlib import Path

import cv2
import numpy as np

from src.analyze_c2_601_segment_gid_ranking import (
    analyze_gid,
    build_segment_embedding,
    build_segment_tracklet,
    cosine_similarity,
    rank_segment,
)

from src.analyze_c2_gap_identity_changes import (
    strong_gid_candidates,
    summarize_segment,
)

from src.analyze_c3_gap_identity_changes import (
    build_validated_reference_state,
)

from src.analyze_c3_geometry_candidates import (
    load_camera_tracklets,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.reid_extractor import (
    ReIDExtractor,
)

from src.run_c2_production_policy_experiment import (
    get_member_trust_safe,
)

from src.terrace_geometry import (
    detection_ground_point,
    ground_distance,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMERA_ID = 3

TARGET_TRACK_ID = 104

BEFORE_END = 1821

AFTER_START = 1848

TARGET_GID = 11

FPS = 25.0

TOP_ALTERNATIVE_GIDS = 10

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c3.avi"
)


# ============================================================
# MEMBER KEY
# ============================================================

def member_key(
    member,
):

    return (
        normalize_camera_id(
            member.camera_id
        ),
        member.local_track_id,
    )


# ============================================================
# FORMAT OPTIONAL FLOAT
# ============================================================

def format_optional(
    value,
    digits=2,
):

    if value is None:
        return "None"

    return (
        f"{float(value):.{digits}f}"
    )


# ============================================================
# PRINT IDENTITY COMPOSITION
# ============================================================

def print_identity_composition(
    manager,
    gid,
):

    identity = manager.identities.get(
        gid
    )

    if identity is None:

        raise RuntimeError(
            f"GID {gid} does not exist"
        )


    core_keys = manager.core_members.get(
        gid,
        set(),
    )


    rows = []


    for member in identity.members:

        key = member_key(
            member
        )

        trust = get_member_trust_safe(
            manager,
            gid,
            key,
        )

        rows.append(
            {
                "camera_id":
                    key[
                        0
                    ],

                "track_id":
                    key[
                        1
                    ],

                "core":
                    key
                    in core_keys,

                "trust":
                    trust,
            }
        )


    rows.sort(
        key=lambda row: (
            row[
                "camera_id"
            ],

            row[
                "track_id"
            ],
        )
    )


    print()
    print(
        f"GID {gid} MEMBERS"
    )

    print(
        "=" * 72
    )


    for row in rows:

        marker = (
            "CORE"
            if row[
                "core"
            ]
            else "NONCORE"
        )

        trust = (
            row[
                "trust"
            ]
            if row[
                "trust"
            ]
            is not None
            else "-"
        )


        print(
            f"c{row['camera_id']}:"
            f"{row['track_id']:<8}"
            f" | {marker:<7}"
            f" | trust={trust}"
        )


    print()
    print(
        "Total members:",
        len(
            rows
        ),
    )

    print(
        "CORE members:",
        sum(
            1

            for row in rows

            if row[
                "core"
            ]
        ),
    )


# ============================================================
# PRINT ONE GID SUMMARY
# ============================================================

def print_gid_summary(
    title,
    result,
):

    print()
    print(
        title
    )

    print(
        "=" * 72
    )


    if result is None:

        print(
            "NO RESULT"
        )

        return


    print(
        f"GID:                  "
        f"{result['gid']}"
    )

    print(
        f"CORE gallery count:   "
        f"{result['core_count']}"
    )

    print(
        f"CORE gallery max:     "
        f"{result['core_max']:.4f}"
    )

    print(
        f"CORE gallery top3:    "
        f"{result['core_top3']:.4f}"
    )

    print(
        f"CORE gallery mean:    "
        f"{result['core_mean']:.4f}"
    )

    print(
        f"Geometry supports:    "
        f"{result['geometry_support_count']}"
    )

    print(
        f"Joint supports:       "
        f"{result['joint_support_count']}"
    )

    print(
        f"CORE contradictions:  "
        f"{result['core_contradiction_count']}"
    )

    print(
        f"Identity veto:        "
        f"{result['identity_veto']}"
    )


# ============================================================
# PRINT EXACT JOINT SUPPORT MEMBERS
# ============================================================

def print_joint_support_members(
    title,
    result,
):

    print()
    print(
        title
    )

    print(
        "=" * 72
    )


    if result is None:

        print(
            "NO RESULT"
        )

        return


    rows = result[
        "joint_rows"
    ]


    if not rows:

        print(
            "NONE"
        )

        return


    for row in rows:

        print(
            f"c{row['camera_id']}:"
            f"{row['track_id']:<8}"
            f" | ReID="
            f"{row['direct_similarity']:.4f}"
            f" | geom="
            f"{row['geometry_status']:<12}"
            f" | shared="
            f"{row['shared_frames']:<4}"
            f" | median="
            f"{format_optional(row['median_distance'])}"
            f" | JOINT+"
        )


# ============================================================
# INDEX CORE RELATIONS BY MEMBER
# ============================================================

def core_relation_map(
    result,
):

    if result is None:

        return {}


    return {
        (
            row[
                "camera_id"
            ],
            row[
                "track_id"
            ],
        ):
            row

        for row in result[
            "core_rows"
        ]
    }


# ============================================================
# PRINT BEFORE / AFTER GID CORE EVIDENCE SIDE BY SIDE
# ============================================================

def print_core_comparison(
    before_result,
    after_result,
):

    before_rows = core_relation_map(
        before_result
    )

    after_rows = core_relation_map(
        after_result
    )


    keys = sorted(
        set(
            before_rows
        )
        |
        set(
            after_rows
        )
    )


    print()
    print(
        f"GID {TARGET_GID} CORE MEMBER EVIDENCE:"
        f" BEFORE vs AFTER"
    )

    print(
        "=" * 72
    )


    if not keys:

        print(
            "NO CORE RELATIONS AVAILABLE"
        )

        return


    for key in keys:

        camera_id, track_id = key

        before = before_rows.get(
            key
        )

        after = after_rows.get(
            key
        )


        print()
        print(
            f"CORE member c{camera_id}:{track_id}"
        )


        if before is None:

            print(
                "  BEFORE: unavailable"
            )

        else:

            before_flags = []

            if before[
                "geometry_supported"
            ]:

                before_flags.append(
                    "GEOM+"
                )

            if before[
                "interesting_joint"
            ]:

                before_flags.append(
                    "JOINT+"
                )

            if (
                before[
                    "geometry_status"
                ]
                == "CONTRADICTED"
            ):

                before_flags.append(
                    "CONTRADICTED"
                )


            print(
                "  BEFORE:"
                f" ReID="
                f"{before['direct_similarity']:.4f}"
                f" | geom="
                f"{before['geometry_status']}"
                f" | shared="
                f"{before['shared_frames']}"
                f" | median="
                f"{format_optional(before['median_distance'])}"
                f" | flags="
                f"{','.join(before_flags) if before_flags else '-'}"
            )


        if after is None:

            print(
                "  AFTER:  unavailable"
            )

        else:

            after_flags = []

            if after[
                "geometry_supported"
            ]:

                after_flags.append(
                    "GEOM+"
                )

            if after[
                "interesting_joint"
            ]:

                after_flags.append(
                    "JOINT+"
                )

            if (
                after[
                    "geometry_status"
                ]
                == "CONTRADICTED"
            ):

                after_flags.append(
                    "CONTRADICTED"
                )


            print(
                "  AFTER: "
                f" ReID="
                f"{after['direct_similarity']:.4f}"
                f" | geom="
                f"{after['geometry_status']}"
                f" | shared="
                f"{after['shared_frames']}"
                f" | median="
                f"{format_optional(after['median_distance'])}"
                f" | flags="
                f"{','.join(after_flags) if after_flags else '-'}"
            )


# ============================================================
# SORT GID RANKING
# ============================================================

def gid_rank_key(
    row,
):

    return (
        0
        if row[
            "identity_veto"
        ]
        else 1,

        row[
            "joint_support_count"
        ],

        row[
            "geometry_support_count"
        ],

        row[
            "core_top3"
        ],

        row[
            "core_max"
        ],
    )


# ============================================================
# PRINT TOP GID ALTERNATIVES
# ============================================================

def print_top_gid_alternatives(
    title,
    ranked_rows,
    exclude_gid=None,
):

    rows = [
        row

        for row in ranked_rows

        if (
            exclude_gid is None
            or row[
                "gid"
            ]
            != exclude_gid
        )
    ]


    rows.sort(
        key=gid_rank_key,
        reverse=True,
    )


    rows = rows[
        :TOP_ALTERNATIVE_GIDS
    ]


    print()
    print(
        title
    )

    print(
        "=" * 72
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for row in rows:

        print(
            f"GID {row['gid']:<3}"
            f" | veto="
            f"{str(row['identity_veto']):<5}"
            f" | joint="
            f"{row['joint_support_count']:<2}"
            f" | geom+="
            f"{row['geometry_support_count']:<2}"
            f" | contradictions="
            f"{row['core_contradiction_count']:<2}"
            f" | max="
            f"{row['core_max']:.4f}"
            f" | top3="
            f"{row['core_top3']:.4f}"
            f" | mean="
            f"{row['core_mean']:.4f}"
        )


# ============================================================
# FIND BOUNDARY DETECTIONS
# ============================================================

def find_boundary_detections(
    source_tracklet,
):

    detections = sorted(
        source_tracklet.detections,
        key=lambda detection:
            int(
                detection.frame
            ),
    )


    before_candidates = [
        detection

        for detection in detections

        if (
            int(
                detection.frame
            )
            <= BEFORE_END
        )
    ]


    after_candidates = [
        detection

        for detection in detections

        if (
            int(
                detection.frame
            )
            >= AFTER_START
        )
    ]


    if not before_candidates:

        raise RuntimeError(
            "No BEFORE boundary detection"
        )


    if not after_candidates:

        raise RuntimeError(
            "No AFTER boundary detection"
        )


    return (
        before_candidates[
            -1
        ],
        after_candidates[
            0
        ],
    )


# ============================================================
# DETECTION CENTER
# ============================================================

def detection_center(
    detection,
):

    return (
        (
            detection.x1
            + detection.x2
        )
        / 2.0,

        (
            detection.y1
            + detection.y2
        )
        / 2.0,
    )


# ============================================================
# DETECTION BOTTOM CENTER
# ============================================================

def detection_bottom_center(
    detection,
):

    return (
        (
            detection.x1
            + detection.x2
        )
        / 2.0,

        detection.y2,
    )


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def point_distance(
    point_a,
    point_b,
):

    return (
        (
            point_b[
                0
            ]
            - point_a[
                0
            ]
        ) ** 2
        +
        (
            point_b[
                1
            ]
            - point_a[
                1
            ]
        ) ** 2
    ) ** 0.5


# ============================================================
# BBOX DIAGONAL
# ============================================================

def bbox_diagonal(
    detection,
):

    width = (
        detection.x2
        - detection.x1
    )

    height = (
        detection.y2
        - detection.y1
    )


    return (
        width ** 2
        +
        height ** 2
    ) ** 0.5


# ============================================================
# PRINT BOUNDARY MOTION
#
# No automatic motion threshold is applied here.
#
# We print the raw same-camera transition evidence only.
# ============================================================

def print_boundary_motion(
    manager,
    source_tracklet,
):

    (
        before_detection,
        after_detection,
    ) = find_boundary_detections(
        source_tracklet
    )


    before_frame = int(
        before_detection.frame
    )

    after_frame = int(
        after_detection.frame
    )


    frame_gap = (
        after_frame
        - before_frame
    )


    seconds = (
        frame_gap
        / FPS
    )


    before_center = detection_center(
        before_detection
    )

    after_center = detection_center(
        after_detection
    )


    before_bottom = detection_bottom_center(
        before_detection
    )

    after_bottom = detection_bottom_center(
        after_detection
    )


    center_jump = point_distance(
        before_center,
        after_center,
    )

    bottom_jump = point_distance(
        before_bottom,
        after_bottom,
    )


    scale = max(
        bbox_diagonal(
            before_detection
        ),

        bbox_diagonal(
            after_detection
        ),

        1.0,
    )


    normalized_center_jump = (
        center_jump
        / scale
    )

    normalized_bottom_jump = (
        bottom_jump
        / scale
    )


    camera_id = normalize_camera_id(
        source_tracklet.camera_id
    )


    before_ground = detection_ground_point(
        before_detection,
        camera_id,
        manager.homographies,
    )

    after_ground = detection_ground_point(
        after_detection,
        camera_id,
        manager.homographies,
    )


    ground_jump = ground_distance(
        before_ground,
        after_ground,
    )


    print()
    print(
        "c3:104 BOUNDARY MOTION"
    )

    print(
        "=" * 72
    )


    print(
        "Last BEFORE detection frame:",
        before_frame,
    )

    print(
        "First AFTER detection frame:",
        after_frame,
    )

    print(
        "Frame gap:",
        frame_gap,
    )

    print(
        "Elapsed time at 25 FPS:",
        f"{seconds:.3f} s",
    )


    print()
    print(
        "BEFORE center:",
        (
            f"({before_center[0]:.2f}, "
            f"{before_center[1]:.2f})"
        ),
    )

    print(
        "AFTER center:",
        (
            f"({after_center[0]:.2f}, "
            f"{after_center[1]:.2f})"
        ),
    )

    print(
        "Center jump:",
        f"{center_jump:.2f} px",
    )

    print(
        "Normalized center jump:",
        f"{normalized_center_jump:.3f}",
    )


    print()
    print(
        "BEFORE bottom-center:",
        (
            f"({before_bottom[0]:.2f}, "
            f"{before_bottom[1]:.2f})"
        ),
    )

    print(
        "AFTER bottom-center:",
        (
            f"({after_bottom[0]:.2f}, "
            f"{after_bottom[1]:.2f})"
        ),
    )

    print(
        "Bottom-center jump:",
        f"{bottom_jump:.2f} px",
    )

    print(
        "Normalized bottom jump:",
        f"{normalized_bottom_jump:.3f}",
    )


    print()
    print(
        "Ground-plane displacement:",
        f"{ground_jump:.2f}",
    )

    print(
        "Ground units are Terrace ground-coordinate units, "
        "not metres."
    )

    print(
        "No same-camera motion threshold is applied "
        "by this diagnostic."
    )


# ============================================================
# PRINT SUMMARY OBJECT
# ============================================================

def print_segment_summary(
    title,
    summary,
):

    print()
    print(
        title
    )

    print(
        "=" * 72
    )


    print(
        "Status:",
        summary[
            "status"
        ],
    )

    print(
        "GID:",
        summary.get(
            "gid"
        ),
    )

    print(
        "Strong candidate count:",
        summary[
            "candidate_count"
        ],
    )


    candidate_gids = summary.get(
        "candidate_gids"
    )


    if candidate_gids is not None:

        print(
            "Candidate GIDs:",
            candidate_gids,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TERRACE c3:104 FOCUSED SPLIT-EVIDENCE AUDIT"
    )

    print(
        "==========================================="
    )


    print()
    print(
        "Target track: c3:104"
    )

    print(
        "BEFORE segment: start -> frame",
        BEFORE_END,
    )

    print(
        "AFTER segment: frame",
        AFTER_START,
        "-> end",
    )

    print(
        "Target gained identity: GID",
        TARGET_GID,
    )


    print()
    print(
        "This diagnostic does NOT assign c3 identities."
    )

    print(
        "This diagnostic does NOT persist a split."
    )

    print(
        "This diagnostic does NOT change production thresholds."
    )


    # ========================================================
    # REBUILD VALIDATED c0+c1+c2 REFERENCE
    # ========================================================

    (
        manager,
        tracklet_lookup,
    ) = build_validated_reference_state()


    # ========================================================
    # LOAD c3:104
    # ========================================================

    c3_tracklets = load_camera_tracklets(
        CAMERA_ID
    )


    source_tracklet = c3_tracklets.get(
        TARGET_TRACK_ID
    )


    if source_tracklet is None:

        raise RuntimeError(
            f"Could not find c3:{TARGET_TRACK_ID}"
        )


    print()
    print(
        "SOURCE TRACKLET"
    )

    print(
        "=" * 72
    )

    print(
        "Track:",
        f"c3:{TARGET_TRACK_ID}",
    )

    print(
        "Start frame:",
        source_tracklet.start_frame,
    )

    print(
        "End frame:",
        source_tracklet.end_frame,
    )

    print(
        "Detections:",
        len(
            source_tracklet.detections
        ),
    )


    # ========================================================
    # BUILD BEFORE / AFTER TRACKLETS
    # ========================================================

    before_tracklet = build_segment_tracklet(
        source_tracklet=
            source_tracklet,

        start_frame=
            source_tracklet.start_frame,

        end_frame=
            BEFORE_END,

        synthetic_track_id=
            940104,
    )


    after_tracklet = build_segment_tracklet(
        source_tracklet=
            source_tracklet,

        start_frame=
            AFTER_START,

        end_frame=
            source_tracklet.end_frame,

        synthetic_track_id=
            940105,
    )


    print()
    print(
        "SEGMENT SIZES"
    )

    print(
        "=" * 72
    )

    print(
        "BEFORE detections:",
        len(
            before_tracklet.detections
        ),
    )

    print(
        "AFTER detections:",
        len(
            after_tracklet.detections
        ),
    )


    # ========================================================
    # BUILD SEGMENT EMBEDDINGS
    # ========================================================

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
            f"Could not open video: {VIDEO_PATH}"
        )


    try:

        (
            before_embedding,
            before_embedding_frames,
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
            after_embedding_frames,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                after_tracklet,
        )


    finally:

        video.release()


    before_after_similarity = (
        cosine_similarity(
            before_embedding,
            after_embedding,
        )
    )


    print()
    print(
        "SEGMENT EMBEDDINGS"
    )

    print(
        "=" * 72
    )

    print(
        "BEFORE sampled frames:",
        before_embedding_frames,
    )

    print(
        "AFTER sampled frames:",
        after_embedding_frames,
    )

    print(
        "BEFORE <-> AFTER ReID:",
        f"{before_after_similarity:.4f}",
    )


    # ========================================================
    # GID 11 COMPOSITION
    # ========================================================

    print_identity_composition(
        manager=
            manager,

        gid=
            TARGET_GID,
    )


    # ========================================================
    # ANALYZE BEFORE / AFTER DIRECTLY AGAINST GID 11
    # ========================================================

    before_gid11 = analyze_gid(
        manager=
            manager,

        gid=
            TARGET_GID,

        candidate_tracklet=
            before_tracklet,

        candidate_embedding=
            before_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    after_gid11 = analyze_gid(
        manager=
            manager,

        gid=
            TARGET_GID,

        candidate_tracklet=
            after_tracklet,

        candidate_embedding=
            after_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    print_gid_summary(
        "BEFORE SEGMENT -> GID 11",
        before_gid11,
    )


    print_gid_summary(
        "AFTER SEGMENT -> GID 11",
        after_gid11,
    )


    print_joint_support_members(
        "BEFORE GID 11 JOINT SUPPORT MEMBERS",
        before_gid11,
    )


    print_joint_support_members(
        "AFTER GID 11 JOINT SUPPORT MEMBERS",
        after_gid11,
    )


    print_core_comparison(
        before_result=
            before_gid11,

        after_result=
            after_gid11,
    )


    # ========================================================
    # RANK BOTH SEGMENTS AGAINST ALL ANCHORED GIDS
    # ========================================================

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


    before_summary = summarize_segment(
        before_ranked
    )


    after_summary = summarize_segment(
        after_ranked
    )


    print_segment_summary(
        "BEFORE STRONG IDENTITY SUMMARY",
        before_summary,
    )


    print_segment_summary(
        "AFTER STRONG IDENTITY SUMMARY",
        after_summary,
    )


    before_strong = strong_gid_candidates(
        before_ranked
    )

    after_strong = strong_gid_candidates(
        after_ranked
    )


    print()
    print(
        "STRONG CANDIDATE GIDS"
    )

    print(
        "=" * 72
    )

    print(
        "BEFORE:",
        [
            row[
                "gid"
            ]

            for row in before_strong
        ],
    )

    print(
        "AFTER:",
        [
            row[
                "gid"
            ]

            for row in after_strong
        ],
    )


    print_top_gid_alternatives(
        title=
            "TOP BEFORE ALTERNATIVE GIDS "
            "(excluding GID 11)",

        ranked_rows=
            before_ranked,

        exclude_gid=
            TARGET_GID,
    )


    print_top_gid_alternatives(
        title=
            "TOP AFTER ALTERNATIVE GIDS "
            "(excluding GID 11)",

        ranked_rows=
            after_ranked,

        exclude_gid=
            TARGET_GID,
    )


    # ========================================================
    # SAME-CAMERA BOUNDARY MOTION
    # ========================================================

    print_boundary_motion(
        manager=
            manager,

        source_tracklet=
            source_tracklet,
    )


    # ========================================================
    # FINAL SIGNAL SUMMARY
    # ========================================================

    print()
    print(
        "FINAL c3:104 EVIDENCE SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        "BEFORE <-> AFTER ReID:",
        f"{before_after_similarity:.4f}",
    )


    print(
        "BEFORE strong status:",
        before_summary[
            "status"
        ],
    )

    print(
        "BEFORE strong GID:",
        before_summary.get(
            "gid"
        ),
    )


    print(
        "AFTER strong status:",
        after_summary[
            "status"
        ],
    )

    print(
        "AFTER strong GID:",
        after_summary.get(
            "gid"
        ),
    )


    if before_gid11 is not None:

        print(
            "BEFORE -> GID11 joint supports:",
            before_gid11[
                "joint_support_count"
            ],
        )

        print(
            "BEFORE -> GID11 contradictions:",
            before_gid11[
                "core_contradiction_count"
            ],
        )

        print(
            "BEFORE -> GID11 identity veto:",
            before_gid11[
                "identity_veto"
            ],
        )


    if after_gid11 is not None:

        print(
            "AFTER -> GID11 joint supports:",
            after_gid11[
                "joint_support_count"
            ],
        )

        print(
            "AFTER -> GID11 contradictions:",
            after_gid11[
                "core_contradiction_count"
            ],
        )

        print(
            "AFTER -> GID11 identity veto:",
            after_gid11[
                "identity_veto"
            ],
        )


    print(
        "BEFORE alternative strong GIDs:",
        [
            row[
                "gid"
            ]

            for row in before_strong

            if row[
                "gid"
            ]
            != TARGET_GID
        ],
    )


    print()
    print(
        "No automatic split decision is made by this script."
    )

    print(
        "A GAINED_IDENTITY boundary is not by itself "
        "proof of a physical-person change."
    )

    print(
        "No identities were modified."
    )

    print(
        "No splits were persisted."
    )


if __name__ == "__main__":
    main()
