from collections import Counter
from pathlib import Path

import cv2

from src.analyze_c2_601_segment_gid_ranking import (
    build_segment_embedding,
    build_segment_tracklet,
)

from src.analyze_c3_gap_identity_changes import (
    build_validated_reference_state,
)

from src.analyze_c3_geometry_candidates import (
    load_camera_tracklets,
    load_embedding,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.reid_extractor import (
    ReIDExtractor,
)

from src.run_c2_production_policy_experiment import (
    find_member_gid,
    get_member_trust_safe,
    get_result_field,
    pending_keys,
    register_population_tracklets,
    result_reason,
    run_final_pending_pass,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMERA_ID = 3

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c3.avi"
)

EXPECTED_C3_USEFUL = 94

EXPECTED_C3_SPLIT_TRACKS = {
    104,
}

EXPECTED_C3_POPULATION = 95


# ============================================================
# VALIDATED c3 SPLIT
# ============================================================

SPLIT_TRACK_ID = 104

SPLIT_BEFORE_END = 1821

SPLIT_AFTER_START = 1848

SPLIT_BEFORE_SYNTHETIC_ID = 940104

SPLIT_AFTER_SYNTHETIC_ID = 940105


# ============================================================
# LABELS
# ============================================================

def whole_label(
    track_id,
):

    return f"c3:{track_id}"


def before_label(
    track_id,
):

    return f"c3:{track_id}a"


def after_label(
    track_id,
):

    return f"c3:{track_id}b"


# ============================================================
# LOAD RAW c3 ROWS
# ============================================================

def load_c3_rows():

    tracklets = load_camera_tracklets(
        CAMERA_ID
    )

    rows = []

    missing_embeddings = []


    for track_id, tracklet in (
        tracklets.items()
    ):

        embedding = load_embedding(
            CAMERA_ID,
            track_id,
        )


        if embedding is None:

            missing_embeddings.append(
                track_id
            )

            continue


        rows.append(
            (
                tracklet,
                embedding,
            )
        )


    rows.sort(
        key=lambda item: (
            item[
                0
            ].start_frame,

            item[
                0
            ].local_track_id,
        )
    )


    print()
    print(
        "c3 INPUT"
    )

    print(
        "========"
    )

    print(
        "Useful tracklets:",
        len(
            tracklets
        ),
    )

    print(
        "Whole-track embeddings:",
        len(
            rows
        ),
    )

    print(
        "Missing embeddings:",
        len(
            missing_embeddings
        ),
    )


    if missing_embeddings:

        print(
            "Missing IDs:",
            sorted(
                missing_embeddings
            ),
        )


    if (
        len(
            tracklets
        )
        != EXPECTED_C3_USEFUL
    ):

        raise RuntimeError(
            "c3 useful-tracklet regression: "
            f"expected {EXPECTED_C3_USEFUL}, "
            f"got {len(tracklets)}"
        )


    if (
        len(
            rows
        )
        != EXPECTED_C3_USEFUL
    ):

        raise RuntimeError(
            "c3 embedding regression: "
            f"expected {EXPECTED_C3_USEFUL}, "
            f"got {len(rows)}"
        )


    return rows


# ============================================================
# BUILD VALIDATED c3:104 SPLIT
# ============================================================

def build_validated_c3_104_split(
    source_tracklet,
):

    before_tracklet = build_segment_tracklet(
        source_tracklet=
            source_tracklet,

        start_frame=
            source_tracklet.start_frame,

        end_frame=
            SPLIT_BEFORE_END,

        synthetic_track_id=
            SPLIT_BEFORE_SYNTHETIC_ID,
    )


    after_tracklet = build_segment_tracklet(
        source_tracklet=
            source_tracklet,

        start_frame=
            SPLIT_AFTER_START,

        end_frame=
            source_tracklet.end_frame,

        synthetic_track_id=
            SPLIT_AFTER_SYNTHETIC_ID,
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
            f"Could not open video: {VIDEO_PATH}"
        )


    try:

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


    finally:

        video.release()


    print()
    print(
        "VALIDATED c3:104 SPLIT"
    )

    print(
        "======================"
    )

    print(
        "Before:",
        before_tracklet.start_frame,
        "->",
        before_tracklet.end_frame,
        "| detections=",
        len(
            before_tracklet.detections
        ),
    )

    print(
        "After: ",
        after_tracklet.start_frame,
        "->",
        after_tracklet.end_frame,
        "| detections=",
        len(
            after_tracklet.detections
        ),
    )

    print(
        "Before embedding frames:",
        before_frames,
    )

    print(
        "After embedding frames:",
        after_frames,
    )


    return {
        "before_tracklet":
            before_tracklet,

        "after_tracklet":
            after_tracklet,

        "before_embedding":
            before_embedding,

        "after_embedding":
            after_embedding,
    }


# ============================================================
# BUILD c3 PRODUCTION POPULATION
# ============================================================

def build_c3_population(
    c3_rows,
):

    source_104 = None


    for source_tracklet, _ in c3_rows:

        if (
            source_tracklet.local_track_id
            == SPLIT_TRACK_ID
        ):

            source_104 = source_tracklet

            break


    if source_104 is None:

        raise RuntimeError(
            "Validated split source c3:104 not found"
        )


    split = build_validated_c3_104_split(
        source_104
    )


    population = []


    for source_tracklet, source_embedding in (
        c3_rows
    ):

        track_id = (
            source_tracklet.local_track_id
        )


        if (
            track_id
            != SPLIT_TRACK_ID
        ):

            population.append(
                {
                    "source_track_id":
                        track_id,

                    "label":
                        whole_label(
                            track_id
                        ),

                    "kind":
                        "WHOLE",

                    "tracklet":
                        source_tracklet,

                    "embedding":
                        source_embedding,

                    "split_boundary":
                        None,
                }
            )

            continue


        population.append(
            {
                "source_track_id":
                    track_id,

                "label":
                    before_label(
                        track_id
                    ),

                "kind":
                    "SPLIT_BEFORE",

                "tracklet":
                    split[
                        "before_tracklet"
                    ],

                "embedding":
                    split[
                        "before_embedding"
                    ],

                "split_boundary":
                    {
                        "before_frame":
                            SPLIT_BEFORE_END,

                        "after_frame":
                            SPLIT_AFTER_START,
                    },
            }
        )


        population.append(
            {
                "source_track_id":
                    track_id,

                "label":
                    after_label(
                        track_id
                    ),

                "kind":
                    "SPLIT_AFTER",

                "tracklet":
                    split[
                        "after_tracklet"
                    ],

                "embedding":
                    split[
                        "after_embedding"
                    ],

                "split_boundary":
                    {
                        "before_frame":
                            SPLIT_BEFORE_END,

                        "after_frame":
                            SPLIT_AFTER_START,
                    },
            }
        )


    population.sort(
        key=lambda item: (
            item[
                "tracklet"
            ].start_frame,

            item[
                "source_track_id"
            ],

            item[
                "kind"
            ],
        )
    )


    if (
        len(
            population
        )
        != EXPECTED_C3_POPULATION
    ):

        raise RuntimeError(
            "c3 production-population regression: "
            f"expected {EXPECTED_C3_POPULATION}, "
            f"got {len(population)}"
        )


    return population


# ============================================================
# PROCESS c3 THROUGH EXACT PRODUCTION ASSIGNMENT
# ============================================================

def process_c3_population(
    manager,
    population,
    tracklet_lookup,
):

    records = []


    for index, item in enumerate(
        population,
        start=1,
    ):

        tracklet = item[
            "tracklet"
        ]

        embedding = item[
            "embedding"
        ]


        camera_id = normalize_camera_id(
            tracklet.camera_id
        )

        local_track_id = (
            tracklet.local_track_id
        )

        key = (
            camera_id,
            local_track_id,
        )


        result = manager.assign_tracklet(
            camera_id=
                camera_id,

            local_track_id=
                local_track_id,

            start_frame=
                tracklet.start_frame,

            end_frame=
                tracklet.end_frame,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )


        status = get_result_field(
            result,
            [
                "status",
            ],
            default="UNKNOWN",
        )


        merged = bool(
            get_result_field(
                result,
                [
                    "merged",
                ],
                default=False,
            )
        )


        created_new = bool(
            get_result_field(
                result,
                [
                    "created_new",
                ],
                default=False,
            )
        )


        immediate_gid = find_member_gid(
            manager,
            key,
        )


        immediate_trust = (
            get_member_trust_safe(
                manager,
                immediate_gid,
                key,
            )
        )


        records.append(
            {
                "label":
                    item[
                        "label"
                    ],

                "kind":
                    item[
                        "kind"
                    ],

                "source_track_id":
                    item[
                        "source_track_id"
                    ],

                "key":
                    key,

                "start_frame":
                    tracklet.start_frame,

                "end_frame":
                    tracklet.end_frame,

                "initial_status":
                    status,

                "merged":
                    merged,

                "created_new":
                    created_new,

                "reason":
                    result_reason(
                        result
                    ),

                "immediate_gid":
                    immediate_gid,

                "immediate_trust":
                    immediate_trust,
            }
        )


        # ====================================================
        # SAME ORDINARY PRODUCTION REEVALUATION
        # ====================================================

        if (
            merged
            or created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    tracklet_lookup,

                include_same_camera=
                    False,

                include_dual_evidence=
                    False,

                include_core_gallery=
                    False,
            )


        if (
            index % 25
            == 0
            or index
            == len(
                population
            )
        ):

            print(
                f"Processed c3 "
                f"{index}/"
                f"{len(population)}"
                f" | GIDs="
                f"{len(manager.identities)}"
                f" | pending="
                f"{len(manager.pending_tracklets)}"
            )


    return records


# ============================================================
# FINALIZE c3 RECORDS
# ============================================================

def finalize_c3_records(
    manager,
    records,
    baseline_gid_ids,
):

    pending = pending_keys(
        manager
    )


    for record in records:

        key = record[
            "key"
        ]


        gid = find_member_gid(
            manager,
            key,
        )


        trust = get_member_trust_safe(
            manager,
            gid,
            key,
        )


        record[
            "final_gid"
        ] = gid

        record[
            "final_trust"
        ] = trust

        record[
            "is_pending"
        ] = (
            key
            in pending
        )


        if gid is not None:

            if gid in baseline_gid_ids:

                record[
                    "gid_origin"
                ] = "EXISTING"

            else:

                record[
                    "gid_origin"
                ] = "NEW"

        else:

            record[
                "gid_origin"
            ] = "-"


        if gid is not None:

            record[
                "final_state"
            ] = (
                trust
                if trust is not None
                else "ASSIGNED"
            )

        elif record[
            "is_pending"
        ]:

            record[
                "final_state"
            ] = "PENDING"

        else:

            record[
                "final_state"
            ] = "UNASSIGNED"


    return records


# ============================================================
# PRINT INITIAL STATUS COUNTS
# ============================================================

def print_initial_status_counts(
    records,
):

    counts = Counter(
        record[
            "initial_status"
        ]

        for record in records
    )


    print()
    print(
        "INITIAL c3 ASSIGNMENT STATUS"
    )

    print(
        "============================"
    )


    for status, count in sorted(
        counts.items()
    ):

        print(
            f"{status:<20}"
            f"{count}"
        )


# ============================================================
# PRINT FINAL STATE COUNTS
# ============================================================

def print_final_state_counts(
    records,
):

    counts = Counter(
        record[
            "final_state"
        ]

        for record in records
    )


    print()
    print(
        "FINAL c3 STATE COUNTS"
    )

    print(
        "====================="
    )


    for state, count in sorted(
        counts.items()
    ):

        print(
            f"{state:<20}"
            f"{count}"
        )


# ============================================================
# PRINT EXISTING-GID ASSOCIATIONS
# ============================================================

def print_existing_gid_associations(
    records,
):

    rows = [
        record

        for record in records

        if (
            record[
                "final_gid"
            ]
            is not None

            and

            record[
                "gid_origin"
            ]
            == "EXISTING"
        )
    ]


    rows.sort(
        key=lambda record: (
            record[
                "final_gid"
            ],

            record[
                "start_frame"
            ],

            record[
                "label"
            ],
        )
    )


    print()
    print(
        "c3 -> EXISTING GID ASSOCIATIONS"
    )

    print(
        "==============================="
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for record in rows:

        print(
            f"{record['label']:<12}"
            f" -> GID "
            f"{record['final_gid']:<3}"
            f" | trust="
            f"{record['final_trust']}"
            f" | initial="
            f"{record['initial_status']}"
            f" | reason="
            f"{record['reason']}"
        )


# ============================================================
# PRINT NEW-GID MEMBERS
# ============================================================

def print_new_gid_members(
    records,
):

    rows = [
        record

        for record in records

        if (
            record[
                "final_gid"
            ]
            is not None

            and

            record[
                "gid_origin"
            ]
            == "NEW"
        )
    ]


    rows.sort(
        key=lambda record: (
            record[
                "final_gid"
            ],

            record[
                "start_frame"
            ],
        )
    )


    print()
    print(
        "c3 -> NEW GID MEMBERS"
    )

    print(
        "====================="
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for record in rows:

        print(
            f"{record['label']:<12}"
            f" -> GID "
            f"{record['final_gid']:<3}"
            f" | trust="
            f"{record['final_trust']}"
            f" | initial="
            f"{record['initial_status']}"
            f" | reason="
            f"{record['reason']}"
        )


# ============================================================
# PRINT PENDING c3
# ============================================================

def print_pending_records(
    records,
):

    rows = [
        record

        for record in records

        if record[
            "is_pending"
        ]
    ]


    rows.sort(
        key=lambda record: (
            record[
                "start_frame"
            ],

            record[
                "label"
            ],
        )
    )


    print()
    print(
        "FINAL c3 PENDING"
    )

    print(
        "================"
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for record in rows:

        print(
            f"{record['label']:<12}"
            f" | "
            f"{record['start_frame']}"
            f" -> "
            f"{record['end_frame']}"
            f" | initial="
            f"{record['initial_status']}"
            f" | reason="
            f"{record['reason']}"
        )


# ============================================================
# PRINT VALIDATED SPLIT OUTCOME
# ============================================================

def print_split_outcome(
    records,
):

    rows = [
        record

        for record in records

        if (
            record[
                "source_track_id"
            ]
            == SPLIT_TRACK_ID
        )
    ]


    rows.sort(
        key=lambda record:
            record[
                "start_frame"
            ]
    )


    print()
    print(
        "VALIDATED c3:104 SPLIT OUTCOME"
    )

    print(
        "=============================="
    )


    for record in rows:

        print(
            f"{record['label']:<12}"
            f" | "
            f"{record['start_frame']}"
            f" -> "
            f"{record['end_frame']}"
            f" | final_gid="
            f"{record['final_gid']}"
            f" | trust="
            f"{record['final_trust']}"
            f" | pending="
            f"{record['is_pending']}"
            f" | reason="
            f"{record['reason']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c3 PRODUCTION POLICY EXPERIMENT"
    )

    print(
        "==============================="
    )


    print()
    print(
        "Production manager code is unchanged."
    )

    print(
        "Production thresholds are unchanged."
    )

    print(
        "The validated c3:104 split exists only in memory."
    )


    # ========================================================
    # REBUILD VALIDATED c0+c1+c2 STATE
    # ========================================================

    (
        manager,
        tracklet_lookup,
    ) = build_validated_reference_state()


    baseline_gid_ids = set(
        manager.identities.keys()
    )


    baseline_gids = len(
        manager.identities
    )

    baseline_pending = len(
        manager.pending_tracklets
    )

    baseline_anchored = len(
        manager.anchored_gids
    )


    print()
    print(
        "VALIDATED c0+c1+c2 START STATE"
    )

    print(
        "============================="
    )

    print(
        "GIDs:",
        baseline_gids,
    )

    print(
        "Pending:",
        baseline_pending,
    )

    print(
        "Anchored:",
        baseline_anchored,
    )


    # ========================================================
    # LOAD + BUILD c3 POPULATION
    # ========================================================

    c3_rows = load_c3_rows()


    population = build_c3_population(
        c3_rows
    )


    print()
    print(
        "IN-MEMORY c3 POPULATION"
    )

    print(
        "======================="
    )

    print(
        "Original useful tracklets:",
        len(
            c3_rows
        ),
    )

    print(
        "Validated split tracks:",
        sorted(
            EXPECTED_C3_SPLIT_TRACKS
        ),
    )

    print(
        "Production input segments:",
        len(
            population
        ),
    )


    # ========================================================
    # REGISTER c3 POPULATION
    # ========================================================

    collisions = register_population_tracklets(
        population=
            population,

        tracklet_lookup=
            tracklet_lookup,
    )


    if collisions:

        raise RuntimeError(
            "Unexpected c3 tracklet lookup collisions: "
            f"{collisions}"
        )


    # ========================================================
    # PROCESS c3
    # ========================================================

    print()
    print(
        "PROCESSING c3 WITH UNCHANGED "
        "PRODUCTION MANAGER..."
    )


    records = process_c3_population(
        manager=
            manager,

        population=
            population,

        tracklet_lookup=
            tracklet_lookup,
    )


    print_initial_status_counts(
        records
    )


    pre_final_gids = len(
        manager.identities
    )

    pre_final_pending = len(
        manager.pending_tracklets
    )

    pre_final_anchored = len(
        manager.anchored_gids
    )


    print()
    print(
        "STATE BEFORE FINAL CONTINUATION"
    )

    print(
        "==============================="
    )

    print(
        "GIDs:",
        pre_final_gids,
    )

    print(
        "Pending:",
        pre_final_pending,
    )

    print(
        "Anchored:",
        pre_final_anchored,
    )


    # ========================================================
    # FINAL CONTINUATION PASS
    # ========================================================

    final_recoveries = run_final_pending_pass(
        manager=
            manager,

        tracklet_lookup=
            tracklet_lookup,
    )


    finalize_c3_records(
        manager=
            manager,

        records=
            records,

        baseline_gid_ids=
            baseline_gid_ids,
    )


    print()
    print(
        "FINAL CONTINUATION RECOVERIES"
    )

    print(
        "============================="
    )

    print(
        "Recoveries:",
        len(
            final_recoveries
        )
        if final_recoveries is not None
        else 0,
    )


    # ========================================================
    # REPORT c3
    # ========================================================

    print_final_state_counts(
        records
    )


    print_existing_gid_associations(
        records
    )


    print_new_gid_members(
        records
    )


    print_pending_records(
        records
    )


    print_split_outcome(
        records
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    existing_count = sum(
        1

        for record in records

        if (
            record[
                "final_gid"
            ]
            is not None

            and

            record[
                "gid_origin"
            ]
            == "EXISTING"
        )
    )


    new_count = sum(
        1

        for record in records

        if (
            record[
                "final_gid"
            ]
            is not None

            and

            record[
                "gid_origin"
            ]
            == "NEW"
        )
    )


    pending_count = sum(
        1

        for record in records

        if record[
            "is_pending"
        ]
    )


    final_gids = len(
        manager.identities
    )

    final_pending = len(
        manager.pending_tracklets
    )

    final_anchored = len(
        manager.anchored_gids
    )


    print()
    print(
        "FINAL c3 PRODUCTION SUMMARY"
    )

    print(
        "==========================="
    )

    print(
        "c3 production segments:",
        len(
            records
        ),
    )

    print(
        "c3 -> existing GID:",
        existing_count,
    )

    print(
        "c3 -> new GID:",
        new_count,
    )

    print(
        "c3 pending:",
        pending_count,
    )


    print()
    print(
        "GID delta:",
        final_gids
        - baseline_gids,
    )

    print(
        "Pending delta:",
        final_pending
        - baseline_pending,
    )

    print(
        "Anchored delta:",
        final_anchored
        - baseline_anchored,
    )


    print()
    print(
        "Final total GIDs:",
        final_gids,
    )

    print(
        "Final total pending:",
        final_pending,
    )

    print(
        "Final total anchored:",
        final_anchored,
    )


    print()
    print(
        "Production manager source was not modified."
    )

    print(
        "Production thresholds were not modified."
    )

    print(
        "No c3 split was persisted to disk."
    )


if __name__ == "__main__":
    main()
