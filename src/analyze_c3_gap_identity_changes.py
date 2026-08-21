from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.analyze_c2_gap_identity_changes import (
    find_gap_boundaries,
    format_summary,
)

from src.analyze_c3_geometry_candidates import (
    load_camera_tracklets,
    load_embedding,
)

from src.reid_extractor import (
    ReIDExtractor,
)

from src.run_c2_identity_aware_split_experiment import (
    build_split_population,
    discover_gap_evidence,
    evaluate_boundary,
    select_split_proposals,
)

from src.run_c2_production_policy_experiment import (
    finalize_records,
    process_c2_population,
    register_population_tracklets,
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


# ============================================================
# VALIDATED REFERENCE EXPECTATIONS
#
# These are regression guards for the already-validated
# c0+c1+c2 production state.
#
# If one of these changes, this c3 diagnostic must stop.
# We do not want to analyze c3 against a silently changed
# reference identity population.
# ============================================================

EXPECTED_C0_C1_TRACKLETS = 222

EXPECTED_C0_C1_GIDS = 76

EXPECTED_C0_C1_PENDING = 92

EXPECTED_C0_C1_ANCHORED = 22

EXPECTED_C2_USEFUL = 105

EXPECTED_C2_SPLIT_TRACKS = {
    9,
    601,
}

EXPECTED_C2_POPULATION = 107

EXPECTED_FINAL_GIDS = 107

EXPECTED_FINAL_PENDING = 164

EXPECTED_FINAL_ANCHORED = 22

EXPECTED_C3_USEFUL = 94


# ============================================================
# KNOWN VALIDATED c2 ASSOCIATIONS
#
# These are important identity anchors from the already
# validated c2 production experiment.
# ============================================================

EXPECTED_C2_ASSOCIATIONS = {
    "c2:9a": 1,
    "c2:9b": 11,
    "c2:508": 65,
    "c2:601b": 73,
}


# ============================================================
# REBUILD VALIDATED c0+c1+c2 REFERENCE STATE
# ============================================================

def build_validated_reference_state():

    print()
    print(
        "REBUILDING VALIDATED c0+c1+c2 REFERENCE STATE"
    )

    print(
        "============================================"
    )


    # ========================================================
    # REPLAY VALIDATED c0+c1
    # ========================================================

    (
        manager,
        baseline_tracklets,
        tracklet_lookup,
        _,
    ) = replay_final_production()


    print()
    print(
        "c0+c1 STATE"
    )

    print(
        "==========="
    )

    print(
        "Tracklets:",
        len(
            baseline_tracklets
        ),
    )

    print(
        "GIDs:",
        len(
            manager.identities
        ),
    )

    print(
        "Pending:",
        len(
            manager.pending_tracklets
        ),
    )

    print(
        "Anchored:",
        len(
            manager.anchored_gids
        ),
    )


    # ========================================================
    # REGRESSION GUARDS
    # ========================================================

    if (
        len(
            baseline_tracklets
        )
        != EXPECTED_C0_C1_TRACKLETS
    ):

        raise RuntimeError(
            "c0+c1 tracklet regression: "
            f"expected {EXPECTED_C0_C1_TRACKLETS}, "
            f"got {len(baseline_tracklets)}"
        )


    if (
        len(
            manager.identities
        )
        != EXPECTED_C0_C1_GIDS
    ):

        raise RuntimeError(
            "c0+c1 GID regression: "
            f"expected {EXPECTED_C0_C1_GIDS}, "
            f"got {len(manager.identities)}"
        )


    if (
        len(
            manager.pending_tracklets
        )
        != EXPECTED_C0_C1_PENDING
    ):

        raise RuntimeError(
            "c0+c1 pending regression: "
            f"expected {EXPECTED_C0_C1_PENDING}, "
            f"got {len(manager.pending_tracklets)}"
        )


    if (
        len(
            manager.anchored_gids
        )
        != EXPECTED_C0_C1_ANCHORED
    ):

        raise RuntimeError(
            "c0+c1 anchored regression: "
            f"expected {EXPECTED_C0_C1_ANCHORED}, "
            f"got {len(manager.anchored_gids)}"
        )


    baseline_gid_ids = set(
        manager.identities.keys()
    )


    # ========================================================
    # LOAD RAW c2
    # ========================================================

    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()


    print()
    print(
        "c2 INPUT"
    )

    print(
        "========"
    )

    print(
        "Useful c2 tracklets:",
        len(
            c2_rows
        ),
    )


    if (
        len(
            c2_rows
        )
        != EXPECTED_C2_USEFUL
    ):

        raise RuntimeError(
            "c2 useful-tracklet regression: "
            f"expected {EXPECTED_C2_USEFUL}, "
            f"got {len(c2_rows)}"
        )


    tracklet_lookup.update(
        c2_lookup
    )


    # ========================================================
    # REDISCOVER VALIDATED c2 IDENTITY-AWARE SPLITS
    #
    # This intentionally reuses the exact already-validated
    # c2 split-discovery implementation.
    # ========================================================

    print()
    print(
        "Rediscovering validated c2 identity-aware splits..."
    )


    boundary_results = (
        discover_gap_evidence(
            manager=
                manager,

            c2_rows=
                c2_rows,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    (
        selected_splits,
        rejected_ambiguous,
    ) = select_split_proposals(
        boundary_results
    )


    selected_ids = set(
        selected_splits.keys()
    )


    print()
    print(
        "Selected c2 split tracks:",
        sorted(
            selected_ids
        ),
    )

    print(
        "Rejected ambiguous c2 split tracks:",
        sorted(
            rejected_ambiguous.keys()
        ),
    )


    if (
        selected_ids
        != EXPECTED_C2_SPLIT_TRACKS
    ):

        raise RuntimeError(
            "Validated c2 split regression: "
            f"expected "
            f"{sorted(EXPECTED_C2_SPLIT_TRACKS)}, "
            f"got {sorted(selected_ids)}"
        )


    # ========================================================
    # BUILD VALIDATED IN-MEMORY c2 POPULATION
    # ========================================================

    population = build_split_population(
        c2_rows=
            c2_rows,

        selected_splits=
            selected_splits,
    )


    print(
        "c2 production population:",
        len(
            population
        ),
    )


    if (
        len(
            population
        )
        != EXPECTED_C2_POPULATION
    ):

        raise RuntimeError(
            "c2 production-population regression: "
            f"expected {EXPECTED_C2_POPULATION}, "
            f"got {len(population)}"
        )


    # ========================================================
    # REGISTER c2 SPLIT SEGMENTS
    # ========================================================

    collisions = (
        register_population_tracklets(
            population=
                population,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if collisions:

        raise RuntimeError(
            "Unexpected c2 tracklet-lookup collisions: "
            f"{collisions}"
        )


    # ========================================================
    # PROCESS c2 THROUGH THE REAL PRODUCTION MANAGER
    # ========================================================

    print()
    print(
        "Processing validated c2 production population..."
    )


    records = process_c2_population(
        manager=
            manager,

        population=
            population,

        tracklet_lookup=
            tracklet_lookup,
    )


    # ========================================================
    # FINAL PRODUCTION CONTINUATION PASS
    # ========================================================

    run_final_pending_pass(
        manager=
            manager,

        tracklet_lookup=
            tracklet_lookup,
    )


    finalize_records(
        manager=
            manager,

        records=
            records,

        baseline_gid_ids=
            baseline_gid_ids,
    )


    # ========================================================
    # VALIDATE KNOWN c2 ASSOCIATIONS
    # ========================================================

    records_by_label = {
        record[
            "label"
        ]:
            record

        for record in records
    }


    print()
    print(
        "VALIDATED c2 ASSOCIATION GUARDS"
    )

    print(
        "==============================="
    )


    for label, expected_gid in (
        EXPECTED_C2_ASSOCIATIONS.items()
    ):

        record = records_by_label.get(
            label
        )


        if record is None:

            raise RuntimeError(
                f"Missing validated c2 record: {label}"
            )


        actual_gid = record[
            "final_gid"
        ]


        print(
            f"{label:<10}"
            f" -> expected GID "
            f"{expected_gid:<3}"
            f" | actual GID "
            f"{actual_gid}"
        )


        if (
            actual_gid
            != expected_gid
        ):

            raise RuntimeError(
                f"{label} GID regression: "
                f"expected {expected_gid}, "
                f"got {actual_gid}"
            )


    # ========================================================
    # FINAL REFERENCE-STATE REGRESSION GUARDS
    # ========================================================

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
        "FINAL c0+c1+c2 REFERENCE STATE"
    )

    print(
        "=============================="
    )

    print(
        "GIDs:",
        final_gids
    )

    print(
        "Pending:",
        final_pending
    )

    print(
        "Anchored:",
        final_anchored
    )


    if (
        final_gids
        != EXPECTED_FINAL_GIDS
    ):

        raise RuntimeError(
            "Final GID regression: "
            f"expected {EXPECTED_FINAL_GIDS}, "
            f"got {final_gids}"
        )


    if (
        final_pending
        != EXPECTED_FINAL_PENDING
    ):

        raise RuntimeError(
            "Final pending regression: "
            f"expected {EXPECTED_FINAL_PENDING}, "
            f"got {final_pending}"
        )


    if (
        final_anchored
        != EXPECTED_FINAL_ANCHORED
    ):

        raise RuntimeError(
            "Final anchored regression: "
            f"expected {EXPECTED_FINAL_ANCHORED}, "
            f"got {final_anchored}"
        )


    print()
    print(
        "REFERENCE STATE: PASS"
    )


    return (
        manager,
        tracklet_lookup,
    )


# ============================================================
# LOAD c3 USEFUL TRACKLETS + WHOLE EMBEDDINGS
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
            "Missing track IDs:",
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
# DISCOVER c3 INTERNAL-GAP IDENTITY EVIDENCE
#
# Uses the exact generic boundary evaluator already validated
# during c2 work, but points it at the c3 video.
# ============================================================

def discover_c3_gap_evidence(
    manager,
    c3_rows,
    tracklet_lookup,
):

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


    results = []

    track_boundary_counts = Counter()

    synthetic_id = 930000


    try:

        for source_tracklet, _ in c3_rows:

            boundaries = find_gap_boundaries(
                source_tracklet
            )


            track_boundary_counts[
                source_tracklet.local_track_id
            ] = len(
                boundaries
            )


            for boundary in boundaries:

                synthetic_id += 2


                result = evaluate_boundary(
                    manager=
                        manager,

                    extractor=
                        extractor,

                    video=
                        video,

                    source_tracklet=
                        source_tracklet,

                    boundary=
                        boundary,

                    tracklet_lookup=
                        tracklet_lookup,

                    synthetic_id_start=
                        synthetic_id,
                )


                results.append(
                    result
                )


    finally:

        video.release()


    return (
        results,
        track_boundary_counts,
    )


# ============================================================
# FORMAT ONE BOUNDARY
# ============================================================

def print_boundary(
    row,
):

    print(
        f"c3:{row['track_id']:<4}"
        f" | "
        f"{row['before_frame']:>4}"
        f" -> "
        f"{row['after_frame']:<4}"
        f" | gap="
        f"{row['gap']:<3}"
        f" | before_n="
        f"{row['before_count']:<4}"
        f" | after_n="
        f"{row['after_count']:<4}"
        f" | jump="
        f"{row['normalized_jump']:.3f}"
        f" | "
        f"{row['classification']}"
    )


    print(
        "    before:",
        format_summary(
            row[
                "before_summary"
            ]
        ),
    )


    print(
        "    after: ",
        format_summary(
            row[
                "after_summary"
            ]
        ),
    )


# ============================================================
# PRINT CLASSIFICATION REPORT
# ============================================================

def print_classification_report(
    boundary_results,
):

    counts = Counter(
        row[
            "classification"
        ]

        for row
        in boundary_results
    )


    print()
    print(
        "c3 GAP CLASSIFICATION COUNTS"
    )

    print(
        "============================"
    )


    for classification in [
        "IDENTITY_SHIFT",
        "GAINED_IDENTITY",
        "LOST_IDENTITY",
        "SAME_IDENTITY",
        "AMBIGUOUS",
        "UNRESOLVED",
    ]:

        print(
            f"{classification:<18}: "
            f"{counts[classification]}"
        )


    return counts


# ============================================================
# PRINT ALL IDENTITY-INFORMATIVE BOUNDARIES
# ============================================================

def print_identity_informative_boundaries(
    boundary_results,
):

    informative = [
        row

        for row in boundary_results

        if row[
            "classification"
        ]
        != "UNRESOLVED"
    ]


    informative.sort(
        key=lambda row: (
            row[
                "track_id"
            ],

            row[
                "before_frame"
            ],

            row[
                "after_frame"
            ],
        )
    )


    print()
    print(
        "IDENTITY-INFORMATIVE c3 GAP BOUNDARIES"
    )

    print(
        "======================================"
    )


    if not informative:

        print(
            "NONE"
        )

        return


    for row in informative:

        print_boundary(
            row
        )


# ============================================================
# PRINT SPLIT-ELIGIBLE BOUNDARIES
# ============================================================

def print_split_eligible_boundaries(
    boundary_results,
):

    eligible = [
        row

        for row in boundary_results

        if row[
            "classification"
        ]
        in {
            "IDENTITY_SHIFT",
            "GAINED_IDENTITY",
        }
    ]


    eligible.sort(
        key=lambda row: (
            0
            if row[
                "classification"
            ]
            == "IDENTITY_SHIFT"
            else 1,

            row[
                "track_id"
            ],

            -row[
                "gap"
            ],
        )
    )


    print()
    print(
        "SPLIT-ELIGIBLE c3 BOUNDARIES"
    )

    print(
        "============================"
    )


    if not eligible:

        print(
            "NONE"
        )

        return


    for row in eligible:

        print_boundary(
            row
        )


# ============================================================
# PRINT SELECTED SPLIT PROPOSALS
# ============================================================

def print_selected_split_proposals(
    selected_splits,
):

    print()
    print(
        "SELECTED c3 SPLIT PROPOSALS"
    )

    print(
        "==========================="
    )


    if not selected_splits:

        print(
            "NONE"
        )

        return


    for track_id in sorted(
        selected_splits
    ):

        row = selected_splits[
            track_id
        ]


        print_boundary(
            row
        )


# ============================================================
# PRINT AMBIGUOUS SPLIT-PROPOSAL TRACKS
# ============================================================

def print_ambiguous_split_tracks(
    rejected_ambiguous,
):

    print()
    print(
        "REJECTED AMBIGUOUS c3 SPLIT TRACKS"
    )

    print(
        "=================================="
    )


    if not rejected_ambiguous:

        print(
            "NONE"
        )

        return


    for track_id in sorted(
        rejected_ambiguous
    ):

        print()
        print(
            f"c3:{track_id}"
        )


        for row in rejected_ambiguous[
            track_id
        ]:

            print_boundary(
                row
            )


# ============================================================
# PRINT TRACKS WITH VIABLE GAPS
# ============================================================

def print_gap_population(
    track_boundary_counts,
):

    rows = [
        (
            track_id,
            count,
        )

        for track_id, count
        in track_boundary_counts.items()

        if count > 0
    ]


    rows.sort(
        key=lambda item: (
            -item[
                1
            ],

            item[
                0
            ],
        )
    )


    print()
    print(
        "c3 TRACKS WITH VIABLE INTERNAL GAPS"
    )

    print(
        "=================================="
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for track_id, count in rows:

        print(
            f"c3:{track_id:<4}"
            f" boundaries="
            f"{count}"
        )


# ============================================================
# EXTRA CAUTION REPORT FOR LARGE-GAP TRACKS
# ============================================================

def print_large_gap_tracks(
    boundary_results,
):

    by_track = defaultdict(
        list
    )


    for row in boundary_results:

        by_track[
            row[
                "track_id"
            ]
        ].append(
            row
        )


    rows = []


    for track_id, track_rows in (
        by_track.items()
    ):

        largest = max(
            track_rows,
            key=lambda row:
                row[
                    "gap"
                ],
        )


        rows.append(
            largest
        )


    rows.sort(
        key=lambda row:
            row[
                "gap"
            ],
        reverse=True,
    )


    print()
    print(
        "LARGEST VIABLE GAP PER c3 TRACK"
    )

    print(
        "================================"
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for row in rows:

        print_boundary(
            row
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TERRACE c3 IDENTITY-AWARE GAP DIAGNOSTIC"
    )

    print(
        "========================================"
    )


    print()
    print(
        "This script does NOT assign c3 identities."
    )

    print(
        "This script does NOT modify production thresholds."
    )

    print(
        "This script does NOT persist tracklet splits."
    )


    # ========================================================
    # REBUILD TRUSTED c0+c1+c2 REFERENCE
    # ========================================================

    (
        manager,
        tracklet_lookup,
    ) = build_validated_reference_state()


    # ========================================================
    # LOAD c3
    # ========================================================

    c3_rows = load_c3_rows()


    # ========================================================
    # SCAN c3 GAPS
    # ========================================================

    print()
    print(
        "SCANNING c3 INTERNAL GAPS..."
    )


    (
        boundary_results,
        track_boundary_counts,
    ) = discover_c3_gap_evidence(
        manager=
            manager,

        c3_rows=
            c3_rows,

        tracklet_lookup=
            tracklet_lookup,
    )


    print()
    print(
        "Total viable c3 gap boundaries:",
        len(
            boundary_results
        ),
    )


    print(
        "c3 tracks with >=1 viable gap:",
        sum(
            1

            for count
            in track_boundary_counts.values()

            if count > 0
        ),
    )


    print_gap_population(
        track_boundary_counts
    )


    print_classification_report(
        boundary_results
    )


    print_identity_informative_boundaries(
        boundary_results
    )


    print_split_eligible_boundaries(
        boundary_results
    )


    # ========================================================
    # APPLY THE SAME VALIDATED SPLIT-SELECTION POLICY
    #
    # Diagnostic only. Nothing is persisted.
    # ========================================================

    (
        selected_splits,
        rejected_ambiguous,
    ) = select_split_proposals(
        boundary_results
    )


    print_selected_split_proposals(
        selected_splits
    )


    print_ambiguous_split_tracks(
        rejected_ambiguous
    )


    print_large_gap_tracks(
        boundary_results
    )


    print()
    print(
        "FINAL c3 GAP DIAGNOSTIC SUMMARY"
    )

    print(
        "==============================="
    )


    classification_counts = Counter(
        row[
            "classification"
        ]

        for row
        in boundary_results
    )


    print(
        "Useful c3 tracklets:",
        len(
            c3_rows
        ),
    )

    print(
        "Viable gap boundaries:",
        len(
            boundary_results
        ),
    )

    print(
        "Identity shifts:",
        classification_counts[
            "IDENTITY_SHIFT"
        ],
    )

    print(
        "Gained identity:",
        classification_counts[
            "GAINED_IDENTITY"
        ],
    )

    print(
        "Lost identity:",
        classification_counts[
            "LOST_IDENTITY"
        ],
    )

    print(
        "Same identity:",
        classification_counts[
            "SAME_IDENTITY"
        ],
    )

    print(
        "Ambiguous:",
        classification_counts[
            "AMBIGUOUS"
        ],
    )

    print(
        "Unresolved:",
        classification_counts[
            "UNRESOLVED"
        ],
    )

    print(
        "Selected split proposals:",
        len(
            selected_splits
        ),
    )

    print(
        "Rejected ambiguous split tracks:",
        len(
            rejected_ambiguous
        ),
    )


    print()
    print(
        "No c3 identities were assigned."
    )

    print(
        "No c3 splits were persisted."
    )

    print(
        "Production policy was not modified."
    )


if __name__ == "__main__":
    main()
