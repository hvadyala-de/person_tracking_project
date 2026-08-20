from collections import Counter, defaultdict
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

from src.analyze_c2_gap_identity_changes import (
    classify_boundary,
    find_gap_boundaries,
    format_summary,
    summarize_segment,
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


# ============================================================
# EXPERIMENT POLICY
#
# IDENTITY_SHIFT:
#   strong identity A before
#   strong identity B after
#   A != B
#
# GAINED_IDENTITY:
#   unresolved before
#   strong identity after
#
# These two classifications may propose a split.
#
# LOST_IDENTITY is intentionally NOT automatically split.
# Losing evidence is not sufficient proof that identity changed.
#
# UNRESOLVED / SAME_IDENTITY / AMBIGUOUS are also not split.
#
# Nothing here modifies production.
# ============================================================

SPLIT_CLASSIFICATIONS = {
    "IDENTITY_SHIFT",
    "GAINED_IDENTITY",
}


# ============================================================
# PRETTY LABELS
# ============================================================

def whole_label(
    track_id,
):

    return f"c2:{track_id}"


def before_label(
    track_id,
):

    return f"c2:{track_id}a"


def after_label(
    track_id,
):

    return f"c2:{track_id}b"


# ============================================================
# RANK ONE TRACKLET
# ============================================================

def analyze_tracklet(
    manager,
    tracklet,
    embedding,
    tracklet_lookup,
):

    ranked = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            tracklet,

        candidate_embedding=
            embedding,

        tracklet_lookup=
            tracklet_lookup,
    )

    summary = summarize_segment(
        ranked
    )

    return {
        "ranked":
            ranked,

        "summary":
            summary,
    }


# ============================================================
# BUILD BASELINE UNSPLIT c2 POPULATION
# ============================================================

def build_baseline_population(
    c2_rows,
):

    population = []

    for tracklet, embedding in c2_rows:

        population.append(
            {
                "source_track_id":
                    tracklet.local_track_id,

                "label":
                    whole_label(
                        tracklet.local_track_id
                    ),

                "kind":
                    "WHOLE",

                "tracklet":
                    tracklet,

                "embedding":
                    embedding,

                "split_boundary":
                    None,
            }
        )

    return population


# ============================================================
# ANALYZE BASELINE POPULATION
# ============================================================

def analyze_population(
    manager,
    population,
    tracklet_lookup,
):

    results = []

    for item in population:

        analysis = analyze_tracklet(
            manager=
                manager,

            tracklet=
                item[
                    "tracklet"
                ],

            embedding=
                item[
                    "embedding"
                ],

            tracklet_lookup=
                tracklet_lookup,
        )

        results.append(
            {
                **item,

                "analysis":
                    analysis,
            }
        )

    return results


# ============================================================
# EVALUATE ONE GAP BOUNDARY
# ============================================================

def evaluate_boundary(
    manager,
    extractor,
    video,
    source_tracklet,
    boundary,
    tracklet_lookup,
    synthetic_id_start,
):

    before_tracklet = build_segment_tracklet(
        source_tracklet=
            source_tracklet,

        start_frame=
            source_tracklet.start_frame,

        end_frame=
            boundary[
                "before_frame"
            ],

        synthetic_track_id=
            synthetic_id_start,
    )

    after_tracklet = build_segment_tracklet(
        source_tracklet=
            source_tracklet,

        start_frame=
            boundary[
                "after_frame"
            ],

        end_frame=
            source_tracklet.end_frame,

        synthetic_track_id=
            synthetic_id_start + 1,
    )

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

    classification = classify_boundary(
        before_summary,
        after_summary,
    )

    return {
        "track_id":
            source_tracklet.local_track_id,

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

        "classification":
            classification,

        "before_tracklet":
            before_tracklet,

        "after_tracklet":
            after_tracklet,

        "before_embedding":
            before_embedding,

        "after_embedding":
            after_embedding,

        "before_embedding_frames":
            before_embedding_frames,

        "after_embedding_frames":
            after_embedding_frames,

        "before_ranked":
            before_ranked,

        "after_ranked":
            after_ranked,

        "before_summary":
            before_summary,

        "after_summary":
            after_summary,
    }


# ============================================================
# DISCOVER ALL GAP EVIDENCE
# ============================================================

def discover_gap_evidence(
    manager,
    c2_rows,
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
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    results = []

    synthetic_id = 910000

    try:

        for source_tracklet, _ in c2_rows:

            boundaries = find_gap_boundaries(
                source_tracklet
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

    return results


# ============================================================
# SELECT AT MOST ONE SPLIT PER TRACKLET
# ============================================================

def select_split_proposals(
    boundary_results,
):

    by_track = defaultdict(
        list
    )

    for row in boundary_results:

        if (
            row[
                "classification"
            ]
            not in SPLIT_CLASSIFICATIONS
        ):

            continue

        by_track[
            row[
                "track_id"
            ]
        ].append(
            row
        )

    selected = {}

    rejected_ambiguous_tracks = {}


    for track_id, rows in by_track.items():

        # ----------------------------------------------------
        # IDENTITY_SHIFT has priority over GAINED_IDENTITY.
        # ----------------------------------------------------

        identity_shift_rows = [
            row

            for row in rows

            if row[
                "classification"
            ]
            == "IDENTITY_SHIFT"
        ]

        if identity_shift_rows:

            transitions = {
                (
                    row[
                        "before_summary"
                    ][
                        "gid"
                    ],
                    row[
                        "after_summary"
                    ][
                        "gid"
                    ],
                )

                for row
                in identity_shift_rows
            }

            # ------------------------------------------------
            # If different boundaries claim different
            # identity transitions, do not auto-select.
            # ------------------------------------------------

            if len(
                transitions
            ) > 1:

                rejected_ambiguous_tracks[
                    track_id
                ] = identity_shift_rows

                continue

            # ------------------------------------------------
            # Same identity transition observed at multiple
            # nearby gaps:
            #
            # choose the largest gap.
            #
            # This resolves the c2:9 case:
            #
            #   1284 -> 1316  gap 32
            #   1321 -> 1357  gap 36
            #
            # Both suggested GID1 -> GID11, but the second
            # larger gap is the visually validated boundary.
            # ------------------------------------------------

            best = max(
                identity_shift_rows,
                key=lambda row: (
                    row[
                        "gap"
                    ],
                    row[
                        "normalized_jump"
                    ],
                    row[
                        "after_frame"
                    ],
                ),
            )

            selected[
                track_id
            ] = best

            continue


        # ----------------------------------------------------
        # No identity shift. Consider GAINED_IDENTITY.
        # ----------------------------------------------------

        gained_rows = [
            row

            for row in rows

            if row[
                "classification"
            ]
            == "GAINED_IDENTITY"
        ]

        if gained_rows:

            gained_gids = {
                row[
                    "after_summary"
                ][
                    "gid"
                ]

                for row
                in gained_rows
            }

            if len(
                gained_gids
            ) > 1:

                rejected_ambiguous_tracks[
                    track_id
                ] = gained_rows

                continue

            best = max(
                gained_rows,
                key=lambda row: (
                    row[
                        "gap"
                    ],
                    row[
                        "normalized_jump"
                    ],
                    row[
                        "after_frame"
                    ],
                ),
            )

            selected[
                track_id
            ] = best


    return (
        selected,
        rejected_ambiguous_tracks,
    )


# ============================================================
# BUILD IDENTITY-AWARE SPLIT POPULATION
# ============================================================

def build_split_population(
    c2_rows,
    selected_splits,
):

    population = []

    for source_tracklet, source_embedding in c2_rows:

        track_id = (
            source_tracklet.local_track_id
        )

        proposal = selected_splits.get(
            track_id
        )

        if proposal is None:

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
                    proposal[
                        "before_tracklet"
                    ],

                "embedding":
                    proposal[
                        "before_embedding"
                    ],

                "split_boundary":
                    proposal,
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
                    proposal[
                        "after_tracklet"
                    ],

                "embedding":
                    proposal[
                        "after_embedding"
                    ],

                "split_boundary":
                    proposal,
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

    return population


# ============================================================
# COUNT POPULATION ASSOCIATION STATES
# ============================================================

def population_counts(
    results,
):

    counts = Counter()

    for row in results:

        status = row[
            "analysis"
        ][
            "summary"
        ][
            "status"
        ]

        counts[
            status
        ] += 1

    return counts


# ============================================================
# MAP RESULTS BY LABEL
# ============================================================

def results_by_label(
    results,
):

    return {
        row[
            "label"
        ]:
            row

        for row
        in results
    }


# ============================================================
# FORMAT RESULT
# ============================================================

def format_result(
    row,
):

    if row is None:

        return "MISSING"

    summary = row[
        "analysis"
    ][
        "summary"
    ]

    return format_summary(
        summary
    )


# ============================================================
# PRINT SPLIT PROPOSAL
# ============================================================

def print_split_proposal(
    row,
):

    print(
        f"c2:{row['track_id']}"
        f" | "
        f"{row['before_frame']}"
        f"->{row['after_frame']}"
        f" | gap="
        f"{row['gap']}"
        f" | norm="
        f"{row['normalized_jump']:.3f}"
        f" | "
        f"{row['classification']}"
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


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_checks(
    selected_splits,
    split_results,
):

    result_map = results_by_label(
        split_results
    )

    checks = []


    # ========================================================
    # c2:9 expected split boundary
    # ========================================================

    c2_9_split = selected_splits.get(
        9
    )

    checks.append(
        (
            "c2:9 split boundary 1321->1357",

            c2_9_split is not None

            and

            c2_9_split[
                "before_frame"
            ] == 1321

            and

            c2_9_split[
                "after_frame"
            ] == 1357,
        )
    )


    # ========================================================
    # c2:9a -> GID 1
    # ========================================================

    c2_9a = result_map.get(
        "c2:9a"
    )

    checks.append(
        (
            "c2:9a -> GID 1",

            c2_9a is not None

            and

            c2_9a[
                "analysis"
            ][
                "summary"
            ][
                "status"
            ] == "UNIQUE"

            and

            c2_9a[
                "analysis"
            ][
                "summary"
            ][
                "gid"
            ] == 1,
        )
    )


    # ========================================================
    # c2:9b -> GID 11
    # ========================================================

    c2_9b = result_map.get(
        "c2:9b"
    )

    checks.append(
        (
            "c2:9b -> GID 11",

            c2_9b is not None

            and

            c2_9b[
                "analysis"
            ][
                "summary"
            ][
                "status"
            ] == "UNIQUE"

            and

            c2_9b[
                "analysis"
            ][
                "summary"
            ][
                "gid"
            ] == 11,
        )
    )


    # ========================================================
    # c2:601 expected split
    # ========================================================

    c2_601_split = selected_splits.get(
        601
    )

    checks.append(
        (
            "c2:601 split boundary 4016->4049",

            c2_601_split is not None

            and

            c2_601_split[
                "before_frame"
            ] == 4016

            and

            c2_601_split[
                "after_frame"
            ] == 4049,
        )
    )


    # ========================================================
    # c2:601a remains unresolved
    # ========================================================

    c2_601a = result_map.get(
        "c2:601a"
    )

    checks.append(
        (
            "c2:601a unresolved",

            c2_601a is not None

            and

            c2_601a[
                "analysis"
            ][
                "summary"
            ][
                "status"
            ] == "NONE",
        )
    )


    # ========================================================
    # c2:601b -> GID 73
    # ========================================================

    c2_601b = result_map.get(
        "c2:601b"
    )

    checks.append(
        (
            "c2:601b -> GID 73",

            c2_601b is not None

            and

            c2_601b[
                "analysis"
            ][
                "summary"
            ][
                "status"
            ] == "UNIQUE"

            and

            c2_601b[
                "analysis"
            ][
                "summary"
            ][
                "gid"
            ] == 73,
        )
    )


    # ========================================================
    # c2:508 should stay whole and map GID65
    # ========================================================

    c2_508 = result_map.get(
        "c2:508"
    )

    checks.append(
        (
            "c2:508 remains whole -> GID 65",

            c2_508 is not None

            and

            c2_508[
                "kind"
            ] == "WHOLE"

            and

            c2_508[
                "analysis"
            ][
                "summary"
            ][
                "status"
            ] == "UNIQUE"

            and

            c2_508[
                "analysis"
            ][
                "summary"
            ][
                "gid"
            ] == 65,
        )
    )


    # ========================================================
    # c2:635 LOST_IDENTITY should NOT be auto-split
    # ========================================================

    c2_635 = result_map.get(
        "c2:635"
    )

    checks.append(
        (
            "c2:635 remains unsplit",

            635 not in selected_splits

            and

            c2_635 is not None

            and

            c2_635[
                "kind"
            ] == "WHOLE",
        )
    )


    print()
    print(
        "SANITY CHECKS"
    )

    print(
        "============="
    )


    passed_count = 0


    for description, passed in checks:

        if passed:

            passed_count += 1

        print(
            "PASS"
            if passed
            else "FAIL",
            "|",
            description,
        )


    print()
    print(
        f"{passed_count}/"
        f"{len(checks)} checks passed"
    )


    return (
        passed_count
        == len(
            checks
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2 IDENTITY-AWARE SPLIT EXPERIMENT"
    )

    print(
        "=================================="
    )


    # ========================================================
    # REPLAY VALIDATED c0+c1 PRODUCTION
    # ========================================================

    (
        manager,
        baseline_tracklets,
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


    print()
    print(
        "c0+c1 BASELINE"
    )

    print(
        "=============="
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
        "Anchored GIDs:",
        len(
            manager.anchored_gids
        ),
    )


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


    # ========================================================
    # UNSPLIT BASELINE
    # ========================================================

    print()
    print(
        "ANALYZING UNSPLIT c2 BASELINE..."
    )


    baseline_population = (
        build_baseline_population(
            c2_rows
        )
    )


    baseline_results = analyze_population(
        manager=
            manager,

        population=
            baseline_population,

        tracklet_lookup=
            tracklet_lookup,
    )


    baseline_counts = population_counts(
        baseline_results
    )


    # ========================================================
    # GAP ANALYSIS
    # ========================================================

    print()
    print(
        "EVALUATING VIABLE c2 GAP BOUNDARIES..."
    )


    boundary_results = discover_gap_evidence(
        manager=
            manager,

        c2_rows=
            c2_rows,

        tracklet_lookup=
            tracklet_lookup,
    )


    boundary_counts = Counter(
        row[
            "classification"
        ]

        for row
        in boundary_results
    )


    print()
    print(
        "BOUNDARY CLASSIFICATION COUNTS"
    )

    print(
        "=============================="
    )


    for classification in [
        "IDENTITY_SHIFT",
        "GAINED_IDENTITY",
        "LOST_IDENTITY",
        "AMBIGUOUS",
        "SAME_IDENTITY",
        "UNRESOLVED",
    ]:

        print(
            f"{classification:<18}: "
            f"{boundary_counts.get(classification, 0)}"
        )


    # ========================================================
    # SELECT SPLITS
    # ========================================================

    (
        selected_splits,
        rejected_ambiguous_tracks,
    ) = select_split_proposals(
        boundary_results
    )


    print()
    print(
        "SELECTED IDENTITY-AWARE SPLITS"
    )

    print(
        "=============================="
    )


    if not selected_splits:

        print(
            "NONE"
        )


    else:

        for track_id in sorted(
            selected_splits
        ):

            print()

            print_split_proposal(
                selected_splits[
                    track_id
                ]
            )


    print()
    print(
        "TRACKLETS REJECTED DUE TO "
        "CONFLICTING SPLIT PROPOSALS"
    )

    print(
        "=========================================="
    )


    if not rejected_ambiguous_tracks:

        print(
            "NONE"
        )


    else:

        for track_id, rows in (
            sorted(
                rejected_ambiguous_tracks.items()
            )
        ):

            print()
            print(
                f"c2:{track_id}"
            )

            for row in rows:

                print_split_proposal(
                    row
                )


    # ========================================================
    # BUILD SPLIT POPULATION
    # ========================================================

    split_population = (
        build_split_population(
            c2_rows=
                c2_rows,

            selected_splits=
                selected_splits,
        )
    )


    split_results = analyze_population(
        manager=
            manager,

        population=
            split_population,

        tracklet_lookup=
            tracklet_lookup,
    )


    split_counts = population_counts(
        split_results
    )


    # ========================================================
    # POPULATION COMPARISON
    # ========================================================

    print()
    print(
        "UNSPLIT vs IDENTITY-AWARE SPLIT"
    )

    print(
        "==============================="
    )


    print(
        f"{'Metric':<26}"
        f"{'Unsplit':>10}"
        f"{'Split':>10}"
        f"{'Delta':>10}"
    )

    print(
        "-" * 56
    )


    metrics = [
        (
            "Population items",
            len(
                baseline_results
            ),
            len(
                split_results
            ),
        ),

        (
            "UNIQUE strong GID",
            baseline_counts.get(
                "UNIQUE",
                0,
            ),
            split_counts.get(
                "UNIQUE",
                0,
            ),
        ),

        (
            "AMBIGUOUS",
            baseline_counts.get(
                "AMBIGUOUS",
                0,
            ),
            split_counts.get(
                "AMBIGUOUS",
                0,
            ),
        ),

        (
            "NONE / unresolved",
            baseline_counts.get(
                "NONE",
                0,
            ),
            split_counts.get(
                "NONE",
                0,
            ),
        ),
    ]


    for (
        name,
        baseline_value,
        split_value,
    ) in metrics:

        delta = (
            split_value
            - baseline_value
        )

        print(
            f"{name:<26}"
            f"{baseline_value:>10}"
            f"{split_value:>10}"
            f"{delta:>+10}"
        )


    # ========================================================
    # SPLIT TRACKLET BEFORE / AFTER COMPARISON
    # ========================================================

    baseline_map = results_by_label(
        baseline_results
    )

    split_map = results_by_label(
        split_results
    )


    print()
    print(
        "CHANGES FOR SELECTED SPLIT TRACKLETS"
    )

    print(
        "===================================="
    )


    for track_id in sorted(
        selected_splits
    ):

        whole = baseline_map.get(
            whole_label(
                track_id
            )
        )

        before = split_map.get(
            before_label(
                track_id
            )
        )

        after = split_map.get(
            after_label(
                track_id
            )
        )


        proposal = selected_splits[
            track_id
        ]


        print()
        print(
            f"c2:{track_id}"
            f" | split "
            f"{proposal['before_frame']}"
            f"->{proposal['after_frame']}"
            f" | "
            f"{proposal['classification']}"
        )


        print(
            "  UNSPLIT:",
            format_result(
                whole
            ),
        )


        print(
            "  BEFORE :",
            format_result(
                before
            ),
        )


        print(
            "  AFTER  :",
            format_result(
                after
            ),
        )


    # ========================================================
    # ALL UNIQUE ASSOCIATIONS AFTER SPLITTING
    # ========================================================

    print()
    print(
        "UNIQUE c2 -> EXISTING GID ASSOCIATIONS "
        "AFTER SPLITTING"
    )

    print(
        "=============================================="
    )


    unique_rows = [
        row

        for row in split_results

        if (
            row[
                "analysis"
            ][
                "summary"
            ][
                "status"
            ]
            == "UNIQUE"
        )
    ]


    unique_rows.sort(
        key=lambda row: (
            row[
                "analysis"
            ][
                "summary"
            ][
                "gid"
            ],

            row[
                "tracklet"
            ].start_frame,

            row[
                "label"
            ],
        )
    )


    if not unique_rows:

        print(
            "NONE"
        )


    else:

        for row in unique_rows:

            summary = row[
                "analysis"
            ][
                "summary"
            ]

            candidate = summary[
                "row"
            ]

            print(
                f"{row['label']:<12}"
                f" -> GID "
                f"{summary['gid']:<3}"
                f" | joint="
                f"{candidate['joint_support_count']}"
                f" | geom="
                f"{candidate['geometry_support_count']}"
                f" | max="
                f"{candidate['core_max']:.4f}"
                f" | top3="
                f"{candidate['core_top3']:.4f}"
            )


    # ========================================================
    # SANITY
    # ========================================================

    sanity_pass = run_sanity_checks(
        selected_splits=
            selected_splits,

        split_results=
            split_results,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "FINAL EXPERIMENT SUMMARY"
    )

    print(
        "========================"
    )


    print(
        "Original useful c2 tracklets:",
        len(
            c2_rows
        ),
    )


    print(
        "Selected split tracklets:",
        len(
            selected_splits
        ),
    )


    print(
        "Resulting in-memory c2 segments:",
        len(
            split_population
        ),
    )


    print(
        "Unique strong associations before:",
        baseline_counts.get(
            "UNIQUE",
            0,
        ),
    )


    print(
        "Unique strong associations after:",
        split_counts.get(
            "UNIQUE",
            0,
        ),
    )


    print(
        "Ambiguous associations after:",
        split_counts.get(
            "AMBIGUOUS",
            0,
        ),
    )


    print(
        "Unresolved after:",
        split_counts.get(
            "NONE",
            0,
        ),
    )


    print(
        "Sanity:",
        (
            "PASS"
            if sanity_pass
            else "FAIL / REVIEW REQUIRED"
        ),
    )


    print()
    print(
        "IDENTITIES MODIFIED: NO"
    )

    print(
        "TRACK CSV MODIFIED: NO"
    )

    print(
        "EMBEDDING FILES MODIFIED: NO"
    )

    print(
        "All splits existed in memory only."
    )

    print(
        "Experiment only."
    )


if __name__ == "__main__":
    main()
