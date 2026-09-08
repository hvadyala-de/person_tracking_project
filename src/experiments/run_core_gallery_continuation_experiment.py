from src.experiments.analyze_final_pending_population import (
    replay_final_production,
)

from src.experiments.analyze_core_gallery_near_miss_gate import (
    CORE_GALLERY_MAX_MIN,
    CORE_GALLERY_TOP3_MIN,
    core_gallery_scores,
    cross_core_support,
)

from src.identity_compatibility import (
    identity_has_conflict,
    normalize_camera_id,
)


# ============================================================
# EXPECTED VALIDATED TARGET
# ============================================================

EXPECTED_CAMERA_ID = 1
EXPECTED_TRACK_ID = 787
EXPECTED_GID = 73

EXPECTED_BASELINE_GIDS = 76
EXPECTED_BASELINE_PENDING = 93
EXPECTED_CORE_MEMBERS = 68

EXPECTED_POST_PENDING = 92


# ============================================================
# COUNT CORE MEMBERS
# ============================================================

def count_core_members(
    manager,
):

    return sum(
        len(
            members
        )

        for members
        in manager.core_members.values()
    )


# ============================================================
# FIND CORE-ONLY HIGH-GALLERY CANDIDATES
#
# This is intentionally the same diagnostic rule validated by:
#
# src.analyze_core_gallery_near_miss_gate
#
# Positive evidence:
#
#   * CORE-only gallery max
#   * CORE-only gallery top-3
#   * cross-camera CORE geometry
#   * cross-camera CORE direct appearance
#
# Negative evidence:
#
#   * same-camera identity conflict veto
#   * geometry contradiction veto
#
# RELAXED members do NOT provide positive evidence here.
# ============================================================

def find_candidates(
    manager,
    tracklet_lookup,
):

    candidates = []
    ambiguous = []


    for key, pending in sorted(
        manager.pending_tracklets.items()
    ):

        camera_id = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )

        local_track_id = pending[
            "local_track_id"
        ]


        tracklet = pending.get(
            "candidate_tracklet"
        )


        if tracklet is None:

            tracklet = manager.get_tracklet(
                camera_id,
                local_track_id,
                tracklet_lookup,
            )


        if tracklet is None:

            continue


        qualifying = []


        for gid in sorted(
            manager.anchored_gids
        ):

            identity = manager.identities.get(
                gid
            )


            if identity is None:

                continue


            # ------------------------------------------------
            # Preserve current negative evidence.
            # ------------------------------------------------

            if identity_has_conflict(
                tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            if manager.identity_geometry_contradicted(
                tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            # ------------------------------------------------
            # CORE-ONLY gallery appearance.
            # ------------------------------------------------

            scores = core_gallery_scores(
                manager,
                gid,
                pending[
                    "embedding"
                ],
            )


            if scores is None:

                continue


            if (
                scores[
                    "max"
                ]
                < CORE_GALLERY_MAX_MIN
            ):

                continue


            if (
                scores[
                    "top3"
                ]
                < CORE_GALLERY_TOP3_MIN
            ):

                continue


            # ------------------------------------------------
            # Independent cross-camera CORE geometry support.
            # ------------------------------------------------

            cross_rows = cross_core_support(
                manager=
                    manager,

                gid=
                    gid,

                pending_tracklet=
                    tracklet,

                pending_embedding=
                    pending[
                        "embedding"
                    ],

                tracklet_lookup=
                    tracklet_lookup,
            )


            if not cross_rows:

                continue


            cross_rows.sort(
                key=lambda row: (
                    row[
                        "direct"
                    ],

                    row[
                        "shared"
                    ],

                    -row[
                        "median"
                    ],
                ),
                reverse=True,
            )


            qualifying.append(
                {
                    "gid":
                        gid,

                    "scores":
                        scores,

                    "cross":
                        cross_rows,

                    "best_cross":
                        cross_rows[
                            0
                        ],
                }
            )


        if len(
            qualifying
        ) == 1:

            candidates.append(
                {
                    "key":
                        key,

                    "pending":
                        pending,

                    "camera_id":
                        camera_id,

                    "local_track_id":
                        local_track_id,

                    **qualifying[
                        0
                    ],
                }
            )


        elif len(
            qualifying
        ) > 1:

            ambiguous.append(
                {
                    "key":
                        key,

                    "camera_id":
                        camera_id,

                    "local_track_id":
                        local_track_id,

                    "candidates":
                        qualifying,
                }
            )


    return (
        candidates,
        ambiguous,
    )


# ============================================================
# APPLY ONE FINAL-ONLY RELAXED CONTINUATION
# ============================================================

def apply_candidate(
    manager,
    candidate,
):

    pending = candidate[
        "pending"
    ]

    gid = candidate[
        "gid"
    ]


    manager.add_to_identity(
        global_id=
            gid,

        camera_id=
            pending[
                "camera_id"
            ],

        local_track_id=
            pending[
                "local_track_id"
            ],

        start_frame=
            pending[
                "start_frame"
            ],

        end_frame=
            pending[
                "end_frame"
            ],

        embedding=
            pending[
                "embedding"
            ],

        trust_level=
            "RELAXED",
    )


    del manager.pending_tracklets[
        candidate[
            "key"
        ]
    ]


# ============================================================
# PRINT GID MEMBERS
# ============================================================

def print_gid_members(
    manager,
    gid,
):

    identity = manager.identities[
        gid
    ]


    for member in identity.members:

        camera_id = normalize_camera_id(
            member.camera_id
        )


        trust = manager.get_member_trust(
            gid,
            camera_id,
            member.local_track_id,
        )


        print(
            f"c{camera_id}:"
            f"{member.local_track_id}"
            f" [{trust}]"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "CORE-ONLY HIGH-GALLERY CONTINUATION EXPERIMENT"
    )

    print(
        "=============================================="
    )


    (
        manager,
        tracklets,
        tracklet_lookup,
        final_results,
    ) = replay_final_production()


    baseline_gids = len(
        manager.identities
    )


    baseline_pending = len(
        manager.pending_tracklets
    )


    baseline_core = count_core_members(
        manager
    )


    print()
    print(
        "BASELINE FINAL PRODUCTION STATE"
    )

    print(
        "==============================="
    )


    print(
        "Tracklets:",
        len(
            tracklets
        ),
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
        "CORE members:",
        baseline_core,
    )


    # ========================================================
    # FIND CANDIDATES
    # ========================================================

    (
        candidates,
        ambiguous,
    ) = find_candidates(
        manager,
        tracklet_lookup,
    )


    print()
    print(
        "EXPERIMENT CANDIDATES"
    )

    print(
        "====================="
    )


    if not candidates:

        print(
            "NONE"
        )


    for candidate in candidates:

        scores = candidate[
            "scores"
        ]


        cross = candidate[
            "best_cross"
        ]


        print(
            f"c{candidate['camera_id']}:"
            f"{candidate['local_track_id']}"
            f" -> GID "
            f"{candidate['gid']}"
            f" | CORE max="
            f"{scores['max']:.4f}"
            f" | CORE top3="
            f"{scores['top3']:.4f}"
            f" | CROSS CORE="
            f"c{cross['camera_id']}:"
            f"{cross['track_id']}"
            f" | direct="
            f"{cross['direct']:.4f}"
            f" | shared="
            f"{cross['shared']}"
            f" | median="
            f"{cross['median']:.2f}"
        )


    print()
    print(
        "AMBIGUOUS CANDIDATES"
    )

    print(
        "===================="
    )


    if not ambiguous:

        print(
            "NONE"
        )


    else:

        for row in ambiguous:

            gids = [
                candidate[
                    "gid"
                ]

                for candidate
                in row[
                    "candidates"
                ]
            ]


            print(
                f"c{row['camera_id']}:"
                f"{row['local_track_id']}"
                f" -> GIDs "
                f"{gids}"
            )


    # ========================================================
    # REQUIRE EXACTLY ONE UNIQUE CANDIDATE
    # ========================================================

    if (
        len(
            candidates
        )
        != 1

        or

        len(
            ambiguous
        )
        != 0
    ):

        print()
        print(
            "EXPERIMENT VERDICT: FAIL / REVIEW REQUIRED"
        )

        print(
            "Expected exactly one unique candidate "
            "and zero ambiguities."
        )

        return


    candidate = candidates[
        0
    ]


    expected_candidate = (
        candidate[
            "camera_id"
        ]
        == EXPECTED_CAMERA_ID

        and

        candidate[
            "local_track_id"
        ]
        == EXPECTED_TRACK_ID

        and

        candidate[
            "gid"
        ]
        == EXPECTED_GID
    )


    if not expected_candidate:

        print()
        print(
            "EXPERIMENT VERDICT: FAIL / REVIEW REQUIRED"
        )

        print(
            "Unique candidate did not match "
            "expected c1:787 -> GID 73."
        )

        return


    # ========================================================
    # APPLY FINAL-ONLY RELAXED ADDITION
    # ========================================================

    print()
    print(
        "APPLY FINAL-ONLY EXPERIMENTAL CONTINUATION"
    )

    print(
        "=========================================="
    )


    apply_candidate(
        manager,
        candidate,
    )


    target_key = (
        EXPECTED_CAMERA_ID,
        EXPECTED_TRACK_ID,
    )


    target_trust = (
        manager.get_member_trust(
            EXPECTED_GID,
            EXPECTED_CAMERA_ID,
            EXPECTED_TRACK_ID,
        )
    )


    post_add_pending = len(
        manager.pending_tracklets
    )


    post_add_core = count_core_members(
        manager
    )


    print(
        f"c{EXPECTED_CAMERA_ID}:"
        f"{EXPECTED_TRACK_ID}"
        f" -> GID "
        f"{EXPECTED_GID}"
    )


    print(
        "Trust:",
        target_trust,
    )


    print(
        "Still pending:",
        target_key
        in manager.pending_tracklets,
    )


    print(
        "GIDs:",
        len(
            manager.identities
        ),
    )


    print(
        "Pending:",
        post_add_pending,
    )


    print(
        "CORE members:",
        post_add_core,
    )


    # ========================================================
    # GID 73 AFTER EXPERIMENTAL ADDITION
    # ========================================================

    print()
    print(
        "GID 73 AFTER EXPERIMENT"
    )

    print(
        "======================="
    )


    print_gid_members(
        manager,
        EXPECTED_GID,
    )


    # ========================================================
    # DELIBERATE CASCADE SAFETY TEST
    #
    # Production would STOP after this final-only addition.
    #
    # We deliberately run the ordinary cross-camera pending
    # sweep anyway to test whether the new RELAXED embedding
    # could accidentally unlock another identity.
    #
    # Same-camera and dual-evidence stages remain disabled.
    # ========================================================

    pending_before_cascade_test = len(
        manager.pending_tracklets
    )


    cascade_results = (
        manager.reevaluate_pending(
            tracklet_lookup=
                tracklet_lookup,

            include_same_camera=
                False,

            include_dual_evidence=
                False,
        )
    )


    cascade_resolutions = [
        result

        for result
        in cascade_results

        if result.get(
            "reason"
        )
        == "RELAXED_CORE_PENDING_SWEEP"
    ]


    pending_after_cascade_test = len(
        manager.pending_tracklets
    )


    print()
    print(
        "POST-ADDITION CASCADE SAFETY TEST"
    )

    print(
        "================================"
    )


    print(
        "Pending before test:",
        pending_before_cascade_test,
    )


    print(
        "Pending after test:",
        pending_after_cascade_test,
    )


    print(
        "Cascade resolutions:",
        len(
            cascade_resolutions
        ),
    )


    if cascade_resolutions:

        for result in cascade_resolutions:

            print(
                f"UNEXPECTED CASCADE | "
                f"c{result['camera_id']}:"
                f"{result['local_track_id']}"
                f" -> GID "
                f"{result['global_id']}"
            )


    else:

        print(
            "POST-ADDITION CASCADE RESOLUTIONS: NONE"
        )


    # ========================================================
    # FINAL SAFETY VERDICT
    # ========================================================

    final_core = count_core_members(
        manager
    )


    passed = (
        baseline_gids
        == EXPECTED_BASELINE_GIDS

        and

        baseline_pending
        == EXPECTED_BASELINE_PENDING

        and

        baseline_core
        == EXPECTED_CORE_MEMBERS

        and

        post_add_pending
        == EXPECTED_POST_PENDING

        and

        target_key
        not in manager.pending_tracklets

        and

        target_trust
        == "RELAXED"

        and

        post_add_core
        == EXPECTED_CORE_MEMBERS

        and

        final_core
        == EXPECTED_CORE_MEMBERS

        and

        len(
            manager.identities
        )
        == EXPECTED_BASELINE_GIDS

        and

        len(
            cascade_resolutions
        )
        == 0

        and

        pending_after_cascade_test
        == EXPECTED_POST_PENDING
    )


    print()
    print(
        "EXPERIMENT SUMMARY"
    )

    print(
        "=================="
    )


    print(
        "Baseline GIDs:",
        baseline_gids,
    )


    print(
        "Baseline pending:",
        baseline_pending,
    )


    print(
        "Experimental continuations:",
        1,
    )


    print(
        "Post-continuation pending:",
        post_add_pending,
    )


    print(
        "CORE members:",
        final_core,
    )


    print(
        "Post-addition cascade resolutions:",
        len(
            cascade_resolutions
        ),
    )


    print()
    print(
        "EXPERIMENT VERDICT:",
        (
            "PASS"
            if passed
            else "FAIL / REVIEW REQUIRED"
        ),
    )


if __name__ == "__main__":
    main()
