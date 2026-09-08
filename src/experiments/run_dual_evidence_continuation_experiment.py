from src.experiments.analyze_dual_evidence_gate import (
    evaluate_gid,
    replay_production,
)

from src.identity_compatibility import (
    normalize_camera_id,
)


# ============================================================
# HELPERS
# ============================================================

def core_member_count(
    manager,
):

    return sum(
        len(members)
        for members
        in manager.core_members.values()
    )


def find_gid(
    manager,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            member_camera = (
                normalize_camera_id(
                    member.camera_id
                )
            )


            if (
                member_camera
                == camera_id

                and

                member.local_track_id
                == local_track_id
            ):

                return gid


    return None


def find_pending_key(
    manager,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )


    for key, pending in (
        manager.pending_tracklets.items()
    ):

        pending_camera = (
            normalize_camera_id(
                pending[
                    "camera_id"
                ]
            )
        )


        if (
            pending_camera
            == camera_id

            and

            pending[
                "local_track_id"
            ]
            == local_track_id
        ):

            return key


    return None


def is_pending(
    manager,
    camera_id,
    local_track_id,
):

    return (
        find_pending_key(
            manager,
            camera_id,
            local_track_id,
        )
        is not None
    )


def is_core_member(
    manager,
    gid,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )


    key = (
        camera_id,
        local_track_id,
    )


    return (
        key
        in manager.core_members[
            gid
        ]
    )


# ============================================================
# COLLECT DUAL-EVIDENCE CANDIDATES
#
# Reuses exactly the diagnostic gate already tested in:
#
# src/analyze_dual_evidence_gate.py
#
# No production thresholds are changed.
# ============================================================

def collect_dual_candidates(
    manager,
    lookup,
):

    unique = []

    ambiguous = []


    for key, pending in list(
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
                lookup,
            )


        if tracklet is None:

            continue


        qualifying = []


        for gid in sorted(
            manager.anchored_gids
        ):

            result = evaluate_gid(
                manager=manager,

                gid=gid,

                pending_tracklet=
                    tracklet,

                pending_embedding=
                    pending[
                        "embedding"
                    ],

                lookup=lookup,
            )


            if result is not None:

                qualifying.append(
                    result
                )


        if len(qualifying) == 1:

            unique.append(
                {
                    "key":
                        key,

                    "pending":
                        pending,

                    "result":
                        qualifying[0],
                }
            )


        elif len(qualifying) > 1:

            ambiguous.append(
                {
                    "key":
                        key,

                    "pending":
                        pending,

                    "results":
                        qualifying,
                }
            )


    return (
        unique,
        ambiguous,
    )


# ============================================================
# ADD UNIQUE DUAL-EVIDENCE MEMBER AS RELAXED
# ============================================================

def apply_dual_candidate(
    manager,
    candidate,
):

    key = candidate[
        "key"
    ]


    pending = candidate[
        "pending"
    ]


    result = candidate[
        "result"
    ]


    gid = result[
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
        key
    ]


    return {
        "camera_id":
            normalize_camera_id(
                pending[
                    "camera_id"
                ]
            ),

        "local_track_id":
            pending[
                "local_track_id"
            ],

        "global_id":
            gid,

        "same":
            result[
                "same"
            ],

        "cross":
            result[
                "cross"
            ],
    }


# ============================================================
# REGRESSION HELPERS
# ============================================================

def group_gid(
    manager,
    members,
):

    gids = []


    for (
        camera_id,
        local_track_id,
    ) in members:

        gid = find_gid(
            manager,
            camera_id,
            local_track_id,
        )


        gids.append(
            gid
        )


    if (
        not gids
        or any(
            gid is None
            for gid in gids
        )
    ):

        return None


    if len(
        set(
            gids
        )
    ) != 1:

        return None


    return gids[0]


def check_together(
    manager,
    name,
    members,
):

    gid = group_gid(
        manager,
        members,
    )


    passed = (
        gid is not None
    )


    return (
        passed,
        name,
        (
            f"GID {gid}"
            if gid is not None
            else "NOT TOGETHER"
        ),
    )


def check_separate(
    manager,
    name,
    first,
    second,
):

    first_gid = find_gid(
        manager,
        first[0],
        first[1],
    )


    second_gid = find_gid(
        manager,
        second[0],
        second[1],
    )


    passed = (
        first_gid is not None
        and second_gid is not None
        and first_gid != second_gid
    )


    detail = (
        f"GID {first_gid}"
        f" vs "
        f"GID {second_gid}"
    )


    return (
        passed,
        name,
        detail,
    )


def run_regression_checks(
    manager,
):

    checks = []


    checks.append(
        check_together(
            manager,
            "255 family together",
            [
                (0, 255),
                (1, 298),
                (1, 319),
            ],
        )
    )


    checks.append(
        check_together(
            manager,
            "399 family together",
            [
                (0, 399),
                (1, 382),
                (1, 369),
            ],
        )
    )


    checks.append(
        check_together(
            manager,
            "469/503/600 together",
            [
                (0, 469),
                (1, 503),
                (0, 600),
            ],
        )
    )


    checks.append(
        check_together(
            manager,
            "492/511 together",
            [
                (0, 492),
                (1, 511),
            ],
        )
    )


    checks.append(
        check_together(
            manager,
            "651/560 together",
            [
                (0, 651),
                (1, 560),
            ],
        )
    )


    checks.append(
        check_together(
            manager,
            "701 family together",
            [
                (0, 701),
                (0, 740),
                (1, 639),
                (1, 721),
            ],
        )
    )


    checks.append(
        check_together(
            manager,
            "1 family together",
            [
                (0, 1),
                (1, 1),
                (1, 66),
            ],
        )
    )


    checks.append(
        check_separate(
            manager,
            "255 separate from 399",
            (0, 255),
            (0, 399),
        )
    )


    checks.append(
        check_separate(
            manager,
            "492 separate from 701",
            (0, 492),
            (0, 701),
        )
    )


    checks.append(
        check_separate(
            manager,
            "651 separate from 701",
            (0, 651),
            (0, 701),
        )
    )


    return checks


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "DUAL-EVIDENCE CONTINUATION EXPERIMENT"
    )

    print(
        "====================================="
    )


    (
        manager,
        items,
        lookup,
    ) = replay_production()


    baseline_gids = len(
        manager.identities
    )


    baseline_pending = len(
        manager.pending_tracklets
    )


    baseline_core = (
        core_member_count(
            manager
        )
    )


    print()
    print(
        "BASELINE"
    )

    print(
        "========"
    )


    print(
        "Tracklets:",
        len(
            items
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


    print()
    print(
        "CHAIN MEMBERS BEFORE"
    )

    print(
        "===================="
    )


    print(
        "c1:556 pending:",
        is_pending(
            manager,
            1,
            556,
        ),
    )


    print(
        "c0:622 pending:",
        is_pending(
            manager,
            0,
            622,
        ),
    )


    print(
        "c1:549 pending:",
        is_pending(
            manager,
            1,
            549,
        ),
    )


    # ========================================================
    # DISCOVER DUAL-EVIDENCE CANDIDATES
    # ========================================================

    (
        unique,
        ambiguous,
    ) = collect_dual_candidates(
        manager,
        lookup,
    )


    print()
    print(
        "DUAL-EVIDENCE DISCOVERY"
    )

    print(
        "======================="
    )


    print(
        "Unique candidates:",
        len(
            unique
        ),
    )


    print(
        "Ambiguous candidates:",
        len(
            ambiguous
        ),
    )


    for candidate in unique:

        pending = candidate[
            "pending"
        ]


        result = candidate[
            "result"
        ]


        same = result[
            "same"
        ]


        cross = result[
            "cross"
        ]


        camera_id = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )


        print()
        print(
            f"UNIQUE "
            f"c{camera_id}:"
            f"{pending['local_track_id']}"
            f" -> GID "
            f"{result['gid']}"
        )


        print(
            f"  SAME CORE "
            f"c{same['camera_id']}:"
            f"{same['track_id']}"
            f" | gap="
            f"{same['gap']}"
            f" | ReID="
            f"{same['reid']:.4f}"
            f" | center="
            f"{same['center']:.2f}"
            f" | bottom="
            f"{same['bottom']:.2f}"
        )


        print(
            f"  CROSS CORE "
            f"c{cross['camera_id']}:"
            f"{cross['track_id']}"
            f" | ReID="
            f"{cross['reid']:.4f}"
            f" | shared="
            f"{cross['shared']}"
            f" | median="
            f"{cross['median']:.2f}"
        )


    for candidate in ambiguous:

        pending = candidate[
            "pending"
        ]


        camera_id = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )


        gids = [
            result[
                "gid"
            ]
            for result
            in candidate[
                "results"
            ]
        ]


        print(
            f"AMBIGUOUS "
            f"c{camera_id}:"
            f"{pending['local_track_id']}"
            f" -> "
            f"{gids}"
        )


    # ========================================================
    # SAFETY:
    # DO NOT APPLY ANYTHING IF THE GATE IS AMBIGUOUS.
    # ========================================================

    if ambiguous:

        print()
        print(
            "EXPERIMENT ABORTED"
        )

        print(
            "Dual-evidence ambiguity exists."
        )

        return


    # ========================================================
    # APPLY UNIQUE DUAL-EVIDENCE CANDIDATES AS RELAXED
    # ========================================================

    applied = []


    for candidate in unique:

        applied.append(
            apply_dual_candidate(
                manager,
                candidate,
            )
        )


    print()
    print(
        "DUAL-EVIDENCE ADDITIONS"
    )

    print(
        "======================="
    )


    if not applied:

        print(
            "NONE"
        )


    for result in applied:

        print(
            f"c{result['camera_id']}:"
            f"{result['local_track_id']}"
            f" -> GID "
            f"{result['global_id']}"
            f" | trust=RELAXED"
        )


    after_dual_gids = len(
        manager.identities
    )


    after_dual_pending = len(
        manager.pending_tracklets
    )


    after_dual_core = (
        core_member_count(
            manager
        )
    )


    print()
    print(
        "POST-DUAL STATE"
    )

    print(
        "==============="
    )


    print(
        "GIDs:",
        after_dual_gids,
    )


    print(
        "Pending:",
        after_dual_pending,
    )


    print(
        "CORE members:",
        after_dual_core,
    )


    # ========================================================
    # VERIFY c1:556 TRUST
    # ========================================================

    gid_556 = find_gid(
        manager,
        1,
        556,
    )


    is_556_core = False


    if gid_556 is not None:

        is_556_core = is_core_member(
            manager,
            gid_556,
            1,
            556,
        )


    print()
    print(
        "c1:556 TRUST CHECK"
    )

    print(
        "=================="
    )


    print(
        "Assigned GID:",
        gid_556,
    )


    print(
        "Is CORE:",
        is_556_core,
    )


    print(
        "Expected:",
        "GID 63, RELAXED",
    )


    # ========================================================
    # CASCADE TEST
    #
    # IMPORTANT:
    #
    # include_same_camera=False
    #
    # so we are testing only whether adding c1:556 to the
    # gallery causes new cross-camera RELAXED/CORE resolutions.
    # ========================================================

    cascade_results = (
        manager.reevaluate_pending(
            tracklet_lookup=
                lookup,

            include_same_camera=
                False,
        )
    )


    cascade_resolutions = [
        result

        for result
        in cascade_results

        if (
            result.get(
                "reason"
            )
            == "RELAXED_CORE_PENDING_SWEEP"
        )
    ]


    print()
    print(
        "POST-DUAL CASCADE RESOLUTIONS"
    )

    print(
        "============================="
    )


    if not cascade_resolutions:

        print(
            "NONE"
        )


    else:

        for result in (
            cascade_resolutions
        ):

            camera_id = (
                normalize_camera_id(
                    result[
                        "camera_id"
                    ]
                )
            )


            print(
                f"c{camera_id}:"
                f"{result['local_track_id']}"
                f" -> GID "
                f"{result['global_id']}"
                f" | reason="
                f"{result['reason']}"
            )


    # ========================================================
    # CHAIN SAFETY
    # ========================================================

    print()
    print(
        "TRANSITIVE CHAIN SAFETY"
    )

    print(
        "======================="
    )


    print(
        "c1:556 GID:",
        find_gid(
            manager,
            1,
            556,
        ),
    )


    print(
        "c1:556 CORE:",
        (
            is_core_member(
                manager,
                find_gid(
                    manager,
                    1,
                    556,
                ),
                1,
                556,
            )
            if find_gid(
                manager,
                1,
                556,
            )
            is not None
            else False
        ),
    )


    print(
        "c0:622 pending:",
        is_pending(
            manager,
            0,
            622,
        ),
    )


    print(
        "c0:622 GID:",
        find_gid(
            manager,
            0,
            622,
        ),
    )


    print(
        "c1:549 pending:",
        is_pending(
            manager,
            1,
            549,
        ),
    )


    print(
        "c1:549 GID:",
        find_gid(
            manager,
            1,
            549,
        ),
    )


    # ========================================================
    # REGRESSION CHECKS
    # ========================================================

    regression_checks = (
        run_regression_checks(
            manager
        )
    )


    passed_count = sum(
        1

        for passed, _, _
        in regression_checks

        if passed
    )


    print()
    print(
        "REGRESSION CHECKS"
    )

    print(
        "================="
    )


    for (
        passed,
        name,
        detail,
    ) in regression_checks:

        status = (
            "PASS"
            if passed
            else "FAIL"
        )


        print(
            f"{status}"
            f" | "
            f"{name}"
            f" | "
            f"{detail}"
        )


    # ========================================================
    # TARGET IDENTITY CHECK
    # ========================================================

    target_members = [
        (0, 666),
        (0, 757),
        (1, 635),
        (1, 556),
    ]


    target_gid = group_gid(
        manager,
        target_members,
    )


    print()
    print(
        "GID 63 TARGET CHECK"
    )

    print(
        "==================="
    )


    print(
        "666 / 757 / 635 / 556 together:",
        target_gid is not None,
    )


    print(
        "Resolved GID:",
        target_gid,
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_gids = len(
        manager.identities
    )


    final_pending = len(
        manager.pending_tracklets
    )


    final_core = (
        core_member_count(
            manager
        )
    )


    print()
    print(
        "FINAL RESULT"
    )

    print(
        "============"
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
        "Baseline CORE:",
        baseline_core,
    )


    print(
        "Dual-evidence additions:",
        len(
            applied
        ),
    )


    print(
        "Cascade resolutions:",
        len(
            cascade_resolutions
        ),
    )


    print(
        "Final GIDs:",
        final_gids,
    )


    print(
        "Final pending:",
        final_pending,
    )


    print(
        "Final CORE:",
        final_core,
    )


    print(
        "Regression passed:",
        f"{passed_count}/"
        f"{len(regression_checks)}",
    )


    # ========================================================
    # EXPECTED SAFETY SIGNALS
    # ========================================================

    expected = (
        baseline_gids == 76

        and

        baseline_pending == 94

        and

        baseline_core == 68

        and

        len(
            unique
        ) == 1

        and

        len(
            ambiguous
        ) == 0

        and

        len(
            applied
        ) == 1

        and

        gid_556 == 63

        and

        not is_556_core

        and

        len(
            cascade_resolutions
        ) == 0

        and

        final_gids == 76

        and

        final_pending == 93

        and

        final_core == 68

        and

        is_pending(
            manager,
            0,
            622,
        )

        and

        is_pending(
            manager,
            1,
            549,
        )

        and

        passed_count
        == len(
            regression_checks
        )

        and

        target_gid == 63
    )


    print()
    print(
        "EXPERIMENT VERDICT"
    )

    print(
        "=================="
    )


    if expected:

        print(
            "PASS"
        )


        print(
            "Dual-evidence continuation "
            "resolved c1:556 safely."
        )


        print(
            "No CORE promotion occurred."
        )


        print(
            "No transitive cascade occurred."
        )


    else:

        print(
            "FAIL / REVIEW REQUIRED"
        )


        print(
            "One or more expected safety "
            "conditions did not hold."
        )


if __name__ == "__main__":
    main()
