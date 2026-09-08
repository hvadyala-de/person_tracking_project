from collections import Counter

from src.experiments.analyze_final_pending_population import (
    replay_final_production,
)

from src.experiments.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.experiments.analyze_c2_601_segment_gid_ranking import (
    rank_segment,
)

from src.experiments.analyze_c2_production_gate_failures import (
    build_population,
    register_population,
    snapshot_members,
    audit_snapshot,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.experiments.run_c2_multicore_consensus_experiment import (
    CONSENSUS_ACCEPT_MIN_JOINT,
    CONSENSUS_CORE_MAX_MIN,
    CONSENSUS_CORE_TOP3_MIN,
    finalize_records,
    process_population,
)


# ============================================================
# PURPOSE
#
# Negative/adversarial audit for the proposed multi-CORE
# consensus rule.
#
# Important correction:
#
# Some known high-ReID / bad-geometry c0/c1 tracklets are
# pending/unassigned and therefore do NOT have embeddings in
# manager.member_embeddings.
#
# That is completely valid.
#
# For the negative safety audit we only need to prove that the
# known bad cross-camera pair has CONTRADICTED geometry.
#
# CONTRADICTED geometry can never provide positive joint CORE
# support to the proposed consensus rule.
#
# If the other tracklet happens to be an assigned GID member,
# we additionally audit that GID's identity veto.
#
# Production source and thresholds are NOT modified.
# ============================================================


# ============================================================
# KNOWN HIGH-ReID / BAD-GEOMETRY PAIRS
#
# ReID values were already measured in the earlier c2 geometry
# diagnostic. We intentionally do NOT recompute them here.
# ============================================================

KNOWN_NEGATIVE_PAIRS = [
    {
        "name":
            "c1:313 <-> c2:195",

        "prior_reid":
            0.8873,

        "c2_track_id":
            195,

        "other_camera":
            1,

        "other_track_id":
            313,
    },

    {
        "name":
            "c1:793 <-> c2:555",

        "prior_reid":
            0.8602,

        "c2_track_id":
            555,

        "other_camera":
            1,

        "other_track_id":
            793,
    },

    {
        "name":
            "c1:812 <-> c2:555",

        "prior_reid":
            0.8554,

        "c2_track_id":
            555,

        "other_camera":
            1,

        "other_track_id":
            812,
    },

    {
        "name":
            "c1:792 <-> c2:555",

        "prior_reid":
            0.8517,

        "c2_track_id":
            555,

        "other_camera":
            1,

        "other_track_id":
            792,
    },

    {
        "name":
            "c1:434 <-> c2:321",

        "prior_reid":
            0.8498,

        "c2_track_id":
            321,

        "other_camera":
            1,

        "other_track_id":
            434,
    },

    {
        "name":
            "c0:420 <-> c2:419",

        "prior_reid":
            0.8420,

        "c2_track_id":
            419,

        "other_camera":
            0,

        "other_track_id":
            420,
    },
]


EXPECTED_EXPERIMENTAL_ACCEPTS = {
    "c2:9a",
    "c2:508",
    "c2:601b",
}


EXPECTED_TWO_SUPPORT_HOLDS = {
    "c2:1",
    "c2:53",
    "c2:371",
    "c2:513",
}


# ============================================================
# HELPERS
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


def find_member_gid(
    manager,
    camera_id,
    local_track_id,
):

    target = (
        normalize_camera_id(
            camera_id
        ),
        local_track_id,
    )


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            if member_key(
                member
            ) == target:

                return gid


    return None


def status_text(
    value,
):

    if value is None:

        return "None"


    if hasattr(
        value,
        "value",
    ):

        return str(
            value.value
        )


    return str(
        value
    )


def fmt(
    value,
    digits=4,
):

    if value is None:

        return "None"


    return (
        f"{value:.{digits}f}"
    )


# ============================================================
# c2 LOOKUP
# ============================================================

def build_c2_map(
    c2_rows,
):

    return {
        tracklet.local_track_id:
            (
                tracklet,
                embedding,
            )

        for tracklet, embedding
        in c2_rows
    }


# ============================================================
# FIND ONE GID RANK ROW
# ============================================================

def find_gid_rank_row(
    manager,
    tracklet,
    embedding,
    lookup,
    gid,
):

    rows = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            tracklet,

        candidate_embedding=
            embedding,

        tracklet_lookup=
            lookup,
    )


    for row in rows:

        if row[
            "gid"
        ] == gid:

            return row


    return None


# ============================================================
# WOULD A SPECIFIC ASSIGNED GID PASS MULTI-CORE?
#
# This is only meaningful when the "other" negative tracklet
# already belongs to an existing GID.
# ============================================================

def target_gid_would_pass_consensus(
    manager,
    candidate_tracklet,
    gid,
    row,
):

    if row is None:

        return False


    camera_id = (
        normalize_camera_id(
            candidate_tracklet.camera_id
        )
    )


    identity = manager.identities[
        gid
    ]


    same_camera_member = any(
        normalize_camera_id(
            member.camera_id
        )
        == camera_id

        for member
        in identity.members
    )


    return (
        gid in manager.anchored_gids

        and

        not bool(
            row[
                "identity_veto"
            ]
        )

        and

        not same_camera_member

        and

        int(
            row[
                "joint_support_count"
            ]
        )
        >= CONSENSUS_ACCEPT_MIN_JOINT

        and

        float(
            row[
                "core_max"
            ]
        )
        >= CONSENSUS_CORE_MAX_MIN

        and

        float(
            row[
                "core_top3"
            ]
        )
        >= CONSENSUS_CORE_TOP3_MIN
    )


# ============================================================
# KNOWN NEGATIVE PAIR AUDIT
# ============================================================

def audit_known_negative_pairs(
    manager,
    c2_rows,
    lookup,
):

    c2_map = build_c2_map(
        c2_rows
    )


    results = []


    print()
    print(
        "KNOWN HIGH-ReID / BAD-GEOMETRY NEGATIVE PAIRS"
    )

    print(
        "============================================="
    )


    for spec in KNOWN_NEGATIVE_PAIRS:

        (
            c2_tracklet,
            c2_embedding,
        ) = c2_map[
            spec[
                "c2_track_id"
            ]
        ]


        other_key = (
            normalize_camera_id(
                spec[
                    "other_camera"
                ]
            ),

            spec[
                "other_track_id"
            ],
        )


        other_tracklet = (
            lookup.get(
                other_key
            )
        )


        if other_tracklet is None:

            raise RuntimeError(
                "Missing tracklet for "
                f"{spec['name']}"
            )


        # ----------------------------------------------------
        # Pair geometry.
        #
        # This is the critical negative-safety evidence.
        # ----------------------------------------------------

        geometry = (
            evaluate_cross_camera_geometry(
                c2_tracklet,
                other_tracklet,
                manager.homographies,
            )
        )


        geometry_status = (
            status_text(
                geometry.status
            )
        )


        pair_can_be_positive_support = (
            geometry_status
            == "SUPPORTED"
        )


        # ----------------------------------------------------
        # The other tracklet may or may not belong to an
        # assigned GID.
        # ----------------------------------------------------

        gid = find_member_gid(
            manager=
                manager,

            camera_id=
                spec[
                    "other_camera"
                ],

            local_track_id=
                spec[
                    "other_track_id"
                ],
        )


        identity_veto = None
        rank_row = None
        would_accept_gid = False


        if gid is not None:

            identity = manager.identities[
                gid
            ]


            identity_veto = (
                manager.identity_geometry_contradicted(
                    c2_tracklet,
                    identity,
                    lookup,
                )
            )


            rank_row = find_gid_rank_row(
                manager=
                    manager,

                tracklet=
                    c2_tracklet,

                embedding=
                    c2_embedding,

                lookup=
                    lookup,

                gid=
                    gid,
            )


            would_accept_gid = (
                target_gid_would_pass_consensus(
                    manager=
                        manager,

                    candidate_tracklet=
                        c2_tracklet,

                    gid=
                        gid,

                    row=
                        rank_row,
                )
            )


        result = {
            "name":
                spec[
                    "name"
                ],

            "prior_reid":
                spec[
                    "prior_reid"
                ],

            "geometry_status":
                geometry_status,

            "shared":
                geometry.shared_frames,

            "median":
                geometry.median_distance,

            "pair_can_be_positive_support":
                pair_can_be_positive_support,

            "gid":
                gid,

            "identity_veto":
                identity_veto,

            "rank_row":
                rank_row,

            "would_accept_gid":
                would_accept_gid,
        }


        results.append(
            result
        )


        print()
        print(
            spec[
                "name"
            ]
        )


        print(
            "  prior ReID:",
            fmt(
                spec[
                    "prior_reid"
                ]
            ),
        )


        print(
            "  geometry:",
            geometry_status,
            "| shared=",
            geometry.shared_frames,
            "| median=",
            fmt(
                geometry.median_distance,
                2,
            ),
        )


        print(
            "  can contribute positive geometry support:",
            pair_can_be_positive_support,
        )


        if gid is None:

            print(
                "  existing GID for other track:",
                "NONE / pending or unassigned",
            )

            print(
                "  identity-level consensus audit:",
                "N/A",
            )

        else:

            print(
                "  existing GID for other track:",
                gid,
            )


            print(
                "  identity veto:",
                identity_veto,
            )


            if rank_row is None:

                print(
                    "  GID rank row:",
                    "NONE",
                )

            else:

                print(
                    "  GID row:"
                    f" joint="
                    f"{rank_row['joint_support_count']}"
                    f" | max="
                    f"{fmt(rank_row['core_max'])}"
                    f" | top3="
                    f"{fmt(rank_row['core_top3'])}"
                    f" | veto="
                    f"{rank_row['identity_veto']}"
                )


            print(
                "  target GID passes multi-CORE:",
                (
                    "YES - REVIEW"
                    if would_accept_gid
                    else "NO - SAFE"
                ),
            )


    return results


# ============================================================
# SEQUENTIAL CONSENSUS OUTCOME COUNTS
# ============================================================

def print_consensus_outcomes(
    records,
):

    counts = Counter()


    for record in records:

        consensus = record.get(
            "consensus"
        )


        if consensus is None:

            counts[
                "PRODUCTION_PRIORITY"
            ] += 1

        else:

            counts[
                consensus[
                    "reason"
                ]
            ] += 1


    print()
    print(
        "SEQUENTIAL CONSENSUS OUTCOME COUNTS"
    )

    print(
        "==================================="
    )


    for reason, count in sorted(
        counts.items()
    ):

        print(
            f"{reason:<45}"
            f": {count}"
        )


# ============================================================
# SAME-CAMERA SAFETY
# ============================================================

def print_same_camera_safety(
    records,
):

    blocked = []


    for record in records:

        consensus = record.get(
            "consensus"
        )


        if consensus is None:

            continue


        if (
            consensus[
                "reason"
            ]
            == "MULTICORE_BLOCKED_SAME_CAMERA_MEMBER"
        ):

            blocked.append(
                record
            )


    print()
    print(
        "SAME-CAMERA CONSENSUS SAFETY"
    )

    print(
        "============================"
    )


    if not blocked:

        print(
            "No explicit same-camera block occurred "
            "in this c2 population."
        )

        print(
            "Accepted rows are still checked below "
            "for same-camera-member violations."
        )

    else:

        for record in blocked:

            consensus = record[
                "consensus"
            ]


            print(
                f"{record['label']:<10}"
                f" | GID="
                f"{consensus['gid']}"
                f" | joint="
                f"{consensus['joint_supports']}"
                f" | blocked=YES"
            )


    return blocked


# ============================================================
# ACCEPTANCE INVARIANT AUDIT
# ============================================================

def audit_acceptance_invariants(
    records,
):

    violations = []


    for record in records:

        if (
            record[
                "decision"
            ]
            != "MULTICORE_RELAXED"
        ):

            continue


        consensus = record.get(
            "consensus"
        )


        if consensus is None:

            violations.append(
                (
                    record[
                        "label"
                    ],
                    "missing consensus",
                )
            )

            continue


        if consensus[
            "identity_veto"
        ]:

            violations.append(
                (
                    record[
                        "label"
                    ],
                    "accepted despite identity veto",
                )
            )


        if consensus[
            "same_camera_member"
        ]:

            violations.append(
                (
                    record[
                        "label"
                    ],
                    "accepted despite same-camera member",
                )
            )


        if (
            consensus[
                "joint_supports"
            ]
            < CONSENSUS_ACCEPT_MIN_JOINT
        ):

            violations.append(
                (
                    record[
                        "label"
                    ],
                    "joint support below threshold",
                )
            )


        if (
            consensus[
                "core_max"
            ]
            < CONSENSUS_CORE_MAX_MIN
        ):

            violations.append(
                (
                    record[
                        "label"
                    ],
                    "CORE max below threshold",
                )
            )


        if (
            consensus[
                "core_top3"
            ]
            < CONSENSUS_CORE_TOP3_MIN
        ):

            violations.append(
                (
                    record[
                        "label"
                    ],
                    "CORE top3 below threshold",
                )
            )


    print()
    print(
        "ACCEPTANCE INVARIANT AUDIT"
    )

    print(
        "=========================="
    )


    if not violations:

        print(
            "PASS - no acceptance invariant violations"
        )

    else:

        for label, reason in violations:

            print(
                "FAIL",
                "|",
                label,
                "|",
                reason,
            )


    return violations


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2 MULTI-CORE CONSENSUS NEGATIVE AUDIT"
    )

    print(
        "======================================"
    )


    # ========================================================
    # VALIDATED c0+c1 BASELINE
    # ========================================================

    (
        manager,
        baseline_tracklets,
        lookup,
        _,
    ) = replay_final_production()


    baseline_gid_ids = set(
        manager.identities.keys()
    )


    baseline_snapshot = (
        snapshot_members(
            manager
        )
    )


    print()
    print(
        "VALIDATED c0+c1 START STATE"
    )

    print(
        "==========================="
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
    # LOAD c2
    # ========================================================

    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()


    lookup.update(
        c2_lookup
    )


    # ========================================================
    # KNOWN NEGATIVE PAIRS
    #
    # This happens BEFORE the manager is changed by c2.
    # ========================================================

    negative_results = (
        audit_known_negative_pairs(
            manager=
                manager,

            c2_rows=
                c2_rows,

            lookup=
                lookup,
        )
    )


    # ========================================================
    # BUILD SAME 107-ITEM IDENTITY-AWARE c2 POPULATION
    # ========================================================

    print()
    print(
        "BUILDING IDENTITY-AWARE c2 POPULATION..."
    )


    population = build_population(
        c2_rows
    )


    register_population(
        population,
        lookup,
    )


    print(
        "Population:",
        len(
            population
        ),
    )


    if len(
        population
    ) != 107:

        raise RuntimeError(
            "Expected 107 c2 segments"
        )


    # ========================================================
    # RUN EXACT SAME MULTI-CORE EXPERIMENTAL POLICY
    # ========================================================

    print()
    print(
        "RUNNING SEQUENTIAL MULTI-CORE POLICY..."
    )


    records = process_population(
        manager=
            manager,

        population=
            population,

        lookup=
            lookup,
    )


    # ========================================================
    # EXISTING FINAL CONTINUATION PASS
    # ========================================================

    manager.reevaluate_pending(
        tracklet_lookup=
            lookup,

        include_same_camera=
            True,

        include_dual_evidence=
            True,

        include_core_gallery=
            True,
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
    # OUTCOMES
    # ========================================================

    print_consensus_outcomes(
        records
    )


    print_same_camera_safety(
        records
    )


    accepted = {
        record[
            "label"
        ]

        for record in records

        if (
            record[
                "decision"
            ]
            == "MULTICORE_RELAXED"
        )
    }


    pending_holds = {
        record[
            "label"
        ]

        for record in records

        if (
            record[
                "decision"
            ]
            == "MULTICORE_PENDING"
        )
    }


    print()
    print(
        "EXPERIMENTAL ACCEPTED SET"
    )

    print(
        "========================="
    )


    for label in sorted(
        accepted
    ):

        print(
            label
        )


    print()
    print(
        "EXPERIMENTAL TWO-SUPPORT HOLDS"
    )

    print(
        "=============================="
    )


    for label in sorted(
        pending_holds
    ):

        print(
            label
        )


    violations = (
        audit_acceptance_invariants(
            records
        )
    )


    # ========================================================
    # EXISTING c0+c1 REGRESSION
    # ========================================================

    changes = audit_snapshot(
        manager=
            manager,

        snapshot=
            baseline_snapshot,
    )


    print()
    print(
        "EXISTING c0+c1 REGRESSION AUDIT"
    )

    print(
        "================================"
    )


    if not changes:

        print(
            "PASS - existing c0+c1 "
            "members unchanged"
        )

    else:

        print(
            "FAIL - changed members:",
            len(
                changes
            ),
        )


        for change in changes:

            print(
                change
            )


    # ========================================================
    # SANITY
    # ========================================================

    all_negative_contradicted = all(
        row[
            "geometry_status"
        ]
        == "CONTRADICTED"

        for row
        in negative_results
    )


    no_negative_positive_support = all(
        not row[
            "pair_can_be_positive_support"
        ]

        for row
        in negative_results
    )


    no_assigned_negative_gid_accepted = all(
        not row[
            "would_accept_gid"
        ]

        for row
        in negative_results

        if row[
            "gid"
        ]
        is not None
    )


    record_map = {
        record[
            "label"
        ]:
            record

        for record
        in records
    }


    checks = [
        (
            "all known negative pairs are CONTRADICTED",
            all_negative_contradicted,
        ),

        (
            "no known negative pair can supply positive geometry support",
            no_negative_positive_support,
        ),

        (
            "no assigned negative-pair GID passes multi-CORE",
            no_assigned_negative_gid_accepted,
        ),

        (
            "accepted set is exactly c2:9a, c2:508, c2:601b",
            accepted
            == EXPECTED_EXPERIMENTAL_ACCEPTS,
        ),

        (
            "two-support hold set is exactly expected",
            pending_holds
            == EXPECTED_TWO_SUPPORT_HOLDS,
        ),

        (
            "c2:1 remains pending",
            (
                record_map[
                    "c2:1"
                ][
                    "final_gid"
                ]
                is None

                and

                record_map[
                    "c2:1"
                ][
                    "pending"
                ]
            ),
        ),

        (
            "c2:635 remains unresolved/pending",
            (
                record_map[
                    "c2:635"
                ][
                    "final_gid"
                ]
                is None

                and

                record_map[
                    "c2:635"
                ][
                    "pending"
                ]
            ),
        ),

        (
            "no acceptance invariant violations",
            len(
                violations
            )
            == 0,
        ),

        (
            "existing c0+c1 members unchanged",
            len(
                changes
            )
            == 0,
        ),
    ]


    print()
    print(
        "NEGATIVE AUDIT SANITY"
    )

    print(
        "====================="
    )


    passed = 0


    for description, ok in checks:

        if ok:

            passed += 1


        print(
            "PASS"
            if ok
            else "FAIL",
            "|",
            description,
        )


    print()
    print(
        f"{passed}/"
        f"{len(checks)} checks passed"
    )


    print()
    print(
        "PRODUCTION THRESHOLDS MODIFIED: NO"
    )

    print(
        "global_identity_manager.py MODIFIED: NO"
    )

    print(
        "TRACK CSV MODIFIED: NO"
    )

    print(
        "EMBEDDING FILES MODIFIED: NO"
    )

    print(
        "SPLITS PERSISTED: NO"
    )

    print(
        "Negative audit only."
    )


if __name__ == "__main__":
    main()
