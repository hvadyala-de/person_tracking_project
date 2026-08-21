from collections import Counter

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.analyze_c2_601_segment_gid_ranking import (
    rank_segment,
)

from src.analyze_c2_gap_identity_changes import (
    summarize_segment,
)

from src.analyze_c2_production_gate_failures import (
    build_population,
    register_population,
    snapshot_members,
    audit_snapshot,
    find_member_gid,
)

from src.identity_compatibility import (
    normalize_camera_id,
)


# ============================================================
# PURPOSE
#
# Test a conservative multi-CORE consensus rule for c2.
#
# IMPORTANT:
#
# - Existing production manager source is NOT changed.
# - Existing production thresholds are NOT changed.
# - Existing STRICT / RELAXED / pending-anchor rules retain
#   priority.
#
# Experimental behavior is considered ONLY when the existing
# trusted production routes cannot assign the tracklet.
#
#
# EXPERIMENTAL STRONG CONSENSUS:
#
#   anchored existing GID
#   diagnostic UNIQUE candidate
#   no identity veto
#   >= 3 joint CORE supports
#   CORE gallery max >= .82
#   CORE gallery top3 >= .80
#   no previous member from this same camera in that GID
#
#   -> add as RELAXED
#
#
# EXPERIMENTAL TWO-SUPPORT HOLD:
#
#   diagnostic UNIQUE candidate
#   >= 2 joint CORE supports
#
#   -> HOLD PENDING instead of creating a new singleton
#
# This second rule is specifically intended to prevent cases
# like:
#
#   c2:1 -> NEW GID 77
#
# when multiple existing CORE members already provide useful
# cross-camera evidence.
#
# Nothing is persisted.
# ============================================================


# ============================================================
# REFERENCE RESULT FROM UNCHANGED PRODUCTION POLICY
# ============================================================

REFERENCE_FINAL_GIDS = 108
REFERENCE_FINAL_PENDING = 166

REFERENCE_C2_EXISTING = 1
REFERENCE_C2_NEW = 32
REFERENCE_C2_PENDING = 74


# ============================================================
# EXPERIMENTAL CONSENSUS THRESHOLDS
#
# These do NOT replace production thresholds.
# ============================================================

CONSENSUS_ACCEPT_MIN_JOINT = 3

CONSENSUS_HOLD_MIN_JOINT = 2

CONSENSUS_CORE_MAX_MIN = 0.82

CONSENSUS_CORE_TOP3_MIN = 0.80


# ============================================================
# HELPERS
# ============================================================

def tracklet_key(
    tracklet,
):

    return (
        normalize_camera_id(
            tracklet.camera_id
        ),
        tracklet.local_track_id,
    )


def gid_has_camera_member(
    manager,
    gid,
    camera_id,
):

    identity = manager.identities[
        gid
    ]


    for member in identity.members:

        member_camera = (
            normalize_camera_id(
                member.camera_id
            )
        )

        if member_camera == camera_id:

            return True


    return False


# ============================================================
# EXISTING PRODUCTION TRUSTED ROUTES
#
# Before applying the experimental rule, check whether the
# production manager already has exactly one trusted:
#
#   STRICT existing candidate
#   RELAXED existing candidate
#   STRICT current <-> pending candidate
#
# If yes, production keeps priority.
# ============================================================

def production_trusted_candidate_exists(
    manager,
    tracklet,
    embedding,
    lookup,
):

    strict_candidates = (
        manager.find_strict_existing_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )
    )


    relaxed_candidates = (
        manager.find_relaxed_core_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )
    )


    camera_id = normalize_camera_id(
        tracklet.camera_id
    )


    pending_candidates = (
        manager.find_strict_pending_candidates(
            camera_id=
                camera_id,

            local_track_id=
                tracklet.local_track_id,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )
    )


    return {
        "strict":
            strict_candidates,

        "relaxed":
            relaxed_candidates,

        "pending":
            pending_candidates,

        "has_unique_trusted_route":
            (
                len(strict_candidates) == 1
                or
                len(relaxed_candidates) == 1
                or
                len(pending_candidates) == 1
            ),
    }


# ============================================================
# MULTI-CORE CONSENSUS ANALYSIS
# ============================================================

def analyze_consensus(
    manager,
    tracklet,
    embedding,
    lookup,
):

    ranked = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            tracklet,

        candidate_embedding=
            embedding,

        tracklet_lookup=
            lookup,
    )


    summary = summarize_segment(
        ranked
    )


    if summary[
        "status"
    ] != "UNIQUE":

        return {
            "status":
                summary[
                    "status"
                ],

            "gid":
                None,

            "row":
                None,

            "accept":
                False,

            "hold":
                False,

            "reason":
                (
                    "NO_UNIQUE_MULTICORE_CANDIDATE"
                ),
        }


    gid = summary[
        "gid"
    ]

    row = summary[
        "row"
    ]


    camera_id = (
        normalize_camera_id(
            tracklet.camera_id
        )
    )


    anchored = (
        gid in manager.anchored_gids
    )


    identity_veto = bool(
        row[
            "identity_veto"
        ]
    )


    joint_supports = int(
        row[
            "joint_support_count"
        ]
    )


    core_max = float(
        row[
            "core_max"
        ]
    )


    core_top3 = float(
        row[
            "core_top3"
        ]
    )


    same_camera_member = (
        gid_has_camera_member(
            manager=
                manager,

            gid=
                gid,

            camera_id=
                camera_id,
        )
    )


    accept = (
        anchored

        and

        not identity_veto

        and

        not same_camera_member

        and

        joint_supports
        >= CONSENSUS_ACCEPT_MIN_JOINT

        and

        core_max
        >= CONSENSUS_CORE_MAX_MIN

        and

        core_top3
        >= CONSENSUS_CORE_TOP3_MIN
    )


    hold = (
        anchored

        and

        not identity_veto

        and

        not same_camera_member

        and

        not accept

        and

        joint_supports
        >= CONSENSUS_HOLD_MIN_JOINT

        and

        core_max
        >= CONSENSUS_CORE_MAX_MIN

        and

        core_top3
        >= CONSENSUS_CORE_TOP3_MIN
    )


    if accept:

        reason = (
            "MULTICORE_CONSENSUS_RELAXED"
        )

    elif hold:

        reason = (
            "MULTICORE_CONSENSUS_PENDING_HOLD"
        )

    elif same_camera_member:

        reason = (
            "MULTICORE_BLOCKED_SAME_CAMERA_MEMBER"
        )

    elif identity_veto:

        reason = (
            "MULTICORE_BLOCKED_IDENTITY_VETO"
        )

    else:

        reason = (
            "MULTICORE_INSUFFICIENT_EVIDENCE"
        )


    return {
        "status":
            "UNIQUE",

        "gid":
            gid,

        "row":
            row,

        "anchored":
            anchored,

        "identity_veto":
            identity_veto,

        "same_camera_member":
            same_camera_member,

        "joint_supports":
            joint_supports,

        "core_max":
            core_max,

        "core_top3":
            core_top3,

        "accept":
            accept,

        "hold":
            hold,

        "reason":
            reason,
    }


# ============================================================
# EXPERIMENTAL RELAXED ASSIGNMENT
# ============================================================

def add_multicore_relaxed(
    manager,
    tracklet,
    embedding,
    gid,
):

    camera_id = (
        normalize_camera_id(
            tracklet.camera_id
        )
    )


    manager.add_to_identity(
        global_id=
            gid,

        camera_id=
            camera_id,

        local_track_id=
            tracklet.local_track_id,

        start_frame=
            tracklet.start_frame,

        end_frame=
            tracklet.end_frame,

        embedding=
            embedding,

        trust_level=
            "RELAXED",
    )


    current_key = (
        camera_id,
        tracklet.local_track_id,
    )


    manager.pending_tracklets.pop(
        current_key,
        None,
    )


# ============================================================
# EXPERIMENTAL PENDING HOLD
# ============================================================

def add_multicore_pending(
    manager,
    tracklet,
    embedding,
):

    camera_id = (
        normalize_camera_id(
            tracklet.camera_id
        )
    )


    manager.add_pending(
        camera_id=
            camera_id,

        local_track_id=
            tracklet.local_track_id,

        start_frame=
            tracklet.start_frame,

        end_frame=
            tracklet.end_frame,

        embedding=
            embedding,

        candidate_tracklet=
            tracklet,
    )


# ============================================================
# PROCESS POPULATION
# ============================================================

def process_population(
    manager,
    population,
    lookup,
):

    records = []


    for index, item in enumerate(
        population,
        start=1,
    ):

        label = item[
            "label"
        ]

        tracklet = item[
            "tracklet"
        ]

        embedding = item[
            "embedding"
        ]


        camera_id = (
            normalize_camera_id(
                tracklet.camera_id
            )
        )


        key = (
            camera_id,
            tracklet.local_track_id,
        )


        trusted = (
            production_trusted_candidate_exists(
                manager=
                    manager,

                tracklet=
                    tracklet,

                embedding=
                    embedding,

                lookup=
                    lookup,
            )
        )


        consensus = None


        # ====================================================
        # PRODUCTION TRUSTED ROUTES RETAIN PRIORITY
        # ====================================================

        if trusted[
            "has_unique_trusted_route"
        ]:

            result = (
                manager.assign_tracklet(
                    camera_id=
                        camera_id,

                    local_track_id=
                        tracklet.local_track_id,

                    start_frame=
                        tracklet.start_frame,

                    end_frame=
                        tracklet.end_frame,

                    embedding=
                        embedding,

                    candidate_tracklet=
                        tracklet,

                    tracklet_lookup=
                        lookup,
                )
            )


            decision = (
                "PRODUCTION"
            )

            status = (
                result.status
            )

            reason = (
                result.reason
            )

            gid = (
                result.global_id
            )

            merged = bool(
                result.merged
            )

            created_new = bool(
                result.created_new
            )


        else:

            consensus = (
                analyze_consensus(
                    manager=
                        manager,

                    tracklet=
                        tracklet,

                    embedding=
                        embedding,

                    lookup=
                        lookup,
                )
            )


            # =================================================
            # EXPERIMENTAL >= 3 CORE CONSENSUS
            # =================================================

            if consensus[
                "accept"
            ]:

                gid = (
                    consensus[
                        "gid"
                    ]
                )


                add_multicore_relaxed(
                    manager=
                        manager,

                    tracklet=
                        tracklet,

                    embedding=
                        embedding,

                    gid=
                        gid,
                )


                decision = (
                    "MULTICORE_RELAXED"
                )

                status = (
                    "STRONG"
                )

                reason = (
                    "MULTICORE_CONSENSUS_RELAXED"
                )

                merged = True

                created_new = False


            # =================================================
            # EXPERIMENTAL 2-CORE HOLD
            #
            # Do NOT create a new singleton when two trusted
            # CORE supports already point toward an existing
            # identity.
            # =================================================

            elif consensus[
                "hold"
            ]:

                gid = (
                    consensus[
                        "gid"
                    ]
                )


                add_multicore_pending(
                    manager=
                        manager,

                    tracklet=
                        tracklet,

                    embedding=
                        embedding,
                )


                decision = (
                    "MULTICORE_PENDING"
                )

                status = (
                    "PENDING"
                )

                reason = (
                    "MULTICORE_CONSENSUS_PENDING_HOLD"
                )

                merged = False

                created_new = False


            # =================================================
            # OTHERWISE USE NORMAL PRODUCTION
            # =================================================

            else:

                result = (
                    manager.assign_tracklet(
                        camera_id=
                            camera_id,

                        local_track_id=
                            tracklet.local_track_id,

                        start_frame=
                            tracklet.start_frame,

                        end_frame=
                            tracklet.end_frame,

                        embedding=
                            embedding,

                        candidate_tracklet=
                            tracklet,

                        tracklet_lookup=
                            lookup,
                    )
                )


                decision = (
                    "PRODUCTION"
                )

                status = (
                    result.status
                )

                reason = (
                    result.reason
                )

                gid = (
                    result.global_id
                )

                merged = bool(
                    result.merged
                )

                created_new = bool(
                    result.created_new
                )


        records.append(
            {
                "label":
                    label,

                "key":
                    key,

                "decision":
                    decision,

                "status":
                    status,

                "reason":
                    reason,

                "initial_gid":
                    gid,

                "consensus":
                    consensus,
            }
        )


        # ====================================================
        # SAME ORDINARY REEVALUATION RULE USED BY PRODUCTION
        # ====================================================

        if (
            merged
            or
            created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup,

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
            or
            index
            == len(
                population
            )
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{len(population)}"
                f" | GIDs="
                f"{len(manager.identities)}"
                f" | pending="
                f"{len(manager.pending_tracklets)}"
            )


    return records


# ============================================================
# FINALIZE RECORDS
# ============================================================

def finalize_records(
    manager,
    records,
    baseline_gid_ids,
):

    for record in records:

        key = record[
            "key"
        ]


        gid = find_member_gid(
            manager,
            key,
        )


        pending = (
            key
            in manager.pending_tracklets
        )


        trust = None


        if gid is not None:

            trust = (
                manager.get_member_trust(
                    gid,
                    key[0],
                    key[1],
                )
            )


        record[
            "final_gid"
        ] = gid

        record[
            "final_trust"
        ] = trust

        record[
            "pending"
        ] = pending


        if gid is None:

            record[
                "gid_origin"
            ] = None

        elif gid in baseline_gid_ids:

            record[
                "gid_origin"
            ] = "EXISTING"

        else:

            record[
                "gid_origin"
            ] = "NEW"


# ============================================================
# PRINT EXPERIMENTAL DECISIONS
# ============================================================

def print_experimental_decisions(
    records,
):

    experimental = [
        record

        for record in records

        if record[
            "decision"
        ]
        != "PRODUCTION"
    ]


    print()
    print(
        "EXPERIMENTAL MULTI-CORE DECISIONS"
    )

    print(
        "================================="
    )


    if not experimental:

        print(
            "NONE"
        )

        return


    for record in experimental:

        consensus = record[
            "consensus"
        ]


        print(
            f"{record['label']:<10}"
            f" | "
            f"{record['decision']:<18}"
            f" | GID "
            f"{consensus['gid']:<3}"
            f" | joint="
            f"{consensus['joint_supports']}"
            f" | max="
            f"{consensus['core_max']:.4f}"
            f" | top3="
            f"{consensus['core_top3']:.4f}"
            f" | veto="
            f"{consensus['identity_veto']}"
        )


# ============================================================
# PRINT TARGET STATES
# ============================================================

def print_key_states(
    records,
):

    targets = [
        "c2:1",
        "c2:9a",
        "c2:9b",
        "c2:53",
        "c2:153",
        "c2:371",
        "c2:513",
        "c2:508",
        "c2:601a",
        "c2:601b",
        "c2:635",
        "c2:701",
    ]


    record_map = {
        record[
            "label"
        ]:
            record

        for record
        in records
    }


    print()
    print(
        "KEY c2 FINAL STATES"
    )

    print(
        "==================="
    )


    for label in targets:

        record = (
            record_map.get(
                label
            )
        )


        if record is None:

            print(
                f"{label:<10}"
                " | NOT FOUND"
            )

            continue


        print(
            f"{label:<10}"
            f" | decision="
            f"{record['decision']:<18}"
            f" | final_gid="
            f"{str(record['final_gid']):<4}"
            f" | trust="
            f"{str(record['final_trust']):<10}"
            f" | pending="
            f"{record['pending']}"
            f" | reason="
            f"{record['reason']}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    manager,
    records,
    baseline_gids,
    baseline_pending,
):

    existing_count = sum(
        1

        for record in records

        if record[
            "gid_origin"
        ]
        == "EXISTING"
    )


    new_count = sum(
        1

        for record in records

        if record[
            "gid_origin"
        ]
        == "NEW"
    )


    pending_count = sum(
        1

        for record in records

        if (
            record[
                "final_gid"
            ]
            is None

            and

            record[
                "pending"
            ]
        )
    )


    multicore_relaxed = sum(
        1

        for record in records

        if record[
            "decision"
        ]
        == "MULTICORE_RELAXED"
    )


    multicore_pending = sum(
        1

        for record in records

        if record[
            "decision"
        ]
        == "MULTICORE_PENDING"
    )


    print()
    print(
        "UNCHANGED PRODUCTION vs MULTI-CORE EXPERIMENT"
    )

    print(
        "============================================="
    )


    print(
        f"{'Metric':<34}"
        f"{'Production':>12}"
        f"{'Experiment':>12}"
        f"{'Delta':>10}"
    )

    print(
        "-" * 68
    )


    rows = [
        (
            "Final GIDs",
            REFERENCE_FINAL_GIDS,
            len(
                manager.identities
            ),
        ),

        (
            "Final global pending",
            REFERENCE_FINAL_PENDING,
            len(
                manager.pending_tracklets
            ),
        ),

        (
            "c2 -> existing GID",
            REFERENCE_C2_EXISTING,
            existing_count,
        ),

        (
            "c2 -> new GID",
            REFERENCE_C2_NEW,
            new_count,
        ),

        (
            "c2 pending",
            REFERENCE_C2_PENDING,
            pending_count,
        ),
    ]


    for (
        name,
        old,
        new,
    ) in rows:

        print(
            f"{name:<34}"
            f"{old:>12}"
            f"{new:>12}"
            f"{new - old:>+10}"
        )


    print()
    print(
        "Experimental MULTICORE_RELAXED:",
        multicore_relaxed,
    )

    print(
        "Experimental MULTICORE_PENDING:",
        multicore_pending,
    )


    print()
    print(
        "c0+c1 starting GIDs:",
        baseline_gids,
    )

    print(
        "c0+c1 starting pending:",
        baseline_pending,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2 MULTI-CORE CONSENSUS EXPERIMENT"
    )

    print(
        "=================================="
    )


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


    baseline_gids = len(
        manager.identities
    )


    baseline_pending = len(
        manager.pending_tracklets
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
        baseline_gids,
    )

    print(
        "Pending:",
        baseline_pending,
    )

    print(
        "Anchored:",
        len(
            manager.anchored_gids
        ),
    )


    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()


    lookup.update(
        c2_lookup
    )


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
        "c2 input segments:",
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


    print()
    print(
        "EXPERIMENTAL RULE"
    )

    print(
        "================="
    )

    print(
        "Accept RELAXED:"
        f" joint >= "
        f"{CONSENSUS_ACCEPT_MIN_JOINT}"
        f", core max >= "
        f"{CONSENSUS_CORE_MAX_MIN:.2f}"
        f", top3 >= "
        f"{CONSENSUS_CORE_TOP3_MIN:.2f}"
    )

    print(
        "Pending hold:"
        f" joint >= "
        f"{CONSENSUS_HOLD_MIN_JOINT}"
    )

    print(
        "Positive support propagation:"
        " CORE only"
    )

    print(
        "Experimental RELAXED members:"
        " do not become CORE"
    )


    print()
    print(
        "PROCESSING c2..."
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
    # EXACT EXISTING FINAL CONTINUATION PASS
    # ========================================================

    print()
    print(
        "RUNNING EXISTING FINAL CONTINUATION PASS..."
    )


    final_results = (
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
    )


    final_reasons = Counter(
        str(
            row.get(
                "reason",
                "UNKNOWN",
            )
        )

        for row in final_results
    )


    print()
    print(
        "FINAL CONTINUATION REASONS"
    )

    print(
        "=========================="
    )


    for reason, count in sorted(
        final_reasons.items()
    ):

        print(
            f"{reason:<42}: "
            f"{count}"
        )


    finalize_records(
        manager=
            manager,

        records=
            records,

        baseline_gid_ids=
            baseline_gid_ids,
    )


    print_experimental_decisions(
        records
    )


    print_key_states(
        records
    )


    print_summary(
        manager=
            manager,

        records=
            records,

        baseline_gids=
            baseline_gids,

        baseline_pending=
            baseline_pending,
    )


    # ========================================================
    # c0+c1 REGRESSION
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


        for (
            key,
            old_value,
            new_value,
        ) in changes:

            print(
                key,
                old_value,
                "->",
                new_value,
            )


    # ========================================================
    # SANITY
    # ========================================================

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
            "c2 population = 107",
            len(
                population
            )
            == 107,
        ),

        (
            "c2:9a -> existing GID 1",
            record_map[
                "c2:9a"
            ][
                "final_gid"
            ]
            == 1,
        ),

        (
            "c2:9b -> existing GID 11",
            record_map[
                "c2:9b"
            ][
                "final_gid"
            ]
            == 11,
        ),

        (
            "c2:508 -> existing GID 65",
            record_map[
                "c2:508"
            ][
                "final_gid"
            ]
            == 65,
        ),

        (
            "c2:601b -> existing GID 73",
            record_map[
                "c2:601b"
            ][
                "final_gid"
            ]
            == 73,
        ),

        (
            "c2:1 did not create duplicate new GID",
            (
                record_map[
                    "c2:1"
                ][
                    "final_gid"
                ]
                is None
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
            ),
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
        "SANITY"
    )

    print(
        "======"
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
        "Experimental policy only."
    )


if __name__ == "__main__":
    main()
