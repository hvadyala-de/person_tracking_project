from collections import Counter
from itertools import combinations

from src.experiments.analyze_c2_601_segment_gid_ranking import (
    cosine_similarity,
    normalize,
)

from src.experiments.analyze_c3_gap_identity_changes import (
    build_validated_reference_state,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.experiments.run_c2_production_policy_experiment import (
    audit_existing_members,
    register_population_tracklets,
    run_final_pending_pass,
    snapshot_existing_members,
)

from src.experiments.run_c3_production_policy_experiment import (
    build_c3_population,
    finalize_c3_records,
    load_c3_rows,
    process_c3_population,
)


# ============================================================
# OBSERVED c3 REFERENCE VALUES
#
# These are values from the first successful c3 production
# experiment.
#
# They are NOT being declared permanent production constants
# yet. This audit determines whether they are safe to freeze.
# ============================================================

OBSERVED_C3_POPULATION = 95

OBSERVED_FINAL_GIDS = 134

OBSERVED_FINAL_PENDING = 173

OBSERVED_FINAL_ANCHORED = 47

OBSERVED_C3_EXISTING = 13

OBSERVED_C3_NEW = 33

OBSERVED_C3_PENDING = 49


# ============================================================
# VALIDATED SPLIT EXPECTATIONS
# ============================================================

C3_104_BEFORE_KEY = (
    3,
    940104,
)

C3_104_AFTER_KEY = (
    3,
    940105,
)

EXPECTED_C3_104_AFTER_GID = 11


# ============================================================
# AUDIT-ONLY JOINT EVIDENCE SETTINGS
#
# These do NOT modify production thresholds.
#
# They are deliberately broad safety-audit requirements:
#
#   geometry SUPPORTED
#   >= 10 shared frames
#   direct ReID >= 0.80
#
# A newly anchored identity should have at least one such
# cross-camera CORE relation.
# ============================================================

AUDIT_MIN_SHARED_FRAMES = 10

AUDIT_MIN_DIRECT_REID = 0.80


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
# CURRENT MEMBER -> GID MAP
# ============================================================

def build_member_gid_map(
    manager,
):

    result = {}


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            result[
                member_key(
                    member
                )
            ] = gid


    return result


# ============================================================
# FIND GID FOR MEMBER KEY
# ============================================================

def find_gid_for_key(
    manager,
    key,
):

    mapping = build_member_gid_map(
        manager
    )

    return mapping.get(
        key
    )


# ============================================================
# SPLIT EXISTING-MEMBER CHANGES
#
# A trust-only change is reported separately from a GID move.
#
# Moving a pre-c3 member to another GID is always a hard
# regression.
# ============================================================

def classify_existing_member_changes(
    changes,
):

    gid_moves = []

    trust_only = []


    for row in changes:

        if (
            row[
                "old_gid"
            ]
            != row[
                "new_gid"
            ]
        ):

            gid_moves.append(
                row
            )

        else:

            trust_only.append(
                row
            )


    return (
        gid_moves,
        trust_only,
    )


# ============================================================
# GET TRACKLET SAFELY
# ============================================================

def get_member_tracklet(
    manager,
    member,
    tracklet_lookup,
):

    try:

        return manager.get_member_tracklet(
            member,
            tracklet_lookup,
        )

    except Exception:

        return None


# ============================================================
# GET MEMBER EMBEDDING SAFELY
# ============================================================

def get_member_embedding(
    manager,
    key,
):

    embedding = manager.member_embeddings.get(
        key
    )


    if embedding is None:

        return None


    return normalize(
        embedding
    )


# ============================================================
# CORE MEMBER ROWS
# ============================================================

def get_core_member_rows(
    manager,
    gid,
    tracklet_lookup,
):

    identity = manager.identities.get(
        gid
    )


    if identity is None:

        return []


    core_keys = manager.core_members.get(
        gid,
        set(),
    )


    rows = []


    for member in identity.members:

        key = member_key(
            member
        )


        if key not in core_keys:

            continue


        rows.append(
            {
                "member":
                    member,

                "key":
                    key,

                "camera_id":
                    key[
                        0
                    ],

                "track_id":
                    key[
                        1
                    ],

                "tracklet":
                    get_member_tracklet(
                        manager,
                        member,
                        tracklet_lookup,
                    ),

                "embedding":
                    get_member_embedding(
                        manager,
                        key,
                    ),
            }
        )


    return rows


# ============================================================
# AUDIT CROSS-CAMERA CORE PAIRS
# ============================================================

def audit_core_cross_camera_pairs(
    manager,
    gid,
    tracklet_lookup,
):

    core_rows = get_core_member_rows(
        manager,
        gid,
        tracklet_lookup,
    )


    pair_rows = []


    for row_a, row_b in combinations(
        core_rows,
        2,
    ):

        if (
            row_a[
                "camera_id"
            ]
            == row_b[
                "camera_id"
            ]
        ):

            continue


        tracklet_a = row_a[
            "tracklet"
        ]

        tracklet_b = row_b[
            "tracklet"
        ]


        embedding_a = row_a[
            "embedding"
        ]

        embedding_b = row_b[
            "embedding"
        ]


        if (
            tracklet_a is None
            or tracklet_b is None
        ):

            continue


        geometry = evaluate_cross_camera_geometry(
            tracklet_a,
            tracklet_b,
            manager.homographies,
        )


        direct_similarity = None


        if (
            embedding_a is not None
            and embedding_b is not None
        ):

            direct_similarity = cosine_similarity(
                embedding_a,
                embedding_b,
            )


        geometry_supported = (
            geometry.status
            == "SUPPORTED"

            and

            geometry.shared_frames
            >= AUDIT_MIN_SHARED_FRAMES
        )


        joint_supported = (
            geometry_supported

            and

            direct_similarity is not None

            and

            direct_similarity
            >= AUDIT_MIN_DIRECT_REID
        )


        pair_rows.append(
            {
                "key_a":
                    row_a[
                        "key"
                    ],

                "key_b":
                    row_b[
                        "key"
                    ],

                "geometry_status":
                    geometry.status,

                "shared_frames":
                    geometry.shared_frames,

                "median_distance":
                    geometry.median_distance,

                "direct_similarity":
                    direct_similarity,

                "geometry_supported":
                    geometry_supported,

                "joint_supported":
                    joint_supported,
            }
        )


    return pair_rows


# ============================================================
# SAME-CAMERA TEMPORAL OVERLAP AUDIT
#
# Multiple fragments from the same camera in one GID are
# allowed when they are sequential.
#
# The safety violation is temporal overlap between different
# same-camera CORE members.
# ============================================================

def audit_same_camera_core_overlaps(
    manager,
    gid,
    tracklet_lookup,
):

    core_rows = get_core_member_rows(
        manager,
        gid,
        tracklet_lookup,
    )


    conflicts = []


    for row_a, row_b in combinations(
        core_rows,
        2,
    ):

        if (
            row_a[
                "camera_id"
            ]
            != row_b[
                "camera_id"
            ]
        ):

            continue


        tracklet_a = row_a[
            "tracklet"
        ]

        tracklet_b = row_b[
            "tracklet"
        ]


        if (
            tracklet_a is None
            or tracklet_b is None
        ):

            continue


        overlap_start = max(
            tracklet_a.start_frame,
            tracklet_b.start_frame,
        )

        overlap_end = min(
            tracklet_a.end_frame,
            tracklet_b.end_frame,
        )


        if overlap_end < overlap_start:

            continue


        conflicts.append(
            {
                "gid":
                    gid,

                "key_a":
                    row_a[
                        "key"
                    ],

                "key_b":
                    row_b[
                        "key"
                    ],

                "overlap_start":
                    overlap_start,

                "overlap_end":
                    overlap_end,

                "overlap_frames":
                    (
                        overlap_end
                        - overlap_start
                        + 1
                    ),
            }
        )


    return conflicts


# ============================================================
# PRINT EXISTING-MEMBER REGRESSION
# ============================================================

def print_existing_member_audit(
    gid_moves,
    trust_only,
):

    print()
    print(
        "PRE-c3 MEMBER REGRESSION AUDIT"
    )

    print(
        "=============================="
    )


    print(
        "GID moves:",
        len(
            gid_moves
        ),
    )

    print(
        "Trust-only changes:",
        len(
            trust_only
        ),
    )


    if gid_moves:

        print()
        print(
            "GID MOVES"
        )


        for row in gid_moves:

            print(
                f"{row['key']}"
                f" | GID "
                f"{row['old_gid']}"
                f" -> "
                f"{row['new_gid']}"
                f" | trust "
                f"{row['old_trust']}"
                f" -> "
                f"{row['new_trust']}"
            )


    if trust_only:

        print()
        print(
            "TRUST-ONLY CHANGES"
        )


        for row in trust_only:

            print(
                f"{row['key']}"
                f" | GID "
                f"{row['old_gid']}"
                f" | trust "
                f"{row['old_trust']}"
                f" -> "
                f"{row['new_trust']}"
            )


# ============================================================
# PRINT FINAL CONTINUATION INTERPRETATION
# ============================================================

def print_final_continuation_audit(
    results,
):

    actual_recoveries = [
        row

        for row in results

        if row.get(
            "status"
        )
        != "PENDING"
    ]


    remaining_rows = [
        row

        for row in results

        if row.get(
            "status"
        )
        == "PENDING"
    ]


    reason_counts = Counter(
        row.get(
            "reason",
            "UNKNOWN",
        )

        for row in actual_recoveries
    )


    print()
    print(
        "FINAL CONTINUATION INTERPRETATION"
    )

    print(
        "================================="
    )


    print(
        "Returned result rows:",
        len(
            results
        ),
    )

    print(
        "Actual recoveries:",
        len(
            actual_recoveries
        ),
    )

    print(
        "Still-pending rows:",
        len(
            remaining_rows
        ),
    )


    if reason_counts:

        print()
        print(
            "Actual recovery reasons:"
        )


        for reason, count in sorted(
            reason_counts.items()
        ):

            print(
                f"  {reason}: {count}"
            )


    return (
        actual_recoveries,
        remaining_rows,
    )


# ============================================================
# PRINT NEWLY ANCHORED GID AUDIT
# ============================================================

def print_new_anchor_audit(
    manager,
    newly_anchored,
    tracklet_lookup,
):

    missing_joint_support = []

    contradicted_gids = []

    same_camera_conflicts = []


    print()
    print(
        "NEWLY ANCHORED GID SAFETY AUDIT"
    )

    print(
        "==============================="
    )


    print(
        "Newly anchored GIDs:",
        len(
            newly_anchored
        ),
    )


    for gid in sorted(
        newly_anchored
    ):

        pair_rows = audit_core_cross_camera_pairs(
            manager=
                manager,

            gid=
                gid,

            tracklet_lookup=
                tracklet_lookup,
        )


        joint_rows = [
            row

            for row in pair_rows

            if row[
                "joint_supported"
            ]
        ]


        contradictions = [
            row

            for row in pair_rows

            if row[
                "geometry_status"
            ]
            == "CONTRADICTED"
        ]


        overlaps = audit_same_camera_core_overlaps(
            manager=
                manager,

            gid=
                gid,

            tracklet_lookup=
                tracklet_lookup,
        )


        if not joint_rows:

            missing_joint_support.append(
                gid
            )


        if contradictions:

            contradicted_gids.append(
                (
                    gid,
                    contradictions,
                )
            )


        if overlaps:

            same_camera_conflicts.extend(
                overlaps
            )


        best_joint = None


        if joint_rows:

            best_joint = max(
                joint_rows,
                key=lambda row: (
                    row[
                        "direct_similarity"
                    ],

                    row[
                        "shared_frames"
                    ],
                ),
            )


        print()
        print(
            f"GID {gid}"
        )

        print(
            f"  cross-camera CORE pairs: "
            f"{len(pair_rows)}"
        )

        print(
            f"  joint-supported pairs:   "
            f"{len(joint_rows)}"
        )

        print(
            f"  contradicted pairs:      "
            f"{len(contradictions)}"
        )

        print(
            f"  same-camera overlaps:    "
            f"{len(overlaps)}"
        )


        if best_joint is not None:

            median = best_joint[
                "median_distance"
            ]


            median_text = (
                "None"

                if median is None

                else f"{median:.2f}"
            )


            print(
                "  best joint support:       "
                f"c{best_joint['key_a'][0]}:"
                f"{best_joint['key_a'][1]}"
                f" <-> "
                f"c{best_joint['key_b'][0]}:"
                f"{best_joint['key_b'][1]}"
                f" | ReID="
                f"{best_joint['direct_similarity']:.4f}"
                f" | shared="
                f"{best_joint['shared_frames']}"
                f" | median="
                f"{median_text}"
            )


    if contradicted_gids:

        print()
        print(
            "NEW-ANCHOR CORE CONTRADICTIONS"
        )

        print(
            "=============================="
        )


        for gid, rows in contradicted_gids:

            for row in rows:

                median = row[
                    "median_distance"
                ]


                median_text = (
                    "None"

                    if median is None

                    else f"{median:.2f}"
                )


                direct = row[
                    "direct_similarity"
                ]


                direct_text = (
                    "None"

                    if direct is None

                    else f"{direct:.4f}"
                )


                print(
                    f"GID {gid}"
                    f" | "
                    f"c{row['key_a'][0]}:"
                    f"{row['key_a'][1]}"
                    f" <-> "
                    f"c{row['key_b'][0]}:"
                    f"{row['key_b'][1]}"
                    f" | ReID="
                    f"{direct_text}"
                    f" | shared="
                    f"{row['shared_frames']}"
                    f" | median="
                    f"{median_text}"
                )


    if same_camera_conflicts:

        print()
        print(
            "SAME-CAMERA CORE OVERLAPS"
        )

        print(
            "========================="
        )


        for row in same_camera_conflicts:

            print(
                f"GID {row['gid']}"
                f" | "
                f"c{row['key_a'][0]}:"
                f"{row['key_a'][1]}"
                f" <-> "
                f"c{row['key_b'][0]}:"
                f"{row['key_b'][1]}"
                f" | overlap="
                f"{row['overlap_start']}"
                f".."
                f"{row['overlap_end']}"
                f" | frames="
                f"{row['overlap_frames']}"
            )


    return {
        "missing_joint_support":
            missing_joint_support,

        "contradicted_gids":
            contradicted_gids,

        "same_camera_conflicts":
            same_camera_conflicts,
    }


# ============================================================
# CHECK GIDs CONTAINING c3 FOR SAME-CAMERA CORE OVERLAP
# ============================================================

def audit_all_c3_identity_overlaps(
    manager,
    tracklet_lookup,
):

    c3_gids = set()


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            if (
                normalize_camera_id(
                    member.camera_id
                )
                == 3
            ):

                c3_gids.add(
                    gid
                )

                break


    conflicts = []


    for gid in sorted(
        c3_gids
    ):

        conflicts.extend(
            audit_same_camera_core_overlaps(
                manager=
                    manager,

                gid=
                    gid,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


    return (
        c3_gids,
        conflicts,
    )


# ============================================================
# PRINT CHECKS
# ============================================================

def print_checks(
    checks,
):

    print()
    print(
        "FOUR-CAMERA SAFETY CHECKS"
    )

    print(
        "========================="
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
        f"{len(checks)}"
        " checks passed"
    )


    return (
        passed
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
        "c3 FOUR-CAMERA PRODUCTION SAFETY AUDIT"
    )

    print(
        "====================================="
    )


    print()
    print(
        "No production thresholds are modified."
    )

    print(
        "No identities are persisted."
    )

    print(
        "The validated c3:104 split remains in memory only."
    )


    # ========================================================
    # REBUILD VALIDATED c0+c1+c2
    # ========================================================

    (
        manager,
        tracklet_lookup,
    ) = build_validated_reference_state()


    baseline_member_snapshot = snapshot_existing_members(
        manager
    )


    baseline_anchored_gids = set(
        manager.anchored_gids
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
        "PRE-c3 REFERENCE STATE"
    )

    print(
        "======================"
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

    print(
        "Assigned member snapshot:",
        len(
            baseline_member_snapshot
        ),
    )


    # ========================================================
    # BUILD c3 POPULATION
    # ========================================================

    c3_rows = load_c3_rows()


    population = build_c3_population(
        c3_rows
    )


    print()
    print(
        "c3 PRODUCTION INPUT"
    )

    print(
        "==================="
    )

    print(
        "Original useful tracklets:",
        len(
            c3_rows
        ),
    )

    print(
        "Production segments:",
        len(
            population
        ),
    )


    collisions = register_population_tracklets(
        population=
            population,

        tracklet_lookup=
            tracklet_lookup,
    )


    # ========================================================
    # PROCESS c3
    # ========================================================

    records = process_c3_population(
        manager=
            manager,

        population=
            population,

        tracklet_lookup=
            tracklet_lookup,
    )


    pre_final_pending = len(
        manager.pending_tracklets
    )


    # ========================================================
    # FINAL CONTINUATION
    # ========================================================

    final_results = run_final_pending_pass(
        manager=
            manager,

        tracklet_lookup=
            tracklet_lookup,
    )


    (
        actual_recoveries,
        remaining_result_rows,
    ) = print_final_continuation_audit(
        final_results
    )


    # ========================================================
    # FINALIZE c3 RECORDS
    # ========================================================

    baseline_gid_ids = set(
        range(
            1,
            baseline_gids + 1,
        )
    )


    finalize_c3_records(
        manager=
            manager,

        records=
            records,

        baseline_gid_ids=
            baseline_gid_ids,
    )


    # ========================================================
    # AUDIT PRE-c3 MEMBERS
    # ========================================================

    existing_changes = audit_existing_members(
        manager=
            manager,

        baseline_snapshot=
            baseline_member_snapshot,
    )


    (
        gid_moves,
        trust_only_changes,
    ) = classify_existing_member_changes(
        existing_changes
    )


    print_existing_member_audit(
        gid_moves=
            gid_moves,

        trust_only=
            trust_only_changes,
    )


    # ========================================================
    # c3:104 SPLIT
    # ========================================================

    c3_104a_gid = find_gid_for_key(
        manager,
        C3_104_BEFORE_KEY,
    )

    c3_104b_gid = find_gid_for_key(
        manager,
        C3_104_AFTER_KEY,
    )


    print()
    print(
        "c3:104 SPLIT SAFETY"
    )

    print(
        "==================="
    )

    print(
        "c3:104a GID:",
        c3_104a_gid,
    )

    print(
        "c3:104b GID:",
        c3_104b_gid,
    )


    # ========================================================
    # NEW ANCHORS
    # ========================================================

    newly_anchored = (
        set(
            manager.anchored_gids
        )
        - baseline_anchored_gids
    )


    new_anchor_audit = print_new_anchor_audit(
        manager=
            manager,

        newly_anchored=
            newly_anchored,

        tracklet_lookup=
            tracklet_lookup,
    )


    # ========================================================
    # ALL GIDs CONTAINING c3:
    # SAME-CAMERA CORE OVERLAP
    # ========================================================

    (
        c3_member_gids,
        c3_identity_overlaps,
    ) = audit_all_c3_identity_overlaps(
        manager=
            manager,

        tracklet_lookup=
            tracklet_lookup,
    )


    print()
    print(
        "ALL c3-CONTAINING GID OVERLAP AUDIT"
    )

    print(
        "=================================="
    )

    print(
        "GIDs containing c3 members:",
        len(
            c3_member_gids
        ),
    )

    print(
        "Same-camera CORE overlaps:",
        len(
            c3_identity_overlaps
        ),
    )


    if c3_identity_overlaps:

        for row in c3_identity_overlaps:

            print(
                f"GID {row['gid']}"
                f" | "
                f"c{row['key_a'][0]}:"
                f"{row['key_a'][1]}"
                f" <-> "
                f"c{row['key_b'][0]}:"
                f"{row['key_b'][1]}"
                f" | "
                f"{row['overlap_start']}"
                f".."
                f"{row['overlap_end']}"
            )


    # ========================================================
    # FINAL COUNTS
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


    c3_existing = sum(
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


    c3_new = sum(
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


    c3_pending = sum(
        1

        for record in records

        if record[
            "is_pending"
        ]
    )


    print()
    print(
        "FINAL FOUR-CAMERA STATE"
    )

    print(
        "======================="
    )

    print(
        "GIDs:",
        final_gids,
    )

    print(
        "Pending:",
        final_pending,
    )

    print(
        "Anchored:",
        final_anchored,
    )


    print()
    print(
        "c3 -> existing GID:",
        c3_existing,
    )

    print(
        "c3 -> new GID:",
        c3_new,
    )

    print(
        "c3 pending:",
        c3_pending,
    )


    print()
    print(
        "Pending before final continuation:",
        pre_final_pending,
    )

    print(
        "Actual final recoveries:",
        len(
            actual_recoveries
        ),
    )

    print(
        "Pending after final continuation:",
        final_pending,
    )


    # ========================================================
    # HARD SAFETY CHECKS
    # ========================================================

    checks = []


    checks.append(
        (
            "c3 production population has 95 segments",

            len(
                population
            )
            == OBSERVED_C3_POPULATION,
        )
    )


    checks.append(
        (
            "no synthetic c3 tracklet collisions",

            len(
                collisions
            )
            == 0,
        )
    )


    checks.append(
        (
            "no pre-c3 assigned member moved to another GID",

            len(
                gid_moves
            )
            == 0,
        )
    )


    checks.append(
        (
            "c3:104a and c3:104b remain separated",

            c3_104a_gid is not None

            and

            c3_104b_gid is not None

            and

            c3_104a_gid
            != c3_104b_gid,
        )
    )


    checks.append(
        (
            "c3:104b resolves to validated GID 11",

            c3_104b_gid
            == EXPECTED_C3_104_AFTER_GID,
        )
    )


    checks.append(
        (
            "c3:104a does not resolve to GID 11",

            c3_104a_gid
            != EXPECTED_C3_104_AFTER_GID,
        )
    )


    checks.append(
        (
            "every newly anchored GID has cross-camera "
            "joint CORE support",

            len(
                new_anchor_audit[
                    "missing_joint_support"
                ]
            )
            == 0,
        )
    )


    checks.append(
        (
            "newly anchored GIDs have no contradicted "
            "cross-camera CORE pairs",

            len(
                new_anchor_audit[
                    "contradicted_gids"
                ]
            )
            == 0,
        )
    )


    checks.append(
        (
            "GIDs containing c3 have no overlapping "
            "same-camera CORE members",

            len(
                c3_identity_overlaps
            )
            == 0,
        )
    )


    checks.append(
        (
            "final GID count reproduces observed 134",

            final_gids
            == OBSERVED_FINAL_GIDS,
        )
    )


    checks.append(
        (
            "final pending count reproduces observed 173",

            final_pending
            == OBSERVED_FINAL_PENDING,
        )
    )


    checks.append(
        (
            "final anchored count reproduces observed 47",

            final_anchored
            == OBSERVED_FINAL_ANCHORED,
        )
    )


    checks.append(
        (
            "c3 existing association count reproduces 13",

            c3_existing
            == OBSERVED_C3_EXISTING,
        )
    )


    checks.append(
        (
            "c3 new-GID member count reproduces 33",

            c3_new
            == OBSERVED_C3_NEW,
        )
    )


    checks.append(
        (
            "c3 pending count reproduces 49",

            c3_pending
            == OBSERVED_C3_PENDING,
        )
    )


    checks.append(
        (
            "final continuation return is interpreted "
            "consistently",

            (
                len(
                    actual_recoveries
                )
                +
                len(
                    remaining_result_rows
                )
            )
            == len(
                final_results
            )

            and

            len(
                remaining_result_rows
            )
            == final_pending,
        )
    )


    all_passed = print_checks(
        checks
    )


    print()
    print(
        "FINAL SAFETY AUDIT RESULT"
    )

    print(
        "========================="
    )


    if all_passed:

        print(
            "PASS"
        )

    else:

        print(
            "FAIL - inspect failed checks before "
            "freezing four-camera behavior"
        )


    print()
    print(
        "Production manager source was not modified."
    )

    print(
        "Production thresholds were not modified."
    )

    print(
        "No c3 split was persisted."
    )


if __name__ == "__main__":
    main()
