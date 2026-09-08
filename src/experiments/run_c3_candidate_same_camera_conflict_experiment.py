from collections import Counter
from itertools import combinations

import numpy as np

import src.identity_compatibility as identity_compatibility

from src.experiments.analyze_c3_gap_identity_changes import (
    build_validated_reference_state,
)

from src.identity_compatibility import (
    MIN_SEPARATED_FRAMES,
    MIN_SEPARATED_RATIO,
    MIN_SHARED_FRAMES,
    bbox_diagonal,
    bbox_iou,
    center_distance,
    detections_by_frame,
    normalize_camera_id,
)

from src.experiments.run_c2_production_policy_experiment import (
    audit_existing_members,
    find_member_gid,
    get_member_trust_safe,
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
# PURPOSE
#
# Replay the validated c0+c1+c2+c3 production policy using
# an AUDIT-ONLY candidate same-camera conflict rule.
#
# Production source files are NOT modified.
#
# The candidate rule is installed temporarily by replacing:
#
#   identity_compatibility.same_camera_conflict
#
# for this process only.
# ============================================================


# ============================================================
# OBSERVED REFERENCE STATE
# ============================================================

EXPECTED_PRE_C3_GIDS = 107
EXPECTED_PRE_C3_PENDING = 164
EXPECTED_PRE_C3_ANCHORED = 22

EXPECTED_C3_POPULATION = 95

EXPECTED_C3_104_AFTER_GID = 11


# ============================================================
# CANDIDATE FRAME-LEVEL RULE
#
# This is the same audit-only rule evaluated by:
#
#   analyze_c3_same_camera_conflict_thresholds.py
#
# These are NOT production thresholds.
# ============================================================

CANDIDATE_MAX_IOU = 0.10

CANDIDATE_MIN_ABSOLUTE_DISTANCE = 20.0

CANDIDATE_MIN_RELATIVE_DISTANCE = 0.25


# ============================================================
# TARGET KEYS
# ============================================================

C3_12_KEY = (
    3,
    12,
)

C3_15_KEY = (
    3,
    15,
)

C3_104A_KEY = (
    3,
    940104,
)

C3_104B_KEY = (
    3,
    940105,
)


# ============================================================
# BOX HELPERS
# ============================================================

def detection_box(
    detection,
):

    return (
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    )


def bottom_center(
    box,
):

    x1, y1, x2, y2 = box

    return np.asarray(
        [
            (
                x1
                + x2
            )
            / 2.0,

            y2,
        ],
        dtype=np.float64,
    )


def bottom_center_distance(
    box_a,
    box_b,
):

    return float(
        np.linalg.norm(
            bottom_center(
                box_a
            )
            -
            bottom_center(
                box_b
            )
        )
    )


# ============================================================
# CANDIDATE FRAME-LEVEL SPATIAL SEPARATION
# ============================================================

def candidate_spatially_separated(
    detection_a,
    detection_b,
):

    box_a = detection_box(
        detection_a
    )

    box_b = detection_box(
        detection_b
    )


    iou = bbox_iou(
        box_a,
        box_b,
    )


    if (
        iou
        > CANDIDATE_MAX_IOU
    ):

        return False


    scale = max(
        bbox_diagonal(
            box_a
        ),

        bbox_diagonal(
            box_b
        ),

        1.0,
    )


    required_distance = max(
        CANDIDATE_MIN_ABSOLUTE_DISTANCE,

        CANDIDATE_MIN_RELATIVE_DISTANCE
        * scale,
    )


    center = center_distance(
        box_a,
        box_b,
    )


    bottom = bottom_center_distance(
        box_a,
        box_b,
    )


    return (
        center
        >= required_distance

        or

        bottom
        >= required_distance
    )


# ============================================================
# CANDIDATE SAME-CAMERA CONFLICT
# ============================================================

def candidate_same_camera_conflict(
    candidate_tracklet,
    existing_tracklet,
):

    candidate_camera = normalize_camera_id(
        candidate_tracklet.camera_id
    )

    existing_camera = normalize_camera_id(
        existing_tracklet.camera_id
    )


    if (
        candidate_camera
        != existing_camera
    ):

        return False


    candidate_frames = detections_by_frame(
        candidate_tracklet
    )

    existing_frames = detections_by_frame(
        existing_tracklet
    )


    shared_frames = sorted(
        set(
            candidate_frames
        )
        &
        set(
            existing_frames
        )
    )


    if (
        len(
            shared_frames
        )
        < MIN_SHARED_FRAMES
    ):

        return False


    separated_count = 0


    for frame in shared_frames:

        if candidate_spatially_separated(
            candidate_frames[
                frame
            ],

            existing_frames[
                frame
            ],
        ):

            separated_count += 1


    separated_ratio = (
        separated_count
        / len(
            shared_frames
        )
    )


    return (
        separated_count
        >= MIN_SEPARATED_FRAMES

        and

        separated_ratio
        >= MIN_SEPARATED_RATIO
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
# GET MEMBER TRACKLET
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

        key = member_key(
            member
        )


        tracklet = tracklet_lookup.get(
            key
        )


        if tracklet is not None:

            return tracklet


        return tracklet_lookup.get(
            (
                f"c{key[0]}",
                key[1],
            )
        )


# ============================================================
# AUDIT SAME-GID SAME-CAMERA PAIRS
# ============================================================

def audit_same_gid_same_camera_pairs(
    manager,
    tracklet_lookup,
    original_conflict_function,
):

    rows = []


    for gid, identity in (
        manager.identities.items()
    ):

        for member_a, member_b in combinations(
            identity.members,
            2,
        ):

            key_a = member_key(
                member_a
            )

            key_b = member_key(
                member_b
            )


            if (
                key_a[
                    0
                ]
                != key_b[
                    0
                ]
            ):

                continue


            tracklet_a = get_member_tracklet(
                manager,
                member_a,
                tracklet_lookup,
            )

            tracklet_b = get_member_tracklet(
                manager,
                member_b,
                tracklet_lookup,
            )


            if (
                tracklet_a is None
                or tracklet_b is None
            ):

                continue


            frames_a = detections_by_frame(
                tracklet_a
            )

            frames_b = detections_by_frame(
                tracklet_b
            )


            shared_frames = sorted(
                set(
                    frames_a
                )
                &
                set(
                    frames_b
                )
            )


            current_conflict = (
                original_conflict_function(
                    tracklet_a,
                    tracklet_b,
                )
            )


            candidate_conflict = (
                candidate_same_camera_conflict(
                    tracklet_a,
                    tracklet_b,
                )
            )


            rows.append(
                {
                    "gid":
                        gid,

                    "key_a":
                        key_a,

                    "key_b":
                        key_b,

                    "shared":
                        len(
                            shared_frames
                        ),

                    "current_conflict":
                        current_conflict,

                    "candidate_conflict":
                        candidate_conflict,
                }
            )


    return rows


# ============================================================
# PRINT SAME-GID PAIR AUDIT
# ============================================================

def print_same_gid_pair_audit(
    rows,
):

    candidate_conflicts = [
        row

        for row in rows

        if row[
            "candidate_conflict"
        ]
    ]


    current_conflicts = [
        row

        for row in rows

        if row[
            "current_conflict"
        ]
    ]


    print()
    print(
        "FINAL SAME-GID SAME-CAMERA PAIR AUDIT"
    )

    print(
        "====================================="
    )


    print(
        "Same-camera member pairs inside GIDs:",
        len(
            rows
        ),
    )

    print(
        "Current-rule conflicts still inside GIDs:",
        len(
            current_conflicts
        ),
    )

    print(
        "Candidate-rule conflicts still inside GIDs:",
        len(
            candidate_conflicts
        ),
    )


    if candidate_conflicts:

        print()
        print(
            "CANDIDATE CONFLICTS STILL INSIDE IDENTITIES"
        )

        print(
            "=========================================="
        )


        for row in candidate_conflicts:

            print(
                f"GID {row['gid']}"
                f" | "
                f"c{row['key_a'][0]}:"
                f"{row['key_a'][1]}"
                f" <-> "
                f"c{row['key_b'][0]}:"
                f"{row['key_b'][1]}"
                f" | shared="
                f"{row['shared']}"
            )


    return (
        current_conflicts,
        candidate_conflicts,
    )


# ============================================================
# PRINT PRE-c3 MEMBER CHANGES
# ============================================================

def print_existing_member_changes(
    changes,
):

    gid_moves = []

    trust_changes = []


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

            trust_changes.append(
                row
            )


    print()
    print(
        "PRE-c3 MEMBER REGRESSION"
    )

    print(
        "========================"
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
            trust_changes
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
                f" | "
                f"GID {row['old_gid']}"
                f" -> "
                f"{row['new_gid']}"
                f" | trust "
                f"{row['old_trust']}"
                f" -> "
                f"{row['new_trust']}"
            )


    return (
        gid_moves,
        trust_changes,
    )


# ============================================================
# FIND RECORD
# ============================================================

def find_record(
    records,
    label,
):

    for record in records:

        if (
            record[
                "label"
            ]
            == label
        ):

            return record


    return None


# ============================================================
# PRINT TARGET OUTCOMES
# ============================================================

def print_target_outcomes(
    manager,
    records,
):

    gid_12 = find_member_gid(
        manager,
        C3_12_KEY,
    )

    gid_15 = find_member_gid(
        manager,
        C3_15_KEY,
    )

    gid_104a = find_member_gid(
        manager,
        C3_104A_KEY,
    )

    gid_104b = find_member_gid(
        manager,
        C3_104B_KEY,
    )


    row_12 = find_record(
        records,
        "c3:12",
    )

    row_15 = find_record(
        records,
        "c3:15",
    )

    row_104a = find_record(
        records,
        "c3:104a",
    )

    row_104b = find_record(
        records,
        "c3:104b",
    )


    print()
    print(
        "TARGET OUTCOMES"
    )

    print(
        "==============="
    )


    for label, key, gid, row in (
        (
            "c3:12",
            C3_12_KEY,
            gid_12,
            row_12,
        ),
        (
            "c3:15",
            C3_15_KEY,
            gid_15,
            row_15,
        ),
        (
            "c3:104a",
            C3_104A_KEY,
            gid_104a,
            row_104a,
        ),
        (
            "c3:104b",
            C3_104B_KEY,
            gid_104b,
            row_104b,
        ),
    ):

        trust = get_member_trust_safe(
            manager,
            gid,
            key,
        )


        if row is None:

            print(
                f"{label:<10}"
                f" | GID={gid}"
                f" | trust={trust}"
                f" | record=MISSING"
            )

            continue


        print(
            f"{label:<10}"
            f" | GID={gid}"
            f" | trust={trust}"
            f" | pending="
            f"{row['is_pending']}"
            f" | initial="
            f"{row['initial_status']}"
            f" | reason="
            f"{row['reason']}"
        )


    target_safe = not (
        gid_12 is not None

        and

        gid_15 is not None

        and

        gid_12
        == gid_15
    )


    return {
        "gid_12":
            gid_12,

        "gid_15":
            gid_15,

        "gid_104a":
            gid_104a,

        "gid_104b":
            gid_104b,

        "target_safe":
            target_safe,
    }


# ============================================================
# PRINT CHECKS
# ============================================================

def print_checks(
    checks,
):

    print()
    print(
        "CANDIDATE-RULE REPLAY CHECKS"
    )

    print(
        "============================"
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
        "c3 CANDIDATE SAME-CAMERA CONFLICT POLICY REPLAY"
    )

    print(
        "=============================================="
    )


    print()
    print(
        "Production source files are unchanged."
    )

    print(
        "The candidate rule exists only in this process."
    )


    original_same_camera_conflict = (
        identity_compatibility.same_camera_conflict
    )


    # ========================================================
    # INSTALL AUDIT-ONLY RULE
    # ========================================================

    identity_compatibility.same_camera_conflict = (
        candidate_same_camera_conflict
    )


    try:

        print()
        print(
            "Candidate same-camera conflict rule installed."
        )


        # ====================================================
        # REBUILD c0+c1+c2 UNDER CANDIDATE RULE
        # ====================================================

        (
            manager,
            tracklet_lookup,
        ) = build_validated_reference_state()


        pre_c3_gids = len(
            manager.identities
        )

        pre_c3_pending = len(
            manager.pending_tracklets
        )

        pre_c3_anchored = len(
            manager.anchored_gids
        )


        print()
        print(
            "CANDIDATE-RULE PRE-c3 STATE"
        )

        print(
            "==========================="
        )

        print(
            "GIDs:",
            pre_c3_gids,
        )

        print(
            "Pending:",
            pre_c3_pending,
        )

        print(
            "Anchored:",
            pre_c3_anchored,
        )


        baseline_snapshot = snapshot_existing_members(
            manager
        )


        baseline_gid_ids = set(
            manager.identities.keys()
        )


        # ====================================================
        # BUILD c3 POPULATION
        # ====================================================

        c3_rows = load_c3_rows()


        population = build_c3_population(
            c3_rows
        )


        collisions = register_population_tracklets(
            population=
                population,

            tracklet_lookup=
                tracklet_lookup,
        )


        print()
        print(
            "CANDIDATE-RULE c3 INPUT"
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
            "Production segments:",
            len(
                population
            ),
        )

        print(
            "Synthetic collisions:",
            len(
                collisions
            ),
        )


        # ====================================================
        # PROCESS c3
        # ====================================================

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


        final_results = run_final_pending_pass(
            manager=
                manager,

            tracklet_lookup=
                tracklet_lookup,
        )


        actual_final_recoveries = [
            row

            for row in final_results

            if row.get(
                "status"
            )
            != "PENDING"
        ]


        finalize_c3_records(
            manager=
                manager,

            records=
                records,

            baseline_gid_ids=
                baseline_gid_ids,
        )


        # ====================================================
        # PRE-c3 MEMBER STABILITY
        # ====================================================

        existing_changes = audit_existing_members(
            manager=
                manager,

            baseline_snapshot=
                baseline_snapshot,
        )


        (
            gid_moves,
            trust_changes,
        ) = print_existing_member_changes(
            existing_changes
        )


        # ====================================================
        # TARGETS
        # ====================================================

        targets = print_target_outcomes(
            manager=
                manager,

            records=
                records,
        )


        # ====================================================
        # SAME-GID CONFLICT AUDIT
        # ====================================================

        same_gid_rows = (
            audit_same_gid_same_camera_pairs(
                manager=
                    manager,

                tracklet_lookup=
                    tracklet_lookup,

                original_conflict_function=
                    original_same_camera_conflict,
            )
        )


        (
            current_conflicts_inside,
            candidate_conflicts_inside,
        ) = print_same_gid_pair_audit(
            same_gid_rows
        )


        # ====================================================
        # FINAL COUNTS
        # ====================================================

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


        state_counts = Counter(
            record[
                "final_state"
            ]

            for record in records
        )


        print()
        print(
            "CANDIDATE-RULE FINAL STATE"
        )

        print(
            "=========================="
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
            "c3 final states:"
        )


        for state, count in sorted(
            state_counts.items()
        ):

            print(
                f"  {state}: {count}"
            )


        print()
        print(
            "Pending before final continuation:",
            pre_final_pending,
        )

        print(
            "Actual final recoveries:",
            len(
                actual_final_recoveries
            ),
        )

        print(
            "Pending after final continuation:",
            final_pending,
        )


        # ====================================================
        # CHECKS
        # ====================================================

        checks = []


        checks.append(
            (
                "candidate rule preserves validated "
                "pre-c3 GID count",

                pre_c3_gids
                == EXPECTED_PRE_C3_GIDS,
            )
        )


        checks.append(
            (
                "candidate rule preserves validated "
                "pre-c3 pending count",

                pre_c3_pending
                == EXPECTED_PRE_C3_PENDING,
            )
        )


        checks.append(
            (
                "candidate rule preserves validated "
                "pre-c3 anchored count",

                pre_c3_anchored
                == EXPECTED_PRE_C3_ANCHORED,
            )
        )


        checks.append(
            (
                "c3 production population remains 95",

                len(
                    population
                )
                == EXPECTED_C3_POPULATION,
            )
        )


        checks.append(
            (
                "no synthetic c3 collisions",

                len(
                    collisions
                )
                == 0,
            )
        )


        checks.append(
            (
                "no pre-c3 assigned member moves GID "
                "after adding c3",

                len(
                    gid_moves
                )
                == 0,
            )
        )


        checks.append(
            (
                "candidate rule blocks c3:12 and c3:15 "
                "from sharing one GID",

                targets[
                    "target_safe"
                ],
            )
        )


        checks.append(
            (
                "c3:104a and c3:104b remain separated",

                not (
                    targets[
                        "gid_104a"
                    ]
                    is not None

                    and

                    targets[
                        "gid_104b"
                    ]
                    is not None

                    and

                    targets[
                        "gid_104a"
                    ]
                    == targets[
                        "gid_104b"
                    ]
                ),
            )
        )


        checks.append(
            (
                "c3:104b still resolves to GID 11",

                targets[
                    "gid_104b"
                ]
                == EXPECTED_C3_104_AFTER_GID,
            )
        )


        checks.append(
            (
                "c3:104a does not resolve to GID 11",

                targets[
                    "gid_104a"
                ]
                != EXPECTED_C3_104_AFTER_GID,
            )
        )


        checks.append(
            (
                "no candidate-rule same-camera conflict "
                "remains inside any final GID",

                len(
                    candidate_conflicts_inside
                )
                == 0,
            )
        )


        all_passed = print_checks(
            checks
        )


        print()
        print(
            "CANDIDATE-RULE EXPERIMENT RESULT"
        )

        print(
            "================================"
        )


        if all_passed:

            print(
                "PASS"
            )

        else:

            print(
                "FAIL - do not modify production policy"
            )


    finally:

        identity_compatibility.same_camera_conflict = (
            original_same_camera_conflict
        )


        print()
        print(
            "Original production same-camera conflict "
            "function restored."
        )


    print()
    print(
        "No production source file was modified."
    )


if __name__ == "__main__":
    main()
