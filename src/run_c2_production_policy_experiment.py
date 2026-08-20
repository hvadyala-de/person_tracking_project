from collections import Counter

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.run_c2_identity_aware_split_experiment import (
    build_split_population,
    discover_gap_evidence,
    select_split_proposals,
)


# ============================================================
# PURPOSE
#
# Start from the validated final c0+c1 production state.
#
# Then:
#
#   1. load useful c2 tracklets
#   2. run the already-validated identity-aware split proposal
#   3. build the split c2 population in memory
#   4. feed those c2 items through the REAL production
#      GlobalIdentityManager.assign_tracklet()
#   5. use the SAME ordinary pending reevaluation used by
#      run_global_id_offline.py
#   6. run the SAME final continuation cascade once
#   7. audit final c2 membership and trust
#
# IMPORTANT:
#
# No production threshold is changed here.
# No CSV is changed.
# No embedding file is changed.
# No production source file is changed.
# ============================================================


# ============================================================
# BASIC HELPERS
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
# BUILD CURRENT IDENTITY MEMBER MAP
#
# Returns:
#
#   (camera_id, local_track_id) -> gid
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
# FIND ONE MEMBER'S CURRENT GID
# ============================================================

def find_member_gid(
    manager,
    key,
):

    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            if (
                member_key(
                    member
                )
                == key
            ):

                return gid

    return None


# ============================================================
# GET TRUST SAFELY
# ============================================================

def get_member_trust_safe(
    manager,
    gid,
    key,
):

    if gid is None:

        return None

    camera_id, local_track_id = key

    try:

        return manager.get_member_trust(
            gid,
            camera_id,
            local_track_id,
        )

    except Exception:

        return None


# ============================================================
# EXTRACT PENDING KEYS
#
# The production manager owns the exact pending structure.
#
# This diagnostic intentionally handles several possible
# container/object layouts so that we do not make the
# experiment depend on internal representation details.
# ============================================================

def extract_key_from_object(
    value,
):

    if value is None:

        return None


    # --------------------------------------------------------
    # Direct object with camera_id / local_track_id
    # --------------------------------------------------------

    if (
        hasattr(
            value,
            "camera_id",
        )
        and
        hasattr(
            value,
            "local_track_id",
        )
    ):

        try:

            return (
                normalize_camera_id(
                    value.camera_id
                ),
                int(
                    value.local_track_id
                ),
            )

        except Exception:

            pass


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        camera_id = value.get(
            "camera_id"
        )

        local_track_id = value.get(
            "local_track_id"
        )

        if (
            camera_id is not None
            and
            local_track_id is not None
        ):

            try:

                return (
                    normalize_camera_id(
                        camera_id
                    ),
                    int(
                        local_track_id
                    ),
                )

            except Exception:

                pass


        for nested_key in [
            "tracklet",
            "candidate_tracklet",
        ]:

            if nested_key in value:

                nested = (
                    extract_key_from_object(
                        value[
                            nested_key
                        ]
                    )
                )

                if nested is not None:

                    return nested


    # --------------------------------------------------------
    # Object containing a tracklet
    # --------------------------------------------------------

    for attribute in [
        "tracklet",
        "candidate_tracklet",
    ]:

        if hasattr(
            value,
            attribute,
        ):

            nested = (
                extract_key_from_object(
                    getattr(
                        value,
                        attribute,
                    )
                )
            )

            if nested is not None:

                return nested


    return None


def pending_keys(
    manager,
):

    result = set()

    pending = (
        manager.pending_tracklets
    )


    # --------------------------------------------------------
    # Dictionary-style pending storage
    # --------------------------------------------------------

    if isinstance(
        pending,
        dict,
    ):

        for key, value in (
            pending.items()
        ):

            if (
                isinstance(
                    key,
                    tuple,
                )
                and
                len(
                    key
                )
                >= 2
            ):

                try:

                    result.add(
                        (
                            normalize_camera_id(
                                key[0]
                            ),
                            int(
                                key[1]
                            ),
                        )
                    )

                except Exception:

                    pass


            extracted = (
                extract_key_from_object(
                    value
                )
            )

            if extracted is not None:

                result.add(
                    extracted
                )


        return result


    # --------------------------------------------------------
    # Sequence-style pending storage
    # --------------------------------------------------------

    try:

        iterator = iter(
            pending
        )

    except TypeError:

        return result


    for value in iterator:

        extracted = (
            extract_key_from_object(
                value
            )
        )

        if extracted is not None:

            result.add(
                extracted
            )

    return result


# ============================================================
# READ ASSIGNMENT RESULT FIELDS SAFELY
# ============================================================

def get_result_field(
    result,
    names,
    default=None,
):

    for name in names:

        if hasattr(
            result,
            name,
        ):

            return getattr(
                result,
                name,
            )

    return default


def result_reason(
    result,
):

    value = get_result_field(
        result,
        [
            "reason",
            "decision_reason",
            "match_reason",
        ],
    )

    if value is None:

        return "-"

    return str(
        value
    )


# ============================================================
# SNAPSHOT EXISTING c0+c1 MEMBERS
# ============================================================

def snapshot_existing_members(
    manager,
):

    gid_map = (
        build_member_gid_map(
            manager
        )
    )

    snapshot = {}

    for key, gid in gid_map.items():

        snapshot[
            key
        ] = {
            "gid":
                gid,

            "trust":
                get_member_trust_safe(
                    manager,
                    gid,
                    key,
                ),
        }

    return snapshot


# ============================================================
# CHECK THAT ORIGINAL ASSIGNED MEMBERS DID NOT MOVE
# ============================================================

def audit_existing_members(
    manager,
    baseline_snapshot,
):

    changed = []

    current_map = (
        build_member_gid_map(
            manager
        )
    )

    for key, old in (
        baseline_snapshot.items()
    ):

        current_gid = (
            current_map.get(
                key
            )
        )

        current_trust = (
            get_member_trust_safe(
                manager,
                current_gid,
                key,
            )
        )


        if (
            current_gid
            != old[
                "gid"
            ]
            or
            current_trust
            != old[
                "trust"
            ]
        ):

            changed.append(
                {
                    "key":
                        key,

                    "old_gid":
                        old[
                            "gid"
                        ],

                    "new_gid":
                        current_gid,

                    "old_trust":
                        old[
                            "trust"
                        ],

                    "new_trust":
                        current_trust,
                }
            )


    return changed


# ============================================================
# REGISTER IN-MEMORY c2 SEGMENTS IN LOOKUP
#
# This is necessary because once a c2 fragment becomes a
# member, later manager logic must be able to retrieve its
# Tracklet object.
# ============================================================

def register_population_tracklets(
    population,
    tracklet_lookup,
):

    collisions = []

    for item in population:

        tracklet = item[
            "tracklet"
        ]

        key = tracklet_key(
            tracklet
        )

        existing = (
            tracklet_lookup.get(
                key
            )
        )


        # Synthetic split IDs should not collide.
        #
        # WHOLE tracklets already exist in c2_lookup and are
        # therefore expected to have the exact same object/key.
        if (
            existing is not None
            and
            item[
                "kind"
            ]
            != "WHOLE"
        ):

            collisions.append(
                (
                    item[
                        "label"
                    ],
                    key,
                )
            )


        tracklet_lookup[
            key
        ] = tracklet


    return collisions


# ============================================================
# PROCESS c2 THROUGH ACTUAL PRODUCTION MANAGER
# ============================================================

def process_c2_population(
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

        camera_id = (
            normalize_camera_id(
                tracklet.camera_id
            )
        )

        local_track_id = (
            tracklet.local_track_id
        )

        key = (
            camera_id,
            local_track_id,
        )


        # ====================================================
        # EXACT PRODUCTION ASSIGNMENT CALL
        #
        # Same call shape as run_global_id_offline.py.
        # ====================================================

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


        immediate_gid = (
            find_member_gid(
                manager,
                key,
            )
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
        # EXACT ORDINARY PRODUCTION REEVALUATION
        #
        # Same-camera continuation remains disabled during
        # chronological construction.
        # ====================================================

        if (
            merged
            or
            created_new
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
            or
            index
            == len(
                population
            )
        ):

            print(
                f"Processed c2 "
                f"{index}/"
                f"{len(population)}"
                f" | GIDs="
                f"{len(manager.identities)}"
                f" | pending="
                f"{len(manager.pending_tracklets)}"
            )


    return records


# ============================================================
# FINAL PRODUCTION CONTINUATION PASS
# ============================================================

def run_final_pending_pass(
    manager,
    tracklet_lookup,
):

    return manager.reevaluate_pending(
        tracklet_lookup=
            tracklet_lookup,

        include_same_camera=
            True,

        include_dual_evidence=
            True,

        include_core_gallery=
            True,
    )


# ============================================================
# FINALIZE c2 RECORDS
# ============================================================

def finalize_records(
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
                ] = (
                    "EXISTING"
                )

            else:

                record[
                    "gid_origin"
                ] = (
                    "NEW"
                )

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
# PRINT SELECTED SPLITS
# ============================================================

def print_selected_splits(
    selected_splits,
):

    print()
    print(
        "IDENTITY-AWARE c2 SPLITS"
    )

    print(
        "========================"
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

        before_summary = row[
            "before_summary"
        ]

        after_summary = row[
            "after_summary"
        ]


        print()

        print(
            f"c2:{track_id}"
            f" | "
            f"{row['before_frame']}"
            f"->{row['after_frame']}"
            f" | gap="
            f"{row['gap']}"
            f" | "
            f"{row['classification']}"
        )


        if (
            before_summary[
                "status"
            ]
            == "UNIQUE"
        ):

            print(
                "  diagnostic BEFORE:"
                f" GID "
                f"{before_summary['gid']}"
            )

        else:

            print(
                "  diagnostic BEFORE:",
                before_summary[
                    "status"
                ],
            )


        if (
            after_summary[
                "status"
            ]
            == "UNIQUE"
        ):

            print(
                "  diagnostic AFTER :"
                f" GID "
                f"{after_summary['gid']}"
            )

        else:

            print(
                "  diagnostic AFTER :",
                after_summary[
                    "status"
                ],
            )


# ============================================================
# PRINT INITIAL PRODUCTION STATUS COUNTS
# ============================================================

def print_initial_status_counts(
    records,
):

    counts = Counter(
        str(
            record[
                "initial_status"
            ]
        )

        for record in records
    )


    print()
    print(
        "INITIAL PRODUCTION ASSIGNMENT STATUS"
    )

    print(
        "===================================="
    )


    for status, count in sorted(
        counts.items()
    ):

        print(
            f"{status:<20}: "
            f"{count}"
        )


# ============================================================
# PRINT FINAL STATE COUNTS
# ============================================================

def print_final_state_counts(
    records,
):

    state_counts = Counter(
        str(
            record[
                "final_state"
            ]
        )

        for record in records
    )

    origin_counts = Counter(
        record[
            "gid_origin"
        ]

        for record in records

        if record[
            "final_gid"
        ]
        is not None
    )


    print()
    print(
        "FINAL c2 MEMBER STATE COUNTS"
    )

    print(
        "============================"
    )


    for state, count in sorted(
        state_counts.items()
    ):

        print(
            f"{state:<20}: "
            f"{count}"
        )


    print()
    print(
        "FINAL c2 GID ORIGIN"
    )

    print(
        "==================="
    )


    print(
        "Assigned to existing c0+c1 GID:",
        origin_counts.get(
            "EXISTING",
            0,
        ),
    )

    print(
        "Assigned to new GID:",
        origin_counts.get(
            "NEW",
            0,
        ),
    )

    print(
        "Not assigned:",
        sum(
            1

            for record
            in records

            if record[
                "final_gid"
            ]
            is None
        ),
    )


# ============================================================
# PRINT EXISTING-GID c2 ASSOCIATIONS
# ============================================================

def print_existing_gid_associations(
    records,
):

    rows = [
        record

        for record
        in records

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
        "c2 -> EXISTING c0+c1 GID"
    )

    print(
        "========================"
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
            f" | frames="
            f"{record['start_frame']}"
            f"-"
            f"{record['end_frame']}"
        )


# ============================================================
# PRINT NEW c2 GIDS
# ============================================================

def print_new_gid_members(
    records,
):

    rows = [
        record

        for record
        in records

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
        "c2 MEMBERS ASSIGNED TO NEW GIDS"
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
            f" -> NEW GID "
            f"{record['final_gid']:<3}"
            f" | trust="
            f"{record['final_trust']}"
            f" | initial="
            f"{record['initial_status']}"
        )


# ============================================================
# PRINT PENDING c2
# ============================================================

def print_pending_records(
    records,
):

    rows = [
        record

        for record
        in records

        if (
            record[
                "final_gid"
            ]
            is None

            and

            record[
                "is_pending"
            ]
        )
    ]


    print()
    print(
        "FINAL PENDING c2 SEGMENTS"
    )

    print(
        "========================="
    )


    print(
        "Count:",
        len(
            rows
        ),
    )


    for record in rows:

        print(
            f"{record['label']:<12}"
            f" | initial="
            f"{record['initial_status']}"
            f" | frames="
            f"{record['start_frame']}"
            f"-"
            f"{record['end_frame']}"
        )


# ============================================================
# IMPORTANT KNOWN c2 CANDIDATES
# ============================================================

def print_key_candidate_audit(
    records,
):

    by_label = {
        record[
            "label"
        ]:
            record

        for record
        in records
    }


    labels = [
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
        "c2:701",
        "c2:635",
    ]


    print()
    print(
        "KEY c2 CANDIDATE PRODUCTION AUDIT"
    )

    print(
        "================================="
    )


    for label in labels:

        record = by_label.get(
            label
        )

        if record is None:

            print(
                f"{label:<12}"
                " | NOT PRESENT"
            )

            continue


        print(
            f"{label:<12}"
            f" | initial="
            f"{record['initial_status']}"
            f" | final_gid="
            f"{record['final_gid']}"
            f" | trust="
            f"{record['final_trust']}"
            f" | origin="
            f"{record['gid_origin']}"
            f" | pending="
            f"{record['is_pending']}"
            f" | reason="
            f"{record['reason']}"
        )


# ============================================================
# FINAL CONTINUATION REASON COUNTS
# ============================================================

def print_final_pending_results(
    final_results,
):

    print()
    print(
        "FINAL CONTINUATION PASS"
    )

    print(
        "======================="
    )


    print(
        "Result rows:",
        len(
            final_results
        ),
    )


    reason_counts = Counter()


    for result in final_results:

        if isinstance(
            result,
            dict,
        ):

            reason = result.get(
                "reason",
                "UNKNOWN",
            )

        else:

            reason = getattr(
                result,
                "reason",
                "UNKNOWN",
            )


        reason_counts[
            str(
                reason
            )
        ] += 1


    if not reason_counts:

        print(
            "No continuation assignments."
        )

        return


    for reason, count in sorted(
        reason_counts.items()
    ):

        print(
            f"{reason:<40}: "
            f"{count}"
        )


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    selected_splits,
    population,
    collisions,
    existing_member_changes,
):

    checks = []


    # --------------------------------------------------------
    # Population size
    # --------------------------------------------------------

    checks.append(
        (
            "identity-aware c2 population has 107 items",

            len(
                population
            )
            == 107,
        )
    )


    # --------------------------------------------------------
    # c2:9
    # --------------------------------------------------------

    c2_9 = selected_splits.get(
        9
    )

    checks.append(
        (
            "c2:9 split is 1321->1357",

            c2_9 is not None

            and

            c2_9[
                "before_frame"
            ]
            == 1321

            and

            c2_9[
                "after_frame"
            ]
            == 1357,
        )
    )


    # --------------------------------------------------------
    # c2:601
    # --------------------------------------------------------

    c2_601 = selected_splits.get(
        601
    )

    checks.append(
        (
            "c2:601 split is 4016->4049",

            c2_601 is not None

            and

            c2_601[
                "before_frame"
            ]
            == 4016

            and

            c2_601[
                "after_frame"
            ]
            == 4049,
        )
    )


    # --------------------------------------------------------
    # c2:635 not split
    # --------------------------------------------------------

    checks.append(
        (
            "c2:635 is not automatically split",

            635
            not in selected_splits,
        )
    )


    # --------------------------------------------------------
    # c2:508 not split
    # --------------------------------------------------------

    checks.append(
        (
            "c2:508 remains whole",

            508
            not in selected_splits,
        )
    )


    # --------------------------------------------------------
    # Synthetic key collisions
    # --------------------------------------------------------

    checks.append(
        (
            "no synthetic split-ID collisions",

            len(
                collisions
            )
            == 0,
        )
    )


    # --------------------------------------------------------
    # Existing assigned c0+c1 members must never move.
    # --------------------------------------------------------

    checks.append(
        (
            "existing assigned c0+c1 members unchanged",

            len(
                existing_member_changes
            )
            == 0,
        )
    )


    print()
    print(
        "SANITY CHECKS"
    )

    print(
        "============="
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
        "c2 PRODUCTION POLICY EXPERIMENT"
    )

    print(
        "==============================="
    )


    # ========================================================
    # REPLAY VALIDATED FINAL c0+c1 STATE
    # ========================================================

    (
        manager,
        baseline_tracklets,
        tracklet_lookup,
        _,
    ) = replay_final_production()


    baseline_gid_ids = set(
        manager.identities.keys()
    )

    baseline_member_snapshot = (
        snapshot_existing_members(
            manager
        )
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
        "Anchored GIDs:",
        baseline_anchored,
    )

    print(
        "Assigned members:",
        len(
            baseline_member_snapshot
        ),
    )


    # ========================================================
    # LOAD c2
    # ========================================================

    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()


    tracklet_lookup.update(
        c2_lookup
    )


    print()
    print(
        "c2 INPUT"
    )

    print(
        "========"
    )

    print(
        "Useful tracklets:",
        len(
            c2_rows
        ),
    )


    # ========================================================
    # DISCOVER IDENTITY-AWARE SPLITS
    #
    # Same split policy validated by the previous experiment.
    # ========================================================

    print()
    print(
        "DISCOVERING IDENTITY-AWARE SPLITS..."
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


    print_selected_splits(
        selected_splits
    )


    print()
    print(
        "Conflicting split-proposal tracklets:",
        len(
            rejected_ambiguous
        ),
    )


    # ========================================================
    # BUILD IN-MEMORY c2 POPULATION
    # ========================================================

    population = build_split_population(
        c2_rows=
            c2_rows,

        selected_splits=
            selected_splits,
    )


    print()
    print(
        "IN-MEMORY c2 POPULATION"
    )

    print(
        "======================="
    )

    print(
        "Original useful tracklets:",
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
        "Production input segments:",
        len(
            population
        ),
    )


    # ========================================================
    # REGISTER SPLIT TRACKLETS
    # ========================================================

    collisions = (
        register_population_tracklets(
            population=
                population,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    # ========================================================
    # ACTUAL PRODUCTION ASSIGNMENT
    # ========================================================

    print()
    print(
        "PROCESSING c2 WITH UNCHANGED "
        "PRODUCTION MANAGER..."
    )


    records = process_c2_population(
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
        "Anchored GIDs:",
        pre_final_anchored,
    )


    # ========================================================
    # EXACT FINAL PRODUCTION CASCADE
    # ========================================================

    final_results = (
        run_final_pending_pass(
            manager=
                manager,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    print_final_pending_results(
        final_results
    )


    # ========================================================
    # FINAL c2 STATE
    # ========================================================

    finalize_records(
        manager=
            manager,

        records=
            records,

        baseline_gid_ids=
            baseline_gid_ids,
    )


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


    print_key_candidate_audit(
        records
    )


    # ========================================================
    # REGRESSION AUDIT
    # ========================================================

    existing_member_changes = (
        audit_existing_members(
            manager=
                manager,

            baseline_snapshot=
                baseline_member_snapshot,
        )
    )


    print()
    print(
        "EXISTING c0+c1 MEMBER REGRESSION AUDIT"
    )

    print(
        "======================================"
    )


    if not existing_member_changes:

        print(
            "PASS - no existing assigned "
            "member changed GID/trust"
        )

    else:

        for row in (
            existing_member_changes
        ):

            camera_id, local_track_id = (
                row[
                    "key"
                ]
            )

            print(
                f"c{camera_id}:"
                f"{local_track_id}"
                f" | GID "
                f"{row['old_gid']}"
                f" -> "
                f"{row['new_gid']}"
                f" | trust "
                f"{row['old_trust']}"
                f" -> "
                f"{row['new_trust']}"
            )


    sanity = run_sanity_checks(
        selected_splits=
            selected_splits,

        population=
            population,

        collisions=
            collisions,

        existing_member_changes=
            existing_member_changes,
    )


    # ========================================================
    # FINAL SUMMARY
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


    existing_c2 = sum(
        1

        for record
        in records

        if record[
            "gid_origin"
        ]
        == "EXISTING"
    )


    new_c2 = sum(
        1

        for record
        in records

        if record[
            "gid_origin"
        ]
        == "NEW"
    )


    pending_c2 = sum(
        1

        for record
        in records

        if record[
            "final_gid"
        ]
        is None

        and

        record[
            "is_pending"
        ]
    )


    unassigned_c2 = sum(
        1

        for record
        in records

        if record[
            "final_gid"
        ]
        is None

        and

        not record[
            "is_pending"
        ]
    )


    print()
    print(
        "FINAL PRODUCTION-POLICY SUMMARY"
    )

    print(
        "==============================="
    )


    print(
        "c0+c1 starting GIDs:",
        baseline_gids,
    )

    print(
        "Final GIDs:",
        final_gids,
    )

    print(
        "GID delta:",
        final_gids
        - baseline_gids,
    )


    print()

    print(
        "c0+c1 starting pending:",
        baseline_pending,
    )

    print(
        "Final global pending:",
        final_pending,
    )

    print(
        "Pending delta:",
        final_pending
        - baseline_pending,
    )


    print()

    print(
        "Starting anchored GIDs:",
        baseline_anchored,
    )

    print(
        "Final anchored GIDs:",
        final_anchored,
    )

    print(
        "Anchored delta:",
        final_anchored
        - baseline_anchored,
    )


    print()

    print(
        "c2 production input segments:",
        len(
            records
        ),
    )

    print(
        "c2 assigned to existing GID:",
        existing_c2,
    )

    print(
        "c2 assigned to new GID:",
        new_c2,
    )

    print(
        "c2 pending:",
        pending_c2,
    )

    print(
        "c2 unassigned:",
        unassigned_c2,
    )


    print()

    print(
        "Sanity:",
        (
            "PASS"
            if sanity
            else "FAIL / REVIEW REQUIRED"
        ),
    )


    print()
    print(
        "PRODUCTION THRESHOLDS MODIFIED: NO"
    )

    print(
        "PRODUCTION MANAGER SOURCE MODIFIED: NO"
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
        "All c2 additions existed in memory only."
    )


if __name__ == "__main__":
    main()
