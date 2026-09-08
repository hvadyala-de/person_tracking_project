from collections import Counter

import src.identity_compatibility as identity_compatibility

from src.experiments.analyze_c3_same_camera_conflict_thresholds import (
    analyze_pair,
)

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.experiments.run_c3_candidate_same_camera_conflict_experiment import (
    candidate_same_camera_conflict,
)

from src.run_global_id_offline import (
    load_all_tracklets,
)

from src.terrace_geometry import (
    load_ground_homographies,
)


# ============================================================
# EXPECTED VALIDATED ORIGINAL STATE
# ============================================================

EXPECTED_ORIGINAL_GIDS = 76
EXPECTED_ORIGINAL_PENDING = 92
EXPECTED_ORIGINAL_ANCHORED = 22


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
# FIND MEMBER GID
# ============================================================

def find_member_gid(
    manager,
    key,
):

    for gid, identity in manager.identities.items():

        for member in identity.members:

            if member_key(
                member
            ) == key:

                return gid


    return None


# ============================================================
# IDENTITY MEMBER SET FOR EVERY ASSIGNED MEMBER
# ============================================================

def build_identity_member_sets(
    manager,
):

    result = {}


    for gid, identity in manager.identities.items():

        members = frozenset(
            member_key(
                member
            )

            for member in identity.members
        )


        for key in members:

            result[
                key
            ] = {
                "gid":
                    gid,

                "members":
                    members,
            }


    return result


# ============================================================
# SAFE RESULT FIELD
# ============================================================

def result_field(
    result,
    name,
    default=None,
):

    if result is None:

        return default


    if isinstance(
        result,
        dict,
    ):

        return result.get(
            name,
            default,
        )


    return getattr(
        result,
        name,
        default,
    )


# ============================================================
# TRACKLET KEY
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


# ============================================================
# FORMAT KEY
# ============================================================

def format_key(
    key,
):

    return (
        f"c{key[0]}:"
        f"{key[1]}"
    )


# ============================================================
# CANDIDATE-RULE TRACE WRAPPER
#
# Record only cases where:
#
#   original rule  -> False
#   candidate rule -> True
#
# These are the NEW vetoes introduced by the candidate rule.
# ============================================================

class CandidateConflictTracer:

    def __init__(
        self,
        original_function,
    ):

        self.original_function = (
            original_function
        )

        self.rows = []


    def __call__(
        self,
        candidate_tracklet,
        existing_tracklet,
    ):

        original_conflict = (
            self.original_function(
                candidate_tracklet,
                existing_tracklet,
            )
        )


        candidate_conflict = (
            candidate_same_camera_conflict(
                candidate_tracklet,
                existing_tracklet,
            )
        )


        if (
            candidate_conflict

            and

            not original_conflict
        ):

            candidate_key = tracklet_key(
                candidate_tracklet
            )

            existing_key = tracklet_key(
                existing_tracklet
            )


            pair = tuple(
                sorted(
                    (
                        candidate_key,
                        existing_key,
                    )
                )
            )


            self.rows.append(
                {
                    "candidate_key":
                        candidate_key,

                    "existing_key":
                        existing_key,

                    "pair":
                        pair,
                }
            )


        return candidate_conflict


# ============================================================
# REPLAY ONE POLICY
# ============================================================

def replay_policy(
    label,
):

    (
        tracklets,
        tracklet_lookup,
    ) = load_all_tracklets()


    tracklets.sort(
        key=lambda item: (
            item[
                0
            ].start_frame,

            normalize_camera_id(
                item[
                    0
                ].camera_id
            ),

            item[
                0
            ].local_track_id,
        )
    )


    homographies = (
        load_ground_homographies()
    )


    manager = GlobalIdentityManager(
        homographies=homographies
    )


    events = []


    for index, (
        tracklet,
        embedding,
    ) in enumerate(
        tracklets,
        start=1,
    ):

        key = tracklet_key(
            tracklet
        )


        result = manager.assign_tracklet(
            camera_id=
                key[
                    0
                ],

            local_track_id=
                key[
                    1
                ],

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


        if (
            result_field(
                result,
                "merged",
                False,
            )

            or

            result_field(
                result,
                "created_new",
                False,
            )
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


        assigned_gid = find_member_gid(
            manager,
            key,
        )


        events.append(
            {
                "index":
                    index,

                "key":
                    key,

                "start_frame":
                    tracklet.start_frame,

                "end_frame":
                    tracklet.end_frame,

                "status":
                    result_field(
                        result,
                        "status",
                    ),

                "reason":
                    result_field(
                        result,
                        "reason",
                    ),

                "merged":
                    bool(
                        result_field(
                            result,
                            "merged",
                            False,
                        )
                    ),

                "created_new":
                    bool(
                        result_field(
                            result,
                            "created_new",
                            False,
                        )
                    ),

                "result_gid":
                    result_field(
                        result,
                        "global_id",
                    ),

                "assigned_gid":
                    assigned_gid,

                "gid_count":
                    len(
                        manager.identities
                    ),

                "pending_count":
                    len(
                        manager.pending_tracklets
                    ),

                "anchored_count":
                    len(
                        manager.anchored_gids
                    ),
            }
        )


    pre_final = {
        "gids":
            len(
                manager.identities
            ),

        "pending":
            len(
                manager.pending_tracklets
            ),

        "anchored":
            len(
                manager.anchored_gids
            ),
    }


    final_results = manager.reevaluate_pending(
        tracklet_lookup=
            tracklet_lookup,

        include_same_camera=
            True,

        include_dual_evidence=
            True,

        include_core_gallery=
            True,
    )


    final_state = {
        "gids":
            len(
                manager.identities
            ),

        "pending":
            len(
                manager.pending_tracklets
            ),

        "anchored":
            len(
                manager.anchored_gids
            ),
    }


    print()
    print(
        label
    )

    print(
        "="
        * len(
            label
        )
    )

    print(
        "Tracklets:",
        len(
            tracklets
        ),
    )

    print(
        "Before final pass:",
        pre_final,
    )

    print(
        "Final:",
        final_state,
    )


    return {
        "manager":
            manager,

        "tracklets":
            tracklets,

        "tracklet_lookup":
            tracklet_lookup,

        "events":
            events,

        "pre_final":
            pre_final,

        "final_results":
            final_results,

        "final_state":
            final_state,

        "member_sets":
            build_identity_member_sets(
                manager
            ),
    }


# ============================================================
# FIRST CHRONOLOGICAL DIVERGENCE
# ============================================================

def find_first_event_divergence(
    original_events,
    candidate_events,
):

    comparable_fields = (
        "status",
        "reason",
        "merged",
        "created_new",
        "result_gid",
        "assigned_gid",
        "gid_count",
        "pending_count",
        "anchored_count",
    )


    for original, candidate in zip(
        original_events,
        candidate_events,
    ):

        if (
            original[
                "key"
            ]
            != candidate[
                "key"
            ]
        ):

            return (
                original,
                candidate,
            )


        for field in comparable_fields:

            if (
                original[
                    field
                ]
                != candidate[
                    field
                ]
            ):

                return (
                    original,
                    candidate,
                )


    return None


# ============================================================
# PRINT EVENT
# ============================================================

def print_event(
    prefix,
    event,
):

    print(
        f"{prefix}: "
        f"#{event['index']} "
        f"{format_key(event['key'])}"
        f" | frames="
        f"{event['start_frame']}"
        f".."
        f"{event['end_frame']}"
        f" | status="
        f"{event['status']}"
        f" | reason="
        f"{event['reason']}"
        f" | result_gid="
        f"{event['result_gid']}"
        f" | assigned_gid="
        f"{event['assigned_gid']}"
        f" | GIDs="
        f"{event['gid_count']}"
        f" | pending="
        f"{event['pending_count']}"
        f" | anchored="
        f"{event['anchored_count']}"
    )


# ============================================================
# FINAL IDENTITY DIFFERENCES
# ============================================================

def final_identity_differences(
    original,
    candidate,
):

    original_sets = original[
        "member_sets"
    ]

    candidate_sets = candidate[
        "member_sets"
    ]


    all_keys = sorted(
        set(
            original_sets
        )
        |
        set(
            candidate_sets
        )
    )


    differences = []


    for key in all_keys:

        original_row = original_sets.get(
            key
        )

        candidate_row = candidate_sets.get(
            key
        )


        original_members = (
            None

            if original_row is None

            else original_row[
                "members"
            ]
        )


        candidate_members = (
            None

            if candidate_row is None

            else candidate_row[
                "members"
            ]
        )


        if (
            original_members
            == candidate_members
        ):

            continue


        differences.append(
            {
                "key":
                    key,

                "original_gid":
                    (
                        None

                        if original_row is None

                        else original_row[
                            "gid"
                        ]
                    ),

                "candidate_gid":
                    (
                        None

                        if candidate_row is None

                        else candidate_row[
                            "gid"
                        ]
                    ),

                "original_members":
                    original_members,

                "candidate_members":
                    candidate_members,
            }
        )


    return differences


# ============================================================
# UNIQUE TRACE PAIRS
# ============================================================

def unique_trace_pairs(
    trace_rows,
):

    unique = {}


    for row in trace_rows:

        unique[
            row[
                "pair"
            ]
        ] = row


    return list(
        unique.values()
    )


# ============================================================
# REGRESSION-RELEVANT NEW VETOES
#
# A new candidate-rule conflict is especially important if
# the ORIGINAL validated replay ultimately placed the pair
# inside the same GID.
# ============================================================

def regression_relevant_vetoes(
    trace_rows,
    original,
    candidate,
):

    original_sets = original[
        "member_sets"
    ]

    candidate_sets = candidate[
        "member_sets"
    ]


    relevant = []


    for row in unique_trace_pairs(
        trace_rows
    ):

        key_a, key_b = row[
            "pair"
        ]


        original_a = original_sets.get(
            key_a
        )

        original_b = original_sets.get(
            key_b
        )


        if (
            original_a is None
            or original_b is None
        ):

            continue


        originally_same_identity = (
            original_a[
                "members"
            ]
            ==
            original_b[
                "members"
            ]
        )


        if not originally_same_identity:

            continue


        candidate_a = candidate_sets.get(
            key_a
        )

        candidate_b = candidate_sets.get(
            key_b
        )


        candidate_same_identity = (
            candidate_a is not None

            and

            candidate_b is not None

            and

            candidate_a[
                "members"
            ]
            ==
            candidate_b[
                "members"
            ]
        )


        relevant.append(
            {
                "key_a":
                    key_a,

                "key_b":
                    key_b,

                "original_gid":
                    original_a[
                        "gid"
                    ],

                "candidate_same_identity":
                    candidate_same_identity,

                "candidate_gid_a":
                    (
                        None

                        if candidate_a is None

                        else candidate_a[
                            "gid"
                        ]
                    ),

                "candidate_gid_b":
                    (
                        None

                        if candidate_b is None

                        else candidate_b[
                            "gid"
                        ]
                    ),
            }
        )


    return relevant


# ============================================================
# TRACKLET FROM LOOKUP
# ============================================================

def get_tracklet(
    lookup,
    key,
):

    tracklet = lookup.get(
        key
    )


    if tracklet is not None:

        return tracklet


    return lookup.get(
        (
            f"c{key[0]}",
            key[1],
        )
    )


# ============================================================
# PRINT PAIR SPATIAL ANALYSIS
# ============================================================

def print_pair_analysis(
    row,
    tracklet_lookup,
):

    key_a = row[
        "key_a"
    ]

    key_b = row[
        "key_b"
    ]


    tracklet_a = get_tracklet(
        tracklet_lookup,
        key_a,
    )

    tracklet_b = get_tracklet(
        tracklet_lookup,
        key_b,
    )


    print()
    print(
        f"{format_key(key_a)}"
        f" <-> "
        f"{format_key(key_b)}"
    )


    print(
        f"Original GID: "
        f"{row['original_gid']}"
    )

    print(
        "Candidate final GIDs:",
        row[
            "candidate_gid_a"
        ],
        "/",
        row[
            "candidate_gid_b"
        ],
    )


    if (
        tracklet_a is None
        or tracklet_b is None
    ):

        print(
            "Tracklet lookup unavailable."
        )

        return


    analysis = analyze_pair(
        tracklet_a,
        tracklet_b,
    )


    if analysis is None:

        print(
            "Fewer than 10 simultaneous detections."
        )

        return


    print(
        "Shared detections:",
        analysis[
            "shared"
        ],
    )

    print(
        "Shared range:",
        analysis[
            "first_shared"
        ],
        "->",
        analysis[
            "last_shared"
        ],
    )

    print(
        "Current conflict:",
        analysis[
            "current_conflict"
        ],
    )

    print(
        "Candidate conflict:",
        analysis[
            "proposed_conflict"
        ],
    )

    print(
        "Candidate separated:",
        f"{analysis['proposed_separated']}"
        f"/"
        f"{analysis['shared']}"
        f" "
        f"({analysis['proposed_ratio']:.3f})",
    )

    print(
        "IoU median/max:",
        f"{analysis['iou_median']:.4f}",
        "/",
        f"{analysis['iou_max']:.4f}",
    )

    print(
        "Center median:",
        f"{analysis['center_median']:.2f}",
        "px",
        f"normalized="
        f"{analysis['normalized_center_median']:.3f}",
    )

    print(
        "Bottom median:",
        f"{analysis['bottom_median']:.2f}",
        "px",
        f"normalized="
        f"{analysis['normalized_bottom_median']:.3f}",
    )


# ============================================================
# FINAL RESULT REASON COUNTS
# ============================================================

def final_result_reason_counts(
    results,
):

    return Counter(
        row.get(
            "reason",
            "UNKNOWN",
        )

        for row in results

        if row.get(
            "status"
        )
        != "PENDING"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c0+c1 CANDIDATE CONFLICT REGRESSION DIAGNOSTIC"
    )

    print(
        "============================================="
    )


    original_function = (
        identity_compatibility.same_camera_conflict
    )


    # ========================================================
    # ORIGINAL REPLAY
    # ========================================================

    identity_compatibility.same_camera_conflict = (
        original_function
    )


    original = replay_policy(
        "ORIGINAL PRODUCTION RULE"
    )


    # ========================================================
    # ORIGINAL REGRESSION GUARD
    # ========================================================

    original_ok = (
        original[
            "final_state"
        ][
            "gids"
        ]
        == EXPECTED_ORIGINAL_GIDS

        and

        original[
            "final_state"
        ][
            "pending"
        ]
        == EXPECTED_ORIGINAL_PENDING

        and

        original[
            "final_state"
        ][
            "anchored"
        ]
        == EXPECTED_ORIGINAL_ANCHORED
    )


    print()
    print(
        "ORIGINAL REFERENCE GUARD"
    )

    print(
        "========================"
    )

    print(
        "PASS"
        if original_ok
        else "FAIL"
    )


    if not original_ok:

        raise RuntimeError(
            "Original c0+c1 production replay no longer "
            "matches validated 76 / 92 / 22 state"
        )


    # ========================================================
    # CANDIDATE REPLAY WITH TRACE
    # ========================================================

    tracer = CandidateConflictTracer(
        original_function
    )


    identity_compatibility.same_camera_conflict = (
        tracer
    )


    try:

        candidate = replay_policy(
            "CANDIDATE SAME-CAMERA RULE"
        )

    finally:

        identity_compatibility.same_camera_conflict = (
            original_function
        )


    # ========================================================
    # FIRST CHRONOLOGICAL DIVERGENCE
    # ========================================================

    divergence = find_first_event_divergence(
        original[
            "events"
        ],
        candidate[
            "events"
        ],
    )


    print()
    print(
        "FIRST CHRONOLOGICAL DIVERGENCE"
    )

    print(
        "=============================="
    )


    if divergence is None:

        print(
            "NONE during chronological assignment."
        )

        print(
            "The regression is introduced during "
            "the final continuation pass."
        )

    else:

        original_event, candidate_event = (
            divergence
        )


        print_event(
            "ORIGINAL ",
            original_event,
        )

        print_event(
            "CANDIDATE",
            candidate_event,
        )


    # ========================================================
    # FINAL MEMBER DIFFERENCES
    # ========================================================

    differences = final_identity_differences(
        original,
        candidate,
    )


    print()
    print(
        "FINAL IDENTITY MEMBERSHIP DIFFERENCES"
    )

    print(
        "====================================="
    )

    print(
        "Members with changed identity membership:",
        len(
            differences
        ),
    )


    for row in differences[:30]:

        print()
        print(
            format_key(
                row[
                    "key"
                ]
            )
        )

        print(
            "  original GID:",
            row[
                "original_gid"
            ],
        )

        print(
            "  candidate GID:",
            row[
                "candidate_gid"
            ],
        )

        print(
            "  original members:",
            sorted(
                row[
                    "original_members"
                ]
            )
            if row[
                "original_members"
            ]
            is not None
            else None,
        )

        print(
            "  candidate members:",
            sorted(
                row[
                    "candidate_members"
                ]
            )
            if row[
                "candidate_members"
            ]
            is not None
            else None,
        )


    # ========================================================
    # NEW VETO TRACE
    # ========================================================

    unique_vetoes = unique_trace_pairs(
        tracer.rows
    )


    print()
    print(
        "CANDIDATE-RULE NEW VETO TRACE"
    )

    print(
        "============================="
    )

    print(
        "Unique newly vetoed pairs encountered:",
        len(
            unique_vetoes
        ),
    )


    # ========================================================
    # REGRESSION-RELEVANT VETOES
    # ========================================================

    relevant = regression_relevant_vetoes(
        trace_rows=
            tracer.rows,

        original=
            original,

        candidate=
            candidate,
    )


    print()
    print(
        "REGRESSION-RELEVANT NEW VETOES"
    )

    print(
        "=============================="
    )

    print(
        "Count:",
        len(
            relevant
        ),
    )


    if not relevant:

        print(
            "NONE"
        )


    for row in relevant:

        print_pair_analysis(
            row=
                row,

            tracklet_lookup=
                original[
                    "tracklet_lookup"
                ],
        )


    # ========================================================
    # FINAL-PASS RECOVERY DIFFERENCES
    # ========================================================

    original_reasons = (
        final_result_reason_counts(
            original[
                "final_results"
            ]
        )
    )

    candidate_reasons = (
        final_result_reason_counts(
            candidate[
                "final_results"
            ]
        )
    )


    print()
    print(
        "FINAL CONTINUATION RECOVERY COUNTS"
    )

    print(
        "=================================="
    )

    print(
        "Original:",
        dict(
            sorted(
                original_reasons.items()
            )
        ),
    )

    print(
        "Candidate:",
        dict(
            sorted(
                candidate_reasons.items()
            )
        ),
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "REGRESSION SUMMARY"
    )

    print(
        "=================="
    )

    print(
        "Original final:",
        original[
            "final_state"
        ],
    )

    print(
        "Candidate final:",
        candidate[
            "final_state"
        ],
    )

    print(
        "New veto pairs encountered:",
        len(
            unique_vetoes
        ),
    )

    print(
        "Regression-relevant veto pairs:",
        len(
            relevant
        ),
    )


    if relevant:

        print()
        print(
            "The pair(s) above were members of the same "
            "validated original identity but are blocked "
            "by the candidate rule."
        )


    print()
    print(
        "Original production conflict function restored."
    )

    print(
        "No production source file was modified."
    )


if __name__ == "__main__":
    main()
