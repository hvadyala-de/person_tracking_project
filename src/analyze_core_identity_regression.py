from collections import defaultdict

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.terrace_geometry import (
    load_ground_homographies,
)

from src.run_core_anchor_expansion_experiment import (
    load_data,
    cosine_similarity,
    track_key,
    find_strict_existing_candidates,
    find_relaxed_core_candidates,
    find_strict_pending_candidates,
    create_core_anchor_from_pending,
    sweep_pending_into_core_anchors,
    add_to_existing_gid,
    hold_pending,
    build_embedding_lookup,
    build_member_gid_lookup,
)


# ============================================================
# CONFIGURATION
# ============================================================

LOWEST_APPEARANCE_ROWS = 25


# ============================================================
# REPLAY EXACT CORE-ONLY POLICY
# ============================================================

def replay_policy():

    items, tracklet_lookup = (
        load_data()
    )


    embedding_lookup = (
        build_embedding_lookup(
            items
        )
    )


    homographies = (
        load_ground_homographies()
    )


    manager = GlobalIdentityManager(
        homographies=
            homographies
    )


    anchored_gids = set()


    core_members = defaultdict(
        set
    )


    counters = {
        "strict_existing": 0,
        "strict_pending": 0,
        "relaxed_live": 0,
        "relaxed_sweep": 0,
        "strong_held": 0,
        "pending_held": 0,
        "new_gid": 0,
        "ambiguous_strict_existing": 0,
        "ambiguous_relaxed": 0,
        "ambiguous_strict_pending": 0,
    }


    for tracklet, embedding in items:

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        track_id = (
            tracklet.local_track_id
        )


        key = (
            camera,
            track_id,
        )


        # ====================================================
        # FIRST TRACK
        # ====================================================

        if not manager.identities:

            manager.create_identity(
                camera_id=
                    camera,

                local_track_id=
                    track_id,

                start_frame=
                    tracklet.start_frame,

                end_frame=
                    tracklet.end_frame,

                embedding=
                    embedding,
            )


            counters[
                "new_gid"
            ] += 1


            continue


        # ====================================================
        # PRIORITY 1:
        # STRICT EXISTING GEOMETRY
        # ====================================================

        strict_candidates = (
            find_strict_existing_candidates(
                manager,
                tracklet,
                embedding,
                tracklet_lookup,
                embedding_lookup,
                homographies,
                anchored_gids,
                core_members,
            )
        )


        if len(
            strict_candidates
        ) == 1:

            candidate = (
                strict_candidates[0]
            )


            gid = candidate[
                "gid"
            ]


            # -----------------------------------------------
            # First strict geometry connection bootstraps
            # existing GID members into CORE.
            # -----------------------------------------------

            if gid not in anchored_gids:

                identity = (
                    manager.identities[
                        gid
                    ]
                )


                for member in identity.members:

                    core_members[
                        gid
                    ].add(
                        track_key(
                            member.camera_id,
                            member.local_track_id,
                        )
                    )


                anchored_gids.add(
                    gid
                )


            # -----------------------------------------------
            # Add current member.
            # -----------------------------------------------

            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )


            # Strict member becomes CORE.
            core_members[
                gid
            ].add(
                key
            )


            counters[
                "strict_existing"
            ] += 1


            swept = (
                sweep_pending_into_core_anchors(
                    manager,
                    tracklet_lookup,
                    embedding_lookup,
                    homographies,
                    anchored_gids,
                    core_members,
                )
            )


            counters[
                "relaxed_sweep"
            ] += len(
                swept
            )


            continue


        elif len(
            strict_candidates
        ) > 1:

            counters[
                "ambiguous_strict_existing"
            ] += 1


        # ====================================================
        # PRIORITY 2:
        # RELAXED CORE-SUPPORTED EXISTING GID
        # ====================================================

        relaxed_candidates = (
            find_relaxed_core_candidates(
                manager,
                tracklet,
                embedding,
                tracklet_lookup,
                embedding_lookup,
                homographies,
                anchored_gids,
                core_members,
            )
        )


        if len(
            relaxed_candidates
        ) == 1:

            candidate = (
                relaxed_candidates[0]
            )


            gid = candidate[
                "gid"
            ]


            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )


            # IMPORTANT:
            #
            # Relaxed member is deliberately NOT added
            # to core_members.

            counters[
                "relaxed_live"
            ] += 1


            continue


        elif len(
            relaxed_candidates
        ) > 1:

            counters[
                "ambiguous_relaxed"
            ] += 1


        # ====================================================
        # PRIORITY 3:
        # STRICT PENDING PAIR
        # ====================================================

        pending_candidates = (
            find_strict_pending_candidates(
                manager,
                tracklet,
                embedding,
                tracklet_lookup,
                homographies,
            )
        )


        if len(
            pending_candidates
        ) == 1:

            candidate = (
                pending_candidates[0]
            )


            create_core_anchor_from_pending(
                manager,
                tracklet,
                embedding,
                candidate,
                anchored_gids,
                core_members,
            )


            counters[
                "strict_pending"
            ] += 1


            swept = (
                sweep_pending_into_core_anchors(
                    manager,
                    tracklet_lookup,
                    embedding_lookup,
                    homographies,
                    anchored_gids,
                    core_members,
                )
            )


            counters[
                "relaxed_sweep"
            ] += len(
                swept
            )


            continue


        elif len(
            pending_candidates
        ) > 1:

            counters[
                "ambiguous_strict_pending"
            ] += 1


        # ====================================================
        # PRIORITY 4:
        # APPEARANCE ONLY
        #
        # Appearance NEVER modifies a GID.
        # ====================================================

        match = manager.evaluate(
            camera_id=
                camera,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )


        if match.status == "STRONG":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )


            counters[
                "strong_held"
            ] += 1


            continue


        if match.status == "PENDING":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )


            counters[
                "pending_held"
            ] += 1


            continue


        # ====================================================
        # CLEAR NEW SINGLETON
        # ====================================================

        manager.create_identity(
            camera_id=
                camera,

            local_track_id=
                track_id,

            start_frame=
                tracklet.start_frame,

            end_frame=
                tracklet.end_frame,

            embedding=
                embedding,
        )


        manager.pending_tracklets.pop(
            key,
            None,
        )


        counters[
            "new_gid"
        ] += 1


    # ========================================================
    # FINAL CORE-ONLY SWEEP
    # ========================================================

    swept = (
        sweep_pending_into_core_anchors(
            manager,
            tracklet_lookup,
            embedding_lookup,
            homographies,
            anchored_gids,
            core_members,
        )
    )


    counters[
        "relaxed_sweep"
    ] += len(
        swept
    )


    return (
        items,
        tracklet_lookup,
        embedding_lookup,
        homographies,
        manager,
        anchored_gids,
        core_members,
        counters,
    )


# ============================================================
# MEMBER TRUST LABEL
# ============================================================

def member_trust_label(
    gid,
    camera,
    track_id,
    anchored_gids,
    core_members,
):

    key = (
        camera,
        track_id,
    )


    if gid not in anchored_gids:

        return "UNANCHORED"


    if key in core_members[
        gid
    ]:

        return "CORE"


    return "RELAXED"


# ============================================================
# AUDIT INTERNAL CROSS-CAMERA GEOMETRY
# ============================================================

def audit_internal_geometry(
    manager,
    tracklet_lookup,
    homographies,
    anchored_gids,
    core_members,
):

    all_rows = []

    contradictions = []


    for gid, identity in (
        manager.identities.items()
    ):

        members = (
            identity.members
        )


        for first_index in range(
            len(members)
        ):

            first = members[
                first_index
            ]


            first_camera = normalize_camera_id(
                first.camera_id
            )


            first_tracklet = (
                tracklet_lookup.get(
                    (
                        first_camera,
                        first.local_track_id,
                    )
                )
            )


            if first_tracklet is None:
                continue


            for second_index in range(
                first_index + 1,
                len(members),
            ):

                second = members[
                    second_index
                ]


                second_camera = (
                    normalize_camera_id(
                        second.camera_id
                    )
                )


                # Geometry audit is cross-camera only.
                if (
                    first_camera
                    == second_camera
                ):
                    continue


                second_tracklet = (
                    tracklet_lookup.get(
                        (
                            second_camera,
                            second.local_track_id,
                        )
                    )
                )


                if second_tracklet is None:
                    continue


                evidence = (
                    evaluate_cross_camera_geometry(
                        first_tracklet,
                        second_tracklet,
                        homographies,
                    )
                )


                row = {
                    "gid":
                        gid,

                    "first_camera":
                        first_camera,

                    "first_track":
                        first.local_track_id,

                    "first_trust":
                        member_trust_label(
                            gid,
                            first_camera,
                            first.local_track_id,
                            anchored_gids,
                            core_members,
                        ),

                    "second_camera":
                        second_camera,

                    "second_track":
                        second.local_track_id,

                    "second_trust":
                        member_trust_label(
                            gid,
                            second_camera,
                            second.local_track_id,
                            anchored_gids,
                            core_members,
                        ),

                    "status":
                        evidence.status,

                    "shared":
                        evidence.shared_frames,

                    "median":
                        evidence.median_distance,
                }


                all_rows.append(
                    row
                )


                if (
                    evidence.status
                    == "CONTRADICTED"
                ):

                    contradictions.append(
                        row
                    )


    return (
        all_rows,
        contradictions,
    )


# ============================================================
# AUDIT INTERNAL APPEARANCE
# ============================================================

def audit_internal_appearance(
    manager,
    embedding_lookup,
    anchored_gids,
    core_members,
):

    rows = []


    for gid, identity in (
        manager.identities.items()
    ):

        members = (
            identity.members
        )


        for first_index in range(
            len(members)
        ):

            first = members[
                first_index
            ]


            first_camera = normalize_camera_id(
                first.camera_id
            )


            first_key = (
                first_camera,
                first.local_track_id,
            )


            first_embedding = (
                embedding_lookup.get(
                    first_key
                )
            )


            if first_embedding is None:
                continue


            for second_index in range(
                first_index + 1,
                len(members),
            ):

                second = members[
                    second_index
                ]


                second_camera = (
                    normalize_camera_id(
                        second.camera_id
                    )
                )


                second_key = (
                    second_camera,
                    second.local_track_id,
                )


                second_embedding = (
                    embedding_lookup.get(
                        second_key
                    )
                )


                if second_embedding is None:
                    continue


                similarity = cosine_similarity(
                    first_embedding,
                    second_embedding,
                )


                rows.append(
                    {
                        "gid":
                            gid,

                        "first_camera":
                            first_camera,

                        "first_track":
                            first.local_track_id,

                        "first_trust":
                            member_trust_label(
                                gid,
                                first_camera,
                                first.local_track_id,
                                anchored_gids,
                                core_members,
                            ),

                        "second_camera":
                            second_camera,

                        "second_track":
                            second.local_track_id,

                        "second_trust":
                            member_trust_label(
                                gid,
                                second_camera,
                                second.local_track_id,
                                anchored_gids,
                                core_members,
                            ),

                        "similarity":
                            similarity,
                    }
                )


    rows.sort(
        key=lambda row:
            row[
                "similarity"
            ]
    )


    return rows


# ============================================================
# BUILD GID MINIMUM APPEARANCE SUMMARY
# ============================================================

def gid_minimum_appearance(
    appearance_rows,
):

    result = {}


    for row in appearance_rows:

        gid = row[
            "gid"
        ]


        if gid not in result:

            result[
                gid
            ] = row


    return result


# ============================================================
# REGRESSION HELPERS
# ============================================================

def same_gid_check(
    name,
    keys,
    mapping,
):

    gids = [
        mapping.get(
            key
        )
        for key in keys
    ]


    passed = (
        all(
            gid is not None
            for gid in gids
        )
        and
        len(
            set(
                gids
            )
        ) == 1
    )


    return {
        "name":
            name,

        "passed":
            passed,

        "details":
            ", ".join(
                (
                    f"c{camera}:{track_id}"
                    f"={gid}"
                )
                for (
                    camera,
                    track_id
                ), gid
                in zip(
                    keys,
                    gids,
                )
            ),
    }


def different_gid_check(
    name,
    first,
    second,
    mapping,
):

    first_gid = mapping.get(
        first
    )


    second_gid = mapping.get(
        second
    )


    passed = (
        first_gid is not None
        and
        second_gid is not None
        and
        first_gid != second_gid
    )


    return {
        "name":
            name,

        "passed":
            passed,

        "details":
            (
                f"c{first[0]}:"
                f"{first[1]}="
                f"{first_gid}, "
                f"c{second[0]}:"
                f"{second[1]}="
                f"{second_gid}"
            ),
    }


# ============================================================
# PRINT MEMBER TRUST
# ============================================================

def print_identity_trust(
    gid,
    identity,
    anchored_gids,
    core_members,
):

    parts = []


    for member in identity.members:

        camera = normalize_camera_id(
            member.camera_id
        )


        trust = member_trust_label(
            gid,
            camera,
            member.local_track_id,
            anchored_gids,
            core_members,
        )


        parts.append(
            (
                f"c{camera}:"
                f"{member.local_track_id}"
                f"[{trust}]"
            )
        )


    return ", ".join(
        parts
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "CORE IDENTITY FINAL REGRESSION AUDIT"
    )

    print(
        "===================================="
    )


    (
        items,
        tracklet_lookup,
        embedding_lookup,
        homographies,
        manager,
        anchored_gids,
        core_members,
        counters,
    ) = replay_policy()


    mapping = (
        build_member_gid_lookup(
            manager
        )
    )


    (
        geometry_rows,
        contradictions,
    ) = audit_internal_geometry(
        manager,
        tracklet_lookup,
        homographies,
        anchored_gids,
        core_members,
    )


    appearance_rows = (
        audit_internal_appearance(
            manager,
            embedding_lookup,
            anchored_gids,
            core_members,
        )
    )


    gid_min_rows = (
        gid_minimum_appearance(
            appearance_rows
        )
    )


    # ========================================================
    # BASIC STATE
    # ========================================================

    core_count = sum(
        len(
            members
        )
        for members in (
            core_members.values()
        )
    )


    relaxed_members = []


    for gid in anchored_gids:

        identity = (
            manager.identities[
                gid
            ]
        )


        for member in identity.members:

            camera = normalize_camera_id(
                member.camera_id
            )


            key = (
                camera,
                member.local_track_id,
            )


            if key not in core_members[
                gid
            ]:

                relaxed_members.append(
                    (
                        gid,
                        camera,
                        member.local_track_id,
                    )
                )


    supported_count = sum(
        1
        for row in geometry_rows
        if row["status"] == "SUPPORTED"
    )


    contradicted_count = sum(
        1
        for row in geometry_rows
        if row["status"] == "CONTRADICTED"
    )


    unknown_count = sum(
        1
        for row in geometry_rows
        if row["status"] == "UNKNOWN"
    )


    print()
    print(
        "REPLAY SUMMARY"
    )

    print(
        "=============="
    )


    print(
        "Tracklets:",
        len(items),
    )


    print(
        "Final GIDs:",
        len(
            manager.identities
        ),
    )


    print(
        "Final pending:",
        len(
            manager.pending_tracklets
        ),
    )


    print(
        "Anchored GIDs:",
        len(
            anchored_gids
        ),
    )


    print(
        "CORE members:",
        core_count,
    )


    print(
        "RELAXED members:",
        len(
            relaxed_members
        ),
    )


    print(
        "Strict existing:",
        counters[
            "strict_existing"
        ],
    )


    print(
        "Strict pending anchors:",
        counters[
            "strict_pending"
        ],
    )


    print(
        "Relaxed live:",
        counters[
            "relaxed_live"
        ],
    )


    print(
        "Relaxed sweep:",
        counters[
            "relaxed_sweep"
        ],
    )


    print(
        "Ambiguous strict existing:",
        counters[
            "ambiguous_strict_existing"
        ],
    )


    print(
        "Ambiguous relaxed:",
        counters[
            "ambiguous_relaxed"
        ],
    )


    print(
        "Ambiguous strict pending:",
        counters[
            "ambiguous_strict_pending"
        ],
    )


    # ========================================================
    # RELAXED MEMBERS
    # ========================================================

    print()
    print(
        "RELAXED MEMBERS"
    )

    print(
        "==============="
    )


    if not relaxed_members:

        print(
            "None"
        )


    else:

        for (
            gid,
            camera,
            track_id,
        ) in sorted(
            relaxed_members
        ):

            print(
                f"GID {gid}: "
                f"c{camera}:{track_id}"
            )


    # ========================================================
    # INTERNAL GEOMETRY
    # ========================================================

    print()
    print(
        "INTERNAL CROSS-CAMERA GEOMETRY"
    )

    print(
        "=============================="
    )


    print(
        "Pairs audited:",
        len(
            geometry_rows
        ),
    )


    print(
        "SUPPORTED:",
        supported_count,
    )


    print(
        "CONTRADICTED:",
        contradicted_count,
    )


    print(
        "UNKNOWN:",
        unknown_count,
    )


    print()
    print(
        "INTERNAL CONTRADICTIONS"
    )

    print(
        "======================="
    )


    if not contradictions:

        print(
            "NONE"
        )


    else:

        for row in contradictions:

            median = row[
                "median"
            ]


            median_text = (
                "None"
                if median is None
                else f"{median:.2f}"
            )


            print(
                f"GID {row['gid']} | "
                f"c{row['first_camera']}:"
                f"{row['first_track']}"
                f"[{row['first_trust']}]"
                f" <-> "
                f"c{row['second_camera']}:"
                f"{row['second_track']}"
                f"[{row['second_trust']}]"
                f" | shared="
                f"{row['shared']}"
                f" | median="
                f"{median_text}"
            )


    # ========================================================
    # LOWEST INTERNAL APPEARANCE
    # ========================================================

    print()
    print(
        "LOWEST INTERNAL APPEARANCE PAIRS"
    )

    print(
        "================================"
    )


    print(
        f"{'GID':>5} "
        f"{'First':<18} "
        f"{'Second':<18} "
        f"{'Similarity':>10}"
    )


    print(
        "-" * 58
    )


    for row in appearance_rows[
        :LOWEST_APPEARANCE_ROWS
    ]:

        first_text = (
            f"c{row['first_camera']}:"
            f"{row['first_track']}"
            f"[{row['first_trust']}]"
        )


        second_text = (
            f"c{row['second_camera']}:"
            f"{row['second_track']}"
            f"[{row['second_trust']}]"
        )


        print(
            f"{row['gid']:>5} "
            f"{first_text:<18} "
            f"{second_text:<18} "
            f"{row['similarity']:>10.4f}"
        )


    # ========================================================
    # GID-LEVEL MINIMUM APPEARANCE
    # ========================================================

    print()
    print(
        "LARGEST GIDS WITH MINIMUM APPEARANCE"
    )

    print(
        "===================================="
    )


    identities = sorted(
        manager.identities.items(),

        key=lambda item:
            len(
                item[1].members
            ),

        reverse=True,
    )


    for gid, identity in identities[
        :20
    ]:

        min_row = gid_min_rows.get(
            gid
        )


        if min_row is None:

            min_text = "n/a"

            pair_text = "singleton"

        else:

            min_text = (
                f"{min_row['similarity']:.4f}"
            )


            pair_text = (
                f"c{min_row['first_camera']}:"
                f"{min_row['first_track']}"
                f" <-> "
                f"c{min_row['second_camera']}:"
                f"{min_row['second_track']}"
            )


        print(
            f"GID {gid:>3} | "
            f"members="
            f"{len(identity.members):>2} | "
            f"min_app="
            f"{min_text} | "
            f"{pair_text}"
        )


        print(
            "       ",
            print_identity_trust(
                gid,
                identity,
                anchored_gids,
                core_members,
            ),
        )


    # ========================================================
    # REGRESSION CHECKS
    # ========================================================

    checks = []


    checks.append(
        same_gid_check(
            "255 family together",
            [
                (0, 255),
                (1, 298),
                (1, 319),
            ],
            mapping,
        )
    )


    checks.append(
        same_gid_check(
            "399 family together",
            [
                (0, 399),
                (1, 369),
                (1, 382),
            ],
            mapping,
        )
    )


    checks.append(
        same_gid_check(
            "469/503/600 together",
            [
                (0, 469),
                (1, 503),
                (0, 600),
            ],
            mapping,
        )
    )


    checks.append(
        same_gid_check(
            "492/511 together",
            [
                (0, 492),
                (1, 511),
            ],
            mapping,
        )
    )


    checks.append(
        same_gid_check(
            "651/560 together",
            [
                (0, 651),
                (1, 560),
            ],
            mapping,
        )
    )


    checks.append(
        same_gid_check(
            "701 family together",
            [
                (1, 639),
                (0, 701),
                (0, 740),
                (1, 721),
            ],
            mapping,
        )
    )


    checks.append(
        same_gid_check(
            "1 family together",
            [
                (0, 1),
                (1, 1),
                (1, 66),
            ],
            mapping,
        )
    )


    checks.append(
        different_gid_check(
            "255 family separated from 399",
            (0, 255),
            (0, 399),
            mapping,
        )
    )


    checks.append(
        different_gid_check(
            "492 family separated from 701",
            (0, 492),
            (0, 701),
            mapping,
        )
    )


    checks.append(
        different_gid_check(
            "651 family separated from 701",
            (0, 651),
            (0, 701),
            mapping,
        )
    )


    print()
    print(
        "REGRESSION CHECKS"
    )

    print(
        "================="
    )


    passed_count = 0


    for check in checks:

        label = (
            "PASS"
            if check[
                "passed"
            ]
            else "FAIL"
        )


        if check[
            "passed"
        ]:

            passed_count += 1


        print(
            f"{label:<4} | "
            f"{check['name']} | "
            f"{check['details']}"
        )


    print()
    print(
        "Regression checks passed:",
        f"{passed_count}/{len(checks)}",
    )


    # ========================================================
    # FINAL VERDICT SIGNALS
    # ========================================================

    print()
    print(
        "AUDIT VERDICT SIGNALS"
    )

    print(
        "====================="
    )


    print(
        "Internal geometry contradictions:",
        len(
            contradictions
        ),
    )


    print(
        "Regression failures:",
        len(
            checks
        )
        - passed_count,
    )


    print(
        "Relaxed propagation members:",
        len(
            relaxed_members
        ),
    )


if __name__ == "__main__":
    main()
