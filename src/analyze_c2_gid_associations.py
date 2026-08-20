from collections import Counter

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.build_tracklets import (
    filter_tracklets,
    load_tracklets,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.run_global_id_offline import (
    MIN_DETECTIONS,
    TRACK_DIR,
    load_embedding,
)


# ============================================================
# FIRST c2 -> EXISTING GID ASSOCIATION DIAGNOSTIC
#
# IMPORTANT:
#
# This script does NOT modify identities.
#
# We begin from the validated final c0+c1 production state:
#
#     76 GIDs
#     92 pending
#     68 CORE members
#
# Then we load the 105 useful c2 tracklets and ask:
#
# "Does a c2 tracklet receive independent positive evidence
#  from multiple CORE members of the same existing GID?"
#
# Positive evidence comes from CORE only.
#
# Existing RELAXED members cannot provide positive support.
#
# Existing negative geometry contradiction remains a veto.
# ============================================================


C2_CAMERA_ID = 2


# ============================================================
# BROAD c2 ANALYSIS GATES
#
# These are DIAGNOSTIC thresholds.
#
# They are NOT new production thresholds.
#
# Camera-2 geometry validation showed useful same-identity
# candidates around median 20-24, so we intentionally use the
# established broad geometry-analysis threshold of 30 here.
#
# Geometry values are Terrace ground-coordinate units,
# NOT metres.
# ============================================================

SUPPORT_REID_MIN = 0.84

SUPPORT_MIN_SHARED_FRAMES = 10

SUPPORT_GEOMETRY_MEDIAN_MAX = 30.0

MIN_CORE_SUPPORTS = 2


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
# LOAD c2 USEFUL TRACKLETS + EMBEDDINGS
# ============================================================

def load_c2_tracklets():

    csv_path = (
        TRACK_DIR
        / "tracks_c2.csv"
    )


    tracklets = load_tracklets(
        csv_path
    )


    useful = filter_tracklets(
        tracklets,
        min_detections=
            MIN_DETECTIONS,
    )


    rows = []

    lookup = {}


    for tracklet in useful.values():

        camera_id = normalize_camera_id(
            tracklet.camera_id
        )


        embedding = load_embedding(
            camera_id,
            tracklet.local_track_id,
        )


        if embedding is None:

            continue


        rows.append(
            (
                tracklet,
                embedding,
            )
        )


        lookup[
            (
                camera_id,
                tracklet.local_track_id,
            )
        ] = tracklet


        lookup[
            (
                f"c{camera_id}",
                tracklet.local_track_id,
            )
        ] = tracklet


    rows.sort(
        key=lambda item: (
            item[0].start_frame,
            item[0].local_track_id,
        )
    )


    return (
        rows,
        lookup,
    )


# ============================================================
# CORE RELATIONS FOR ONE c2 TRACKLET -> ONE GID
# ============================================================

def collect_core_relations(
    manager,
    gid,
    identity,
    candidate_tracklet,
    candidate_embedding,
    tracklet_lookup,
):

    rows = []


    candidate_camera = normalize_camera_id(
        candidate_tracklet.camera_id
    )


    core_keys = manager.core_members.get(
        gid,
        set(),
    )


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        # ----------------------------------------------------
        # Cross-camera evidence only.
        # ----------------------------------------------------

        if member_camera == candidate_camera:

            continue


        member_key = (
            member_camera,
            member.local_track_id,
        )


        # ----------------------------------------------------
        # Positive support comes from CORE only.
        # ----------------------------------------------------

        if member_key not in core_keys:

            continue


        member_tracklet = (
            manager.get_member_tracklet(
                member,
                tracklet_lookup,
            )
        )


        member_embedding = (
            manager.member_embeddings.get(
                member_key
            )
        )


        if (
            member_tracklet is None
            or member_embedding is None
        ):

            continue


        direct_similarity = (
            manager.cosine_similarity(
                candidate_embedding,
                member_embedding,
            )
        )


        geometry = (
            evaluate_cross_camera_geometry(
                candidate_tracklet,
                member_tracklet,
                manager.homographies,
            )
        )


        supported = (
            geometry.status
            == "SUPPORTED"

            and

            geometry.shared_frames
            >= SUPPORT_MIN_SHARED_FRAMES

            and

            geometry.median_distance
            is not None

            and

            geometry.median_distance
            <= SUPPORT_GEOMETRY_MEDIAN_MAX

            and

            direct_similarity
            >= SUPPORT_REID_MIN
        )


        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "direct_similarity":
                    direct_similarity,

                "geometry_status":
                    geometry.status,

                "shared_frames":
                    geometry.shared_frames,

                "median_distance":
                    geometry.median_distance,

                "mean_distance":
                    geometry.mean_distance,

                "p90_distance":
                    geometry.p90_distance,

                "supported":
                    supported,
            }
        )


    return rows


# ============================================================
# ANALYZE ONE c2 TRACKLET -> ONE GID
# ============================================================

def analyze_gid(
    manager,
    gid,
    candidate_tracklet,
    candidate_embedding,
    tracklet_lookup,
):

    identity = manager.identities.get(
        gid
    )


    if identity is None:

        return None


    relations = collect_core_relations(
        manager=
            manager,

        gid=
            gid,

        identity=
            identity,

        candidate_tracklet=
            candidate_tracklet,

        candidate_embedding=
            candidate_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    support_rows = [
        row

        for row in relations

        if row[
            "supported"
        ]
    ]


    if not support_rows:

        return None


    # --------------------------------------------------------
    # Preserve the production identity-level geometry veto.
    #
    # This checks the candidate against all identity members.
    # RELAXED members may therefore contribute negative
    # contradiction evidence, but never positive support.
    # --------------------------------------------------------

    geometry_contradicted = (
        manager.identity_geometry_contradicted(
            candidate_tracklet,
            identity,
            tracklet_lookup,
        )
    )


    core_scores = manager.core_gallery_scores(
        gid,
        candidate_embedding,
    )


    support_rows.sort(
        key=lambda row: (
            row[
                "direct_similarity"
            ],

            row[
                "shared_frames"
            ],

            -row[
                "median_distance"
            ],
        ),
        reverse=True,
    )


    support_cameras = sorted(
        {
            row[
                "camera_id"
            ]

            for row
            in support_rows
        }
    )


    return {
        "gid":
            gid,

        "support_count":
            len(
                support_rows
            ),

        "support_camera_count":
            len(
                support_cameras
            ),

        "support_cameras":
            support_cameras,

        "support_rows":
            support_rows,

        "geometry_contradicted":
            geometry_contradicted,

        "core_scores":
            core_scores,

        "best_direct":
            max(
                row[
                    "direct_similarity"
                ]

                for row
                in support_rows
            ),

        "best_shared":
            max(
                row[
                    "shared_frames"
                ]

                for row
                in support_rows
            ),

        "best_median":
            min(
                row[
                    "median_distance"
                ]

                for row
                in support_rows
            ),
    }


# ============================================================
# PRINT GID CANDIDATE
# ============================================================

def print_gid_candidate(
    camera_id,
    local_track_id,
    result,
):

    scores = result[
        "core_scores"
    ]


    print(
        f"c{camera_id}:"
        f"{local_track_id}"
        f" -> GID "
        f"{result['gid']}"
        f" | CORE supports="
        f"{result['support_count']}"
        f" | support cameras="
        f"{result['support_cameras']}"
        f" | veto="
        f"{result['geometry_contradicted']}"
    )


    if scores is not None:

        print(
            f"  CORE gallery:"
            f" max="
            f"{scores['max']:.4f}"
            f" | top3="
            f"{scores['top3']:.4f}"
            f" | mean="
            f"{scores['mean']:.4f}"
            f" | members="
            f"{scores['count']}"
        )


    for row in result[
        "support_rows"
    ]:

        mean_text = (
            f"{row['mean_distance']:.2f}"

            if row[
                "mean_distance"
            ]
            is not None

            else "None"
        )


        p90_text = (
            f"{row['p90_distance']:.2f}"

            if row[
                "p90_distance"
            ]
            is not None

            else "None"
        )


        print(
            f"  CORE "
            f"c{row['camera_id']}:"
            f"{row['track_id']}"
            f" | ReID="
            f"{row['direct_similarity']:.4f}"
            f" | shared="
            f"{row['shared_frames']}"
            f" | median="
            f"{row['median_distance']:.2f}"
            f" | mean="
            f"{mean_text}"
            f" | p90="
            f"{p90_text}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2 -> EXISTING GID ASSOCIATION ANALYSIS"
    )

    print(
        "======================================="
    )


    # ========================================================
    # REPLAY CURRENT VALIDATED c0+c1 PRODUCTION
    # ========================================================

    (
        manager,
        baseline_tracklets,
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
        "c0+c1 FINAL PRODUCTION BASELINE"
    )

    print(
        "=============================="
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
        len(
            manager.anchored_gids
        ),
    )


    print(
        "CORE members:",
        baseline_core,
    )


    # ========================================================
    # LOAD c2
    # ========================================================

    (
        c2_tracklets,
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
        "Useful c2 tracklets with embeddings:",
        len(
            c2_tracklets
        ),
    )


    # ========================================================
    # DIAGNOSTIC SETTINGS
    # ========================================================

    print()
    print(
        "DIAGNOSTIC SUPPORT GATES"
    )

    print(
        "========================"
    )


    print(
        "CORE direct ReID >=",
        SUPPORT_REID_MIN,
    )


    print(
        "Shared frames >=",
        SUPPORT_MIN_SHARED_FRAMES,
    )


    print(
        "Geometry median <=",
        SUPPORT_GEOMETRY_MEDIAN_MAX,
    )


    print(
        "Minimum CORE supports for "
        "multi-CORE candidate:",
        MIN_CORE_SUPPORTS,
    )


    print()
    print(
        "Geometry values are Terrace "
        "ground-coordinate units, not metres."
    )


    # ========================================================
    # ANALYZE EVERY c2 TRACKLET
    # ========================================================

    unique_multi = []

    ambiguous_multi = []

    single_only = []

    vetoed_multi = []

    no_support = []


    all_candidate_rows = []


    for (
        candidate_tracklet,
        candidate_embedding,
    ) in c2_tracklets:

        camera_id = normalize_camera_id(
            candidate_tracklet.camera_id
        )


        local_track_id = (
            candidate_tracklet.local_track_id
        )


        gid_rows = []


        for gid in sorted(
            manager.anchored_gids
        ):

            result = analyze_gid(
                manager=
                    manager,

                gid=
                    gid,

                candidate_tracklet=
                    candidate_tracklet,

                candidate_embedding=
                    candidate_embedding,

                tracklet_lookup=
                    tracklet_lookup,
            )


            if result is not None:

                gid_rows.append(
                    result
                )


        gid_rows.sort(
            key=lambda row: (
                row[
                    "support_count"
                ],

                row[
                    "best_direct"
                ],

                row[
                    "best_shared"
                ],

                -row[
                    "best_median"
                ],
            ),
            reverse=True,
        )


        all_candidate_rows.append(
            {
                "camera_id":
                    camera_id,

                "local_track_id":
                    local_track_id,

                "gids":
                    gid_rows,
            }
        )


        if not gid_rows:

            no_support.append(
                (
                    camera_id,
                    local_track_id,
                )
            )

            continue


        valid_multi = [
            row

            for row in gid_rows

            if (
                row[
                    "support_count"
                ]
                >= MIN_CORE_SUPPORTS

                and

                not row[
                    "geometry_contradicted"
                ]
            )
        ]


        vetoed = [
            row

            for row in gid_rows

            if (
                row[
                    "support_count"
                ]
                >= MIN_CORE_SUPPORTS

                and

                row[
                    "geometry_contradicted"
                ]
            )
        ]


        if vetoed:

            vetoed_multi.append(
                (
                    camera_id,
                    local_track_id,
                    vetoed,
                )
            )


        if len(
            valid_multi
        ) == 1:

            unique_multi.append(
                (
                    camera_id,
                    local_track_id,
                    valid_multi[
                        0
                    ],
                )
            )


        elif len(
            valid_multi
        ) > 1:

            ambiguous_multi.append(
                (
                    camera_id,
                    local_track_id,
                    valid_multi,
                )
            )


        else:

            non_vetoed_single = [
                row

                for row in gid_rows

                if (
                    row[
                        "support_count"
                    ]
                    == 1

                    and

                    not row[
                        "geometry_contradicted"
                    ]
                )
            ]


            if non_vetoed_single:

                single_only.append(
                    (
                        camera_id,
                        local_track_id,
                        non_vetoed_single,
                    )
                )


    # ========================================================
    # UNIQUE MULTI-CORE CANDIDATES
    # ========================================================

    print()
    print(
        "UNIQUE MULTI-CORE c2 -> GID CANDIDATES"
    )

    print(
        "======================================"
    )


    if not unique_multi:

        print(
            "NONE"
        )


    else:

        for (
            camera_id,
            local_track_id,
            result,
        ) in unique_multi:

            print()

            print_gid_candidate(
                camera_id,
                local_track_id,
                result,
            )


    # ========================================================
    # AMBIGUOUS MULTI-CORE
    # ========================================================

    print()
    print(
        "AMBIGUOUS MULTI-CORE CANDIDATES"
    )

    print(
        "==============================="
    )


    if not ambiguous_multi:

        print(
            "NONE"
        )


    else:

        for (
            camera_id,
            local_track_id,
            rows,
        ) in ambiguous_multi:

            gids = [
                row[
                    "gid"
                ]

                for row
                in rows
            ]


            print(
                f"c{camera_id}:"
                f"{local_track_id}"
                f" -> candidate GIDs "
                f"{gids}"
            )


            for row in rows:

                print_gid_candidate(
                    camera_id,
                    local_track_id,
                    row,
                )


    # ========================================================
    # MULTI-CORE CANDIDATES VETOED BY IDENTITY CONTRADICTION
    # ========================================================

    print()
    print(
        "MULTI-CORE CANDIDATES VETOED BY CONTRADICTION"
    )

    print(
        "============================================="
    )


    if not vetoed_multi:

        print(
            "NONE"
        )


    else:

        for (
            camera_id,
            local_track_id,
            rows,
        ) in vetoed_multi:

            for row in rows:

                print()

                print_gid_candidate(
                    camera_id,
                    local_track_id,
                    row,
                )


    # ========================================================
    # SINGLE-CORE-ONLY CASES
    #
    # These are NOT association candidates.
    # They are reported only for later analysis.
    # ========================================================

    print()
    print(
        "SINGLE-CORE-ONLY c2 CASES"
    )

    print(
        "========================="
    )


    print(
        "Count:",
        len(
            single_only
        ),
    )


    ranked_single = []


    for (
        camera_id,
        local_track_id,
        rows,
    ) in single_only:

        for row in rows:

            ranked_single.append(
                (
                    camera_id,
                    local_track_id,
                    row,
                )
            )


    ranked_single.sort(
        key=lambda item: (
            item[2][
                "best_direct"
            ],

            item[2][
                "best_shared"
            ],

            -item[2][
                "best_median"
            ],
        ),
        reverse=True,
    )


    for (
        camera_id,
        local_track_id,
        result,
    ) in ranked_single[
        :20
    ]:

        print()

        print_gid_candidate(
            camera_id,
            local_track_id,
            result,
        )


    # ========================================================
    # SUPPORT COUNT DISTRIBUTION
    # ========================================================

    distribution = Counter()


    for candidate in all_candidate_rows:

        if not candidate[
            "gids"
        ]:

            distribution[
                0
            ] += 1

            continue


        best_support_count = max(
            row[
                "support_count"
            ]

            for row
            in candidate[
                "gids"
            ]

            if not row[
                "geometry_contradicted"
            ]
        ) if any(
            not row[
                "geometry_contradicted"
            ]

            for row
            in candidate[
                "gids"
            ]
        ) else 0


        distribution[
            best_support_count
        ] += 1


    print()
    print(
        "BEST NON-VETOED CORE SUPPORT COUNT DISTRIBUTION"
    )

    print(
        "=============================================="
    )


    for count in sorted(
        distribution
    ):

        print(
            f"{count} CORE support(s): "
            f"{distribution[count]}"
        )


    # ========================================================
    # EXPECTED SANITY TARGETS
    # ========================================================

    expected_targets = {
        (
            2,
            508,
        ):
            65,

        (
            2,
            601,
        ):
            73,
    }


    found_unique = {
        (
            camera_id,
            local_track_id,
        ):
            result[
                "gid"
            ]

        for (
            camera_id,
            local_track_id,
            result,
        ) in unique_multi
    }


    print()
    print(
        "SANITY TARGETS"
    )

    print(
        "=============="
    )


    sanity_pass = True


    for key, expected_gid in (
        expected_targets.items()
    ):

        actual_gid = found_unique.get(
            key
        )


        passed = (
            actual_gid
            == expected_gid
        )


        sanity_pass = (
            sanity_pass
            and passed
        )


        print(
            (
                "PASS"
                if passed
                else "FAIL"
            ),
            "|",
            f"c{key[0]}:"
            f"{key[1]}"
            f" -> expected GID "
            f"{expected_gid}"
            f" | actual="
            f"{actual_gid}",
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "SUMMARY"
    )

    print(
        "======="
    )


    print(
        "c2 tracklets analyzed:",
        len(
            c2_tracklets
        ),
    )


    print(
        "Unique multi-CORE candidates:",
        len(
            unique_multi
        ),
    )


    print(
        "Ambiguous multi-CORE candidates:",
        len(
            ambiguous_multi
        ),
    )


    print(
        "Multi-CORE candidates vetoed:",
        len(
            vetoed_multi
        ),
    )


    print(
        "Tracklets with single-CORE-only support:",
        len(
            single_only
        ),
    )


    print(
        "Tracklets with no qualifying CORE support:",
        len(
            no_support
        ),
    )


    print(
        "Sanity targets:",
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
        "This script is diagnostic only."
    )


if __name__ == "__main__":
    main()
