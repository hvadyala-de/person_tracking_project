from collections import Counter

from src.experiments.analyze_final_pending_population import (
    replay_final_production,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    identity_has_conflict,
    normalize_camera_id,
)


# ============================================================
# DIAGNOSTIC SETTINGS
#
# IMPORTANT:
#
# Geometry thresholds below are the existing production
# RELAXED geometry requirements.
#
# We are NOT relaxing geometry here.
#
# The appearance minimum is diagnostic only. It prevents
# extremely weak pairs from cluttering the report.
# ============================================================

GEOMETRY_MEDIAN_MAX = 18.0
MIN_SHARED_FRAMES = 15

DIAGNOSTIC_DIRECT_REID_MIN = 0.80
DIAGNOSTIC_GALLERY_MIN = 0.80


# ============================================================
# GEOMETRY SUPPORT ROWS FOR ONE PENDING -> GID
# ============================================================

def collect_core_geometry_rows(
    manager,
    gid,
    identity,
    pending_tracklet,
    pending_embedding,
    tracklet_lookup,
):

    rows = []


    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        # Cross-camera evidence only.
        if member_camera == pending_camera:

            continue


        member_key = (
            member_camera,
            member.local_track_id,
        )


        # Positive evidence comes from CORE only.
        if (
            member_key
            not in manager.core_members[
                gid
            ]
        ):

            continue


        member_tracklet = (
            manager.get_member_tracklet(
                member,
                tracklet_lookup,
            )
        )


        if member_tracklet is None:

            continue


        member_embedding = (
            manager.member_embeddings.get(
                member_key
            )
        )


        if member_embedding is None:

            continue


        evidence = (
            evaluate_cross_camera_geometry(
                pending_tracklet,
                member_tracklet,
                manager.homographies,
            )
        )


        direct_similarity = (
            manager.cosine_similarity(
                pending_embedding,
                member_embedding,
            )
        )


        production_geometry_ok = (
            evidence.status
            == "SUPPORTED"

            and

            evidence.median_distance
            is not None

            and

            evidence.median_distance
            <= GEOMETRY_MEDIAN_MAX

            and

            evidence.shared_frames
            >= MIN_SHARED_FRAMES
        )


        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "status":
                    evidence.status,

                "shared":
                    evidence.shared_frames,

                "median":
                    evidence.median_distance,

                "mean":
                    evidence.mean_distance,

                "p90":
                    evidence.p90_distance,

                "direct":
                    direct_similarity,

                "geometry_ok":
                    production_geometry_ok,
            }
        )


    return rows


# ============================================================
# ANALYZE ONE PENDING -> GID
# ============================================================

def analyze_gid(
    manager,
    gid,
    pending_tracklet,
    pending_embedding,
    tracklet_lookup,
):

    identity = manager.identities.get(
        gid
    )


    if identity is None:

        return None


    # --------------------------------------------------------
    # Preserve all current negative evidence.
    # --------------------------------------------------------

    if identity_has_conflict(
        pending_tracklet,
        identity,
        tracklet_lookup,
    ):

        return None


    if manager.identity_geometry_contradicted(
        pending_tracklet,
        identity,
        tracklet_lookup,
    ):

        return None


    scores = manager.score_identity(
        identity,
        pending_embedding,
    )


    if (
        scores["max"]
        < DIAGNOSTIC_GALLERY_MIN
    ):

        return None


    rows = collect_core_geometry_rows(
        manager=manager,

        gid=gid,

        identity=identity,

        pending_tracklet=
            pending_tracklet,

        pending_embedding=
            pending_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    supported = [
        row

        for row in rows

        if (
            row[
                "geometry_ok"
            ]

            and

            row[
                "direct"
            ]
            >= DIAGNOSTIC_DIRECT_REID_MIN
        )
    ]


    if not supported:

        return None


    supported.sort(
        key=lambda row: (
            row["direct"],
            row["shared"],
            -row["median"],
        ),
        reverse=True,
    )


    return {
        "gid":
            gid,

        "gid_max":
            scores["max"],

        "gid_topk":
            scores["topk"],

        "gid_mean":
            scores["mean"],

        "support":
            supported,

        "support_count":
            len(
                supported
            ),

        "best_direct":
            max(
                row["direct"]
                for row in supported
            ),

        "mean_direct":
            sum(
                row["direct"]
                for row in supported
            )
            / len(
                supported
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "MULTI-CORE GEOMETRY SUPPORT ANALYSIS"
    )

    print(
        "===================================="
    )


    (
        manager,
        tracklets,
        tracklet_lookup,
        final_results,
    ) = replay_final_production()


    print(
        "Tracklets:",
        len(
            tracklets
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
        "CORE members:",
        sum(
            len(
                members
            )

            for members
            in manager.core_members.values()
        ),
    )


    print()
    print(
        "DIAGNOSTIC REQUIREMENTS"
    )

    print(
        "======================="
    )


    print(
        "Geometry median <=",
        GEOMETRY_MEDIAN_MAX,
    )


    print(
        "Shared frames >=",
        MIN_SHARED_FRAMES,
    )


    print(
        "Direct ReID >=",
        DIAGNOSTIC_DIRECT_REID_MIN,
    )


    print(
        "Diagnostic GID max >=",
        DIAGNOSTIC_GALLERY_MIN,
    )


    candidates = []


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


        gid_rows = []


        for gid in sorted(
            manager.anchored_gids
        ):

            result = analyze_gid(
                manager=manager,

                gid=gid,

                pending_tracklet=
                    tracklet,

                pending_embedding=
                    pending[
                        "embedding"
                    ],

                tracklet_lookup=
                    tracklet_lookup,
            )


            if result is not None:

                gid_rows.append(
                    result
                )


        if not gid_rows:

            continue


        gid_rows.sort(
            key=lambda row: (
                row[
                    "support_count"
                ],

                row[
                    "gid_topk"
                ],

                row[
                    "best_direct"
                ],
            ),
            reverse=True,
        )


        candidates.append(
            {
                "camera_id":
                    camera_id,

                "local_track_id":
                    local_track_id,

                "gids":
                    gid_rows,
            }
        )


    # ========================================================
    # SUMMARY COUNTS
    # ========================================================

    support_distribution = Counter()


    for candidate in candidates:

        best = candidate[
            "gids"
        ][0]


        support_distribution[
            best[
                "support_count"
            ]
        ] += 1


    print()
    print(
        "SUPPORT COUNT DISTRIBUTION"
    )

    print(
        "=========================="
    )


    if not support_distribution:

        print(
            "NONE"
        )


    else:

        for count in sorted(
            support_distribution
        ):

            print(
                f"{count} CORE geometry support(s): "
                f"{support_distribution[count]}"
            )


    # ========================================================
    # MULTI-CORE CASES
    # ========================================================

    print()
    print(
        "MULTI-CORE GEOMETRY CANDIDATES"
    )

    print(
        "=============================="
    )


    multi_found = False


    for candidate in candidates:

        for gid_result in candidate[
            "gids"
        ]:

            if (
                gid_result[
                    "support_count"
                ]
                < 2
            ):

                continue


            multi_found = True


            print()
            print(
                f"c{candidate['camera_id']}:"
                f"{candidate['local_track_id']}"
                f" -> GID "
                f"{gid_result['gid']}"
            )


            print(
                f"  support_count="
                f"{gid_result['support_count']}"
                f" | gid_max="
                f"{gid_result['gid_max']:.4f}"
                f" | gid_topk="
                f"{gid_result['gid_topk']:.4f}"
                f" | mean_direct="
                f"{gid_result['mean_direct']:.4f}"
            )


            for row in gid_result[
                "support"
            ]:

                mean_text = (
                    f"{row['mean']:.2f}"
                    if row["mean"] is not None
                    else "None"
                )


                p90_text = (
                    f"{row['p90']:.2f}"
                    if row["p90"] is not None
                    else "None"
                )


                print(
                    f"    CORE "
                    f"c{row['camera_id']}:"
                    f"{row['track_id']}"
                    f" | ReID="
                    f"{row['direct']:.4f}"
                    f" | shared="
                    f"{row['shared']}"
                    f" | median="
                    f"{row['median']:.2f}"
                    f" | mean="
                    f"{mean_text}"
                    f" | p90="
                    f"{p90_text}"
                )


    if not multi_found:

        print(
            "NONE"
        )


    # ========================================================
    # ALL GEOMETRY-SUPPORTED PENDING -> GID PAIRS
    # ========================================================

    print()
    print(
        "ALL GEOMETRY-SUPPORTED PENDING CANDIDATES"
    )

    print(
        "========================================="
    )


    flat_rows = []


    for candidate in candidates:

        for gid_result in candidate[
            "gids"
        ]:

            flat_rows.append(
                (
                    candidate[
                        "camera_id"
                    ],

                    candidate[
                        "local_track_id"
                    ],

                    gid_result,
                )
            )


    flat_rows.sort(
        key=lambda item: (
            item[2][
                "support_count"
            ],

            item[2][
                "gid_topk"
            ],

            item[2][
                "best_direct"
            ],
        ),
        reverse=True,
    )


    if not flat_rows:

        print(
            "NONE"
        )


    else:

        for (
            camera_id,
            local_track_id,
            result,
        ) in flat_rows:

            support_ids = ",".join(
                (
                    f"c{row['camera_id']}:"
                    f"{row['track_id']}"
                )

                for row
                in result[
                    "support"
                ]
            )


            print(
                f"c{camera_id}:"
                f"{local_track_id}"
                f" -> GID "
                f"{result['gid']}"
                f" | supports="
                f"{result['support_count']}"
                f" [{support_ids}]"
                f" | gid_max="
                f"{result['gid_max']:.4f}"
                f" | gid_topk="
                f"{result['gid_topk']:.4f}"
                f" | best_direct="
                f"{result['best_direct']:.4f}"
                f" | mean_direct="
                f"{result['mean_direct']:.4f}"
            )


    # ========================================================
    # MULTIPLE GID AMBIGUITY
    # ========================================================

    print()
    print(
        "MULTIPLE-GID GEOMETRY AMBIGUITIES"
    )

    print(
        "================================="
    )


    ambiguity_found = False


    for candidate in candidates:

        if len(
            candidate[
                "gids"
            ]
        ) <= 1:

            continue


        ambiguity_found = True


        gids = [
            result[
                "gid"
            ]

            for result
            in candidate[
                "gids"
            ]
        ]


        print(
            f"c{candidate['camera_id']}:"
            f"{candidate['local_track_id']}"
            f" -> candidate GIDs "
            f"{gids}"
        )


    if not ambiguity_found:

        print(
            "NONE"
        )


    # ========================================================
    # CURRENT 11 NEAR-MISS TARGETS
    # ========================================================

    target_keys = {
        (0, 201),
        (0, 636),
        (0, 663),
        (0, 719),

        (1, 82),
        (1, 318),
        (1, 490),
        (1, 733),
        (1, 787),
        (1, 797),
        (1, 833),
    }


    print()
    print(
        "CURRENT CROSS-REID NEAR-MISS TARGETS"
    )

    print(
        "===================================="
    )


    found_targets = set()


    for (
        camera_id,
        local_track_id,
        result,
    ) in flat_rows:

        key = (
            camera_id,
            local_track_id,
        )


        if key not in target_keys:

            continue


        found_targets.add(
            key
        )


        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> GID "
            f"{result['gid']}"
            f" | supports="
            f"{result['support_count']}"
            f" | gid_max="
            f"{result['gid_max']:.4f}"
            f" | gid_topk="
            f"{result['gid_topk']:.4f}"
            f" | best_direct="
            f"{result['best_direct']:.4f}"
            f" | mean_direct="
            f"{result['mean_direct']:.4f}"
        )


    missing_targets = (
        target_keys
        - found_targets
    )


    if missing_targets:

        print()
        print(
            "Targets with no production-valid "
            "geometry support at diagnostic "
            "ReID >= 0.80:"
        )


        for (
            camera_id,
            local_track_id,
        ) in sorted(
            missing_targets
        ):

            print(
                f"c{camera_id}:"
                f"{local_track_id}"
            )


if __name__ == "__main__":
    main()
