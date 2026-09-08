from collections import defaultdict

from src.experiments.analyze_final_pending_population import (
    replay_final_production,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    normalize_camera_id,
)


# ============================================================
# CURRENT SINGLE-CORE GEOMETRY CANDIDATES
#
# These came directly from:
#
# src/analyze_multi_core_geometry_support.py
#
# Format:
#
# (
#     camera_id,
#     pending_track_id,
#     candidate_gid,
# )
# ============================================================

TARGETS = [

    (1, 787, 73),

    (0, 636, 66),

    (0, 663, 66),

    (0, 201, 30),

    (1, 833, 76),

    (1, 797, 66),

    (1, 490, 47),
]


# ============================================================
# BROAD DIAGNOSTIC SAME-CAMERA WINDOW
#
# THESE ARE NOT PRODUCTION THRESHOLDS.
#
# We use them only to find potential corroborating evidence.
# ============================================================

DIAG_SAME_MAX_GAP = 60

DIAG_SAME_REID_MIN = 0.84

DIAG_SAME_CENTER_MAX = 80.0

DIAG_SAME_BOTTOM_MAX = 80.0


# ============================================================
# PENDING LOOKUP
# ============================================================

def get_pending(
    manager,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )


    return manager.pending_tracklets.get(
        (
            camera_id,
            local_track_id,
        )
    )


# ============================================================
# GET PENDING TRACKLET
# ============================================================

def get_pending_tracklet(
    manager,
    pending,
    tracklet_lookup,
):

    if pending is None:

        return None


    tracklet = pending.get(
        "candidate_tracklet"
    )


    if tracklet is not None:

        return tracklet


    return manager.get_tracklet(
        pending[
            "camera_id"
        ],

        pending[
            "local_track_id"
        ],

        tracklet_lookup,
    )


# ============================================================
# SAME-CAMERA MEMBER RELATION
# ============================================================

def same_camera_relation(
    manager,
    pending_tracklet,
    pending_embedding,
    member,
    member_tracklet,
    member_embedding,
):

    direct = manager.cosine_similarity(
        pending_embedding,
        member_embedding,
    )


    (
        gap,
        overlap,
    ) = manager.same_camera_temporal_stats(
        member_tracklet,
        pending_tracklet,
    )


    center = None
    bottom = None


    if (
        overlap == 0
        and gap >= 0
    ):

        (
            center,
            bottom,
        ) = (
            manager.same_camera_endpoint_distance(
                member_tracklet,
                pending_tracklet,
            )
        )


    near = False


    if (
        overlap == 0

        and

        gap >= 0

        and

        gap <= DIAG_SAME_MAX_GAP

        and

        direct >= DIAG_SAME_REID_MIN

        and

        center is not None

        and

        bottom is not None

        and

        center <= DIAG_SAME_CENTER_MAX

        and

        bottom <= DIAG_SAME_BOTTOM_MAX
    ):

        near = True


    return {
        "direct":
            direct,

        "gap":
            gap,

        "overlap":
            overlap,

        "center":
            center,

        "bottom":
            bottom,

        "near":
            near,
    }


# ============================================================
# CROSS-CAMERA MEMBER RELATION
# ============================================================

def cross_camera_relation(
    manager,
    pending_tracklet,
    pending_embedding,
    member_tracklet,
    member_embedding,
):

    direct = manager.cosine_similarity(
        pending_embedding,
        member_embedding,
    )


    evidence = (
        evaluate_cross_camera_geometry(
            pending_tracklet,
            member_tracklet,
            manager.homographies,
        )
    )


    return {
        "direct":
            direct,

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
    }


# ============================================================
# PRINT MEMBER EVIDENCE FOR TARGET GID
# ============================================================

def analyze_target(
    manager,
    tracklet_lookup,
    camera_id,
    pending_id,
    gid,
):

    camera_id = normalize_camera_id(
        camera_id
    )


    pending = get_pending(
        manager,
        camera_id,
        pending_id,
    )


    print()
    print(
        f"c{camera_id}:"
        f"{pending_id}"
        f" -> GID "
        f"{gid}"
    )


    if pending is None:

        print(
            "  NOT CURRENTLY PENDING"
        )

        return []


    pending_tracklet = (
        get_pending_tracklet(
            manager,
            pending,
            tracklet_lookup,
        )
    )


    if pending_tracklet is None:

        print(
            "  MISSING TRACKLET"
        )

        return []


    identity = manager.identities.get(
        gid
    )


    if identity is None:

        print(
            "  GID MISSING"
        )

        return []


    scores = manager.score_identity(
        identity,
        pending[
            "embedding"
        ],
    )


    print(
        f"  gallery:"
        f" max={scores['max']:.4f}"
        f" mean={scores['mean']:.4f}"
        f" topk={scores['topk']:.4f}"
    )


    corroboration_hits = []


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        member_key = (
            member_camera,
            member.local_track_id,
        )


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


        trust = manager.get_member_trust(
            gid,
            member_camera,
            member.local_track_id,
        )


        # ====================================================
        # SAME CAMERA
        # ====================================================

        if member_camera == camera_id:

            relation = (
                same_camera_relation(
                    manager=
                        manager,

                    pending_tracklet=
                        pending_tracklet,

                    pending_embedding=
                        pending[
                            "embedding"
                        ],

                    member=
                        member,

                    member_tracklet=
                        member_tracklet,

                    member_embedding=
                        member_embedding,
                )
            )


            center_text = (
                f"{relation['center']:.2f}"
                if relation[
                    "center"
                ]
                is not None
                else "None"
            )


            bottom_text = (
                f"{relation['bottom']:.2f}"
                if relation[
                    "bottom"
                ]
                is not None
                else "None"
            )


            print(
                f"  SAME "
                f"{trust:<9}"
                f" c{member_camera}:"
                f"{member.local_track_id}"
                f" | ReID="
                f"{relation['direct']:.4f}"
                f" | gap="
                f"{relation['gap']}"
                f" | overlap="
                f"{relation['overlap']}"
                f" | center="
                f"{center_text}"
                f" | bottom="
                f"{bottom_text}"
                f" | near="
                f"{relation['near']}"
            )


            if relation[
                "near"
            ]:

                corroboration_hits.append(
                    {
                        "type":
                            "SAME_CAMERA",

                        "trust":
                            trust,

                        "camera_id":
                            member_camera,

                        "track_id":
                            member.local_track_id,

                        "reid":
                            relation[
                                "direct"
                            ],

                        "gap":
                            relation[
                                "gap"
                            ],

                        "center":
                            relation[
                                "center"
                            ],

                        "bottom":
                            relation[
                                "bottom"
                            ],
                    }
                )


        # ====================================================
        # CROSS CAMERA
        # ====================================================

        else:

            relation = (
                cross_camera_relation(
                    manager=
                        manager,

                    pending_tracklet=
                        pending_tracklet,

                    pending_embedding=
                        pending[
                            "embedding"
                        ],

                    member_tracklet=
                        member_tracklet,

                    member_embedding=
                        member_embedding,
                )
            )


            median_text = (
                f"{relation['median']:.2f}"
                if relation[
                    "median"
                ]
                is not None
                else "None"
            )


            mean_text = (
                f"{relation['mean']:.2f}"
                if relation[
                    "mean"
                ]
                is not None
                else "None"
            )


            p90_text = (
                f"{relation['p90']:.2f}"
                if relation[
                    "p90"
                ]
                is not None
                else "None"
            )


            print(
                f"  CROSS "
                f"{trust:<9}"
                f" c{member_camera}:"
                f"{member.local_track_id}"
                f" | ReID="
                f"{relation['direct']:.4f}"
                f" | geom="
                f"{relation['status']}"
                f" | shared="
                f"{relation['shared']}"
                f" | median="
                f"{median_text}"
                f" | mean="
                f"{mean_text}"
                f" | p90="
                f"{p90_text}"
            )


    return corroboration_hits


# ============================================================
# PAIR RELATION BETWEEN TWO PENDING TARGETS
# ============================================================

def analyze_pending_pair(
    manager,
    tracklet_lookup,
    first,
    second,
):

    (
        camera_a,
        track_a,
        gid_a,
    ) = first


    (
        camera_b,
        track_b,
        gid_b,
    ) = second


    if gid_a != gid_b:

        return


    camera_a = normalize_camera_id(
        camera_a
    )


    camera_b = normalize_camera_id(
        camera_b
    )


    pending_a = get_pending(
        manager,
        camera_a,
        track_a,
    )


    pending_b = get_pending(
        manager,
        camera_b,
        track_b,
    )


    if (
        pending_a is None
        or pending_b is None
    ):

        return


    tracklet_a = get_pending_tracklet(
        manager,
        pending_a,
        tracklet_lookup,
    )


    tracklet_b = get_pending_tracklet(
        manager,
        pending_b,
        tracklet_lookup,
    )


    if (
        tracklet_a is None
        or tracklet_b is None
    ):

        return


    direct = manager.cosine_similarity(
        pending_a[
            "embedding"
        ],
        pending_b[
            "embedding"
        ],
    )


    # ========================================================
    # SAME-CAMERA PENDING PAIR
    # ========================================================

    if camera_a == camera_b:

        (
            gap,
            overlap,
        ) = (
            manager.same_camera_temporal_stats(
                tracklet_a,
                tracklet_b,
            )
        )


        center = None
        bottom = None


        if (
            overlap == 0
            and gap >= 0
        ):

            (
                center,
                bottom,
            ) = (
                manager.same_camera_endpoint_distance(
                    tracklet_a,
                    tracklet_b,
                )
            )


        center_text = (
            f"{center:.2f}"
            if center is not None
            else "None"
        )


        bottom_text = (
            f"{bottom:.2f}"
            if bottom is not None
            else "None"
        )


        print(
            f"  c{camera_a}:{track_a}"
            f" <-> "
            f"c{camera_b}:{track_b}"
            f" | SAME CAMERA"
            f" | ReID="
            f"{direct:.4f}"
            f" | gap="
            f"{gap}"
            f" | overlap="
            f"{overlap}"
            f" | center="
            f"{center_text}"
            f" | bottom="
            f"{bottom_text}"
        )


    # ========================================================
    # CROSS-CAMERA PENDING PAIR
    # ========================================================

    else:

        evidence = (
            evaluate_cross_camera_geometry(
                tracklet_a,
                tracklet_b,
                manager.homographies,
            )
        )


        median_text = (
            f"{evidence.median_distance:.2f}"
            if evidence.median_distance
            is not None
            else "None"
        )


        mean_text = (
            f"{evidence.mean_distance:.2f}"
            if evidence.mean_distance
            is not None
            else "None"
        )


        p90_text = (
            f"{evidence.p90_distance:.2f}"
            if evidence.p90_distance
            is not None
            else "None"
        )


        print(
            f"  c{camera_a}:{track_a}"
            f" <-> "
            f"c{camera_b}:{track_b}"
            f" | CROSS CAMERA"
            f" | ReID="
            f"{direct:.4f}"
            f" | geom="
            f"{evidence.status}"
            f" | shared="
            f"{evidence.shared_frames}"
            f" | median="
            f"{median_text}"
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
        "SINGLE-SUPPORT CORROBORATION ANALYSIS"
    )

    print(
        "====================================="
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


    print()
    print(
        "BROAD SAME-CAMERA DIAGNOSTIC WINDOW"
    )

    print(
        "==================================="
    )


    print(
        "gap <=",
        DIAG_SAME_MAX_GAP,
    )


    print(
        "ReID >=",
        DIAG_SAME_REID_MIN,
    )


    print(
        "center <=",
        DIAG_SAME_CENTER_MAX,
    )


    print(
        "bottom <=",
        DIAG_SAME_BOTTOM_MAX,
    )


    # ========================================================
    # PER-TARGET GID MEMBER EVIDENCE
    # ========================================================

    print()
    print(
        "TARGET -> GID MEMBER EVIDENCE"
    )

    print(
        "============================="
    )


    all_hits = []


    for (
        camera_id,
        pending_id,
        gid,
    ) in TARGETS:

        hits = analyze_target(
            manager=
                manager,

            tracklet_lookup=
                tracklet_lookup,

            camera_id=
                camera_id,

            pending_id=
                pending_id,

            gid=
                gid,
        )


        for hit in hits:

            all_hits.append(
                {
                    "pending_camera":
                        normalize_camera_id(
                            camera_id
                        ),

                    "pending_id":
                        pending_id,

                    "gid":
                        gid,

                    **hit,
                }
            )


    # ========================================================
    # SAME-GID PENDING CLUSTERS
    # ========================================================

    grouped = defaultdict(
        list
    )


    for target in TARGETS:

        grouped[
            target[2]
        ].append(
            target
        )


    print()
    print(
        "SAME-GID PENDING-TO-PENDING RELATIONS"
    )

    print(
        "====================================="
    )


    cluster_found = False


    for gid, targets in sorted(
        grouped.items()
    ):

        if len(
            targets
        ) < 2:

            continue


        cluster_found = True


        print()
        print(
            f"GID {gid}"
        )


        for first_index in range(
            len(
                targets
            )
        ):

            for second_index in range(
                first_index + 1,
                len(
                    targets
                ),
            ):

                analyze_pending_pair(
                    manager=
                        manager,

                    tracklet_lookup=
                        tracklet_lookup,

                    first=
                        targets[
                            first_index
                        ],

                    second=
                        targets[
                            second_index
                        ],
                )


    if not cluster_found:

        print(
            "NONE"
        )


    # ========================================================
    # BROAD SAME-CAMERA CORROBORATION HITS
    # ========================================================

    print()
    print(
        "SAME-CAMERA CORROBORATION HITS"
    )

    print(
        "=============================="
    )


    if not all_hits:

        print(
            "NONE"
        )


    else:

        all_hits.sort(
            key=lambda row: (
                row[
                    "reid"
                ],

                -row[
                    "gap"
                ],
            ),
            reverse=True,
        )


        for hit in all_hits:

            print(
                f"c{hit['pending_camera']}:"
                f"{hit['pending_id']}"
                f" -> GID "
                f"{hit['gid']}"
                f" | support="
                f"{hit['trust']} "
                f"c{hit['camera_id']}:"
                f"{hit['track_id']}"
                f" | ReID="
                f"{hit['reid']:.4f}"
                f" | gap="
                f"{hit['gap']}"
                f" | center="
                f"{hit['center']:.2f}"
                f" | bottom="
                f"{hit['bottom']:.2f}"
            )


if __name__ == "__main__":
    main()
