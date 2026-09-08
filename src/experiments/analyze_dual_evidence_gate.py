from collections import Counter

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    identity_has_conflict,
    normalize_camera_id,
)

from src.run_global_id_offline import (
    load_all_tracklets,
)

from src.terrace_geometry import (
    load_ground_homographies,
)


# ============================================================
# DIAGNOSTIC DUAL-EVIDENCE THRESHOLDS
#
# THESE DO NOT MODIFY PRODUCTION.
# ============================================================

SAME_MAX_GAP = 15
SAME_REID_MIN = 0.90
SAME_CENTER_MAX = 50.0
SAME_BOTTOM_MAX = 50.0

CROSS_REID_MIN = 0.85
CROSS_MIN_SHARED = 20
CROSS_MEDIAN_MAX = 18.0

GID_APPEARANCE_MIN = 0.84


# ============================================================
# REPLAY CURRENT PRODUCTION STATE
# ============================================================

def replay_production():

    items, lookup = load_all_tracklets()

    items.sort(
        key=lambda item: (
            item[0].start_frame,
            normalize_camera_id(
                item[0].camera_id
            ),
            item[0].local_track_id,
        )
    )

    homographies = (
        load_ground_homographies()
    )

    manager = GlobalIdentityManager(
        homographies=homographies
    )

    for tracklet, embedding in items:

        camera_id = normalize_camera_id(
            tracklet.camera_id
        )

        result = manager.assign_tracklet(
            camera_id=camera_id,
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

        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=lookup,
                include_same_camera=False,
            )

    manager.reevaluate_pending(
        tracklet_lookup=lookup,
        include_same_camera=True,
    )

    return manager, items, lookup


# ============================================================
# SAME-CAMERA CORE SUPPORT
# ============================================================

def same_camera_support_rows(
    manager,
    identity,
    gid,
    pending_tracklet,
    pending_embedding,
    lookup,
):

    rows = []

    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )

    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )

        if member_camera != pending_camera:
            continue

        member_key = (
            member_camera,
            member.local_track_id,
        )

        if (
            member_key
            not in manager.core_members[gid]
        ):
            continue

        member_tracklet = (
            manager.get_member_tracklet(
                member,
                lookup,
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

        gap, overlap = (
            manager.same_camera_temporal_stats(
                member_tracklet,
                pending_tracklet,
            )
        )

        if overlap != 0:
            continue

        if gap < 0:
            continue

        if gap > SAME_MAX_GAP:
            continue

        (
            center_distance,
            bottom_distance,
        ) = (
            manager.same_camera_endpoint_distance(
                member_tracklet,
                pending_tracklet,
            )
        )

        if (
            center_distance is None
            or bottom_distance is None
        ):
            continue

        direct_similarity = (
            manager.cosine_similarity(
                pending_embedding,
                member_embedding,
            )
        )

        if (
            direct_similarity
            < SAME_REID_MIN
        ):
            continue

        if (
            center_distance
            > SAME_CENTER_MAX
        ):
            continue

        if (
            bottom_distance
            > SAME_BOTTOM_MAX
        ):
            continue

        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "gap":
                    gap,

                "reid":
                    direct_similarity,

                "center":
                    center_distance,

                "bottom":
                    bottom_distance,
            }
        )

    return rows


# ============================================================
# CROSS-CAMERA CORE SUPPORT
# ============================================================

def cross_camera_support_rows(
    manager,
    identity,
    gid,
    pending_tracklet,
    pending_embedding,
    lookup,
):

    rows = []

    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )

    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )

        if member_camera == pending_camera:
            continue

        member_key = (
            member_camera,
            member.local_track_id,
        )

        if (
            member_key
            not in manager.core_members[gid]
        ):
            continue

        member_tracklet = (
            manager.get_member_tracklet(
                member,
                lookup,
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

        if (
            evidence.status
            != "SUPPORTED"
        ):
            continue

        if (
            evidence.shared_frames
            < CROSS_MIN_SHARED
        ):
            continue

        if (
            evidence.median_distance
            is None
        ):
            continue

        if (
            evidence.median_distance
            > CROSS_MEDIAN_MAX
        ):
            continue

        if (
            direct_similarity
            < CROSS_REID_MIN
        ):
            continue

        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "reid":
                    direct_similarity,

                "shared":
                    evidence.shared_frames,

                "median":
                    evidence.median_distance,

                "mean":
                    evidence.mean_distance,

                "p90":
                    evidence.p90_distance,
            }
        )

    return rows


# ============================================================
# QUALIFY ONE GID
# ============================================================

def evaluate_gid(
    manager,
    gid,
    pending_tracklet,
    pending_embedding,
    lookup,
):

    identity = manager.identities.get(
        gid
    )

    if identity is None:

        return None

    # --------------------------------------------------------
    # Keep existing negative evidence.
    # --------------------------------------------------------

    if identity_has_conflict(
        pending_tracklet,
        identity,
        lookup,
    ):

        return None

    if manager.identity_geometry_contradicted(
        pending_tracklet,
        identity,
        lookup,
    ):

        return None

    # --------------------------------------------------------
    # Require existing gallery-level compatibility too.
    # --------------------------------------------------------

    scores = manager.score_identity(
        identity,
        pending_embedding,
    )

    if (
        scores["max"]
        < GID_APPEARANCE_MIN
    ):

        return None

    if (
        scores["topk"]
        < GID_APPEARANCE_MIN
    ):

        return None

    same_rows = (
        same_camera_support_rows(
            manager=manager,
            identity=identity,
            gid=gid,
            pending_tracklet=
                pending_tracklet,
            pending_embedding=
                pending_embedding,
            lookup=lookup,
        )
    )

    if not same_rows:

        return None

    cross_rows = (
        cross_camera_support_rows(
            manager=manager,
            identity=identity,
            gid=gid,
            pending_tracklet=
                pending_tracklet,
            pending_embedding=
                pending_embedding,
            lookup=lookup,
        )
    )

    if not cross_rows:

        return None

    same_rows.sort(
        key=lambda row: (
            row["reid"],
            -row["gap"],
        ),
        reverse=True,
    )

    cross_rows.sort(
        key=lambda row: (
            row["reid"],
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

        "same":
            same_rows[0],

        "cross":
            cross_rows[0],

        "same_count":
            len(same_rows),

        "cross_count":
            len(cross_rows),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "DUAL-EVIDENCE GATE ANALYSIS"
    )

    print(
        "==========================="
    )

    manager, items, lookup = (
        replay_production()
    )

    print(
        "Tracklets:",
        len(items),
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
        "DIAGNOSTIC THRESHOLDS"
    )

    print(
        "====================="
    )

    print(
        "same gap <=",
        SAME_MAX_GAP,
    )

    print(
        "same ReID >=",
        SAME_REID_MIN,
    )

    print(
        "same center <=",
        SAME_CENTER_MAX,
    )

    print(
        "same bottom <=",
        SAME_BOTTOM_MAX,
    )

    print(
        "cross ReID >=",
        CROSS_REID_MIN,
    )

    print(
        "cross shared >=",
        CROSS_MIN_SHARED,
    )

    print(
        "cross median <=",
        CROSS_MEDIAN_MAX,
    )

    print(
        "GID max/topk >=",
        GID_APPEARANCE_MIN,
    )

    resolution_counts = Counter()

    resolved = []

    ambiguous = []

    print()
    print(
        "QUALIFYING PENDING TRACKLETS"
    )

    print(
        "============================"
    )

    for key, pending in sorted(
        manager.pending_tracklets.items()
    ):

        camera_id = normalize_camera_id(
            pending["camera_id"]
        )

        local_track_id = (
            pending["local_track_id"]
        )

        pending_tracklet = pending.get(
            "candidate_tracklet"
        )

        if pending_tracklet is None:

            pending_tracklet = (
                manager.get_tracklet(
                    camera_id,
                    local_track_id,
                    lookup,
                )
            )

        if pending_tracklet is None:

            continue

        qualifying = []

        for gid in sorted(
            manager.anchored_gids
        ):

            result = evaluate_gid(
                manager=manager,
                gid=gid,
                pending_tracklet=
                    pending_tracklet,
                pending_embedding=
                    pending["embedding"],
                lookup=lookup,
            )

            if result is not None:

                qualifying.append(
                    result
                )

        if not qualifying:

            resolution_counts[
                "NO_MATCH"
            ] += 1

            continue

        if len(qualifying) > 1:

            resolution_counts[
                "AMBIGUOUS"
            ] += 1

            ambiguous.append(
                (
                    camera_id,
                    local_track_id,
                    qualifying,
                )
            )

            print(
                f"AMBIGUOUS "
                f"c{camera_id}:"
                f"{local_track_id}"
                f" -> "
                f"{[row['gid'] for row in qualifying]}"
            )

            continue

        result = qualifying[0]

        resolution_counts[
            "UNIQUE"
        ] += 1

        resolved.append(
            (
                camera_id,
                local_track_id,
                result,
            )
        )

        same = result["same"]
        cross = result["cross"]

        mean_text = (
            f"{cross['mean']:.2f}"
            if cross["mean"] is not None
            else "None"
        )

        p90_text = (
            f"{cross['p90']:.2f}"
            if cross["p90"] is not None
            else "None"
        )

        print()
        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> GID "
            f"{result['gid']}"
        )

        print(
            "  GID:"
            f" max={result['gid_max']:.4f}"
            f" topk={result['gid_topk']:.4f}"
        )

        print(
            "  SAME CORE:"
            f" c{same['camera_id']}:"
            f"{same['track_id']}"
            f" gap={same['gap']}"
            f" ReID={same['reid']:.4f}"
            f" center={same['center']:.2f}"
            f" bottom={same['bottom']:.2f}"
        )

        print(
            "  CROSS CORE:"
            f" c{cross['camera_id']}:"
            f"{cross['track_id']}"
            f" ReID={cross['reid']:.4f}"
            f" shared={cross['shared']}"
            f" median={cross['median']:.2f}"
            f" mean={mean_text}"
            f" p90={p90_text}"
        )

    print()
    print(
        "SUMMARY"
    )

    print(
        "======="
    )

    print(
        "Unique dual-evidence candidates:",
        resolution_counts[
            "UNIQUE"
        ],
    )

    print(
        "Ambiguous dual-evidence candidates:",
        resolution_counts[
            "AMBIGUOUS"
        ],
    )

    print(
        "No dual-evidence match:",
        resolution_counts[
            "NO_MATCH"
        ],
    )

    print()
    print(
        "UNIQUE CANDIDATE LIST"
    )

    print(
        "====================="
    )

    if not resolved:

        print(
            "NONE"
        )

    else:

        for (
            camera_id,
            local_track_id,
            result,
        ) in resolved:

            print(
                f"c{camera_id}:"
                f"{local_track_id}"
                f" -> GID "
                f"{result['gid']}"
            )

    print()
    print(
        "AMBIGUITY LIST"
    )

    print(
        "=============="
    )

    if not ambiguous:

        print(
            "NONE"
        )

    else:

        for (
            camera_id,
            local_track_id,
            qualifying,
        ) in ambiguous:

            gids = [
                row["gid"]
                for row in qualifying
            ]

            print(
                f"c{camera_id}:"
                f"{local_track_id}"
                f" -> "
                f"{gids}"
            )


if __name__ == "__main__":
    main()
