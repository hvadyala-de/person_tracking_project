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
# DIAGNOSTIC ONLY
#
# Positive evidence is CORE-only.
#
# RELAXED members may still contribute negative contradiction
# evidence through identity_geometry_contradicted(), but they
# are NOT used for positive appearance or geometry support.
# ============================================================

CROSS_CORE_REID_MIN = 0.83
CROSS_SHARED_MIN = 20
CROSS_MEDIAN_MAX = 18.0

CORE_GALLERY_MAX_MIN = 0.90
CORE_GALLERY_TOP3_MIN = 0.90


def core_gallery_scores(
    manager,
    gid,
    embedding,
):

    rows = []

    for member_key in manager.core_members.get(
        gid,
        set(),
    ):

        member_embedding = (
            manager.member_embeddings.get(
                member_key
            )
        )

        if member_embedding is None:
            continue

        similarity = manager.cosine_similarity(
            embedding,
            member_embedding,
        )

        rows.append(
            (
                similarity,
                member_key,
            )
        )

    if not rows:
        return None

    rows.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    similarities = [
        item[0]
        for item in rows
    ]

    top_count = min(
        3,
        len(similarities),
    )

    top3_mean = (
        sum(
            similarities[:top_count]
        )
        / top_count
    )

    return {
        "max":
            similarities[0],

        "mean":
            sum(similarities)
            / len(similarities),

        "top3":
            top3_mean,

        "count":
            len(similarities),

        "rows":
            rows,
    }


def cross_core_support(
    manager,
    gid,
    pending_tracklet,
    pending_embedding,
    tracklet_lookup,
):

    identity = manager.identities[
        gid
    ]

    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )

    rows = []

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
            not in manager.core_members.get(
                gid,
                set(),
            )
        ):
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

        direct = manager.cosine_similarity(
            pending_embedding,
            member_embedding,
        )

        geometry = (
            evaluate_cross_camera_geometry(
                pending_tracklet,
                member_tracklet,
                manager.homographies,
            )
        )

        if geometry.status != "SUPPORTED":
            continue

        if (
            geometry.shared_frames
            < CROSS_SHARED_MIN
        ):
            continue

        if (
            geometry.median_distance is None
            or
            geometry.median_distance
            > CROSS_MEDIAN_MAX
        ):
            continue

        if direct < CROSS_CORE_REID_MIN:
            continue

        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "direct":
                    direct,

                "shared":
                    geometry.shared_frames,

                "median":
                    geometry.median_distance,
            }
        )

    return rows


def main():

    print()
    print(
        "CORE-ONLY HIGH-GALLERY NEAR-MISS ANALYSIS"
    )

    print(
        "========================================"
    )

    (
        manager,
        tracklets,
        tracklet_lookup,
        final_results,
    ) = replay_final_production()

    print(
        "Tracklets:",
        len(tracklets),
    )

    print(
        "GIDs:",
        len(manager.identities),
    )

    print(
        "Pending:",
        len(manager.pending_tracklets),
    )

    print()
    print(
        "DIAGNOSTIC GATES"
    )

    print(
        "================"
    )

    print(
        "Cross CORE ReID >=",
        CROSS_CORE_REID_MIN,
    )

    print(
        "Cross shared >=",
        CROSS_SHARED_MIN,
    )

    print(
        "Cross median <=",
        CROSS_MEDIAN_MAX,
    )

    print(
        "CORE gallery max >=",
        CORE_GALLERY_MAX_MIN,
    )

    print(
        "CORE gallery top3 >=",
        CORE_GALLERY_TOP3_MIN,
    )

    unique = []
    ambiguous = []

    for key, pending in sorted(
        manager.pending_tracklets.items()
    ):

        camera_id = normalize_camera_id(
            pending["camera_id"]
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

        qualifying = []

        for gid in sorted(
            manager.anchored_gids
        ):

            identity = manager.identities[
                gid
            ]

            # Preserve all current negative evidence.
            if identity_has_conflict(
                tracklet,
                identity,
                tracklet_lookup,
            ):
                continue

            if manager.identity_geometry_contradicted(
                tracklet,
                identity,
                tracklet_lookup,
            ):
                continue

            scores = core_gallery_scores(
                manager,
                gid,
                pending["embedding"],
            )

            if scores is None:
                continue

            if (
                scores["max"]
                < CORE_GALLERY_MAX_MIN
            ):
                continue

            if (
                scores["top3"]
                < CORE_GALLERY_TOP3_MIN
            ):
                continue

            cross_rows = cross_core_support(
                manager=
                    manager,

                gid=
                    gid,

                pending_tracklet=
                    tracklet,

                pending_embedding=
                    pending["embedding"],

                tracklet_lookup=
                    tracklet_lookup,
            )

            if not cross_rows:
                continue

            qualifying.append(
                {
                    "gid":
                        gid,

                    "scores":
                        scores,

                    "cross":
                        cross_rows,
                }
            )

        if len(qualifying) == 1:

            unique.append(
                (
                    camera_id,
                    local_track_id,
                    qualifying[0],
                )
            )

        elif len(qualifying) > 1:

            ambiguous.append(
                (
                    camera_id,
                    local_track_id,
                    qualifying,
                )
            )

    print()
    print(
        "UNIQUE CORE-ONLY CANDIDATES"
    )

    print(
        "==========================="
    )

    if not unique:
        print("NONE")

    for (
        camera_id,
        local_track_id,
        result,
    ) in unique:

        scores = result[
            "scores"
        ]

        print()
        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> GID "
            f"{result['gid']}"
        )

        print(
            f"  CORE gallery:"
            f" max={scores['max']:.4f}"
            f" | top3={scores['top3']:.4f}"
            f" | mean={scores['mean']:.4f}"
            f" | members={scores['count']}"
        )

        print(
            "  CORE appearance ranking:"
        )

        for similarity, member_key in (
            scores["rows"]
        ):

            member_camera = member_key[0]
            member_track_id = member_key[1]

            print(
                f"    c{member_camera}:"
                f"{member_track_id}"
                f" | ReID="
                f"{similarity:.4f}"
            )

        print(
            "  CROSS CORE support:"
        )

        for row in result[
            "cross"
        ]:

            print(
                f"    c{row['camera_id']}:"
                f"{row['track_id']}"
                f" | ReID="
                f"{row['direct']:.4f}"
                f" | shared="
                f"{row['shared']}"
                f" | median="
                f"{row['median']:.2f}"
            )

    print()
    print(
        "AMBIGUOUS CORE-ONLY CANDIDATES"
    )

    print(
        "=============================="
    )

    if not ambiguous:
        print("NONE")

    for (
        camera_id,
        local_track_id,
        rows,
    ) in ambiguous:

        gids = [
            row["gid"]
            for row in rows
        ]

        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> GIDs "
            f"{gids}"
        )

    print()
    print(
        "SUMMARY"
    )

    print(
        "======="
    )

    print(
        "Unique candidates:",
        len(unique),
    )

    print(
        "Ambiguous candidates:",
        len(ambiguous),
    )


if __name__ == "__main__":
    main()
