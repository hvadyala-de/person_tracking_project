from src.identity_compatibility import (
    normalize_camera_id,
)

from src.run_global_id_offline import (
    load_all_tracklets,
)

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.terrace_geometry import (
    load_ground_homographies,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)


# ============================================================
# SAME-CAMERA CANDIDATES FROM THE MOTION RANKING
#
# Format:
#
# (
#     pending_camera,
#     pending_track_id,
#     gid,
#     same_camera_core_track_id,
# )
# ============================================================

CANDIDATES = [

    (0, 748, 60, 609),
    (0, 331, 39, 431),
    (1, 565, 47, 503),
    (0, 868, 76, 896),
    (1, 556, 63, 635),
    (1, 622, 38, 453),
    (0, 264, 11, 243),
    (0, 171, 9, 78),
    (0, 622, 63, 666),
    (0, 319, 38, 379),
    (1, 733, 66, 536),
    (1, 420, 25, 298),
]


# ============================================================
# REPLAY CURRENT PRODUCTION STATE
# ============================================================

def replay_production():

    (
        items,
        lookup,
    ) = load_all_tracklets()


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

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        result = manager.assign_tracklet(
            camera_id=
                camera,

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
                tracklet_lookup=
                    lookup,

                include_same_camera=
                    False,
            )


    manager.reevaluate_pending(
        tracklet_lookup=
            lookup,

        include_same_camera=
            True,
    )


    return (
        manager,
        lookup,
    )


# ============================================================
# CROSS-CAMERA EVIDENCE FOR ONE PENDING TRACKLET AGAINST GID
# ============================================================

def collect_cross_camera_evidence(
    manager,
    pending_tracklet,
    pending_embedding,
    gid,
    lookup,
):

    rows = []


    identity = manager.identities.get(
        gid
    )


    if identity is None:

        return rows


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


        # ----------------------------------------------------
        # Positive evidence must still come from CORE.
        # ----------------------------------------------------

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


        geometry = (
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


        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "trust":
                    "CORE",

                "reid":
                    direct_similarity,

                "status":
                    geometry.status,

                "shared":
                    geometry.shared_frames,

                "median":
                    geometry.median_distance,

                "mean":
                    geometry.mean_distance,

                "p90":
                    geometry.p90_distance,
            }
        )


    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "DUAL-EVIDENCE PENDING ANALYSIS"
    )

    print(
        "=============================="
    )


    manager, lookup = replay_production()


    print(
        "Production GIDs:",
        len(
            manager.identities
        ),
    )


    print(
        "Production pending:",
        len(
            manager.pending_tracklets
        ),
    )


    print()
    print(
        "CANDIDATE DETAILS"
    )

    print(
        "================="
    )


    supported_candidates = []


    for (
        camera_id,
        pending_id,
        gid,
        same_core_id,
    ) in CANDIDATES:

        camera_id = normalize_camera_id(
            camera_id
        )


        pending_key = (
            camera_id,
            pending_id,
        )


        pending = (
            manager.pending_tracklets.get(
                pending_key
            )
        )


        print()
        print(
            f"c{camera_id}:"
            f"{pending_id}"
            f" -> GID {gid}"
            f" | same-camera CORE="
            f"c{camera_id}:"
            f"{same_core_id}"
        )


        if pending is None:

            print(
                "  NOT CURRENTLY PENDING"
            )

            continue


        pending_tracklet = pending.get(
            "candidate_tracklet"
        )


        if pending_tracklet is None:

            pending_tracklet = (
                manager.get_tracklet(
                    camera_id,
                    pending_id,
                    lookup,
                )
            )


        if pending_tracklet is None:

            print(
                "  MISSING TRACKLET"
            )

            continue


        rows = collect_cross_camera_evidence(
            manager=
                manager,

            pending_tracklet=
                pending_tracklet,

            pending_embedding=
                pending[
                    "embedding"
                ],

            gid=
                gid,

            lookup=
                lookup,
        )


        if not rows:

            print(
                "  No cross-camera CORE members"
            )

            continue


        has_supported = False

        has_contradicted = False


        for row in rows:

            median_text = (
                f"{row['median']:.2f}"
                if row[
                    "median"
                ]
                is not None
                else "None"
            )


            mean_text = (
                f"{row['mean']:.2f}"
                if row[
                    "mean"
                ]
                is not None
                else "None"
            )


            p90_text = (
                f"{row['p90']:.2f}"
                if row[
                    "p90"
                ]
                is not None
                else "None"
            )


            print(
                f"  vs CORE "
                f"c{row['camera_id']}:"
                f"{row['track_id']}"
                f" | geom="
                f"{row['status']}"
                f" | shared="
                f"{row['shared']}"
                f" | median="
                f"{median_text}"
                f" | mean="
                f"{mean_text}"
                f" | p90="
                f"{p90_text}"
                f" | ReID="
                f"{row['reid']:.4f}"
            )


            if (
                row[
                    "status"
                ]
                == "SUPPORTED"
            ):

                has_supported = True


            if (
                row[
                    "status"
                ]
                == "CONTRADICTED"
            ):

                has_contradicted = True


        if (
            has_supported
            and not has_contradicted
        ):

            supported_candidates.append(
                (
                    camera_id,
                    pending_id,
                    gid,
                    same_core_id,
                )
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "DUAL-EVIDENCE CANDIDATES"
    )

    print(
        "========================"
    )


    if not supported_candidates:

        print(
            "NONE"
        )


    else:

        for (
            camera_id,
            pending_id,
            gid,
            same_core_id,
        ) in supported_candidates:

            print(
                f"c{camera_id}:"
                f"{pending_id}"
                f" -> GID {gid}"
                f" | same-camera CORE="
                f"c{camera_id}:"
                f"{same_core_id}"
                f" | cross-camera geometry="
                f"SUPPORTED"
            )


if __name__ == "__main__":
    main()
