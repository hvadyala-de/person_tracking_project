from collections import Counter
from pathlib import Path

import numpy as np

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.run_global_id_offline import (
    load_all_tracklets,
)

from src.terrace_geometry import (
    load_ground_homographies,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)


# ============================================================
# DIAGNOSTIC LIMITS
#
# These do NOT change production thresholds.
#
# They only define how far outside the current thresholds we
# look when searching for useful "near miss" evidence.
# ============================================================

CROSS_CAMERA_NEAR_REID_MIN = 0.80

CROSS_CAMERA_NEAR_MEDIAN_MAX = 30.0

SAME_CAMERA_NEAR_GAP_MAX = 60

SAME_CAMERA_NEAR_REID_MIN = 0.88

SAME_CAMERA_NEAR_DISTANCE_MAX = 80.0


# ============================================================
# REPLAY CURRENT PRODUCTION PIPELINE
# ============================================================

def replay_production():

    (
        tracklets,
        tracklet_lookup,
    ) = load_all_tracklets()


    tracklets.sort(
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


    for tracklet, embedding in tracklets:

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
                tracklet_lookup,
        )


        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    tracklet_lookup,

                include_same_camera=
                    False,
            )


    # --------------------------------------------------------
    # Final production-only same-camera continuation pass.
    # --------------------------------------------------------

    manager.reevaluate_pending(
        tracklet_lookup=
            tracklet_lookup,

        include_same_camera=
            True,
    )


    return (
        manager,
        tracklets,
        tracklet_lookup,
    )


# ============================================================
# CROSS-CAMERA CORE DIAGNOSTICS
# ============================================================

def cross_camera_core_rows(
    manager,
    pending_tracklet,
    pending_embedding,
    tracklet_lookup,
):

    rows = []


    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )


    for gid in sorted(
        manager.anchored_gids
    ):

        identity = manager.identities.get(
            gid
        )


        if identity is None:
            continue


        gid_scores = manager.score_identity(
            identity,
            pending_embedding,
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


            # ------------------------------------------------
            # Positive diagnostic support is restricted to
            # CORE members, matching production provenance.
            # ------------------------------------------------

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


            evidence = (
                evaluate_cross_camera_geometry(
                    pending_tracklet,
                    member_tracklet,
                    manager.homographies,
                )
            )


            member_embedding = (
                manager.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:
                continue


            direct_similarity = (
                manager.cosine_similarity(
                    pending_embedding,
                    member_embedding,
                )
            )


            rows.append(
                {
                    "gid":
                        gid,

                    "member_camera":
                        member_camera,

                    "member_track_id":
                        member.local_track_id,

                    "geometry_status":
                        evidence.status,

                    "shared_frames":
                        evidence.shared_frames,

                    "median_distance":
                        evidence.median_distance,

                    "direct_similarity":
                        direct_similarity,

                    "gid_max":
                        gid_scores[
                            "max"
                        ],

                    "gid_topk":
                        gid_scores[
                            "topk"
                        ],
                }
            )


    return rows


# ============================================================
# SAME-CAMERA CORE DIAGNOSTICS
# ============================================================

def same_camera_core_rows(
    manager,
    pending_tracklet,
    pending_embedding,
    tracklet_lookup,
):

    rows = []


    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )


    for gid in sorted(
        manager.anchored_gids
    ):

        identity = manager.identities.get(
            gid
        )


        if identity is None:
            continue


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


            (
                gap,
                overlap,
            ) = manager.same_camera_temporal_stats(
                member_tracklet,
                pending_tracklet,
            )


            (
                center_distance,
                bottom_distance,
            ) = (
                manager.same_camera_endpoint_distance(
                    member_tracklet,
                    pending_tracklet,
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
                    "gid":
                        gid,

                    "member_camera":
                        member_camera,

                    "member_track_id":
                        member.local_track_id,

                    "gap":
                        gap,

                    "overlap":
                        overlap,

                    "center_distance":
                        center_distance,

                    "bottom_distance":
                        bottom_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


    return rows


# ============================================================
# CLASSIFY ONE PENDING TRACKLET
# ============================================================

def classify_pending(
    manager,
    pending,
    tracklet_lookup,
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

        return {
            "category":
                "MISSING_TRACKLET",
        }


    embedding = pending[
        "embedding"
    ]


    # ========================================================
    # CURRENT APPEARANCE HYPOTHESIS
    # ========================================================

    appearance = manager.evaluate(
        camera_id=
            camera_id,

        embedding=
            embedding,

        candidate_tracklet=
            tracklet,

        tracklet_lookup=
            tracklet_lookup,
    )


    # ========================================================
    # CHECK WHETHER ANY PRODUCTION TRUSTED CANDIDATE IS
    # AMBIGUOUS.
    #
    # A remaining pending entry should normally have zero
    # accepted candidates, but multiple trusted candidates are
    # worth identifying explicitly.
    # ========================================================

    strict_candidates = (
        manager.find_strict_existing_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    relaxed_candidates = (
        manager.find_relaxed_core_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    same_camera_candidates = (
        manager.find_same_camera_continuation_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if (
        len(strict_candidates) > 1
        or len(relaxed_candidates) > 1
        or len(same_camera_candidates) > 1
    ):

        return {
            "category":
                "AMBIGUOUS_TRUSTED",

            "appearance":
                appearance,

            "strict_count":
                len(
                    strict_candidates
                ),

            "relaxed_count":
                len(
                    relaxed_candidates
                ),

            "same_camera_count":
                len(
                    same_camera_candidates
                ),
        }


    # ========================================================
    # CROSS-CAMERA CORE EVIDENCE
    # ========================================================

    cross_rows = cross_camera_core_rows(
        manager,
        tracklet,
        embedding,
        tracklet_lookup,
    )


    supported_rows = [
        row

        for row in cross_rows

        if (
            row[
                "geometry_status"
            ]
            == "SUPPORTED"

            and

            row[
                "median_distance"
            ]
            is not None

            and

            row[
                "median_distance"
            ]
            <= CROSS_CAMERA_NEAR_MEDIAN_MAX
        )
    ]


    supported_rows.sort(
        key=lambda row: (
            row[
                "direct_similarity"
            ],
            -(
                row[
                    "median_distance"
                ]
                if row[
                    "median_distance"
                ]
                is not None
                else 999999.0
            ),
        ),
        reverse=True,
    )


    # --------------------------------------------------------
    # Best broad geometry-supported CORE relationship.
    # --------------------------------------------------------

    if supported_rows:

        best = supported_rows[0]


        # ----------------------------------------------------
        # Production relaxed direct threshold is .86.
        #
        # Look for cases with meaningful geometry but ReID
        # below that threshold.
        # ----------------------------------------------------

        if (
            best[
                "direct_similarity"
            ]
            >= CROSS_CAMERA_NEAR_REID_MIN

            and

            best[
                "direct_similarity"
            ]
            < manager.relaxed_core_direct_appearance_min
        ):

            return {
                "category":
                    "CROSS_REID_NEAR_MISS",

                "appearance":
                    appearance,

                "best_cross":
                    best,
            }


        # ----------------------------------------------------
        # Direct CORE ReID is sufficient, but some other part
        # of the relaxed production gate failed.
        # ----------------------------------------------------

        if (
            best[
                "direct_similarity"
            ]
            >= manager.relaxed_core_direct_appearance_min
        ):

            strict_geometry_ok = (
                best[
                    "geometry_status"
                ]
                == "SUPPORTED"

                and

                best[
                    "median_distance"
                ]
                is not None

                and

                best[
                    "median_distance"
                ]
                <= manager.relaxed_geometry_median_max

                and

                best[
                    "shared_frames"
                ]
                >= manager.relaxed_min_shared_frames
            )


            gallery_ok = (
                best[
                    "gid_max"
                ]
                >= manager.relaxed_gid_appearance_min

                and

                best[
                    "gid_topk"
                ]
                >= manager.relaxed_gid_appearance_min
            )


            if not strict_geometry_ok:

                return {
                    "category":
                        "CROSS_GEOMETRY_NEAR_MISS",

                    "appearance":
                        appearance,

                    "best_cross":
                        best,
                }


            if not gallery_ok:

                return {
                    "category":
                        "CROSS_GALLERY_NEAR_MISS",

                    "appearance":
                        appearance,

                    "best_cross":
                        best,
                }


    # ========================================================
    # SAME-CAMERA NEAR MISS
    # ========================================================

    same_rows = same_camera_core_rows(
        manager,
        tracklet,
        embedding,
        tracklet_lookup,
    )


    same_near = []


    for row in same_rows:

        if row[
            "overlap"
        ] != 0:

            continue


        if (
            row[
                "gap"
            ] < 0

            or

            row[
                "gap"
            ] > SAME_CAMERA_NEAR_GAP_MAX
        ):

            continue


        if (
            row[
                "direct_similarity"
            ]
            < SAME_CAMERA_NEAR_REID_MIN
        ):

            continue


        if (
            row[
                "center_distance"
            ] is None

            or

            row[
                "bottom_distance"
            ] is None
        ):

            continue


        if (
            row[
                "center_distance"
            ]
            > SAME_CAMERA_NEAR_DISTANCE_MAX
        ):

            continue


        if (
            row[
                "bottom_distance"
            ]
            > SAME_CAMERA_NEAR_DISTANCE_MAX
        ):

            continue


        same_near.append(
            row
        )


    same_near.sort(
        key=lambda row: (
            row[
                "direct_similarity"
            ],
            -row[
                "gap"
            ],
        ),
        reverse=True,
    )


    if same_near:

        return {
            "category":
                "SAME_CAMERA_NEAR_MISS",

            "appearance":
                appearance,

            "best_same":
                same_near[0],
        }


    # ========================================================
    # NO TRUSTED NEAR MISS
    # ========================================================

    if appearance.status == "STRONG":

        category = (
            "APPEARANCE_ONLY_STRONG"
        )


    elif appearance.status == "PENDING":

        category = (
            "APPEARANCE_ONLY_PENDING"
        )


    else:

        category = (
            "NO_USEFUL_EVIDENCE"
        )


    return {
        "category":
            category,

        "appearance":
            appearance,
    }


# ============================================================
# PRINT CROSS-CAMERA ROW
# ============================================================

def print_cross_detail(
    camera_id,
    local_track_id,
    result,
):

    best = result[
        "best_cross"
    ]


    appearance = result[
        "appearance"
    ]


    print(
        f"c{camera_id}:"
        f"{local_track_id}"
        f" | category="
        f"{result['category']}"
        f" | GID="
        f"{best['gid']}"
        f" | core="
        f"c{best['member_camera']}:"
        f"{best['member_track_id']}"
        f" | direct="
        f"{best['direct_similarity']:.4f}"
        f" | shared="
        f"{best['shared_frames']}"
        f" | median="
        f"{best['median_distance']:.2f}"
        f" | gid_max="
        f"{best['gid_max']:.4f}"
        f" | gid_topk="
        f"{best['gid_topk']:.4f}"
        f" | appearance="
        f"{appearance.status}"
    )


# ============================================================
# PRINT SAME-CAMERA ROW
# ============================================================

def print_same_detail(
    camera_id,
    local_track_id,
    result,
):

    best = result[
        "best_same"
    ]


    print(
        f"c{camera_id}:"
        f"{local_track_id}"
        f" | GID="
        f"{best['gid']}"
        f" | CORE="
        f"c{best['member_camera']}:"
        f"{best['member_track_id']}"
        f" | gap="
        f"{best['gap']}"
        f" | ReID="
        f"{best['direct_similarity']:.4f}"
        f" | center="
        f"{best['center_distance']:.2f}"
        f" | bottom="
        f"{best['bottom_distance']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "PENDING POPULATION ANALYSIS"
    )

    print(
        "==========================="
    )


    (
        manager,
        tracklets,
        tracklet_lookup,
    ) = replay_production()


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
        "Anchored GIDs:",
        len(
            manager.anchored_gids
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


    # ========================================================
    # CLASSIFY ALL REMAINING PENDING TRACKLETS
    # ========================================================

    categories = Counter()

    details = []


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


        result = classify_pending(
            manager,
            pending,
            tracklet_lookup,
        )


        categories[
            result[
                "category"
            ]
        ] += 1


        details.append(
            (
                camera_id,
                local_track_id,
                result,
            )
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "CATEGORY COUNTS"
    )

    print(
        "==============="
    )


    for category, count in (
        categories.most_common()
    ):

        print(
            f"{category:<28} "
            f"{count:>4}"
        )


    # ========================================================
    # CROSS-CAMERA NEAR MISSES
    # ========================================================

    print()
    print(
        "CROSS-CAMERA NEAR MISSES"
    )

    print(
        "========================"
    )


    cross_found = False


    for (
        camera_id,
        local_track_id,
        result,
    ) in details:

        if result[
            "category"
        ] not in {
            "CROSS_REID_NEAR_MISS",
            "CROSS_GEOMETRY_NEAR_MISS",
            "CROSS_GALLERY_NEAR_MISS",
        }:

            continue


        cross_found = True


        print_cross_detail(
            camera_id,
            local_track_id,
            result,
        )


    if not cross_found:

        print(
            "NONE"
        )


    # ========================================================
    # SAME-CAMERA NEAR MISSES
    # ========================================================

    print()
    print(
        "SAME-CAMERA NEAR MISSES"
    )

    print(
        "======================="
    )


    same_found = False


    for (
        camera_id,
        local_track_id,
        result,
    ) in details:

        if (
            result[
                "category"
            ]
            != "SAME_CAMERA_NEAR_MISS"
        ):

            continue


        same_found = True


        print_same_detail(
            camera_id,
            local_track_id,
            result,
        )


    if not same_found:

        print(
            "NONE"
        )


    # ========================================================
    # AMBIGUOUS TRUSTED CASES
    # ========================================================

    print()
    print(
        "AMBIGUOUS TRUSTED CASES"
    )

    print(
        "======================="
    )


    ambiguous_found = False


    for (
        camera_id,
        local_track_id,
        result,
    ) in details:

        if (
            result[
                "category"
            ]
            != "AMBIGUOUS_TRUSTED"
        ):

            continue


        ambiguous_found = True


        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" | strict="
            f"{result['strict_count']}"
            f" | relaxed="
            f"{result['relaxed_count']}"
            f" | same-camera="
            f"{result['same_camera_count']}"
        )


    if not ambiguous_found:

        print(
            "NONE"
        )


    # ========================================================
    # APPEARANCE-ONLY POPULATION
    # ========================================================

    print()
    print(
        "APPEARANCE-ONLY SUMMARY"
    )

    print(
        "======================="
    )


    for category in [
        "APPEARANCE_ONLY_STRONG",
        "APPEARANCE_ONLY_PENDING",
        "NO_USEFUL_EVIDENCE",
    ]:

        print(
            f"{category:<28} "
            f"{categories.get(category, 0):>4}"
        )


if __name__ == "__main__":
    main()
