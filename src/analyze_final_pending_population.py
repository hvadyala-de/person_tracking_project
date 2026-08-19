from collections import Counter

from src.analyze_pending_population import (
    classify_pending,
    print_cross_detail,
    print_same_detail,
)

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


# ============================================================
# REPLAY CURRENT FINAL PRODUCTION PIPELINE
#
# Current production order:
#
# chronological processing
#     -> cross-camera pending sweeps only
#
# final pass
#     -> cross-camera relaxed CORE sweep
#     -> same-camera CORE continuation
#     -> dual-evidence continuation
#     -> CORE-only high-gallery continuation
#     -> STOP
#
# Final validated production state:
#
#     GIDs         = 76
#     pending      = 92
#     CORE members = 68
# ============================================================

def replay_final_production():

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


    # ========================================================
    # CHRONOLOGICAL IDENTITY CONSTRUCTION
    #
    # Final-only continuation mechanisms remain disabled.
    # ========================================================

    for tracklet, embedding in tracklets:

        camera_id = normalize_camera_id(
            tracklet.camera_id
        )


        result = manager.assign_tracklet(
            camera_id=
                camera_id,

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

                include_dual_evidence=
                    False,

                include_core_gallery=
                    False,
            )


    # ========================================================
    # FINAL PRODUCTION PASS
    #
    # Order inside GlobalIdentityManager:
    #
    # 1. cross-camera relaxed CORE sweep
    # 2. same-camera CORE continuation
    # 3. dual-evidence continuation
    # 4. CORE-only high-gallery continuation
    # 5. STOP
    #
    # No additional cascade is run afterward.
    # ========================================================

    final_results = (
        manager.reevaluate_pending(
            tracklet_lookup=
                tracklet_lookup,

            include_same_camera=
                True,

            include_dual_evidence=
                True,

            include_core_gallery=
                True,
        )
    )


    return (
        manager,
        tracklets,
        tracklet_lookup,
        final_results,
    )


# ============================================================
# FIND MEMBER GID
# ============================================================

def find_member_gid(
    manager,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            member_camera = (
                normalize_camera_id(
                    member.camera_id
                )
            )


            if (
                member_camera
                == camera_id

                and

                member.local_track_id
                == local_track_id
            ):

                return gid


    return None


# ============================================================
# PRINT FINAL RECOVERY
# ============================================================

def print_final_recovery(
    label,
    result,
):

    print(
        f"{label} | "
        f"c{result['camera_id']}:"
        f"{result['local_track_id']}"
        f" -> GID "
        f"{result['global_id']}"
        f" | trust="
        f"{result.get('trust_level')}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "FINAL PRODUCTION PENDING POPULATION ANALYSIS"
    )

    print(
        "============================================"
    )


    (
        manager,
        tracklets,
        tracklet_lookup,
        final_results,
    ) = replay_final_production()


    # ========================================================
    # FINAL RECOVERY GROUPS
    # ========================================================

    same_camera_results = [
        result

        for result
        in final_results

        if result.get(
            "reason"
        )
        == "SAME_CAMERA_CORE_CONTINUATION"
    ]


    dual_evidence_results = [
        result

        for result
        in final_results

        if result.get(
            "reason"
        )
        == "DUAL_EVIDENCE_CONTINUATION"
    ]


    core_gallery_results = [
        result

        for result
        in final_results

        if result.get(
            "reason"
        )
        == "CORE_GALLERY_CONTINUATION"
    ]


    core_member_count = sum(
        len(
            members
        )

        for members
        in manager.core_members.values()
    )


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
        core_member_count,
    )


    print(
        "Final same-camera resolutions:",
        len(
            same_camera_results
        ),
    )


    print(
        "Final dual-evidence resolutions:",
        len(
            dual_evidence_results
        ),
    )


    print(
        "Final CORE-gallery resolutions:",
        len(
            core_gallery_results
        ),
    )


    # ========================================================
    # FINAL RECOVERIES
    # ========================================================

    print()
    print(
        "FINAL RECOVERIES"
    )

    print(
        "================"
    )


    if same_camera_results:

        for result in same_camera_results:

            print_final_recovery(
                "Same-camera",
                result,
            )

    else:

        print(
            "Same-camera: NONE"
        )


    if dual_evidence_results:

        for result in dual_evidence_results:

            print_final_recovery(
                "Dual-evidence",
                result,
            )

    else:

        print(
            "Dual-evidence: NONE"
        )


    if core_gallery_results:

        for result in core_gallery_results:

            print_final_recovery(
                "CORE-gallery",
                result,
            )

    else:

        print(
            "CORE-gallery: NONE"
        )


    # ========================================================
    # CLASSIFY FINAL REMAINING PENDING TRACKLETS
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
    # CATEGORY COUNTS
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


    total_classified = sum(
        categories.values()
    )


    print()
    print(
        "TOTAL CLASSIFIED:",
        total_classified,
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


    # ========================================================
    # FINAL STATE SANITY CHECK
    #
    # Validate the two final cross-evidence recoveries:
    #
    # c1:556 -> GID 63 [RELAXED]
    # c1:787 -> GID 73 [RELAXED]
    # ========================================================

    key_556 = (
        1,
        556,
    )


    key_787 = (
        1,
        787,
    )


    gid_556 = find_member_gid(
        manager,
        1,
        556,
    )


    gid_787 = find_member_gid(
        manager,
        1,
        787,
    )


    trust_556 = None

    trust_787 = None


    if gid_556 is not None:

        trust_556 = (
            manager.get_member_trust(
                gid_556,
                1,
                556,
            )
        )


    if gid_787 is not None:

        trust_787 = (
            manager.get_member_trust(
                gid_787,
                1,
                787,
            )
        )


    print()
    print(
        "FINAL STATE SANITY CHECK"
    )

    print(
        "========================"
    )


    print(
        "c1:556 still pending:",
        key_556
        in manager.pending_tracklets,
    )


    print(
        "c1:556 assigned GID:",
        gid_556,
    )


    print(
        "c1:556 trust:",
        trust_556,
    )


    print()


    print(
        "c1:787 still pending:",
        key_787
        in manager.pending_tracklets,
    )


    print(
        "c1:787 assigned GID:",
        gid_787,
    )


    print(
        "c1:787 trust:",
        trust_787,
    )


    # ========================================================
    # FINAL VALIDATED PRODUCTION EXPECTATION
    # ========================================================

    expected = (
        len(
            manager.identities
        )
        == 76

        and

        len(
            manager.pending_tracklets
        )
        == 92

        and

        core_member_count
        == 68

        and

        len(
            same_camera_results
        )
        == 2

        and

        len(
            dual_evidence_results
        )
        == 1

        and

        len(
            core_gallery_results
        )
        == 1

        and

        total_classified
        == 92

        and

        key_556
        not in manager.pending_tracklets

        and

        key_787
        not in manager.pending_tracklets

        and

        gid_556
        == 63

        and

        gid_787
        == 73

        and

        trust_556
        == "RELAXED"

        and

        trust_787
        == "RELAXED"
    )


    print()
    print(
        "FINAL PRODUCTION REPLAY:"
    )


    if expected:

        print(
            "PASS"
        )


    else:

        print(
            "FAIL / REVIEW REQUIRED"
        )


if __name__ == "__main__":
    main()
