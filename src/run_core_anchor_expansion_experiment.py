from collections import defaultdict

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    identity_has_conflict,
    normalize_camera_id,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.terrace_geometry import (
    load_ground_homographies,
)

from src.run_anchor_expansion_experiment import (
    STRICT_APPEARANCE_MIN,
    ANCHORED_EXPANSION_APPEARANCE_MIN,
    TRUSTED_GEOMETRY_MEDIAN_MAX,
    load_data,
    cosine_similarity,
    is_trusted_geometry,
    add_to_existing_gid,
    create_anchor_from_pending,
    hold_pending,
    build_member_gid_lookup,
    print_track_state,
)


# ============================================================
# CORE-EXPANSION CONFIGURATION
# ============================================================

# Existing GID-level relaxed threshold.
RELAXED_GID_APPEARANCE_MIN = (
    ANCHORED_EXPANSION_APPEARANCE_MIN
)


# New direct CORE-member requirement.
RELAXED_CORE_DIRECT_APPEARANCE_MIN = 0.86


# Relaxed geometry is slightly stricter than bootstrap.
RELAXED_CORE_GEOMETRY_MEDIAN_MAX = 18.0


# Keep c0:600 <-> c1:503 available.
RELAXED_CORE_MIN_SHARED_FRAMES = 15


# ============================================================
# TRACK KEY
# ============================================================

def track_key(
    camera_id,
    local_track_id,
):

    return (
        normalize_camera_id(
            camera_id
        ),
        local_track_id,
    )


# ============================================================
# EMBEDDING LOOKUP
# ============================================================

def build_embedding_lookup(
    items,
):

    result = {}


    for tracklet, embedding in items:

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        result[
            (
                camera,
                tracklet.local_track_id,
            )
        ] = embedding


    return result


# ============================================================
# GET TRACKLET FOR IDENTITY MEMBER
# ============================================================

def get_member_tracklet(
    member,
    lookup,
):

    camera = normalize_camera_id(
        member.camera_id
    )


    result = lookup.get(
        (
            camera,
            member.local_track_id,
        )
    )


    if result is not None:
        return result


    return lookup.get(
        (
            f"c{camera}",
            member.local_track_id,
        )
    )


# ============================================================
# FORMAT SUPPORT
# ============================================================

def support_text(
    rows,
):

    parts = []


    for row in rows:

        median = row[
            "median"
        ]


        direct = row.get(
            "direct_appearance"
        )


        median_text = (
            "None"
            if median is None
            else f"{median:.2f}"
        )


        direct_text = (
            "None"
            if direct is None
            else f"{direct:.4f}"
        )


        parts.append(
            f"c{row['camera_id']}:"
            f"{row['local_track_id']}"
            f"(app={direct_text},"
            f"shared={row['shared']},"
            f"med={median_text})"
        )


    return "; ".join(
        parts
    )


# ============================================================
# ALL-CROSS-CAMERA CONTRADICTION CHECK
#
# IMPORTANT:
#
# Even relaxed/non-core members remain capable of vetoing a
# candidate if synchronized geometry says CONTRADICTED.
#
# They simply cannot provide positive propagation support.
# ============================================================

def identity_has_geometry_contradiction(
    tracklet,
    identity,
    tracklet_lookup,
    homographies,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )


    contradictions = []


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        if member_camera == camera:
            continue


        member_tracklet = get_member_tracklet(
            member,
            tracklet_lookup,
        )


        if member_tracklet is None:
            continue


        evidence = evaluate_cross_camera_geometry(
            tracklet,
            member_tracklet,
            homographies,
        )


        if evidence.status == "CONTRADICTED":

            contradictions.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "shared":
                        evidence.shared_frames,

                    "median":
                        evidence.median_distance,
                }
            )


    return contradictions


# ============================================================
# STRICT GEOMETRY SUPPORT
#
# For an UNANCHORED singleton:
#     its existing member can bootstrap the anchor.
#
# For an ALREADY ANCHORED GID:
#     positive strict geometry must come from a CORE member.
#
# This prevents a relaxed member from becoming the source of
# later trusted propagation.
# ============================================================

def strict_support_rows(
    gid,
    identity,
    tracklet,
    embedding,
    tracklet_lookup,
    embedding_lookup,
    homographies,
    anchored_gids,
    core_members,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )


    rows = []


    gid_is_anchored = (
        gid in anchored_gids
    )


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        if member_camera == camera:
            continue


        member_key = (
            member_camera,
            member.local_track_id,
        )


        # ----------------------------------------------------
        # Once anchored, only CORE members can provide
        # positive support.
        # ----------------------------------------------------

        if (
            gid_is_anchored
            and
            member_key
            not in core_members[
                gid
            ]
        ):
            continue


        member_tracklet = get_member_tracklet(
            member,
            tracklet_lookup,
        )


        if member_tracklet is None:
            continue


        evidence = evaluate_cross_camera_geometry(
            tracklet,
            member_tracklet,
            homographies,
        )


        if not is_trusted_geometry(
            evidence
        ):
            continue


        member_embedding = embedding_lookup.get(
            member_key
        )


        direct_appearance = None


        if member_embedding is not None:

            direct_appearance = cosine_similarity(
                embedding,
                member_embedding,
            )


        rows.append(
            {
                "camera_id":
                    member_camera,

                "local_track_id":
                    member.local_track_id,

                "shared":
                    evidence.shared_frames,

                "median":
                    evidence.median_distance,

                "direct_appearance":
                    direct_appearance,
            }
        )


    return rows


# ============================================================
# RELAXED CORE SUPPORT
#
# Positive relaxed expansion MUST come directly from CORE.
#
# A relaxed member can never support another relaxed member.
# ============================================================

def relaxed_core_support_rows(
    gid,
    identity,
    tracklet,
    embedding,
    tracklet_lookup,
    embedding_lookup,
    homographies,
    core_members,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )


    rows = []


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        if member_camera == camera:
            continue


        member_key = (
            member_camera,
            member.local_track_id,
        )


        # ----------------------------------------------------
        # CORE ONLY.
        # ----------------------------------------------------

        if member_key not in core_members[
            gid
        ]:
            continue


        member_tracklet = get_member_tracklet(
            member,
            tracklet_lookup,
        )


        if member_tracklet is None:
            continue


        evidence = evaluate_cross_camera_geometry(
            tracklet,
            member_tracklet,
            homographies,
        )


        if evidence.status != "SUPPORTED":
            continue


        if evidence.median_distance is None:
            continue


        if (
            evidence.median_distance
            > RELAXED_CORE_GEOMETRY_MEDIAN_MAX
        ):
            continue


        if (
            evidence.shared_frames
            < RELAXED_CORE_MIN_SHARED_FRAMES
        ):
            continue


        member_embedding = embedding_lookup.get(
            member_key
        )


        if member_embedding is None:
            continue


        direct_appearance = cosine_similarity(
            embedding,
            member_embedding,
        )


        if (
            direct_appearance
            < RELAXED_CORE_DIRECT_APPEARANCE_MIN
        ):
            continue


        rows.append(
            {
                "camera_id":
                    member_camera,

                "local_track_id":
                    member.local_track_id,

                "shared":
                    evidence.shared_frames,

                "median":
                    evidence.median_distance,

                "direct_appearance":
                    direct_appearance,
            }
        )


    return rows


# ============================================================
# FIND STRICT EXISTING CANDIDATES
# ============================================================

def find_strict_existing_candidates(
    manager,
    tracklet,
    embedding,
    tracklet_lookup,
    embedding_lookup,
    homographies,
    anchored_gids,
    core_members,
):

    candidates = []


    for gid, identity in (
        manager.identities.items()
    ):

        # ----------------------------------------------------
        # Same-camera conflict veto.
        # ----------------------------------------------------

        if identity_has_conflict(
            tracklet,
            identity,
            tracklet_lookup,
        ):
            continue


        # ----------------------------------------------------
        # Any geometry contradiction still vetoes.
        # ----------------------------------------------------

        contradictions = (
            identity_has_geometry_contradiction(
                tracklet,
                identity,
                tracklet_lookup,
                homographies,
            )
        )


        if contradictions:
            continue


        supports = strict_support_rows(
            gid,
            identity,
            tracklet,
            embedding,
            tracklet_lookup,
            embedding_lookup,
            homographies,
            anchored_gids,
            core_members,
        )


        if not supports:
            continue


        scores = manager.score_identity(
            identity,
            embedding,
        )


        if (
            scores["max"]
            < STRICT_APPEARANCE_MIN
        ):
            continue


        if (
            scores["topk"]
            < STRICT_APPEARANCE_MIN
        ):
            continue


        candidates.append(
            {
                "gid":
                    gid,

                "max":
                    scores["max"],

                "mean":
                    scores["mean"],

                "topk":
                    scores["topk"],

                "support":
                    supports,
            }
        )


    candidates.sort(
        key=lambda row:
            row["topk"],
        reverse=True,
    )


    return candidates


# ============================================================
# FIND RELAXED CORE CANDIDATES
# ============================================================

def find_relaxed_core_candidates(
    manager,
    tracklet,
    embedding,
    tracklet_lookup,
    embedding_lookup,
    homographies,
    anchored_gids,
    core_members,
):

    candidates = []


    for gid in anchored_gids:

        identity = manager.identities.get(
            gid
        )


        if identity is None:
            continue


        if identity_has_conflict(
            tracklet,
            identity,
            tracklet_lookup,
        ):
            continue


        contradictions = (
            identity_has_geometry_contradiction(
                tracklet,
                identity,
                tracklet_lookup,
                homographies,
            )
        )


        if contradictions:
            continue


        supports = relaxed_core_support_rows(
            gid,
            identity,
            tracklet,
            embedding,
            tracklet_lookup,
            embedding_lookup,
            homographies,
            core_members,
        )


        if not supports:
            continue


        scores = manager.score_identity(
            identity,
            embedding,
        )


        if (
            scores["max"]
            < RELAXED_GID_APPEARANCE_MIN
        ):
            continue


        if (
            scores["topk"]
            < RELAXED_GID_APPEARANCE_MIN
        ):
            continue


        candidates.append(
            {
                "gid":
                    gid,

                "max":
                    scores["max"],

                "mean":
                    scores["mean"],

                "topk":
                    scores["topk"],

                "support":
                    supports,
            }
        )


    candidates.sort(
        key=lambda row:
            row["topk"],
        reverse=True,
    )


    return candidates


# ============================================================
# STRICT CURRENT <-> PENDING PAIR
# ============================================================

def find_strict_pending_candidates(
    manager,
    tracklet,
    embedding,
    tracklet_lookup,
    homographies,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )


    candidates = []


    for key, pending in (
        manager.pending_tracklets.items()
    ):

        pending_camera = normalize_camera_id(
            pending["camera_id"]
        )


        if pending_camera == camera:
            continue


        pending_tracklet = pending.get(
            "candidate_tracklet"
        )


        if pending_tracklet is None:

            pending_tracklet = tracklet_lookup.get(
                (
                    pending_camera,
                    pending[
                        "local_track_id"
                    ],
                )
            )


        if pending_tracklet is None:
            continue


        direct_appearance = cosine_similarity(
            embedding,
            pending[
                "embedding"
            ],
        )


        if (
            direct_appearance
            < STRICT_APPEARANCE_MIN
        ):
            continue


        evidence = evaluate_cross_camera_geometry(
            tracklet,
            pending_tracklet,
            homographies,
        )


        if not is_trusted_geometry(
            evidence
        ):
            continue


        candidates.append(
            {
                "key":
                    key,

                "camera_id":
                    pending_camera,

                "local_track_id":
                    pending[
                        "local_track_id"
                    ],

                "appearance":
                    direct_appearance,

                "evidence":
                    evidence,

                "pending":
                    pending,
            }
        )


    candidates.sort(
        key=lambda row:
            row["appearance"],
        reverse=True,
    )


    return candidates


# ============================================================
# MARK STRICT EXISTING MEMBER AS CORE
# ============================================================

def promote_strict_existing_to_core(
    gid,
    manager,
    tracklet,
    anchored_gids,
    core_members,
):

    # --------------------------------------------------------
    # If this is the first trusted geometry connection for the
    # identity, its pre-existing members become the initial
    # trusted core.
    #
    # Because appearance-only merging is disabled, an
    # unanchored GID should normally be a singleton.
    # --------------------------------------------------------

    if gid not in anchored_gids:

        identity = manager.identities[
            gid
        ]


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


    core_members[
        gid
    ].add(
        track_key(
            tracklet.camera_id,
            tracklet.local_track_id,
        )
    )


# ============================================================
# CREATE STRICT PENDING ANCHOR + CORE
# ============================================================

def create_core_anchor_from_pending(
    manager,
    tracklet,
    embedding,
    candidate,
    anchored_gids,
    core_members,
):

    gid = create_anchor_from_pending(
        manager,
        tracklet,
        embedding,
        candidate,
    )


    anchored_gids.add(
        gid
    )


    core_members[
        gid
    ].add(
        track_key(
            tracklet.camera_id,
            tracklet.local_track_id,
        )
    )


    core_members[
        gid
    ].add(
        (
            candidate[
                "camera_id"
            ],
            candidate[
                "local_track_id"
            ],
        )
    )


    return gid


# ============================================================
# PENDING -> CORE-ANCHORED RELAXED SWEEP
#
# Relaxed members are intentionally NOT added to core_members.
# ============================================================

def sweep_pending_into_core_anchors(
    manager,
    tracklet_lookup,
    embedding_lookup,
    homographies,
    anchored_gids,
    core_members,
):

    resolved = []


    changed = True


    while changed:

        changed = False


        pending_items = list(
            manager.pending_tracklets.items()
        )


        for key, pending in pending_items:

            tracklet = pending.get(
                "candidate_tracklet"
            )


            if tracklet is None:

                camera = normalize_camera_id(
                    pending[
                        "camera_id"
                    ]
                )


                tracklet = tracklet_lookup.get(
                    (
                        camera,
                        pending[
                            "local_track_id"
                        ],
                    )
                )


            if tracklet is None:
                continue


            candidates = (
                find_relaxed_core_candidates(
                    manager,
                    tracklet,
                    pending[
                        "embedding"
                    ],
                    tracklet_lookup,
                    embedding_lookup,
                    homographies,
                    anchored_gids,
                    core_members,
                )
            )


            if len(candidates) != 1:
                continue


            candidate = candidates[0]


            gid = candidate[
                "gid"
            ]


            add_to_existing_gid(
                manager,
                tracklet,
                pending[
                    "embedding"
                ],
                gid,
            )


            # ------------------------------------------------
            # CRITICAL:
            #
            # DO NOT promote this relaxed member into CORE.
            # ------------------------------------------------


            resolved.append(
                {
                    "camera_id":
                        normalize_camera_id(
                            pending[
                                "camera_id"
                            ]
                        ),

                    "local_track_id":
                        pending[
                            "local_track_id"
                        ],

                    "gid":
                        gid,

                    "topk":
                        candidate[
                            "topk"
                        ],

                    "max":
                        candidate[
                            "max"
                        ],

                    "support":
                        candidate[
                            "support"
                        ],
                }
            )


            changed = True


    return resolved


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "CORE-ANCHORED EXPANSION EXPERIMENT"
    )

    print(
        "================================="
    )


    print(
        "Strict appearance minimum:",
        STRICT_APPEARANCE_MIN,
    )


    print(
        "Relaxed GID appearance minimum:",
        RELAXED_GID_APPEARANCE_MIN,
    )


    print(
        "Relaxed direct CORE appearance minimum:",
        RELAXED_CORE_DIRECT_APPEARANCE_MIN,
    )


    print(
        "Relaxed CORE geometry median maximum:",
        RELAXED_CORE_GEOMETRY_MEDIAN_MAX,
    )


    print(
        "Relaxed CORE minimum shared frames:",
        RELAXED_CORE_MIN_SHARED_FRAMES,
    )


    print()
    print(
        "POLICY:"
    )

    print(
        "STRICT members become CORE."
    )

    print(
        "RELAXED members NEVER become CORE."
    )

    print(
        "Appearance-only matches NEVER modify a GID."
    )


    # ========================================================
    # DATA
    # ========================================================

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


    # ========================================================
    # TRUST STATE
    # ========================================================

    anchored_gids = set()


    core_members = defaultdict(
        set
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    strict_existing_count = 0

    strict_pending_count = 0

    relaxed_live_count = 0

    relaxed_sweep_count = 0

    strong_held = 0

    pending_held = 0

    new_gid_count = 0

    ambiguous_strict_existing = 0

    ambiguous_relaxed = 0

    ambiguous_strict_pending = 0


    # ========================================================
    # REPLAY
    # ========================================================

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


        # ----------------------------------------------------
        # FIRST TRACK
        # ----------------------------------------------------

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


            new_gid_count += 1

            continue


        # ====================================================
        # PRIORITY 1
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


            # Mark old identity members as core BEFORE the new
            # member is added if this is the bootstrap event.
            was_anchored = (
                gid in anchored_gids
            )


            if not was_anchored:

                identity = manager.identities[
                    gid
                ]


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


            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )


            # Strict addition becomes CORE.
            core_members[
                gid
            ].add(
                key
            )


            strict_existing_count += 1


            print(
                "STRICT CORE | "
                f"c{camera}:{track_id}"
                f" -> GID {gid} | "
                f"topk="
                f"{candidate['topk']:.4f} | "
                f"support="
                f"{support_text(candidate['support'])}"
            )


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


            for row in swept:

                relaxed_sweep_count += 1


                print(
                    "CORE SWEEP | "
                    f"c{row['camera_id']}:"
                    f"{row['local_track_id']}"
                    f" -> GID {row['gid']} | "
                    f"topk="
                    f"{row['topk']:.4f} | "
                    f"support="
                    f"{support_text(row['support'])}"
                )


            continue


        elif len(
            strict_candidates
        ) > 1:

            ambiguous_strict_existing += 1


        # ====================================================
        # PRIORITY 2
        # RELAXED EXISTING -> CORE SUPPORT ONLY
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


            # -----------------------------------------------
            # CRITICAL:
            # relaxed member does NOT enter core_members.
            # -----------------------------------------------


            relaxed_live_count += 1


            print(
                "RELAXED CORE EXPAND | "
                f"c{camera}:{track_id}"
                f" -> GID {gid} | "
                f"topk="
                f"{candidate['topk']:.4f} | "
                f"support="
                f"{support_text(candidate['support'])}"
            )


            continue


        elif len(
            relaxed_candidates
        ) > 1:

            ambiguous_relaxed += 1


        # ====================================================
        # PRIORITY 3
        # STRICT CURRENT <-> PENDING ANCHOR
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


            gid = (
                create_core_anchor_from_pending(
                    manager,
                    tracklet,
                    embedding,
                    candidate,
                    anchored_gids,
                    core_members,
                )
            )


            strict_pending_count += 1


            evidence = candidate[
                "evidence"
            ]


            print(
                "STRICT CORE ANCHOR | "
                f"GID {gid} | "
                f"c{camera}:{track_id}"
                f" <-> "
                f"c{candidate['camera_id']}:"
                f"{candidate['local_track_id']} | "
                f"appearance="
                f"{candidate['appearance']:.4f} | "
                f"shared="
                f"{evidence.shared_frames} | "
                f"median="
                f"{evidence.median_distance:.2f}"
            )


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


            for row in swept:

                relaxed_sweep_count += 1


                print(
                    "CORE SWEEP | "
                    f"c{row['camera_id']}:"
                    f"{row['local_track_id']}"
                    f" -> GID {row['gid']} | "
                    f"topk="
                    f"{row['topk']:.4f} | "
                    f"support="
                    f"{support_text(row['support'])}"
                )


            continue


        elif len(
            pending_candidates
        ) > 1:

            ambiguous_strict_pending += 1


        # ====================================================
        # PRIORITY 4
        # APPEARANCE-ONLY
        #
        # Still never modifies the identity gallery.
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


            strong_held += 1

            continue


        if match.status == "PENDING":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )


            pending_held += 1

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


        new_gid_count += 1


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


    for row in swept:

        relaxed_sweep_count += 1


        print(
            "FINAL CORE SWEEP | "
            f"c{row['camera_id']}:"
            f"{row['local_track_id']}"
            f" -> GID {row['gid']} | "
            f"topk="
            f"{row['topk']:.4f} | "
            f"support="
            f"{support_text(row['support'])}"
        )


    # ========================================================
    # FINAL MAPPING
    # ========================================================

    mapping = build_member_gid_lookup(
        manager
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "FINAL RESULT"
    )

    print(
        "============"
    )


    print(
        "Tracklets:",
        len(items),
    )


    print(
        "Anchored GIDs:",
        len(
            anchored_gids
        ),
    )


    print(
        "Core members:",
        sum(
            len(members)
            for members
            in core_members.values()
        ),
    )


    print(
        "Strict existing resolutions:",
        strict_existing_count,
    )


    print(
        "Strict pending anchors:",
        strict_pending_count,
    )


    print(
        "Relaxed LIVE core expansions:",
        relaxed_live_count,
    )


    print(
        "Relaxed SWEEP core expansions:",
        relaxed_sweep_count,
    )


    print(
        "Appearance STRONG held:",
        strong_held,
    )


    print(
        "Appearance PENDING held:",
        pending_held,
    )


    print(
        "New singleton GIDs:",
        new_gid_count,
    )


    print(
        "Ambiguous strict existing:",
        ambiguous_strict_existing,
    )


    print(
        "Ambiguous relaxed core:",
        ambiguous_relaxed,
    )


    print(
        "Ambiguous strict pending:",
        ambiguous_strict_pending,
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


    # ========================================================
    # IMPORTANT TRACK MAPPING
    # ========================================================

    print()
    print(
        "IMPORTANT TRACK MAPPING"
    )

    print(
        "======================="
    )


    important = [
        (0, 255),
        (1, 298),
        (1, 319),

        (0, 399),
        (1, 369),
        (1, 382),

        (0, 469),
        (1, 503),
        (0, 600),

        (0, 492),
        (1, 511),

        (0, 431),
        (1, 423),

        (1, 560),
        (0, 651),

        (1, 639),
        (0, 701),
        (0, 740),
        (1, 721),

        (0, 1),
        (1, 1),
        (1, 66),
    ]


    for camera, track_id in important:

        print_track_state(
            mapping,
            manager,
            camera,
            track_id,
        )


    # ========================================================
    # CORE MEMBERS BY GID
    # ========================================================

    print()
    print(
        "CORE MEMBERS BY GID"
    )

    print(
        "==================="
    )


    for gid in sorted(
        anchored_gids
    ):

        members = sorted(
            core_members[
                gid
            ]
        )


        text = ", ".join(
            f"c{camera}:{track_id}"
            for camera, track_id
            in members
        )


        print(
            f"GID {gid}: {text}"
        )


    # ========================================================
    # LARGEST IDENTITIES
    # ========================================================

    print()
    print(
        "LARGEST GLOBAL IDENTITIES"
    )

    print(
        "========================="
    )


    identities = sorted(
        manager.identities.values(),

        key=lambda identity:
            len(
                identity.members
            ),

        reverse=True,
    )


    for identity in identities[:20]:

        print(
            identity.summary()
        )


if __name__ == "__main__":
    main()
