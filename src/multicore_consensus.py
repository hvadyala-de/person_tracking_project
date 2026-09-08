from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    identity_has_conflict,
    normalize_camera_id,
)


# ============================================================
# MULTI-CORE CONSENSUS POLICY
#
# These values are the exact values validated by the c2
# positive and negative experiments.
#
# This module does NOT modify GlobalIdentityManager.
# It provides reusable production-oriented candidate logic.
# ============================================================

MULTICORE_MIN_SHARED_FRAMES = 10

MULTICORE_GEOMETRY_MEDIAN_MAX = 30.0

MULTICORE_DIRECT_REID_MIN = 0.80


MULTICORE_ACCEPT_MIN_JOINT = 3

MULTICORE_HOLD_MIN_JOINT = 2


MULTICORE_CORE_MAX_MIN = 0.82

MULTICORE_CORE_TOP3_MIN = 0.80


# ============================================================
# SAME-CAMERA MEMBER CHECK
# ============================================================

def identity_has_same_camera_member(
    identity,
    candidate_camera,
):

    candidate_camera = normalize_camera_id(
        candidate_camera
    )


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        if member_camera == candidate_camera:

            return True


    return False


# ============================================================
# CORE SUPPORT ROWS
#
# Only CORE members may provide positive support.
#
# RELAXED members never propagate positive trust.
# ============================================================

def multicore_support_rows(
    manager,
    gid,
    identity,
    candidate_tracklet,
    embedding,
    tracklet_lookup,
):

    rows = []


    if candidate_tracklet is None:

        return rows


    if tracklet_lookup is None:

        return rows


    if manager.homographies is None:

        return rows


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


        key = (
            member_camera,
            member.local_track_id,
        )


        # ----------------------------------------------------
        # Positive evidence is CORE only.
        # ----------------------------------------------------

        if key not in core_keys:

            continue


        # ----------------------------------------------------
        # Cross-camera evidence only.
        # ----------------------------------------------------

        if member_camera == candidate_camera:

            continue


        member_embedding = (
            manager.member_embeddings.get(
                key
            )
        )


        member_tracklet = (
            manager.get_member_tracklet(
                member,
                tracklet_lookup,
            )
        )


        if (
            member_embedding is None
            or member_tracklet is None
        ):

            continue


        direct_similarity = (
            manager.cosine_similarity(
                embedding,
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


        geometry_supported = (
            geometry.status
            == "SUPPORTED"

            and

            geometry.shared_frames
            >= MULTICORE_MIN_SHARED_FRAMES

            and

            geometry.median_distance
            is not None

            and

            geometry.median_distance
            <= MULTICORE_GEOMETRY_MEDIAN_MAX
        )


        joint_supported = (
            geometry_supported

            and

            direct_similarity
            >= MULTICORE_DIRECT_REID_MIN
        )


        rows.append(
            {
                "camera_id":
                    member_camera,

                "local_track_id":
                    member.local_track_id,

                "direct_similarity":
                    direct_similarity,

                "geometry_status":
                    geometry.status,

                "shared_frames":
                    geometry.shared_frames,

                "median_distance":
                    geometry.median_distance,

                "geometry_supported":
                    geometry_supported,

                "joint_supported":
                    joint_supported,
            }
        )


    rows.sort(
        key=lambda row: (
            row[
                "joint_supported"
            ],

            row[
                "geometry_supported"
            ],

            row[
                "direct_similarity"
            ],

            row[
                "shared_frames"
            ],
        ),
        reverse=True,
    )


    return rows


# ============================================================
# ANALYZE ONE GID
# ============================================================

def analyze_multicore_gid(
    manager,
    gid,
    candidate_tracklet,
    embedding,
    tracklet_lookup,
):

    identity = manager.identities.get(
        gid
    )


    if identity is None:

        return None


    if gid not in manager.anchored_gids:

        return None


    candidate_camera = normalize_camera_id(
        candidate_tracklet.camera_id
    )


    # --------------------------------------------------------
    # Existing identity compatibility rules remain active.
    # --------------------------------------------------------

    if identity_has_conflict(
        candidate_tracklet,
        identity,
        tracklet_lookup,
    ):

        return None


    # --------------------------------------------------------
    # Any CORE or RELAXED contradiction remains a hard veto.
    # --------------------------------------------------------

    if manager.identity_geometry_contradicted(
        candidate_tracklet,
        identity,
        tracklet_lookup,
    ):

        return None


    # --------------------------------------------------------
    # Do not add another member from the candidate camera.
    #
    # This is the protection that blocked c2:153 and c2:701
    # in the validated negative audit.
    # --------------------------------------------------------

    if identity_has_same_camera_member(
        identity,
        candidate_camera,
    ):

        return None


    core_scores = manager.core_gallery_scores(
        gid,
        embedding,
    )


    if core_scores is None:

        return None


    rows = multicore_support_rows(
        manager=
            manager,

        gid=
            gid,

        identity=
            identity,

        candidate_tracklet=
            candidate_tracklet,

        embedding=
            embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    geometry_rows = [
        row

        for row in rows

        if row[
            "geometry_supported"
        ]
    ]


    joint_rows = [
        row

        for row in rows

        if row[
            "joint_supported"
        ]
    ]


    return {
        "global_id":
            gid,

        "core_count":
            core_scores[
                "count"
            ],

        "core_max":
            core_scores[
                "max"
            ],

        "core_top3":
            core_scores[
                "top3"
            ],

        "core_mean":
            core_scores[
                "mean"
            ],

        "geometry_support_count":
            len(
                geometry_rows
            ),

        "joint_support_count":
            len(
                joint_rows
            ),

        "support":
            rows,

        "geometry_support":
            geometry_rows,

        "joint_support":
            joint_rows,
    }


# ============================================================
# FIND ALL ELIGIBLE GIDS
# ============================================================

def find_multicore_candidates(
    manager,
    embedding,
    candidate_tracklet,
    tracklet_lookup,
):

    candidates = []


    if candidate_tracklet is None:

        return candidates


    if tracklet_lookup is None:

        return candidates


    for gid in sorted(
        manager.anchored_gids
    ):

        row = analyze_multicore_gid(
            manager=
                manager,

            gid=
                gid,

            candidate_tracklet=
                candidate_tracklet,

            embedding=
                embedding,

            tracklet_lookup=
                tracklet_lookup,
        )


        if row is None:

            continue


        if (
            row[
                "core_max"
            ]
            < MULTICORE_CORE_MAX_MIN
        ):

            continue


        if (
            row[
                "core_top3"
            ]
            < MULTICORE_CORE_TOP3_MIN
        ):

            continue


        if (
            row[
                "joint_support_count"
            ]
            < MULTICORE_HOLD_MIN_JOINT
        ):

            continue


        candidates.append(
            row
        )


    candidates.sort(
        key=lambda row: (
            row[
                "joint_support_count"
            ],

            row[
                "core_top3"
            ],

            row[
                "core_max"
            ],
        ),
        reverse=True,
    )


    return candidates


# ============================================================
# CLASSIFY UNIQUE MULTI-CORE CANDIDATE
#
# Multiple eligible GIDs are never accepted automatically.
# ============================================================

def classify_multicore_consensus(
    manager,
    embedding,
    candidate_tracklet,
    tracklet_lookup,
):

    candidates = find_multicore_candidates(
        manager=
            manager,

        embedding=
            embedding,

        candidate_tracklet=
            candidate_tracklet,

        tracklet_lookup=
            tracklet_lookup,
    )


    if len(
        candidates
    ) != 1:

        return {
            "status":
                "NONE"
                if not candidates
                else "AMBIGUOUS",

            "candidate":
                None,

            "candidates":
                candidates,
        }


    candidate = candidates[
        0
    ]


    if (
        candidate[
            "joint_support_count"
        ]
        >= MULTICORE_ACCEPT_MIN_JOINT
    ):

        return {
            "status":
                "RELAXED",

            "candidate":
                candidate,

            "candidates":
                candidates,
        }


    return {
        "status":
            "HOLD",

        "candidate":
            candidate,

        "candidates":
            candidates,
    }
