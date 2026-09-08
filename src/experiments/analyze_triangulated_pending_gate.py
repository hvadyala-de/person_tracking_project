import math

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
# This is deliberately conservative.
#
# A pending tracklet must have:
#
# 1. independent cross-camera support from a CORE member
# 2. strong whole-GID appearance
# 3. strong same-camera overlapping duplicate evidence
#    from another member of THAT SAME GID
# 4. no identity conflict / geometry contradiction
#
# No production code is changed by this script.
# ============================================================


# Independent CORE cross-camera evidence
CROSS_DIRECT_MIN = 0.83
CROSS_SHARED_MIN = 20
CROSS_MEDIAN_MAX = 18.0

# Whole identity appearance
GID_MAX_MIN = 0.90
GID_TOPK_MIN = 0.90

# Same-camera duplicate corroboration
DUPLICATE_REID_MIN = 0.90
DUPLICATE_SHARED_MIN = 5
DUPLICATE_MEDIAN_IOU_MIN = 0.30
DUPLICATE_MEDIAN_CENTER_NORM_MAX = 0.25


def bbox_width(det):
    return max(
        0.0,
        float(det.x2 - det.x1),
    )


def bbox_height(det):
    return max(
        0.0,
        float(det.y2 - det.y1),
    )


def bbox_area(det):
    return (
        bbox_width(det)
        * bbox_height(det)
    )


def bbox_center(det):
    return (
        (det.x1 + det.x2) / 2.0,
        (det.y1 + det.y2) / 2.0,
    )


def euclidean(first, second):
    dx = first[0] - second[0]
    dy = first[1] - second[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def bbox_iou(first, second):

    ix1 = max(first.x1, second.x1)
    iy1 = max(first.y1, second.y1)

    ix2 = min(first.x2, second.x2)
    iy2 = min(first.y2, second.y2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    union = (
        bbox_area(first)
        + bbox_area(second)
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return float(
        intersection / union
    )


def median(values):

    values = sorted(values)

    if not values:
        return None

    size = len(values)

    midpoint = size // 2

    if size % 2:
        return values[midpoint]

    return (
        values[midpoint - 1]
        + values[midpoint]
    ) / 2.0


def detection_map(tracklet):

    return {
        int(det.frame): det

        for det
        in tracklet.detections
    }


# ============================================================
# SAME-CAMERA HIGH-SPATIAL-OVERLAP CORROBORATION
# ============================================================

def duplicate_evidence(
    manager,
    pending_tracklet,
    pending_embedding,
    member_tracklet,
    member_embedding,
):

    similarity = manager.cosine_similarity(
        pending_embedding,
        member_embedding,
    )

    if similarity < DUPLICATE_REID_MIN:
        return None

    pending_map = detection_map(
        pending_tracklet
    )

    member_map = detection_map(
        member_tracklet
    )

    shared_frames = sorted(
        set(pending_map)
        &
        set(member_map)
    )

    if (
        len(shared_frames)
        < DUPLICATE_SHARED_MIN
    ):
        return None

    ious = []
    normalized_centers = []

    for frame in shared_frames:

        first = pending_map[frame]
        second = member_map[frame]

        ious.append(
            bbox_iou(
                first,
                second,
            )
        )

        center_distance = euclidean(
            bbox_center(first),
            bbox_center(second),
        )

        width_a = bbox_width(first)
        height_a = bbox_height(first)

        width_b = bbox_width(second)
        height_b = bbox_height(second)

        diagonal_a = math.sqrt(
            width_a * width_a
            + height_a * height_a
        )

        diagonal_b = math.sqrt(
            width_b * width_b
            + height_b * height_b
        )

        reference_diagonal = max(
            diagonal_a,
            diagonal_b,
            1.0,
        )

        normalized_centers.append(
            center_distance
            / reference_diagonal
        )

    median_iou = median(
        ious
    )

    median_center = median(
        normalized_centers
    )

    if (
        median_iou
        < DUPLICATE_MEDIAN_IOU_MIN
    ):
        return None

    if (
        median_center
        > DUPLICATE_MEDIAN_CENTER_NORM_MAX
    ):
        return None

    iou_30_count = sum(
        1
        for value in ious
        if value >= 0.30
    )

    iou_50_count = sum(
        1
        for value in ious
        if value >= 0.50
    )

    return {
        "reid":
            similarity,

        "shared":
            len(shared_frames),

        "median_iou":
            median_iou,

        "median_center":
            median_center,

        "iou30":
            iou_30_count,

        "iou50":
            iou_50_count,
    }


# ============================================================
# CROSS-CAMERA CORE SUPPORT
# ============================================================

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

        geometry = (
            evaluate_cross_camera_geometry(
                pending_tracklet,
                member_tracklet,
                manager.homographies,
            )
        )

        direct = manager.cosine_similarity(
            pending_embedding,
            member_embedding,
        )

        if (
            geometry.status
            != "SUPPORTED"
        ):
            continue

        if (
            geometry.shared_frames
            < CROSS_SHARED_MIN
        ):
            continue

        if (
            geometry.median_distance
            is None
            or
            geometry.median_distance
            > CROSS_MEDIAN_MAX
        ):
            continue

        if direct < CROSS_DIRECT_MIN:
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


# ============================================================
# SAME-CAMERA MEMBER CORROBORATION
# ============================================================

def same_camera_duplicate_support(
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

        if member_camera != pending_camera:
            continue

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

        evidence = duplicate_evidence(
            manager=
                manager,

            pending_tracklet=
                pending_tracklet,

            pending_embedding=
                pending_embedding,

            member_tracklet=
                member_tracklet,

            member_embedding=
                member_embedding,
        )

        if evidence is None:
            continue

        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
                    member.local_track_id,

                "trust":
                    manager.get_member_trust(
                        gid,
                        member_camera,
                        member.local_track_id,
                    ),

                **evidence,
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TRIANGULATED PENDING EVIDENCE ANALYSIS"
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
        "Cross CORE direct ReID >=",
        CROSS_DIRECT_MIN,
    )

    print(
        "Cross CORE shared >=",
        CROSS_SHARED_MIN,
    )

    print(
        "Cross CORE median <=",
        CROSS_MEDIAN_MAX,
    )

    print(
        "GID max >=",
        GID_MAX_MIN,
    )

    print(
        "GID topk >=",
        GID_TOPK_MIN,
    )

    print(
        "Duplicate ReID >=",
        DUPLICATE_REID_MIN,
    )

    print(
        "Duplicate shared >=",
        DUPLICATE_SHARED_MIN,
    )

    print(
        "Duplicate median IoU >=",
        DUPLICATE_MEDIAN_IOU_MIN,
    )

    print(
        "Duplicate median norm center <=",
        DUPLICATE_MEDIAN_CENTER_NORM_MAX,
    )

    candidates = []
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

        pending_tracklet = pending.get(
            "candidate_tracklet"
        )

        if pending_tracklet is None:

            pending_tracklet = (
                manager.get_tracklet(
                    camera_id,
                    local_track_id,
                    tracklet_lookup,
                )
            )

        if pending_tracklet is None:
            continue

        qualifying_gids = []

        for gid in sorted(
            manager.anchored_gids
        ):

            identity = manager.identities[
                gid
            ]

            # --------------------------------------------
            # Preserve negative evidence.
            # --------------------------------------------

            if identity_has_conflict(
                pending_tracklet,
                identity,
                tracklet_lookup,
            ):
                continue

            if (
                manager.identity_geometry_contradicted(
                    pending_tracklet,
                    identity,
                    tracklet_lookup,
                )
            ):
                continue

            scores = manager.score_identity(
                identity,
                pending["embedding"],
            )

            if (
                scores["max"]
                < GID_MAX_MIN
            ):
                continue

            if (
                scores["topk"]
                < GID_TOPK_MIN
            ):
                continue

            cross_rows = (
                cross_core_support(
                    manager=
                        manager,

                    gid=
                        gid,

                    pending_tracklet=
                        pending_tracklet,

                    pending_embedding=
                        pending["embedding"],

                    tracklet_lookup=
                        tracklet_lookup,
                )
            )

            if not cross_rows:
                continue

            duplicate_rows = (
                same_camera_duplicate_support(
                    manager=
                        manager,

                    gid=
                        gid,

                    pending_tracklet=
                        pending_tracklet,

                    pending_embedding=
                        pending["embedding"],

                    tracklet_lookup=
                        tracklet_lookup,
                )
            )

            if not duplicate_rows:
                continue

            qualifying_gids.append(
                {
                    "gid":
                        gid,

                    "scores":
                        scores,

                    "cross":
                        cross_rows,

                    "duplicate":
                        duplicate_rows,
                }
            )

        if len(qualifying_gids) == 1:

            candidates.append(
                (
                    camera_id,
                    local_track_id,
                    qualifying_gids[0],
                )
            )

        elif len(qualifying_gids) > 1:

            ambiguous.append(
                (
                    camera_id,
                    local_track_id,
                    qualifying_gids,
                )
            )

    print()
    print(
        "UNIQUE TRIANGULATED CANDIDATES"
    )

    print(
        "=============================="
    )

    if not candidates:

        print(
            "NONE"
        )

    for (
        camera_id,
        local_track_id,
        result,
    ) in candidates:

        print()
        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> GID "
            f"{result['gid']}"
        )

        print(
            f"  gallery max="
            f"{result['scores']['max']:.4f}"
            f" | topk="
            f"{result['scores']['topk']:.4f}"
        )

        for row in result[
            "cross"
        ]:

            print(
                f"  CROSS CORE "
                f"c{row['camera_id']}:"
                f"{row['track_id']}"
                f" | ReID="
                f"{row['direct']:.4f}"
                f" | shared="
                f"{row['shared']}"
                f" | median="
                f"{row['median']:.2f}"
            )

        for row in result[
            "duplicate"
        ]:

            print(
                f"  SAME "
                f"{row['trust']} "
                f"c{row['camera_id']}:"
                f"{row['track_id']}"
                f" | ReID="
                f"{row['reid']:.4f}"
                f" | shared="
                f"{row['shared']}"
                f" | median IoU="
                f"{row['median_iou']:.3f}"
                f" | median norm center="
                f"{row['median_center']:.3f}"
                f" | IoU>=.30="
                f"{row['iou30']}/"
                f"{row['shared']}"
                f" | IoU>=.50="
                f"{row['iou50']}/"
                f"{row['shared']}"
            )

    print()
    print(
        "AMBIGUOUS TRIANGULATED CANDIDATES"
    )

    print(
        "================================="
    )

    if not ambiguous:

        print(
            "NONE"
        )

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
        len(candidates),
    )

    print(
        "Ambiguous candidates:",
        len(ambiguous),
    )


if __name__ == "__main__":
    main()
