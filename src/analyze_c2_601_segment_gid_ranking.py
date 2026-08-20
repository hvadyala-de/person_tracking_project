from pathlib import Path

import cv2
import numpy as np
import torch

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.reid_extractor import (
    ReIDExtractor,
)

from src.tracklet import (
    Tracklet,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c2.avi"
)

TARGET_TRACK_ID = 601

EARLY_START = 3796
EARLY_END = 4016

LATE_START = 4049
LATE_END = 4571

CROPS_PER_SEGMENT = 8


# ============================================================
# BROAD DIAGNOSTIC GEOMETRY SETTINGS
#
# These are analysis thresholds only.
# They do NOT change production.
# ============================================================

MIN_SHARED_FRAMES = 10

GEOMETRY_MEDIAN_MAX = 30.0

DIRECT_REID_INTERESTING_MIN = 0.80

TOP_GIDS = 12


# ============================================================
# VECTOR HELPERS
# ============================================================

def normalize(vector):

    if vector is None:
        return None

    if isinstance(
        vector,
        torch.Tensor,
    ):

        vector = (
            vector
            .detach()
            .cpu()
            .numpy()
        )

    vector = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(
        vector
    )

    if norm <= 0.0:
        return None

    return (
        vector
        / norm
    )


def cosine_similarity(
    embedding_a,
    embedding_b,
):

    return float(
        np.dot(
            embedding_a,
            embedding_b,
        )
    )


# ============================================================
# TEMPORARY TRACKLET SEGMENT
# ============================================================

def build_segment_tracklet(
    source_tracklet,
    start_frame,
    end_frame,
    synthetic_track_id,
):

    detections = [
        detection

        for detection
        in source_tracklet.detections

        if (
            start_frame
            <= int(
                detection.frame
            )
            <= end_frame
        )
    ]

    segment = Tracklet(
        camera_id=
            source_tracklet.camera_id,

        local_track_id=
            synthetic_track_id,

        detections=
            list(
                detections
            ),
    )

    segment.sort_detections()

    return segment


# ============================================================
# CROP SELECTION
# ============================================================

def select_detections(
    tracklet,
    count,
):

    detections = list(
        tracklet.detections
    )

    detections.sort(
        key=lambda detection:
            int(
                detection.frame
            )
    )

    if not detections:
        return []

    if len(
        detections
    ) <= count:

        return detections

    indices = np.linspace(
        0,
        len(detections) - 1,
        num=count,
        dtype=int,
    )

    selected = []

    seen_frames = set()

    for index in indices:

        detection = detections[
            int(
                index
            )
        ]

        frame = int(
            detection.frame
        )

        if frame in seen_frames:
            continue

        seen_frames.add(
            frame
        )

        selected.append(
            detection
        )

    return selected


def crop_detection(
    image,
    detection,
):

    height, width = image.shape[
        :2
    ]

    x1 = max(
        0,
        int(
            round(
                detection.x1
            )
        ),
    )

    y1 = max(
        0,
        int(
            round(
                detection.y1
            )
        ),
    )

    x2 = min(
        width,
        int(
            round(
                detection.x2
            )
        ),
    )

    y2 = min(
        height,
        int(
            round(
                detection.y2
            )
        ),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return None

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# BUILD SEGMENT EMBEDDING
#
# Mirrors the normal tracklet embedding strategy:
#
#   crop embeddings
#       ->
#   mean
#       ->
#   L2 normalize
# ============================================================

def build_segment_embedding(
    extractor,
    video,
    tracklet,
):

    selected = select_detections(
        tracklet,
        CROPS_PER_SEGMENT,
    )

    embeddings = []

    selected_frames = []

    for detection in selected:

        frame = int(
            detection.frame
        )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame,
        )

        ok, image = video.read()

        if not ok:
            continue

        crop = crop_detection(
            image,
            detection,
        )

        if crop is None:
            continue

        feature = extractor.extract(
            crop
        )

        feature = normalize(
            feature
        )

        if feature is None:
            continue

        embeddings.append(
            feature
        )

        selected_frames.append(
            frame
        )

    if not embeddings:

        raise RuntimeError(
            "Could not build segment embedding"
        )

    mean_embedding = np.mean(
        np.stack(
            embeddings,
            axis=0,
        ),
        axis=0,
    )

    mean_embedding = normalize(
        mean_embedding
    )

    if mean_embedding is None:

        raise RuntimeError(
            "Could not normalize segment embedding"
        )

    return (
        mean_embedding,
        selected_frames,
    )


# ============================================================
# CORE MEMBER RELATIONS
# ============================================================

def analyze_core_members(
    manager,
    gid,
    candidate_tracklet,
    candidate_embedding,
    tracklet_lookup,
):

    identity = manager.identities[
        gid
    ]

    core_keys = manager.core_members.get(
        gid,
        set(),
    )

    rows = []

    candidate_camera = normalize_camera_id(
        candidate_tracklet.camera_id
    )

    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )

        key = (
            member_camera,
            member.local_track_id,
        )

        if key not in core_keys:
            continue

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

        member_embedding = normalize(
            member_embedding
        )

        direct_similarity = (
            cosine_similarity(
                candidate_embedding,
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
            >= MIN_SHARED_FRAMES

            and

            geometry.median_distance
            is not None

            and

            geometry.median_distance
            <= GEOMETRY_MEDIAN_MAX
        )

        interesting_joint = (
            geometry_supported

            and

            direct_similarity
            >= DIRECT_REID_INTERESTING_MIN
        )

        rows.append(
            {
                "camera_id":
                    member_camera,

                "track_id":
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

                "interesting_joint":
                    interesting_joint,
            }
        )

    rows.sort(
        key=lambda row: (
            row[
                "interesting_joint"
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

def analyze_gid(
    manager,
    gid,
    candidate_tracklet,
    candidate_embedding,
    tracklet_lookup,
):

    identity = manager.identities.get(
        gid
    )

    if identity is None:
        return None

    core_scores = (
        manager.core_gallery_scores(
            gid,
            candidate_embedding,
        )
    )

    if core_scores is None:
        return None

    core_rows = analyze_core_members(
        manager=
            manager,

        gid=
            gid,

        candidate_tracklet=
            candidate_tracklet,

        candidate_embedding=
            candidate_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )

    geometry_supported_rows = [
        row

        for row in core_rows

        if row[
            "geometry_supported"
        ]
    ]

    joint_rows = [
        row

        for row in core_rows

        if row[
            "interesting_joint"
        ]
    ]

    core_contradictions = [
        row

        for row in core_rows

        if row[
            "geometry_status"
        ] == "CONTRADICTED"
    ]

    all_member_veto = (
        manager.identity_geometry_contradicted(
            candidate_tracklet,
            identity,
            tracklet_lookup,
        )
    )

    return {
        "gid":
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
                geometry_supported_rows
            ),

        "joint_support_count":
            len(
                joint_rows
            ),

        "core_contradiction_count":
            len(
                core_contradictions
            ),

        "identity_veto":
            all_member_veto,

        "core_rows":
            core_rows,

        "geometry_supported_rows":
            geometry_supported_rows,

        "joint_rows":
            joint_rows,

        "core_contradictions":
            core_contradictions,
    }


# ============================================================
# RANK ALL ANCHORED GIDS
# ============================================================

def rank_segment(
    manager,
    candidate_tracklet,
    candidate_embedding,
    tracklet_lookup,
):

    rows = []

    for gid in sorted(
        manager.anchored_gids
    ):

        result = analyze_gid(
            manager=
                manager,

            gid=
                gid,

            candidate_tracklet=
                candidate_tracklet,

            candidate_embedding=
                candidate_embedding,

            tracklet_lookup=
                tracklet_lookup,
        )

        if result is not None:

            rows.append(
                result
            )

    return rows


# ============================================================
# PRINT ONE GID
# ============================================================

def print_gid_result(
    result,
):

    print(
        f"GID {result['gid']:<3}"
        f" | gallery max="
        f"{result['core_max']:.4f}"
        f" top3="
        f"{result['core_top3']:.4f}"
        f" mean="
        f"{result['core_mean']:.4f}"
        f" | CORE={result['core_count']}"
        f" | geom+="
        f"{result['geometry_support_count']}"
        f" | joint+="
        f"{result['joint_support_count']}"
        f" | CORE contradictions="
        f"{result['core_contradiction_count']}"
        f" | identity veto="
        f"{result['identity_veto']}"
    )

    for row in result[
        "core_rows"
    ]:

        median = row[
            "median_distance"
        ]

        if median is None:

            median_text = "None"

        else:

            median_text = (
                f"{median:.2f}"
            )

        flags = []

        if row[
            "geometry_supported"
        ]:

            flags.append(
                "GEOM+"
            )

        if row[
            "interesting_joint"
        ]:

            flags.append(
                "JOINT+"
            )

        if (
            row[
                "geometry_status"
            ]
            == "CONTRADICTED"
        ):

            flags.append(
                "CONTRADICTED"
            )

        flag_text = (
            ",".join(
                flags
            )
            if flags
            else "-"
        )

        print(
            f"    CORE "
            f"c{row['camera_id']}:"
            f"{row['track_id']}"
            f" | ReID="
            f"{row['direct_similarity']:.4f}"
            f" | geom="
            f"{row['geometry_status']}"
            f" | shared="
            f"{row['shared_frames']}"
            f" | median="
            f"{median_text}"
            f" | {flag_text}"
        )


# ============================================================
# REPORT SEGMENT
# ============================================================

def report_segment(
    name,
    rows,
):

    print()
    print()
    print(
        "=" * 78
    )

    print(
        f"{name} SEGMENT GID RANKING"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Appearance ranking
    # --------------------------------------------------------

    by_appearance = sorted(
        rows,
        key=lambda row: (
            row[
                "core_top3"
            ],

            row[
                "core_max"
            ],

            row[
                "core_mean"
            ],
        ),
        reverse=True,
    )

    print()
    print(
        "TOP GIDS BY CORE-GALLERY APPEARANCE"
    )

    print(
        "===================================="
    )

    for result in by_appearance[
        :TOP_GIDS
    ]:

        print(
            f"GID {result['gid']:<3}"
            f" | max="
            f"{result['core_max']:.4f}"
            f" | top3="
            f"{result['core_top3']:.4f}"
            f" | mean="
            f"{result['core_mean']:.4f}"
            f" | geom+="
            f"{result['geometry_support_count']}"
            f" | joint+="
            f"{result['joint_support_count']}"
            f" | veto="
            f"{result['identity_veto']}"
        )

    # --------------------------------------------------------
    # Geometry-supported ranking
    # --------------------------------------------------------

    geometry_rows = [
        row

        for row in rows

        if row[
            "geometry_support_count"
        ] > 0
    ]

    geometry_rows.sort(
        key=lambda row: (
            row[
                "joint_support_count"
            ],

            row[
                "geometry_support_count"
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

    print()
    print(
        "TOP GIDS WITH CORE GEOMETRY SUPPORT"
    )

    print(
        "==================================="
    )

    if not geometry_rows:

        print(
            "NONE"
        )

    else:

        for result in geometry_rows[
            :TOP_GIDS
        ]:

            print_gid_result(
                result
            )

    # --------------------------------------------------------
    # Joint geometry + appearance
    # --------------------------------------------------------

    joint_rows = [
        row

        for row in rows

        if row[
            "joint_support_count"
        ] > 0
    ]

    joint_rows.sort(
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

    print()
    print(
        "GIDS WITH JOINT CORE ReID + GEOMETRY SUPPORT"
    )

    print(
        "==========================================="
    )

    if not joint_rows:

        print(
            "NONE"
        )

    else:

        for result in joint_rows:

            print_gid_result(
                result
            )

    # --------------------------------------------------------
    # Explicit GID 73
    # --------------------------------------------------------

    gid73 = next(
        (
            row
            for row in rows
            if row[
                "gid"
            ] == 73
        ),
        None,
    )

    print()
    print(
        "EXPLICIT GID 73 CHECK"
    )

    print(
        "====================="
    )

    if gid73 is None:

        print(
            "GID 73 not available"
        )

    else:

        print_gid_result(
            gid73
        )

    return by_appearance


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2:601 EARLY/LATE -> ALL GID ANALYSIS"
    )

    print(
        "===================================="
    )

    print()
    print(
        "Broad diagnostic thresholds:"
    )

    print(
        "  shared frames >=",
        MIN_SHARED_FRAMES,
    )

    print(
        "  geometry median <=",
        GEOMETRY_MEDIAN_MAX,
    )

    print(
        "  interesting direct ReID >=",
        DIRECT_REID_INTERESTING_MIN,
    )

    print()
    print(
        "Geometry values are Terrace "
        "ground-coordinate units, not metres."
    )

    (
        manager,
        _,
        tracklet_lookup,
        _,
    ) = replay_final_production()

    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()

    tracklet_lookup.update(
        c2_lookup
    )

    c2_map = {
        tracklet.local_track_id:
            tracklet

        for tracklet, _
        in c2_rows
    }

    source = c2_map.get(
        TARGET_TRACK_ID
    )

    if source is None:

        raise RuntimeError(
            f"c2:{TARGET_TRACK_ID} not found"
        )

    early_tracklet = (
        build_segment_tracklet(
            source_tracklet=
                source,

            start_frame=
                EARLY_START,

            end_frame=
                EARLY_END,

            synthetic_track_id=
                601001,
        )
    )

    late_tracklet = (
        build_segment_tracklet(
            source_tracklet=
                source,

            start_frame=
                LATE_START,

            end_frame=
                LATE_END,

            synthetic_track_id=
                601002,
        )
    )

    print()
    print(
        "EARLY temporary tracklet:"
    )

    print(
        early_tracklet.summary()
    )

    print()
    print(
        "LATE temporary tracklet:"
    )

    print(
        late_tracklet.summary()
    )

    extractor = ReIDExtractor(
        device="cuda"
    )

    video = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    try:

        (
            early_embedding,
            early_frames,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                early_tracklet,
        )

        (
            late_embedding,
            late_frames,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                late_tracklet,
        )

    finally:

        video.release()

    print()
    print(
        "EARLY embedding frames:",
        early_frames,
    )

    print(
        "LATE embedding frames:",
        late_frames,
    )

    print()
    print(
        "EARLY <-> LATE ReID:",
        f"{cosine_similarity(early_embedding, late_embedding):.4f}",
    )

    early_rows = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            early_tracklet,

        candidate_embedding=
            early_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )

    late_rows = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            late_tracklet,

        candidate_embedding=
            late_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )

    early_ranking = report_segment(
        "EARLY 3796-4016",
        early_rows,
    )

    late_ranking = report_segment(
        "LATE 4049-4571",
        late_rows,
    )

    print()
    print()
    print(
        "HEAD-TO-HEAD SUMMARY"
    )

    print(
        "===================="
    )

    print(
        "EARLY top appearance GIDs:"
    )

    for row in early_ranking[:5]:

        print(
            f"  GID {row['gid']}"
            f" | top3="
            f"{row['core_top3']:.4f}"
            f" | max="
            f"{row['core_max']:.4f}"
            f" | geom+="
            f"{row['geometry_support_count']}"
            f" | joint+="
            f"{row['joint_support_count']}"
            f" | veto="
            f"{row['identity_veto']}"
        )

    print()
    print(
        "LATE top appearance GIDs:"
    )

    for row in late_ranking[:5]:

        print(
            f"  GID {row['gid']}"
            f" | top3="
            f"{row['core_top3']:.4f}"
            f" | max="
            f"{row['core_max']:.4f}"
            f" | geom+="
            f"{row['geometry_support_count']}"
            f" | joint+="
            f"{row['joint_support_count']}"
            f" | veto="
            f"{row['identity_veto']}"
        )

    print()
    print(
        "IDENTITIES MODIFIED: NO"
    )

    print(
        "Temporary segments existed in memory only."
    )

    print(
        "Diagnostic only."
    )


if __name__ == "__main__":
    main()
