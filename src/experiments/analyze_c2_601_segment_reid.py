from pathlib import Path

import cv2
import numpy as np
import torch

from src.experiments.analyze_final_pending_population import (
    replay_final_production,
)

from src.experiments.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.reid_extractor import (
    ReIDExtractor,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c2.avi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "c2_601_segment_reid"
)


# ============================================================
# TARGET
# ============================================================

TARGET_TRACK_ID = 601
TARGET_GID = 73

EARLY_START = 3796
EARLY_END = 4016

LATE_START = 4049
LATE_END = 4571

CROPS_PER_SEGMENT = 8


# ============================================================
# HELPERS
# ============================================================

def normalize(vector):

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


def feature_to_numpy(
    feature,
):

    if feature is None:
        return None

    if isinstance(
        feature,
        torch.Tensor,
    ):

        feature = (
            feature
            .detach()
            .cpu()
            .numpy()
        )

    return normalize(
        feature
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


def select_detections(
    tracklet,
    start_frame,
    end_frame,
    count,
):

    detections = [
        detection

        for detection
        in tracklet.detections

        if (
            start_frame
            <= int(
                detection.frame
            )
            <= end_frame
        )
    ]


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


    # Avoid duplicates if integer rounding ever produces them.
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
# BUILD EMBEDDING FOR ONE TEMPORAL SEGMENT
# ============================================================

def build_segment_embedding(
    extractor,
    video,
    tracklet,
    name,
    start_frame,
    end_frame,
):

    selected = select_detections(
        tracklet=
            tracklet,

        start_frame=
            start_frame,

        end_frame=
            end_frame,

        count=
            CROPS_PER_SEGMENT,
    )


    segment_dir = (
        OUTPUT_DIR
        / name
    )


    segment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print(
        f"{name.upper()} SEGMENT"
    )

    print(
        "=" * 60
    )


    print(
        "Range:",
        f"{start_frame}-{end_frame}",
    )


    print(
        "Detections in segment:",
        sum(
            1
            for detection
            in tracklet.detections
            if (
                start_frame
                <= int(
                    detection.frame
                )
                <= end_frame
            )
        ),
    )


    print(
        "Selected crop frames:",
        [
            int(
                detection.frame
            )
            for detection
            in selected
        ],
    )


    embeddings = []


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

            print(
                "FAILED frame:",
                frame,
            )

            continue


        crop = crop_detection(
            image,
            detection,
        )


        if crop is None:

            print(
                "INVALID crop:",
                frame,
            )

            continue


        crop_path = (
            segment_dir
            / f"frame_{frame}.jpg"
        )


        cv2.imwrite(
            str(
                crop_path
            ),
            crop,
        )


        feature = extractor.extract(
            crop
        )


        feature = feature_to_numpy(
            feature
        )


        if feature is None:

            print(
                "FAILED embedding:",
                frame,
            )

            continue


        embeddings.append(
            feature
        )


        print(
            "OK",
            f"frame={frame}",
            f"dim={feature.shape[0]}",
        )


    if not embeddings:

        raise RuntimeError(
            f"No usable embeddings for {name}"
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
            f"Could not normalize {name} embedding"
        )


    print(
        "Crops embedded:",
        len(
            embeddings
        ),
    )


    print(
        "Segment embedding dimension:",
        mean_embedding.shape[
            0
        ],
    )


    return (
        mean_embedding,
        embeddings,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2:601 TEMPORAL SEGMENT ReID ANALYSIS"
    )

    print(
        "====================================="
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
            (
                tracklet,
                embedding,
            )

        for tracklet, embedding
        in c2_rows
    }


    if TARGET_TRACK_ID not in c2_map:

        raise RuntimeError(
            f"c2:{TARGET_TRACK_ID} not found"
        )


    (
        tracklet,
        full_embedding,
    ) = c2_map[
        TARGET_TRACK_ID
    ]


    print(
        "Full c2 tracklet:",
        f"c2:{TARGET_TRACK_ID}",
    )


    print(
        "Full range:",
        f"{tracklet.start_frame}-"
        f"{tracklet.end_frame}",
    )


    print(
        "Full detections:",
        len(
            tracklet.detections
        ),
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
            early_crops,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                tracklet,

            name=
                "early",

            start_frame=
                EARLY_START,

            end_frame=
                EARLY_END,
        )


        (
            late_embedding,
            late_crops,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                tracklet,

            name=
                "late",

            start_frame=
                LATE_START,

            end_frame=
                LATE_END,
        )


    finally:

        video.release()


    full_embedding = normalize(
        full_embedding
    )


    # ========================================================
    # EARLY VS LATE
    # ========================================================

    print()
    print(
        "SEGMENT-TO-SEGMENT ReID"
    )

    print(
        "======================="
    )


    print(
        "early <-> late:",
        f"{cosine_similarity(early_embedding, late_embedding):.4f}",
    )


    if full_embedding is not None:

        print(
            "early <-> full c2:601:",
            f"{cosine_similarity(early_embedding, full_embedding):.4f}",
        )

        print(
            "late  <-> full c2:601:",
            f"{cosine_similarity(late_embedding, full_embedding):.4f}",
        )


    # ========================================================
    # WITHIN-SEGMENT CONSISTENCY
    # ========================================================

    print()
    print(
        "WITHIN-SEGMENT CROP CONSISTENCY"
    )

    print(
        "==============================="
    )


    for name, segment_embedding, crops in [
        (
            "early",
            early_embedding,
            early_crops,
        ),
        (
            "late",
            late_embedding,
            late_crops,
        ),
    ]:

        similarities = [
            cosine_similarity(
                segment_embedding,
                crop_embedding,
            )

            for crop_embedding
            in crops
        ]


        print(
            f"{name}:"
            f" n={len(similarities)}"
            f" | min={min(similarities):.4f}"
            f" | mean={np.mean(similarities):.4f}"
            f" | max={max(similarities):.4f}"
        )


    # ========================================================
    # COMPARE AGAINST GID 73
    # ========================================================

    identity = manager.identities[
        TARGET_GID
    ]


    print()
    print(
        "GID 73 MEMBER COMPARISON"
    )

    print(
        "========================"
    )


    print(
        "member       trust      "
        "early     late      full"
    )

    print(
        "-" * 55
    )


    for member in identity.members:

        camera_id = normalize_camera_id(
            member.camera_id
        )


        member_embedding = (
            manager.member_embeddings.get(
                (
                    camera_id,
                    member.local_track_id,
                )
            )
        )


        trust = manager.get_member_trust(
            TARGET_GID,
            camera_id,
            member.local_track_id,
        )


        if member_embedding is None:

            print(
                f"c{camera_id}:"
                f"{member.local_track_id:<7}"
                f"{str(trust):<11}"
                f"MISSING EMBEDDING"
            )

            continue


        member_embedding = normalize(
            member_embedding
        )


        early_score = cosine_similarity(
            early_embedding,
            member_embedding,
        )


        late_score = cosine_similarity(
            late_embedding,
            member_embedding,
        )


        if full_embedding is not None:

            full_score = cosine_similarity(
                full_embedding,
                member_embedding,
            )

            full_text = (
                f"{full_score:.4f}"
            )

        else:

            full_text = "None"


        print(
            f"c{camera_id}:"
            f"{member.local_track_id:<7}"
            f"{str(trust):<11}"
            f"{early_score:8.4f}"
            f"{late_score:10.4f}"
            f"{full_text:>10}"
        )


    # ========================================================
    # CORE-ONLY GALLERY SUMMARY
    # ========================================================

    core_keys = manager.core_members.get(
        TARGET_GID,
        set(),
    )


    core_early = []
    core_late = []


    for member in identity.members:

        camera_id = normalize_camera_id(
            member.camera_id
        )


        key = (
            camera_id,
            member.local_track_id,
        )


        if key not in core_keys:
            continue


        member_embedding = (
            manager.member_embeddings.get(
                key
            )
        )


        if member_embedding is None:
            continue


        member_embedding = normalize(
            member_embedding
        )


        core_early.append(
            cosine_similarity(
                early_embedding,
                member_embedding,
            )
        )


        core_late.append(
            cosine_similarity(
                late_embedding,
                member_embedding,
            )
        )


    print()
    print(
        "CORE-ONLY GID 73 GALLERY SUMMARY"
    )

    print(
        "================================"
    )


    if core_early:

        sorted_early = sorted(
            core_early,
            reverse=True,
        )

        sorted_late = sorted(
            core_late,
            reverse=True,
        )


        print(
            "EARLY:"
            f" max={max(sorted_early):.4f}"
            f" | mean={np.mean(sorted_early):.4f}"
            f" | top3="
            f"{np.mean(sorted_early[:3]):.4f}"
        )


        print(
            "LATE :"
            f" max={max(sorted_late):.4f}"
            f" | mean={np.mean(sorted_late):.4f}"
            f" | top3="
            f"{np.mean(sorted_late[:3]):.4f}"
        )


    print()
    print(
        "Saved diagnostic crops to:"
    )

    print(
        OUTPUT_DIR
    )


    print()
    print(
        "IDENTITIES MODIFIED: NO"
    )

    print(
        "Diagnostic only."
    )


if __name__ == "__main__":
    main()
