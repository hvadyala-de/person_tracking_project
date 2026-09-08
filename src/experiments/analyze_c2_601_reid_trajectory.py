import csv
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
# SETTINGS
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
    / "c2_601_reid_trajectory"
)

CSV_PATH = (
    OUTPUT_DIR
    / "trajectory.csv"
)

TARGET_TRACK_ID = 601
TARGET_GID = 73

SAMPLE_EVERY_N_DETECTIONS = 25

TRANSITION_FRAMES = [
    3990,
    4000,
    4010,
    4016,
    4049,
    4050,
    4060,
    4080,
]


# ============================================================
# HELPERS
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


def crop_detection(
    image,
    detection,
):

    height, width = image.shape[:2]

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


def phase_for_frame(
    frame,
):

    if frame <= 4016:
        return "EARLY"

    if frame < 4049:
        return "GAP"

    return "LATE"


# ============================================================
# SELECT SAMPLE DETECTIONS
# ============================================================

def select_sample_detections(
    tracklet,
):

    detections = sorted(
        tracklet.detections,
        key=lambda detection:
            int(
                detection.frame
            ),
    )

    selected = {}

    # --------------------------------------------------------
    # Regular trajectory samples.
    # --------------------------------------------------------

    for index in range(
        0,
        len(detections),
        SAMPLE_EVERY_N_DETECTIONS,
    ):

        detection = detections[
            index
        ]

        selected[
            int(
                detection.frame
            )
        ] = detection

    # Always include final detection.
    if detections:

        detection = detections[-1]

        selected[
            int(
                detection.frame
            )
        ] = detection

    # --------------------------------------------------------
    # Force transition-region frames if they exist.
    # --------------------------------------------------------

    by_frame = {
        int(
            detection.frame
        ):
            detection

        for detection
        in detections
    }

    for frame in TRANSITION_FRAMES:

        detection = by_frame.get(
            frame
        )

        if detection is not None:

            selected[
                frame
            ] = detection

    return [
        selected[
            frame
        ]

        for frame
        in sorted(
            selected
        )
    ]


# ============================================================
# LOAD GID CORE EMBEDDINGS
# ============================================================

def load_core_gallery(
    manager,
):

    identity = manager.identities[
        TARGET_GID
    ]

    core_keys = manager.core_members.get(
        TARGET_GID,
        set(),
    )

    gallery = []

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

        embedding = (
            manager.member_embeddings.get(
                key
            )
        )

        embedding = normalize(
            embedding
        )

        if embedding is None:
            continue

        gallery.append(
            {
                "camera_id":
                    camera_id,

                "track_id":
                    member.local_track_id,

                "embedding":
                    embedding,
            }
        )

    gallery.sort(
        key=lambda row: (
            row[
                "camera_id"
            ],
            row[
                "track_id"
            ],
        )
    )

    return gallery


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2:601 ReID TRAJECTORY ANALYSIS"
    )

    print(
        "================================"
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

    tracklet, _ = c2_map[
        TARGET_TRACK_ID
    ]

    gallery = load_core_gallery(
        manager
    )

    print()
    print(
        "Target:",
        f"c2:{TARGET_TRACK_ID}",
    )

    print(
        "Range:",
        f"{tracklet.start_frame}-"
        f"{tracklet.end_frame}",
    )

    print(
        "Detections:",
        len(
            tracklet.detections
        ),
    )

    print()
    print(
        "GID:",
        TARGET_GID,
    )

    print(
        "CORE gallery members:",
        len(
            gallery
        ),
    )

    for row in gallery:

        print(
            f"  c{row['camera_id']}:"
            f"{row['track_id']}"
        )

    selected = (
        select_sample_detections(
            tracklet
        )
    )

    print()
    print(
        "Trajectory samples:",
        len(
            selected
        ),
    )

    print(
        "Sample frames:"
    )

    print(
        [
            int(
                detection.frame
            )
            for detection
            in selected
        ]
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

    rows = []

    previous_frame = None

    try:

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

            feature = extractor.extract(
                crop
            )

            embedding = normalize(
                feature
            )

            if embedding is None:

                print(
                    "FAILED embedding:",
                    frame,
                )

                continue

            similarities = []

            per_member = {}

            for member in gallery:

                score = cosine_similarity(
                    embedding,
                    member[
                        "embedding"
                    ],
                )

                similarities.append(
                    score
                )

                label = (
                    f"c"
                    f"{member['camera_id']}"
                    f"_"
                    f"{member['track_id']}"
                )

                per_member[
                    label
                ] = score

            sorted_scores = sorted(
                similarities,
                reverse=True,
            )

            max_score = max(
                sorted_scores
            )

            mean_score = float(
                np.mean(
                    sorted_scores
                )
            )

            top3_score = float(
                np.mean(
                    sorted_scores[:3]
                )
            )

            if previous_frame is None:

                gap_from_previous = 0

            else:

                gap_from_previous = (
                    frame
                    - previous_frame
                )

            previous_frame = frame

            phase = phase_for_frame(
                frame
            )

            row = {
                "frame":
                    frame,

                "phase":
                    phase,

                "gap_from_previous_sample":
                    gap_from_previous,

                "core_max":
                    max_score,

                "core_top3":
                    top3_score,

                "core_mean":
                    mean_score,

                **per_member,
            }

            rows.append(
                row
            )

            crop_path = (
                OUTPUT_DIR
                / (
                    f"frame_{frame}_"
                    f"{phase.lower()}.jpg"
                )
            )

            cv2.imwrite(
                str(
                    crop_path
                ),
                crop,
            )

    finally:

        video.release()

    # ========================================================
    # PRINT TRAJECTORY
    # ========================================================

    print()
    print(
        "CORE-GALLERY ReID TRAJECTORY"
    )

    print(
        "============================"
    )

    print(
        " frame phase  gap"
        "   max"
        "   top3"
        "   mean"
    )

    print(
        "-" * 52
    )

    for row in rows:

        print(
            f"{row['frame']:5d}"
            f" {row['phase']:<5}"
            f" {row['gap_from_previous_sample']:4d}"
            f" {row['core_max']:6.4f}"
            f" {row['core_top3']:7.4f}"
            f" {row['core_mean']:7.4f}"
        )

    # ========================================================
    # PHASE SUMMARY
    # ========================================================

    print()
    print(
        "PHASE SUMMARY"
    )

    print(
        "============="
    )

    for phase in [
        "EARLY",
        "LATE",
    ]:

        phase_rows = [
            row

            for row in rows

            if row[
                "phase"
            ] == phase
        ]

        if not phase_rows:
            continue

        max_scores = [
            row[
                "core_max"
            ]

            for row
            in phase_rows
        ]

        top3_scores = [
            row[
                "core_top3"
            ]

            for row
            in phase_rows
        ]

        mean_scores = [
            row[
                "core_mean"
            ]

            for row
            in phase_rows
        ]

        print()
        print(
            phase
        )

        print(
            f"  samples="
            f"{len(phase_rows)}"
        )

        print(
            f"  CORE max:"
            f" min={min(max_scores):.4f}"
            f" | mean={np.mean(max_scores):.4f}"
            f" | max={max(max_scores):.4f}"
        )

        print(
            f"  CORE top3:"
            f" min={min(top3_scores):.4f}"
            f" | mean={np.mean(top3_scores):.4f}"
            f" | max={max(top3_scores):.4f}"
        )

        print(
            f"  CORE mean:"
            f" min={min(mean_scores):.4f}"
            f" | mean={np.mean(mean_scores):.4f}"
            f" | max={max(mean_scores):.4f}"
        )

    # ========================================================
    # TRANSITION REGION
    # ========================================================

    print()
    print(
        "TRANSITION REGION"
    )

    print(
        "================="
    )

    transition_rows = [
        row

        for row in rows

        if (
            3980
            <= row[
                "frame"
            ]
            <= 4090
        )
    ]

    for row in transition_rows:

        print(
            f"frame={row['frame']}"
            f" | {row['phase']}"
            f" | max={row['core_max']:.4f}"
            f" | top3={row['core_top3']:.4f}"
            f" | mean={row['core_mean']:.4f}"
        )

    # ========================================================
    # WRITE CSV
    # ========================================================

    if rows:

        fieldnames = list(
            rows[
                0
            ].keys()
        )

        with CSV_PATH.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=
                    fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    print()
    print(
        "Saved trajectory CSV:"
    )

    print(
        CSV_PATH
    )

    print()
    print(
        "Saved sampled crops:"
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
