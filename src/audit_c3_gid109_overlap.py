from pathlib import Path

import cv2
import numpy as np

from src.analyze_c3_geometry_candidates import (
    load_camera_tracklets,
    load_embedding,
)

from src.terrace_geometry import (
    detection_ground_point,
    ground_distance,
    load_ground_homographies,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMERA_ID = 3

TRACK_A_ID = 12
TRACK_B_ID = 15

EXPECTED_OVERLAP_START = 970
EXPECTED_OVERLAP_END = 1020

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c3.avi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "c3_gid109_overlap"
)

SAMPLE_FRAMES = 8


# ============================================================
# VECTOR HELPERS
# ============================================================

def normalize(
    vector,
):

    if vector is None:

        return None


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
    a,
    b,
):

    return float(
        np.dot(
            a,
            b,
        )
    )


# ============================================================
# DETECTION LOOKUP
# ============================================================

def detections_by_frame(
    tracklet,
):

    return {
        int(
            detection.frame
        ):
            detection

        for detection
        in tracklet.detections
    }


# ============================================================
# BOX HELPERS
# ============================================================

def bbox_area(
    detection,
):

    width = max(
        0.0,
        detection.x2
        - detection.x1,
    )

    height = max(
        0.0,
        detection.y2
        - detection.y1,
    )


    return (
        width
        * height
    )


def bbox_iou(
    a,
    b,
):

    x1 = max(
        a.x1,
        b.x1,
    )

    y1 = max(
        a.y1,
        b.y1,
    )

    x2 = min(
        a.x2,
        b.x2,
    )

    y2 = min(
        a.y2,
        b.y2,
    )


    intersection_width = max(
        0.0,
        x2 - x1,
    )

    intersection_height = max(
        0.0,
        y2 - y1,
    )


    intersection = (
        intersection_width
        * intersection_height
    )


    union = (
        bbox_area(
            a
        )
        +
        bbox_area(
            b
        )
        -
        intersection
    )


    if union <= 0.0:

        return 0.0


    return float(
        intersection
        / union
    )


def center(
    detection,
):

    return (
        (
            detection.x1
            + detection.x2
        )
        / 2.0,

        (
            detection.y1
            + detection.y2
        )
        / 2.0,
    )


def bottom_center(
    detection,
):

    return (
        (
            detection.x1
            + detection.x2
        )
        / 2.0,

        detection.y2,
    )


def point_distance(
    a,
    b,
):

    return (
        (
            b[
                0
            ]
            - a[
                0
            ]
        ) ** 2
        +
        (
            b[
                1
            ]
            - a[
                1
            ]
        ) ** 2
    ) ** 0.5


def bbox_diagonal(
    detection,
):

    width = (
        detection.x2
        - detection.x1
    )

    height = (
        detection.y2
        - detection.y1
    )


    return (
        width ** 2
        +
        height ** 2
    ) ** 0.5


# ============================================================
# FRAMEWISE OVERLAP ANALYSIS
# ============================================================

def analyze_overlap(
    tracklet_a,
    tracklet_b,
    homographies,
):

    detections_a = detections_by_frame(
        tracklet_a
    )

    detections_b = detections_by_frame(
        tracklet_b
    )


    shared_frames = sorted(
        set(
            detections_a
        )
        &
        set(
            detections_b
        )
    )


    rows = []


    for frame in shared_frames:

        detection_a = detections_a[
            frame
        ]

        detection_b = detections_b[
            frame
        ]


        center_a = center(
            detection_a
        )

        center_b = center(
            detection_b
        )


        bottom_a = bottom_center(
            detection_a
        )

        bottom_b = bottom_center(
            detection_b
        )


        scale = max(
            bbox_diagonal(
                detection_a
            ),

            bbox_diagonal(
                detection_b
            ),

            1.0,
        )


        point_a = detection_ground_point(
            detection_a,
            CAMERA_ID,
            homographies,
        )

        point_b = detection_ground_point(
            detection_b,
            CAMERA_ID,
            homographies,
        )


        rows.append(
            {
                "frame":
                    frame,

                "iou":
                    bbox_iou(
                        detection_a,
                        detection_b,
                    ),

                "center_distance":
                    point_distance(
                        center_a,
                        center_b,
                    ),

                "bottom_distance":
                    point_distance(
                        bottom_a,
                        bottom_b,
                    ),

                "normalized_center_distance":
                    (
                        point_distance(
                            center_a,
                            center_b,
                        )
                        / scale
                    ),

                "normalized_bottom_distance":
                    (
                        point_distance(
                            bottom_a,
                            bottom_b,
                        )
                        / scale
                    ),

                "ground_distance":
                    ground_distance(
                        point_a,
                        point_b,
                    ),
            }
        )


    return rows


# ============================================================
# PRINT NUMERIC SUMMARY
# ============================================================

def print_summary(
    rows,
):

    print()
    print(
        "OVERLAP NUMERIC SUMMARY"
    )

    print(
        "======================="
    )


    if not rows:

        print(
            "NO SHARED FRAMES"
        )

        return


    ious = np.asarray(
        [
            row[
                "iou"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    center_distances = np.asarray(
        [
            row[
                "center_distance"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    bottom_distances = np.asarray(
        [
            row[
                "bottom_distance"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    normalized_centers = np.asarray(
        [
            row[
                "normalized_center_distance"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    normalized_bottoms = np.asarray(
        [
            row[
                "normalized_bottom_distance"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    ground_distances = np.asarray(
        [
            row[
                "ground_distance"
            ]

            for row in rows
        ],
        dtype=np.float64,
    )


    print(
        "Shared frames:",
        len(
            rows
        ),
    )

    print(
        "First shared frame:",
        rows[
            0
        ][
            "frame"
        ],
    )

    print(
        "Last shared frame:",
        rows[
            -1
        ][
            "frame"
        ],
    )


    print()
    print(
        "BBox IoU"
    )

    print(
        "  min:   ",
        f"{np.min(ious):.4f}",
    )

    print(
        "  median:",
        f"{np.median(ious):.4f}",
    )

    print(
        "  mean:  ",
        f"{np.mean(ious):.4f}",
    )

    print(
        "  max:   ",
        f"{np.max(ious):.4f}",
    )


    print()
    print(
        "Center distance"
    )

    print(
        "  median px:",
        f"{np.median(center_distances):.2f}",
    )

    print(
        "  median normalized:",
        f"{np.median(normalized_centers):.3f}",
    )


    print()
    print(
        "Bottom-center distance"
    )

    print(
        "  median px:",
        f"{np.median(bottom_distances):.2f}",
    )

    print(
        "  median normalized:",
        f"{np.median(normalized_bottoms):.3f}",
    )


    print()
    print(
        "Same-camera ground distance"
    )

    print(
        "  min:",
        f"{np.min(ground_distances):.2f}",
    )

    print(
        "  median:",
        f"{np.median(ground_distances):.2f}",
    )

    print(
        "  mean:",
        f"{np.mean(ground_distances):.2f}",
    )

    print(
        "  max:",
        f"{np.max(ground_distances):.2f}",
    )


    print()
    print(
        "Frames with IoU >= 0.50:",
        int(
            np.sum(
                ious
                >= 0.50
            )
        ),
        "/",
        len(
            rows
        ),
    )

    print(
        "Frames with IoU >= 0.70:",
        int(
            np.sum(
                ious
                >= 0.70
            )
        ),
        "/",
        len(
            rows
        ),
    )


# ============================================================
# SELECT VISUAL SAMPLE FRAMES
# ============================================================

def select_sample_frames(
    shared_frames,
):

    if not shared_frames:

        return []


    if (
        len(
            shared_frames
        )
        <= SAMPLE_FRAMES
    ):

        return shared_frames


    indices = np.linspace(
        0,
        len(
            shared_frames
        )
        - 1,
        num=SAMPLE_FRAMES,
        dtype=int,
    )


    return [
        shared_frames[
            int(
                index
            )
        ]

        for index in indices
    ]


# ============================================================
# SAVE VISUAL OVERLAP FRAMES
# ============================================================

def save_visual_samples(
    tracklet_a,
    tracklet_b,
):

    detections_a = detections_by_frame(
        tracklet_a
    )

    detections_b = detections_by_frame(
        tracklet_b
    )


    shared_frames = sorted(
        set(
            detections_a
        )
        &
        set(
            detections_b
        )
    )


    sample_frames = select_sample_frames(
        shared_frames
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    video = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )


    if not video.isOpened():

        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )


    saved = []


    try:

        for frame in sample_frames:

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame,
            )


            ok, image = video.read()


            if not ok:

                continue


            detection_a = detections_a[
                frame
            ]

            detection_b = detections_b[
                frame
            ]


            cv2.rectangle(
                image,
                (
                    int(
                        detection_a.x1
                    ),
                    int(
                        detection_a.y1
                    ),
                ),
                (
                    int(
                        detection_a.x2
                    ),
                    int(
                        detection_a.y2
                    ),
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )


            cv2.putText(
                image,
                "c3:12",
                (
                    int(
                        detection_a.x1
                    ),
                    max(
                        15,
                        int(
                            detection_a.y1
                        )
                        - 5,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    0,
                    255,
                    0,
                ),
                1,
                cv2.LINE_AA,
            )


            cv2.rectangle(
                image,
                (
                    int(
                        detection_b.x1
                    ),
                    int(
                        detection_b.y1
                    ),
                ),
                (
                    int(
                        detection_b.x2
                    ),
                    int(
                        detection_b.y2
                    ),
                ),
                (
                    0,
                    0,
                    255,
                ),
                2,
            )


            cv2.putText(
                image,
                "c3:15",
                (
                    int(
                        detection_b.x1
                    ),
                    max(
                        15,
                        int(
                            detection_b.y1
                        )
                        - 5,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    0,
                    0,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )


            path = (
                OUTPUT_DIR
                / f"frame_{frame:06d}.jpg"
            )


            if cv2.imwrite(
                str(
                    path
                ),
                image,
            ):

                saved.append(
                    path
                )


    finally:

        video.release()


    print()
    print(
        "VISUAL OVERLAP SAMPLES"
    )

    print(
        "======================"
    )


    for path in saved:

        print(
            path
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c3 GID 109 SAME-CAMERA OVERLAP AUDIT"
    )

    print(
        "===================================="
    )


    tracklets = load_camera_tracklets(
        CAMERA_ID
    )


    tracklet_a = tracklets.get(
        TRACK_A_ID
    )

    tracklet_b = tracklets.get(
        TRACK_B_ID
    )


    if (
        tracklet_a is None
        or tracklet_b is None
    ):

        raise RuntimeError(
            "Could not load c3:12 and c3:15"
        )


    print()
    print(
        "TRACKLETS"
    )

    print(
        "========="
    )


    print(
        "c3:12:",
        tracklet_a.start_frame,
        "->",
        tracklet_a.end_frame,
        "| detections=",
        len(
            tracklet_a.detections
        ),
    )


    print(
        "c3:15:",
        tracklet_b.start_frame,
        "->",
        tracklet_b.end_frame,
        "| detections=",
        len(
            tracklet_b.detections
        ),
    )


    embedding_a = load_embedding(
        CAMERA_ID,
        TRACK_A_ID,
    )

    embedding_b = load_embedding(
        CAMERA_ID,
        TRACK_B_ID,
    )


    if (
        embedding_a is None
        or embedding_b is None
    ):

        raise RuntimeError(
            "Could not load c3:12 / c3:15 embeddings"
        )


    embedding_a = normalize(
        embedding_a
    )

    embedding_b = normalize(
        embedding_b
    )


    similarity = cosine_similarity(
        embedding_a,
        embedding_b,
    )


    print()
    print(
        "WHOLE-TRACK ReID"
    )

    print(
        "================"
    )


    print(
        "c3:12 <-> c3:15:",
        f"{similarity:.4f}",
    )


    homographies = load_ground_homographies()


    rows = analyze_overlap(
        tracklet_a,
        tracklet_b,
        homographies,
    )


    shared_frames = [
        row[
            "frame"
        ]

        for row in rows
    ]


    if shared_frames:

        if (
            shared_frames[
                0
            ]
            != EXPECTED_OVERLAP_START
            or
            shared_frames[
                -1
            ]
            != EXPECTED_OVERLAP_END
        ):

            print()
            print(
                "WARNING:"
            )

            print(
                "Observed shared range differs from "
                "970..1020:"
            )

            print(
                shared_frames[
                    0
                ],
                "->",
                shared_frames[
                    -1
                ],
            )


    print_summary(
        rows
    )


    save_visual_samples(
        tracklet_a,
        tracklet_b,
    )


    print()
    print(
        "INTERPRETATION GUIDE"
    )

    print(
        "===================="
    )

    print(
        "High same-frame IoU + very small spatial distance "
        "+ high ReID supports duplicate tracking of one person."
    )

    print(
        "Low IoU + clearly separated positions while both "
        "tracks exist supports two different physical people."
    )

    print(
        "No production decision is made by this diagnostic."
    )


if __name__ == "__main__":
    main()
