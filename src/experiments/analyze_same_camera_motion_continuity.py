from pathlib import Path

import numpy as np

from src.build_tracklets import (
    load_tracklets,
)

from src.identity_compatibility import (
    normalize_camera_id,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


TRACK_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
)


EMBEDDING_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
)


# ============================================================
# CONTROL + TEST PAIRS
#
# POS:
# Known likely same-person fragments.
#
# NEAR:
# Remaining pending same-camera near misses discovered in the
# pending-population analysis.
#
# LONG:
# Known long-gap appearance lookalikes / bridge cases.
# These are included for context, but long-range linear motion
# prediction should NOT be interpreted as reliable.
# ============================================================

PAIRS = [

    # --------------------------------------------------------
    # Verified / likely positive short fragments
    # --------------------------------------------------------

    (
        "POS",
        0,
        701,
        740,
    ),

    (
        "POS",
        0,
        56,
        149,
    ),

    (
        "POS",
        0,
        707,
        746,
    ),

    (
        "POS",
        0,
        188,
        221,
    ),


    # --------------------------------------------------------
    # Current pending same-camera near misses
    # --------------------------------------------------------

    (
        "NEAR",
        0,
        243,
        264,
    ),

    (
        "NEAR",
        0,
        331,
        431,
    ),

    (
        "NEAR",
        1,
        503,
        565,
    ),


    # --------------------------------------------------------
    # Long-gap context
    # --------------------------------------------------------

    (
        "LONG",
        0,
        255,
        399,
    ),

    (
        "LONG",
        0,
        492,
        701,
    ),

    (
        "LONG",
        0,
        331,
        701,
    ),

    (
        "LONG",
        0,
        299,
        701,
    ),

    (
        "BRIDGE",
        0,
        469,
        600,
    ),
]


# ============================================================
# TRAJECTORY CONFIG
# ============================================================

FIT_DETECTIONS = 8


# ============================================================
# LOAD CAMERA TRACKLETS
# ============================================================

def load_camera_tracklets(
    camera_id,
):

    path = (
        TRACK_DIR
        / f"tracks_c{camera_id}.csv"
    )


    raw = load_tracklets(
        path
    )


    return {
        tracklet.local_track_id:
            tracklet

        for tracklet in raw.values()
    }


# ============================================================
# LOAD EMBEDDING
# ============================================================

def load_embedding(
    camera_id,
    track_id,
):

    path = (
        EMBEDDING_DIR
        / f"c{camera_id}"
        / f"track_{track_id}.npy"
    )


    if not path.exists():

        return None


    embedding = np.load(
        path
    ).astype(
        np.float32
    )


    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        return None


    return (
        embedding
        / norm
    )


# ============================================================
# DETECTION HELPERS
# ============================================================

def sorted_detections(
    tracklet,
):

    return sorted(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def bbox_center(
    detection,
):

    return np.asarray(
        [
            (
                detection.x1
                + detection.x2
            ) / 2.0,

            (
                detection.y1
                + detection.y2
            ) / 2.0,
        ],
        dtype=np.float32,
    )


def bbox_bottom_center(
    detection,
):

    return np.asarray(
        [
            (
                detection.x1
                + detection.x2
            ) / 2.0,

            detection.y2,
        ],
        dtype=np.float32,
    )


def bbox_size(
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
        float(width),
        float(height),
    )


# ============================================================
# CHRONOLOGICAL ORDER
# ============================================================

def chronological_pair(
    first,
    second,
):

    if (
        first.start_frame
        <= second.start_frame
    ):

        return (
            first,
            second,
        )


    return (
        second,
        first,
    )


# ============================================================
# TEMPORAL STATS
# ============================================================

def temporal_stats(
    first,
    second,
):

    first, second = (
        chronological_pair(
            first,
            second,
        )
    )


    gap = (
        second.start_frame
        - first.end_frame
        - 1
    )


    overlap_start = max(
        first.start_frame,
        second.start_frame,
    )


    overlap_end = min(
        first.end_frame,
        second.end_frame,
    )


    overlap = max(
        0,
        overlap_end
        - overlap_start
        + 1,
    )


    return (
        gap,
        overlap,
    )


# ============================================================
# LINEAR MOTION FIT
#
# Fit:
#
# x(frame) = ax * frame + bx
# y(frame) = ay * frame + by
#
# to bottom-center locations.
# ============================================================

def fit_bottom_motion(
    detections,
):

    if len(detections) < 2:

        return None


    frames = np.asarray(
        [
            detection.frame

            for detection
            in detections
        ],
        dtype=np.float64,
    )


    points = np.asarray(
        [
            bbox_bottom_center(
                detection
            )

            for detection
            in detections
        ],
        dtype=np.float64,
    )


    x_fit = np.polyfit(
        frames,
        points[:, 0],
        1,
    )


    y_fit = np.polyfit(
        frames,
        points[:, 1],
        1,
    )


    velocity = np.asarray(
        [
            x_fit[0],
            y_fit[0],
        ],
        dtype=np.float64,
    )


    return {
        "x_fit":
            x_fit,

        "y_fit":
            y_fit,

        "velocity":
            velocity,
    }


# ============================================================
# MOTION PREDICTION
# ============================================================

def predict_bottom(
    fit,
    frame,
):

    x = (
        fit[
            "x_fit"
        ][0]
        * frame
        + fit[
            "x_fit"
        ][1]
    )


    y = (
        fit[
            "y_fit"
        ][0]
        * frame
        + fit[
            "y_fit"
        ][1]
    )


    return np.asarray(
        [
            x,
            y,
        ],
        dtype=np.float64,
    )


# ============================================================
# PAIR MOTION METRICS
# ============================================================

def motion_metrics(
    first,
    second,
):

    first, second = (
        chronological_pair(
            first,
            second,
        )
    )


    first_detections = (
        sorted_detections(
            first
        )
    )


    second_detections = (
        sorted_detections(
            second
        )
    )


    if (
        len(first_detections) < 2
        or len(second_detections) < 2
    ):

        return None


    first_tail = (
        first_detections[
            -FIT_DETECTIONS:
        ]
    )


    second_head = (
        second_detections[
            :FIT_DETECTIONS
        ]
    )


    first_fit = fit_bottom_motion(
        first_tail
    )


    second_fit = fit_bottom_motion(
        second_head
    )


    if (
        first_fit is None
        or second_fit is None
    ):

        return None


    first_last = (
        first_detections[-1]
    )


    second_first = (
        second_detections[0]
    )


    # --------------------------------------------------------
    # Raw endpoint distances.
    # --------------------------------------------------------

    endpoint_center_distance = float(
        np.linalg.norm(
            bbox_center(
                first_last
            )
            - bbox_center(
                second_first
            )
        )
    )


    endpoint_bottom_distance = float(
        np.linalg.norm(
            bbox_bottom_center(
                first_last
            )
            - bbox_bottom_center(
                second_first
            )
        )
    )


    # --------------------------------------------------------
    # Predict second start from first trajectory.
    # --------------------------------------------------------

    predicted_second_bottom = (
        predict_bottom(
            first_fit,
            second_first.frame,
        )
    )


    actual_second_bottom = (
        bbox_bottom_center(
            second_first
        ).astype(
            np.float64
        )
    )


    forward_prediction_error = float(
        np.linalg.norm(
            predicted_second_bottom
            - actual_second_bottom
        )
    )


    # --------------------------------------------------------
    # Predict first end backwards from second trajectory.
    # --------------------------------------------------------

    predicted_first_bottom = (
        predict_bottom(
            second_fit,
            first_last.frame,
        )
    )


    actual_first_bottom = (
        bbox_bottom_center(
            first_last
        ).astype(
            np.float64
        )
    )


    backward_prediction_error = float(
        np.linalg.norm(
            predicted_first_bottom
            - actual_first_bottom
        )
    )


    # --------------------------------------------------------
    # Velocity consistency.
    # --------------------------------------------------------

    first_velocity = (
        first_fit[
            "velocity"
        ]
    )


    second_velocity = (
        second_fit[
            "velocity"
        ]
    )


    velocity_difference = float(
        np.linalg.norm(
            first_velocity
            - second_velocity
        )
    )


    first_speed = float(
        np.linalg.norm(
            first_velocity
        )
    )


    second_speed = float(
        np.linalg.norm(
            second_velocity
        )
    )


    # --------------------------------------------------------
    # Bounding-box scale continuity.
    # --------------------------------------------------------

    (
        first_width,
        first_height,
    ) = bbox_size(
        first_last
    )


    (
        second_width,
        second_height,
    ) = bbox_size(
        second_first
    )


    if (
        first_height > 0
        and second_height > 0
    ):

        height_ratio = (
            max(
                first_height,
                second_height,
            )
            /
            min(
                first_height,
                second_height,
            )
        )


    else:

        height_ratio = None


    if (
        first_width > 0
        and second_width > 0
    ):

        width_ratio = (
            max(
                first_width,
                second_width,
            )
            /
            min(
                first_width,
                second_width,
            )
        )


    else:

        width_ratio = None


    # --------------------------------------------------------
    # Normalize prediction error by local person height.
    # --------------------------------------------------------

    mean_height = (
        (
            first_height
            + second_height
        )
        / 2.0
    )


    if mean_height > 0:

        forward_error_height = (
            forward_prediction_error
            / mean_height
        )


        backward_error_height = (
            backward_prediction_error
            / mean_height
        )


        endpoint_bottom_height = (
            endpoint_bottom_distance
            / mean_height
        )


    else:

        forward_error_height = None
        backward_error_height = None
        endpoint_bottom_height = None


    return {
        "endpoint_center":
            endpoint_center_distance,

        "endpoint_bottom":
            endpoint_bottom_distance,

        "forward_error":
            forward_prediction_error,

        "backward_error":
            backward_prediction_error,

        "forward_error_height":
            forward_error_height,

        "backward_error_height":
            backward_error_height,

        "endpoint_bottom_height":
            endpoint_bottom_height,

        "first_speed":
            first_speed,

        "second_speed":
            second_speed,

        "velocity_difference":
            velocity_difference,

        "height_ratio":
            height_ratio,

        "width_ratio":
            width_ratio,
    }


# ============================================================
# REID
# ============================================================

def reid_similarity(
    camera_id,
    first_id,
    second_id,
):

    first = load_embedding(
        camera_id,
        first_id,
    )


    second = load_embedding(
        camera_id,
        second_id,
    )


    if (
        first is None
        or second is None
    ):

        return None


    return float(
        np.dot(
            first,
            second,
        )
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def number_text(
    value,
    decimals=2,
):

    if value is None:

        return "None"


    return (
        f"{value:.{decimals}f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "SAME-CAMERA MOTION CONTINUITY ANALYSIS"
    )

    print(
        "======================================"
    )


    print(
        "Motion-fit detections per endpoint:",
        FIT_DETECTIONS,
    )


    camera_cache = {}


    header = (
        f"{'Type':>7} "
        f"{'Pair':>16} "
        f"{'Gap':>5} "
        f"{'Ov':>4} "
        f"{'ReID':>7} "
        f"{'Ctr':>7} "
        f"{'Bottom':>7} "
        f"{'FwdErr':>7} "
        f"{'BackErr':>7} "
        f"{'Fwd/H':>7} "
        f"{'Back/H':>7} "
        f"{'End/H':>7} "
        f"{'V1':>6} "
        f"{'V2':>6} "
        f"{'dV':>6} "
        f"{'Hrat':>6}"
    )


    print()
    print(
        header
    )


    print(
        "-" * len(
            header
        )
    )


    for (
        label,
        camera_id,
        first_id,
        second_id,
    ) in PAIRS:

        camera_id = normalize_camera_id(
            camera_id
        )


        if camera_id not in camera_cache:

            camera_cache[
                camera_id
            ] = load_camera_tracklets(
                camera_id
            )


        tracks = camera_cache[
            camera_id
        ]


        if (
            first_id not in tracks
            or second_id not in tracks
        ):

            print(
                f"{label:>7} "
                f"c{camera_id}:"
                f"{first_id}-{second_id} "
                f"MISSING"
            )

            continue


        first = tracks[
            first_id
        ]


        second = tracks[
            second_id
        ]


        if (
            second.start_frame
            < first.start_frame
        ):

            first, second = (
                second,
                first,
            )

            first_id, second_id = (
                second_id,
                first_id,
            )


        (
            gap,
            overlap,
        ) = temporal_stats(
            first,
            second,
        )


        metrics = motion_metrics(
            first,
            second,
        )


        similarity = reid_similarity(
            camera_id,
            first_id,
            second_id,
        )


        pair_text = (
            f"c{camera_id}:"
            f"{first_id}"
            f"->{second_id}"
        )


        if metrics is None:

            print(
                f"{label:>7} "
                f"{pair_text:>16} "
                f"{gap:>5} "
                f"{overlap:>4} "
                f"{number_text(similarity, 4):>7} "
                f"MOTION DATA MISSING"
            )

            continue


        print(
            f"{label:>7} "
            f"{pair_text:>16} "
            f"{gap:>5} "
            f"{overlap:>4} "
            f"{number_text(similarity, 4):>7} "
            f"{number_text(metrics['endpoint_center']):>7} "
            f"{number_text(metrics['endpoint_bottom']):>7} "
            f"{number_text(metrics['forward_error']):>7} "
            f"{number_text(metrics['backward_error']):>7} "
            f"{number_text(metrics['forward_error_height']):>7} "
            f"{number_text(metrics['backward_error_height']):>7} "
            f"{number_text(metrics['endpoint_bottom_height']):>7} "
            f"{number_text(metrics['first_speed']):>6} "
            f"{number_text(metrics['second_speed']):>6} "
            f"{number_text(metrics['velocity_difference']):>6} "
            f"{number_text(metrics['height_ratio']):>6}"
        )


    print()
    print(
        "COLUMN NOTES"
    )

    print(
        "============"
    )


    print(
        "FwdErr  = first track trajectory projected to second start"
    )

    print(
        "BackErr = second track trajectory projected back to first end"
    )

    print(
        "Fwd/H and Back/H normalize prediction error by mean bbox height"
    )

    print(
        "End/H normalizes raw bottom-center endpoint distance by bbox height"
    )

    print(
        "V1/V2 are fitted bottom-center speeds in pixels/frame"
    )

    print(
        "dV is the difference between fitted endpoint velocities"
    )

    print(
        "Hrat is the larger endpoint bbox height divided by the smaller"
    )


if __name__ == "__main__":
    main()
