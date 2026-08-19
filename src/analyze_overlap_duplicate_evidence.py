import math
import statistics

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.identity_compatibility import (
    normalize_camera_id,
)


# ============================================================
# PAIRS TO AUDIT
#
# c0:15 <-> c0:26 is our known overlapping duplicate-track
# reference case.
# ============================================================

PAIRS = [
    ("POSITIVE_CONTROL", 0, 15, 0, 26),

    ("TARGET_GID66", 0, 636, 0, 663),

    ("TARGET_GID30", 0, 201, 0, 343),

    ("TARGET_GID73", 1, 787, 1, 755),

    ("TARGET_GID66_B", 1, 797, 1, 730),

    ("TARGET_GID47", 1, 490, 1, 503),
]


# ============================================================
# BASIC HELPERS
# ============================================================

def get_tracklet(
    manager,
    tracklet_lookup,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )

    return manager.get_tracklet(
        camera_id,
        local_track_id,
        tracklet_lookup,
    )


def get_embedding(
    manager,
    camera_id,
    local_track_id,
):

    camera_id = normalize_camera_id(
        camera_id
    )

    key = (
        camera_id,
        local_track_id,
    )

    pending = manager.pending_tracklets.get(
        key
    )

    if pending is not None:

        return pending.get(
            "embedding"
        )

    return manager.member_embeddings.get(
        key
    )


def bbox_width(
    detection,
):

    return max(
        0.0,
        float(
            detection.x2
            - detection.x1
        ),
    )


def bbox_height(
    detection,
):

    return max(
        0.0,
        float(
            detection.y2
            - detection.y1
        ),
    )


def bbox_area(
    detection,
):

    return (
        bbox_width(
            detection
        )
        *
        bbox_height(
            detection
        )
    )


def bbox_center(
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


def bbox_bottom_center(
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


def euclidean(
    first,
    second,
):

    dx = (
        first[0]
        - second[0]
    )

    dy = (
        first[1]
        - second[1]
    )

    return math.sqrt(
        dx * dx
        + dy * dy
    )


def bbox_iou(
    first,
    second,
):

    ix1 = max(
        first.x1,
        second.x1,
    )

    iy1 = max(
        first.y1,
        second.y1,
    )

    ix2 = min(
        first.x2,
        second.x2,
    )

    iy2 = min(
        first.y2,
        second.y2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    union = (
        bbox_area(first)
        + bbox_area(second)
        - intersection
    )

    if union <= 0.0:

        return 0.0

    return float(
        intersection
        / union
    )


def percentile(
    values,
    fraction,
):

    if not values:

        return None

    ordered = sorted(
        values
    )

    if len(
        ordered
    ) == 1:

        return ordered[0]

    index = (
        fraction
        * (
            len(ordered)
            - 1
        )
    )

    low = int(
        math.floor(
            index
        )
    )

    high = int(
        math.ceil(
            index
        )
    )

    if low == high:

        return ordered[
            low
        ]

    weight = (
        index
        - low
    )

    return (
        ordered[low]
        * (
            1.0
            - weight
        )
        +
        ordered[high]
        * weight
    )


def format_value(
    value,
    digits=3,
):

    if value is None:

        return "None"

    return (
        f"{value:.{digits}f}"
    )


# ============================================================
# FRAME MATCHING
# ============================================================

def detection_map(
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
# ANALYZE ONE OVERLAPPING PAIR
# ============================================================

def analyze_pair(
    manager,
    tracklet_lookup,
    label,
    camera_a,
    track_a,
    camera_b,
    track_b,
):

    camera_a = normalize_camera_id(
        camera_a
    )

    camera_b = normalize_camera_id(
        camera_b
    )

    first = get_tracklet(
        manager,
        tracklet_lookup,
        camera_a,
        track_a,
    )

    second = get_tracklet(
        manager,
        tracklet_lookup,
        camera_b,
        track_b,
    )

    print()
    print(
        label
    )

    print(
        "-" * len(label)
    )

    print(
        f"c{camera_a}:{track_a}"
        f" <-> "
        f"c{camera_b}:{track_b}"
    )

    if (
        first is None
        or second is None
    ):

        print(
            "Missing tracklet"
        )

        return

    embedding_a = get_embedding(
        manager,
        camera_a,
        track_a,
    )

    embedding_b = get_embedding(
        manager,
        camera_b,
        track_b,
    )

    if (
        embedding_a is not None
        and embedding_b is not None
    ):

        similarity = (
            manager.cosine_similarity(
                embedding_a,
                embedding_b,
            )
        )

        print(
            f"ReID similarity: "
            f"{similarity:.4f}"
        )

    else:

        print(
            "ReID similarity: unavailable"
        )

    first_map = detection_map(
        first
    )

    second_map = detection_map(
        second
    )

    common_frames = sorted(
        set(
            first_map
        )
        &
        set(
            second_map
        )
    )

    print(
        "Tracklet ranges:",
        f"{first.start_frame}-{first.end_frame}",
        "and",
        f"{second.start_frame}-{second.end_frame}",
    )

    print(
        "Exact shared detection frames:",
        len(
            common_frames
        ),
    )

    if not common_frames:

        print(
            "No exact overlapping detection frames."
        )

        return

    ious = []

    center_distances = []

    bottom_distances = []

    normalized_centers = []

    normalized_bottoms = []

    size_ratios = []

    for frame in common_frames:

        detection_a = first_map[
            frame
        ]

        detection_b = second_map[
            frame
        ]

        iou = bbox_iou(
            detection_a,
            detection_b,
        )

        center_distance = euclidean(
            bbox_center(
                detection_a
            ),
            bbox_center(
                detection_b
            ),
        )

        bottom_distance = euclidean(
            bbox_bottom_center(
                detection_a
            ),
            bbox_bottom_center(
                detection_b
            ),
        )

        width_a = bbox_width(
            detection_a
        )

        height_a = bbox_height(
            detection_a
        )

        width_b = bbox_width(
            detection_b
        )

        height_b = bbox_height(
            detection_b
        )

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

        reference_height = max(
            height_a,
            height_b,
            1.0,
        )

        area_a = max(
            bbox_area(
                detection_a
            ),
            1.0,
        )

        area_b = max(
            bbox_area(
                detection_b
            ),
            1.0,
        )

        size_ratio = (
            max(
                area_a,
                area_b,
            )
            /
            min(
                area_a,
                area_b,
            )
        )

        ious.append(
            iou
        )

        center_distances.append(
            center_distance
        )

        bottom_distances.append(
            bottom_distance
        )

        normalized_centers.append(
            center_distance
            / reference_diagonal
        )

        normalized_bottoms.append(
            bottom_distance
            / reference_height
        )

        size_ratios.append(
            size_ratio
        )

    high_iou_50 = sum(
        1
        for value in ious
        if value >= 0.50
    )

    high_iou_30 = sum(
        1
        for value in ious
        if value >= 0.30
    )

    low_center_25 = sum(
        1
        for value in normalized_centers
        if value <= 0.25
    )

    low_bottom_25 = sum(
        1
        for value in normalized_bottoms
        if value <= 0.25
    )

    count = len(
        common_frames
    )

    print()
    print(
        "IOU"
    )

    print(
        f"  p10="
        f"{format_value(percentile(ious, 0.10))}"
        f" | median="
        f"{format_value(statistics.median(ious))}"
        f" | p90="
        f"{format_value(percentile(ious, 0.90))}"
    )

    print(
        f"  IoU >= 0.50: "
        f"{high_iou_50}/{count}"
        f" ({100.0 * high_iou_50 / count:.1f}%)"
    )

    print(
        f"  IoU >= 0.30: "
        f"{high_iou_30}/{count}"
        f" ({100.0 * high_iou_30 / count:.1f}%)"
    )

    print()
    print(
        "CENTER DISTANCE"
    )

    print(
        f"  pixels median="
        f"{format_value(statistics.median(center_distances), 2)}"
        f" | p90="
        f"{format_value(percentile(center_distances, 0.90), 2)}"
    )

    print(
        f"  normalized median="
        f"{format_value(statistics.median(normalized_centers))}"
        f" | p90="
        f"{format_value(percentile(normalized_centers, 0.90))}"
    )

    print(
        f"  normalized <= 0.25: "
        f"{low_center_25}/{count}"
        f" ({100.0 * low_center_25 / count:.1f}%)"
    )

    print()
    print(
        "BOTTOM-CENTER DISTANCE"
    )

    print(
        f"  pixels median="
        f"{format_value(statistics.median(bottom_distances), 2)}"
        f" | p90="
        f"{format_value(percentile(bottom_distances, 0.90), 2)}"
    )

    print(
        f"  normalized median="
        f"{format_value(statistics.median(normalized_bottoms))}"
        f" | p90="
        f"{format_value(percentile(normalized_bottoms, 0.90))}"
    )

    print(
        f"  normalized <= 0.25: "
        f"{low_bottom_25}/{count}"
        f" ({100.0 * low_bottom_25 / count:.1f}%)"
    )

    print()
    print(
        "SIZE RATIO"
    )

    print(
        f"  median="
        f"{format_value(statistics.median(size_ratios))}"
        f" | p90="
        f"{format_value(percentile(size_ratios, 0.90))}"
    )

    print()
    print(
        "FIRST / MIDDLE / LAST SHARED FRAME"
    )

    sample_indices = sorted(
        set(
            [
                0,
                len(common_frames) // 2,
                len(common_frames) - 1,
            ]
        )
    )

    for index in sample_indices:

        frame = common_frames[
            index
        ]

        detection_a = first_map[
            frame
        ]

        detection_b = second_map[
            frame
        ]

        print(
            f"  frame={frame}"
            f" | IoU="
            f"{bbox_iou(detection_a, detection_b):.3f}"
            f" | center="
            f"{euclidean(bbox_center(detection_a), bbox_center(detection_b)):.2f}"
            f" | bottom="
            f"{euclidean(bbox_bottom_center(detection_a), bbox_bottom_center(detection_b)):.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "OVERLAPPING DUPLICATE-TRACK EVIDENCE"
    )

    print(
        "===================================="
    )

    (
        manager,
        tracklets,
        tracklet_lookup,
        final_results,
    ) = replay_final_production()

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

    for pair in PAIRS:

        analyze_pair(
            manager,
            tracklet_lookup,
            *pair,
        )


if __name__ == "__main__":
    main()
