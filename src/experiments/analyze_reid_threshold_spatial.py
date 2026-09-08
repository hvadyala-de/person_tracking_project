from pathlib import Path

import numpy as np

from build_tracklets import load_tracklets


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_CSV = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
    / "tracks_c0.csv"
)

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
    / "c0"
)


# ============================================================
# VERIFIED SAME-PERSON PAIRS
# ============================================================

VERIFIED_POSITIVE_PAIRS = [
    (701, 740),
    (56, 149),
    (707, 746),
    (188, 221),
]


# ============================================================
# NEGATIVE-PAIR SETTINGS
# ============================================================

# Both tracks must coexist for at least this many frames.
MIN_SHARED_FRAMES = 10

# At least this many shared frames must show strong
# spatial separation.
MIN_SEPARATED_FRAMES = 10

# Fraction of shared frames that must be spatially separated.
MIN_SEPARATED_RATIO = 0.80

# If IoU is very small, boxes are not overlapping much.
MAX_IOU_FOR_DIFFERENT = 0.05

# Absolute minimum center distance in pixels.
MIN_CENTER_DISTANCE = 45.0


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
    )


def bbox_iou(a, b):

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    intersection_w = max(
        0.0,
        x2 - x1,
    )

    intersection_h = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_w
        * intersection_h
    )

    area_a = max(
        0.0,
        a.x2 - a.x1,
    ) * max(
        0.0,
        a.y2 - a.y1,
    )

    area_b = max(
        0.0,
        b.x2 - b.x1,
    ) * max(
        0.0,
        b.y2 - b.y1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


def center_distance(a, b):

    center_ax = (
        a.x1 + a.x2
    ) / 2.0

    center_ay = (
        a.y1 + a.y2
    ) / 2.0

    center_bx = (
        b.x1 + b.x2
    ) / 2.0

    center_by = (
        b.y1 + b.y2
    ) / 2.0

    dx = center_ax - center_bx
    dy = center_ay - center_by

    return float(
        np.sqrt(
            dx * dx
            + dy * dy
        )
    )


def bbox_diagonal(detection):

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

    return float(
        np.sqrt(
            width * width
            + height * height
        )
    )


def spatially_separated(det_a, det_b):

    iou = bbox_iou(
        det_a,
        det_b,
    )

    distance = center_distance(
        det_a,
        det_b,
    )

    diag_a = bbox_diagonal(
        det_a
    )

    diag_b = bbox_diagonal(
        det_b
    )

    # Use both an absolute and person-size-relative threshold.
    required_distance = max(
        MIN_CENTER_DISTANCE,
        0.75 * max(
            diag_a,
            diag_b,
        ),
    )

    return (
        iou <= MAX_IOU_FOR_DIFFERENT
        and distance >= required_distance
    )


def load_embeddings():

    embeddings = {}

    for path in EMBEDDING_DIR.glob(
        "track_*.npy"
    ):

        track_id = int(
            path.stem.replace(
                "track_",
                "",
            )
        )

        embedding = np.load(
            path
        ).astype(
            np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:
            embedding = (
                embedding
                / norm
            )

        embeddings[
            track_id
        ] = embedding

    return embeddings


# ============================================================
# MAIN
# ============================================================

def main():

    raw_tracklets = load_tracklets(
        TRACK_CSV
    )

    tracklets = {
        track.local_track_id: track
        for track
        in raw_tracklets.values()
        if track.num_detections >= 25
    }

    embeddings = load_embeddings()

    print()
    print("SPATIAL REID THRESHOLD ANALYSIS")
    print("===============================")
    print()

    print(
        f"Useful tracklets: {len(tracklets)}"
    )

    print(
        f"Embeddings:       {len(embeddings)}"
    )

    print()

    # ========================================================
    # POSITIVES
    # ========================================================

    positive_scores = []

    print("VERIFIED POSITIVE PAIRS")
    print("-----------------------")

    for id_a, id_b in VERIFIED_POSITIVE_PAIRS:

        similarity = cosine_similarity(
            embeddings[id_a],
            embeddings[id_b],
        )

        positive_scores.append(
            similarity
        )

        print(
            f"{id_a:>5} <-> {id_b:<5} "
            f"similarity={similarity:.4f}"
        )

    print()

    # ========================================================
    # FRAME -> DETECTION LOOKUP
    # ========================================================

    detection_maps = {}

    for track_id, tracklet in tracklets.items():

        detection_maps[
            track_id
        ] = {
            detection.frame: detection
            for detection
            in tracklet.detections
        }

    # ========================================================
    # BUILD STRONG SPATIAL NEGATIVES
    # ========================================================

    track_ids = sorted(
        embeddings.keys()
    )

    negative_pairs = []

    rejected_overlap_pairs = 0

    for i in range(
        len(track_ids)
    ):

        id_a = track_ids[i]

        if id_a not in tracklets:
            continue

        for j in range(
            i + 1,
            len(track_ids),
        ):

            id_b = track_ids[j]

            if id_b not in tracklets:
                continue

            frames_a = set(
                detection_maps[
                    id_a
                ].keys()
            )

            frames_b = set(
                detection_maps[
                    id_b
                ].keys()
            )

            shared_frames = sorted(
                frames_a
                & frames_b
            )

            if (
                len(shared_frames)
                < MIN_SHARED_FRAMES
            ):
                continue

            separated_count = 0

            for frame in shared_frames:

                det_a = (
                    detection_maps[
                        id_a
                    ][frame]
                )

                det_b = (
                    detection_maps[
                        id_b
                    ][frame]
                )

                if spatially_separated(
                    det_a,
                    det_b,
                ):
                    separated_count += 1

            separated_ratio = (
                separated_count
                / len(shared_frames)
            )

            # --------------------------------------------
            # STRONG NEGATIVE
            # --------------------------------------------

            if (
                separated_count
                >= MIN_SEPARATED_FRAMES
                and separated_ratio
                >= MIN_SEPARATED_RATIO
            ):

                similarity = cosine_similarity(
                    embeddings[id_a],
                    embeddings[id_b],
                )

                negative_pairs.append(
                    (
                        similarity,
                        len(shared_frames),
                        separated_count,
                        separated_ratio,
                        id_a,
                        id_b,
                    )
                )

            else:

                rejected_overlap_pairs += 1

    negative_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    negative_scores = np.array(
        [
            pair[0]
            for pair
            in negative_pairs
        ],
        dtype=np.float32,
    )

    # ========================================================
    # NEGATIVE STATS
    # ========================================================

    print("SPATIAL NEGATIVE STATISTICS")
    print("---------------------------")

    print(
        f"Strong negatives:      "
        f"{len(negative_pairs)}"
    )

    print(
        f"Rejected ambiguous:    "
        f"{rejected_overlap_pairs}"
    )

    if len(
        negative_scores
    ) > 0:

        print(
            f"Mean:                  "
            f"{negative_scores.mean():.4f}"
        )

        print(
            f"Median:                "
            f"{np.median(negative_scores):.4f}"
        )

        print(
            f"90th percentile:       "
            f"{np.percentile(negative_scores, 90):.4f}"
        )

        print(
            f"95th percentile:       "
            f"{np.percentile(negative_scores, 95):.4f}"
        )

        print(
            f"99th percentile:       "
            f"{np.percentile(negative_scores, 99):.4f}"
        )

        print(
            f"Maximum:               "
            f"{negative_scores.max():.4f}"
        )

    print()

    # ========================================================
    # POSITIVE STATS
    # ========================================================

    positives = np.array(
        positive_scores,
        dtype=np.float32,
    )

    print("POSITIVE STATISTICS")
    print("-------------------")

    print(
        f"Minimum: "
        f"{positives.min():.4f}"
    )

    print(
        f"Mean:    "
        f"{positives.mean():.4f}"
    )

    print(
        f"Maximum: "
        f"{positives.max():.4f}"
    )

    print()

    # ========================================================
    # HARDEST SPATIALLY VERIFIED NEGATIVES
    # ========================================================

    print("TOP 20 SPATIAL NEGATIVES")
    print("------------------------")

    print(
        f"{'A':>6} "
        f"{'B':>6} "
        f"{'Sim':>8} "
        f"{'Shared':>8} "
        f"{'Sep':>6} "
        f"{'Ratio':>8}"
    )

    print("-" * 48)

    for (
        similarity,
        shared,
        separated,
        ratio,
        id_a,
        id_b,
    ) in negative_pairs[:20]:

        print(
            f"{id_a:>6} "
            f"{id_b:>6} "
            f"{similarity:>8.4f} "
            f"{shared:>8} "
            f"{separated:>6} "
            f"{ratio:>8.2f}"
        )


if __name__ == "__main__":
    main()
