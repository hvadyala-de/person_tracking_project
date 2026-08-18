from pathlib import Path

import numpy as np

from src.build_tracklets import (
    load_tracklets,
)

from src.identity_compatibility import (
    normalize_camera_id,
)


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
# PAIRS TO INSPECT
#
# TRUE / LIKELY FRAGMENTS:
#   known same-camera fragment candidates
#
# SUSPICIOUS:
#   important over-merges seen in the global-ID run
# ============================================================

PAIRS = [

    # ----------------------------------------
    # Positive / likely same-person fragments
    # ----------------------------------------

    ("POS", 0, 701, 740),
    ("POS", 0, 56, 149),
    ("POS", 0, 707, 746),
    ("POS", 0, 188, 221),
    ("POS", 0, 15, 26),

    # ----------------------------------------
    # Suspicious look-alike merges
    # ----------------------------------------

    ("SUSPECT", 0, 255, 399),
    ("SUSPECT", 0, 492, 701),
    ("SUSPECT", 0, 331, 701),
    ("SUSPECT", 0, 299, 701),

    # ----------------------------------------
    # Geometry bridge candidate
    # ----------------------------------------

    ("BRIDGE", 0, 469, 600),
]


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

    return embedding / norm


# ============================================================
# DETECTION HELPERS
# ============================================================

def first_detection(
    tracklet,
):

    if not tracklet.detections:
        return None

    return min(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def last_detection(
    tracklet,
):

    if not tracklet.detections:
        return None

    return max(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def bbox_center(
    detection,
):

    x = (
        detection.x1
        + detection.x2
    ) / 2.0

    y = (
        detection.y1
        + detection.y2
    ) / 2.0

    return np.asarray(
        [x, y],
        dtype=np.float32,
    )


def bbox_bottom_center(
    detection,
):

    x = (
        detection.x1
        + detection.x2
    ) / 2.0

    y = detection.y2

    return np.asarray(
        [x, y],
        dtype=np.float32,
    )


# ============================================================
# TEMPORAL RELATION
# ============================================================

def temporal_stats(
    first,
    second,
):

    # Positive gap:
    # second begins AFTER first ends.
    #
    # Negative:
    # they overlap.

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
# END -> START IMAGE DISTANCE
# ============================================================

def endpoint_distance(
    first,
    second,
):

    end_detection = last_detection(
        first
    )

    start_detection = first_detection(
        second
    )

    if (
        end_detection is None
        or start_detection is None
    ):
        return None, None

    center_distance = float(
        np.linalg.norm(
            bbox_center(end_detection)
            - bbox_center(start_detection)
        )
    )

    bottom_distance = float(
        np.linalg.norm(
            bbox_bottom_center(end_detection)
            - bbox_bottom_center(start_detection)
        )
    )

    return (
        center_distance,
        bottom_distance,
    )


# ============================================================
# REID SIMILARITY
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
# MAIN
# ============================================================

def main():

    camera_cache = {}


    print()
    print("SAME-CAMERA TRANSITION ANALYSIS")
    print("===============================")
    print()


    header = (
        f"{'Type':>8} "
        f"{'Pair':>16} "
        f"{'A frames':>15} "
        f"{'B frames':>15} "
        f"{'Gap':>7} "
        f"{'Overlap':>8} "
        f"{'CtrDist':>9} "
        f"{'Bottom':>9} "
        f"{'ReID':>8}"
    )


    print(
        header
    )

    print(
        "-" * len(header)
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
                f"{label:>8} "
                f"c{camera_id}:"
                f"{first_id}-{second_id} "
                f"MISSING TRACKLET"
            )

            continue


        first = tracks[
            first_id
        ]

        second = tracks[
            second_id
        ]


        # ----------------------------------------------------
        # Put them in chronological order for transition
        # analysis.
        # ----------------------------------------------------

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


        (
            center_distance,
            bottom_distance,
        ) = endpoint_distance(
            first,
            second,
        )


        similarity = reid_similarity(
            camera_id,
            first_id,
            second_id,
        )


        center_text = (
            f"{center_distance:.2f}"
            if center_distance is not None
            else "None"
        )


        bottom_text = (
            f"{bottom_distance:.2f}"
            if bottom_distance is not None
            else "None"
        )


        similarity_text = (
            f"{similarity:.4f}"
            if similarity is not None
            else "None"
        )


        pair_text = (
            f"c{camera_id}:"
            f"{first_id}->{second_id}"
        )


        first_frames = (
            f"{first.start_frame}-"
            f"{first.end_frame}"
        )


        second_frames = (
            f"{second.start_frame}-"
            f"{second.end_frame}"
        )


        print(
            f"{label:>8} "
            f"{pair_text:>16} "
            f"{first_frames:>15} "
            f"{second_frames:>15} "
            f"{gap:>7} "
            f"{overlap:>8} "
            f"{center_text:>9} "
            f"{bottom_text:>9} "
            f"{similarity_text:>8}"
        )


if __name__ == "__main__":
    main()
