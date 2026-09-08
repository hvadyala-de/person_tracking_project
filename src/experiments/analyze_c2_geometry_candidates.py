from pathlib import Path

import numpy as np

from src.build_tracklets import (
    load_tracklets,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.terrace_geometry import (
    detection_ground_point,
    ground_distance,
    load_ground_homographies,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# DIAGNOSTIC SETTINGS
#
# IMPORTANT:
#
# These are broad analysis thresholds.
# They are NOT new production thresholds.
#
# Geometry values are Terrace ground-coordinate units,
# NOT metres.
# ============================================================

MIN_DETECTIONS = 25

MIN_SHARED_FRAMES = 10

SUPPORT_MEDIAN_MAX = 30.0

CONTRADICTION_MEDIAN_MIN = 80.0

INTERESTING_REID_MIN = 0.84

STRONG_REID_MIN = 0.90

TOP_N = 25


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

    tracklets = load_tracklets(
        path
    )

    if isinstance(
        tracklets,
        dict,
    ):

        values = tracklets.values()

    else:

        values = tracklets


    result = {}


    for tracklet in values:

        if (
            len(
                tracklet.detections
            )
            < MIN_DETECTIONS
        ):

            continue


        result[
            tracklet.local_track_id
        ] = tracklet


    return result


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


    embedding = embedding.reshape(
        -1
    )


    norm = np.linalg.norm(
        embedding
    )


    if norm <= 0.0:

        return None


    return (
        embedding
        / norm
    )


# ============================================================
# DETECTIONS BY FRAME
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
# GEOMETRY STATS
# ============================================================

def geometry_stats(
    tracklet_a,
    tracklet_b,
    homographies,
):

    camera_a = normalize_camera_id(
        tracklet_a.camera_id
    )

    camera_b = normalize_camera_id(
        tracklet_b.camera_id
    )


    # --------------------------------------------------------
    # Quick temporal rejection.
    # --------------------------------------------------------

    overlap_start = max(
        tracklet_a.start_frame,
        tracklet_b.start_frame,
    )

    overlap_end = min(
        tracklet_a.end_frame,
        tracklet_b.end_frame,
    )


    if overlap_end < overlap_start:

        return None


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


    if (
        len(
            shared_frames
        )
        < MIN_SHARED_FRAMES
    ):

        return None


    distances = []


    for frame in shared_frames:

        point_a = detection_ground_point(
            detections_a[
                frame
            ],
            camera_a,
            homographies,
        )

        point_b = detection_ground_point(
            detections_b[
                frame
            ],
            camera_b,
            homographies,
        )


        distance = ground_distance(
            point_a,
            point_b,
        )


        if np.isfinite(
            distance
        ):

            distances.append(
                float(
                    distance
                )
            )


    if (
        len(
            distances
        )
        < MIN_SHARED_FRAMES
    ):

        return None


    values = np.asarray(
        distances,
        dtype=np.float64,
    )


    return {
        "shared":
            len(
                shared_frames
            ),

        "valid":
            len(
                values
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90,
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


# ============================================================
# GEOMETRY STATUS
# ============================================================

def geometry_status(
    stats,
):

    if stats is None:

        return "UNKNOWN"


    if (
        stats[
            "valid"
        ]
        < MIN_SHARED_FRAMES
    ):

        return "UNKNOWN"


    if (
        stats[
            "median"
        ]
        <= SUPPORT_MEDIAN_MAX
    ):

        return "SUPPORTED"


    if (
        stats[
            "median"
        ]
        >= CONTRADICTION_MEDIAN_MIN
    ):

        return "CONTRADICTED"


    return "UNKNOWN"


# ============================================================
# SCAN ONE CAMERA PAIR
# ============================================================

def scan_camera_pair(
    camera_a,
    camera_b,
    homographies,
):

    print()
    print(
        f"SCANNING c{camera_a} <-> c{camera_b}"
    )

    print(
        "=" * 60
    )


    tracklets_a = load_camera_tracklets(
        camera_a
    )

    tracklets_b = load_camera_tracklets(
        camera_b
    )


    embeddings_a = {}

    embeddings_b = {}


    for track_id in tracklets_a:

        embedding = load_embedding(
            camera_a,
            track_id,
        )

        if embedding is not None:

            embeddings_a[
                track_id
            ] = embedding


    for track_id in tracklets_b:

        embedding = load_embedding(
            camera_b,
            track_id,
        )

        if embedding is not None:

            embeddings_b[
                track_id
            ] = embedding


    print(
        f"Useful c{camera_a} tracklets:",
        len(
            tracklets_a
        ),
    )

    print(
        f"Embeddings c{camera_a}:",
        len(
            embeddings_a
        ),
    )

    print(
        f"Useful c{camera_b} tracklets:",
        len(
            tracklets_b
        ),
    )

    print(
        f"Embeddings c{camera_b}:",
        len(
            embeddings_b
        ),
    )


    rows = []


    for track_id_a, tracklet_a in (
        tracklets_a.items()
    ):

        embedding_a = embeddings_a.get(
            track_id_a
        )


        if embedding_a is None:

            continue


        for track_id_b, tracklet_b in (
            tracklets_b.items()
        ):

            embedding_b = embeddings_b.get(
                track_id_b
            )


            if embedding_b is None:

                continue


            stats = geometry_stats(
                tracklet_a,
                tracklet_b,
                homographies,
            )


            if stats is None:

                continue


            reid = float(
                np.dot(
                    embedding_a,
                    embedding_b,
                )
            )


            status = geometry_status(
                stats
            )


            rows.append(
                {
                    "camera_a":
                        camera_a,

                    "track_a":
                        track_id_a,

                    "camera_b":
                        camera_b,

                    "track_b":
                        track_id_b,

                    "reid":
                        reid,

                    "status":
                        status,

                    **stats,
                }
            )


    return rows


# ============================================================
# PRINT ROW
# ============================================================

def print_row(
    row,
):

    print(
        f"c{row['camera_a']}:"
        f"{row['track_a']:<4}"
        f" <-> "
        f"c{row['camera_b']}:"
        f"{row['track_b']:<4}"
        f" | {row['status']:<12}"
        f" | ReID="
        f"{row['reid']:.4f}"
        f" | shared="
        f"{row['shared']:<4}"
        f" | median="
        f"{row['median']:8.2f}"
        f" | mean="
        f"{row['mean']:8.2f}"
        f" | p90="
        f"{row['p90']:8.2f}"
    )


# ============================================================
# REPORT ONE CAMERA PAIR
# ============================================================

def report_pair(
    camera_a,
    camera_b,
    rows,
):

    print()
    print(
        f"c{camera_a} <-> c{camera_b} SUMMARY"
    )

    print(
        "=" * 60
    )


    supported = [
        row

        for row in rows

        if row[
            "status"
        ]
        == "SUPPORTED"
    ]


    contradicted = [
        row

        for row in rows

        if row[
            "status"
        ]
        == "CONTRADICTED"
    ]


    unknown = [
        row

        for row in rows

        if row[
            "status"
        ]
        == "UNKNOWN"
    ]


    print(
        "Pairs with >=",
        MIN_SHARED_FRAMES,
        "valid shared frames:",
        len(
            rows
        ),
    )

    print(
        "SUPPORTED:",
        len(
            supported
        ),
    )

    print(
        "UNKNOWN:",
        len(
            unknown
        ),
    )

    print(
        "CONTRADICTED:",
        len(
            contradicted
        ),
    )


    # ========================================================
    # STRONG JOINT GEOMETRY + REID
    # ========================================================

    joint = [
        row

        for row in supported

        if row[
            "reid"
        ]
        >= INTERESTING_REID_MIN
    ]


    joint.sort(
        key=lambda row: (
            row[
                "reid"
            ],

            row[
                "shared"
            ],

            -row[
                "median"
            ],
        ),
        reverse=True,
    )


    print()
    print(
        "SUPPORTED + ReID >= "
        f"{INTERESTING_REID_MIN:.2f}"
    )

    print(
        "=" * 60
    )


    if not joint:

        print(
            "NONE"
        )


    else:

        for row in joint[
            :TOP_N
        ]:

            print_row(
                row
            )


    # ========================================================
    # HIGHEST REID AMONG SUPPORTED GEOMETRY
    # ========================================================

    supported_by_reid = sorted(
        supported,
        key=lambda row: (
            row[
                "reid"
            ],

            row[
                "shared"
            ],
        ),
        reverse=True,
    )


    print()
    print(
        "TOP SUPPORTED PAIRS BY ReID"
    )

    print(
        "=" * 60
    )


    if not supported_by_reid:

        print(
            "NONE"
        )


    else:

        for row in supported_by_reid[
            :TOP_N
        ]:

            print_row(
                row
            )


    # ========================================================
    # BEST GEOMETRY PAIRS
    # ========================================================

    by_geometry = sorted(
        supported,
        key=lambda row: (
            row[
                "median"
            ],

            -row[
                "reid"
            ],
        ),
    )


    print()
    print(
        "LOWEST SUPPORTED GEOMETRY MEDIANS"
    )

    print(
        "=" * 60
    )


    if not by_geometry:

        print(
            "NONE"
        )


    else:

        for row in by_geometry[
            :TOP_N
        ]:

            print_row(
                row
            )


    # ========================================================
    # IMPORTANT SAFETY CHECK:
    # HIGH REID BUT GEOMETRIC CONTRADICTION
    # ========================================================

    dangerous = [
        row

        for row in contradicted

        if row[
            "reid"
        ]
        >= INTERESTING_REID_MIN
    ]


    dangerous.sort(
        key=lambda row: (
            row[
                "reid"
            ],

            row[
                "median"
            ],
        ),
        reverse=True,
    )


    print()
    print(
        "HIGH-ReID GEOMETRY CONTRADICTIONS"
    )

    print(
        "=" * 60
    )


    if not dangerous:

        print(
            "NONE"
        )


    else:

        for row in dangerous[
            :TOP_N
        ]:

            print_row(
                row
            )


    # ========================================================
    # VERY HIGH REID REGARDLESS OF GEOMETRY
    # ========================================================

    strong_reid = [
        row

        for row in rows

        if row[
            "reid"
        ]
        >= STRONG_REID_MIN
    ]


    strong_reid.sort(
        key=lambda row: (
            row[
                "reid"
            ],

            -row[
                "median"
            ],
        ),
        reverse=True,
    )


    print()
    print(
        "ALL ReID >= "
        f"{STRONG_REID_MIN:.2f}"
    )

    print(
        "=" * 60
    )


    if not strong_reid:

        print(
            "NONE"
        )


    else:

        for row in strong_reid[
            :TOP_N
        ]:

            print_row(
                row
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TERRACE c2 GEOMETRY + ReID DIAGNOSTIC"
    )

    print(
        "====================================="
    )


    print(
        "Broad diagnostic thresholds:"
    )

    print(
        "  minimum shared frames:",
        MIN_SHARED_FRAMES,
    )

    print(
        "  SUPPORTED median <=",
        SUPPORT_MEDIAN_MAX,
    )

    print(
        "  CONTRADICTED median >=",
        CONTRADICTION_MEDIAN_MIN,
    )

    print(
        "  interesting ReID >=",
        INTERESTING_REID_MIN,
    )

    print()
    print(
        "Geometry distance units are "
        "Terrace ground-coordinate units, "
        "not metres."
    )


    print()
    print(
        "Loading Terrace homographies..."
    )


    homographies = (
        load_ground_homographies()
    )


    print(
        "Homographies loaded:",
        sorted(
            homographies
        ),
    )


    all_results = {}


    for camera_a, camera_b in [
        (0, 2),
        (1, 2),
    ]:

        rows = scan_camera_pair(
            camera_a,
            camera_b,
            homographies,
        )


        all_results[
            (
                camera_a,
                camera_b,
            )
        ] = rows


    for (
        camera_a,
        camera_b,
    ), rows in all_results.items():

        report_pair(
            camera_a,
            camera_b,
            rows,
        )


    # ========================================================
    # OVERALL c2 SIGNAL
    # ========================================================

    all_rows = []

    for rows in all_results.values():

        all_rows.extend(
            rows
        )


    joint = [
        row

        for row in all_rows

        if (
            row[
                "status"
            ]
            == "SUPPORTED"

            and

            row[
                "reid"
            ]
            >= INTERESTING_REID_MIN
        )
    ]


    contradictions = [
        row

        for row in all_rows

        if (
            row[
                "status"
            ]
            == "CONTRADICTED"

            and

            row[
                "reid"
            ]
            >= INTERESTING_REID_MIN
        )
    ]


    print()
    print(
        "OVERALL c2 DIAGNOSTIC SUMMARY"
    )

    print(
        "============================="
    )


    print(
        "Supported + ReID >=",
        INTERESTING_REID_MIN,
        ":",
        len(
            joint
        ),
    )


    print(
        "High-ReID contradictions:",
        len(
            contradictions
        ),
    )


    print()
    print(
        "No identities were modified."
    )

    print(
        "This script is diagnostic only."
    )


if __name__ == "__main__":
    main()
