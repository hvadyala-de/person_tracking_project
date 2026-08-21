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
# These are the SAME broad diagnostic thresholds used for the
# earlier c2 geometry scan.
#
# They are NOT production identity thresholds.
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
# CAMERA SET
# ============================================================

TARGET_CAMERA = 3

REFERENCE_CAMERAS = [
    0,
    1,
    2,
]


# ============================================================
# KNOWN c2 WHOLE-TRACK CAUTION
#
# c2:9 and c2:601 were previously shown to require
# identity-aware temporal treatment.
#
# The raw c2 embedding files still represent the original
# whole ByteTrack tracklets because splits were deliberately
# not persisted.
#
# Therefore c2<->c3 rows involving these IDs are useful
# diagnostics, but must NOT by themselves be treated as
# final identity evidence.
# ============================================================

KNOWN_C2_HETEROGENEOUS = {
    9,
    601,
}


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


    if not path.exists():

        raise FileNotFoundError(
            f"Track CSV not found: {path}"
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
# PRINT ROW COLLECTION
# ============================================================

def print_rows(
    title,
    rows,
):

    print()
    print(
        title
    )

    print(
        "=" * 60
    )


    if not rows:

        print(
            "NONE"
        )

        return


    for row in rows:

        print_row(
            row
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
    # SUPPORTED + INTERESTING ReID
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
        ),
        reverse=True,
    )


    print_rows(
        (
            "SUPPORTED + ReID >= "
            f"{INTERESTING_REID_MIN:.2f}"
        ),
        joint,
    )


    # ========================================================
    # TOP SUPPORTED BY ReID
    # ========================================================

    top_supported = sorted(
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
    )[
        :TOP_N
    ]


    print_rows(
        "TOP SUPPORTED PAIRS BY ReID",
        top_supported,
    )


    # ========================================================
    # LOWEST SUPPORTED GEOMETRY MEDIANS
    # ========================================================

    lowest_supported = sorted(
        supported,
        key=lambda row: (
            row[
                "median"
            ],

            -row[
                "shared"
            ],
        ),
    )[
        :TOP_N
    ]


    print_rows(
        "LOWEST SUPPORTED GEOMETRY MEDIANS",
        lowest_supported,
    )


    # ========================================================
    # HIGH-ReID GEOMETRY CONTRADICTIONS
    # ========================================================

    high_reid_contradictions = [
        row

        for row in contradicted

        if row[
            "reid"
        ]
        >= INTERESTING_REID_MIN
    ]


    high_reid_contradictions.sort(
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


    print_rows(
        "HIGH-ReID GEOMETRY CONTRADICTIONS",
        high_reid_contradictions,
    )


    # ========================================================
    # ALL VERY STRONG APPEARANCE PAIRS
    #
    # Show geometry regardless of its status.
    # ========================================================

    very_strong = [
        row

        for row in rows

        if row[
            "reid"
        ]
        >= STRONG_REID_MIN
    ]


    very_strong.sort(
        key=lambda row:
            row[
                "reid"
            ],
        reverse=True,
    )


    print_rows(
        (
            "ALL ReID >= "
            f"{STRONG_REID_MIN:.2f}"
        ),
        very_strong,
    )


    return {
        "total":
            len(
                rows
            ),

        "supported":
            len(
                supported
            ),

        "unknown":
            len(
                unknown
            ),

        "contradicted":
            len(
                contradicted
            ),

        "joint":
            len(
                joint
            ),

        "high_reid_contradictions":
            len(
                high_reid_contradictions
            ),
    }


# ============================================================
# c2 WHOLE-TRACK CAUTION REPORT
# ============================================================

def report_c2_caution_rows(
    rows,
):

    caution_rows = [
        row

        for row in rows

        if (
            row[
                "camera_a"
            ]
            == 2

            and

            row[
                "track_a"
            ]
            in KNOWN_C2_HETEROGENEOUS

            and

            (
                (
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

                or

                (
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

                or

                row[
                    "reid"
                ]
                >= STRONG_REID_MIN
            )
        )
    ]


    caution_rows.sort(
        key=lambda row:
            row[
                "reid"
            ],
        reverse=True,
    )


    print()
    print(
        "c2 WHOLE-TRACK CAUTION ROWS"
    )

    print(
        "=" * 60
    )


    if not caution_rows:

        print(
            "NONE"
        )

    else:

        print(
            "These involve raw c2:9 or c2:601."
        )

        print(
            "Do not treat them as final identity evidence "
            "without identity-aware segmentation."
        )

        print()


        for row in caution_rows:

            print_row(
                row
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TERRACE c3 GEOMETRY + ReID DIAGNOSTIC"
    )

    print(
        "====================================="
    )


    print()
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
        "Geometry distance units are Terrace "
        "ground-coordinate units, not metres."
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
            homographies.keys()
        ),
    )


    pair_rows = {}

    summaries = {}


    for reference_camera in REFERENCE_CAMERAS:

        rows = scan_camera_pair(
            reference_camera,
            TARGET_CAMERA,
            homographies,
        )


        pair_rows[
            reference_camera
        ] = rows


    for reference_camera in REFERENCE_CAMERAS:

        summaries[
            reference_camera
        ] = report_pair(
            reference_camera,
            TARGET_CAMERA,
            pair_rows[
                reference_camera
            ],
        )


    # ========================================================
    # SPECIAL c2 WARNING
    # ========================================================

    report_c2_caution_rows(
        pair_rows[
            2
        ]
    )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    total_joint = sum(
        summary[
            "joint"
        ]

        for summary
        in summaries.values()
    )


    total_high_reid_contradictions = sum(
        summary[
            "high_reid_contradictions"
        ]

        for summary
        in summaries.values()
    )


    print()
    print(
        "OVERALL c3 DIAGNOSTIC SUMMARY"
    )

    print(
        "============================="
    )


    for reference_camera in REFERENCE_CAMERAS:

        summary = summaries[
            reference_camera
        ]


        print(
            f"c{reference_camera}<->c3"
            f" | pairs={summary['total']}"
            f" | supported={summary['supported']}"
            f" | unknown={summary['unknown']}"
            f" | contradicted={summary['contradicted']}"
            f" | supported+ReID>={INTERESTING_REID_MIN:.2f}"
            f"={summary['joint']}"
            f" | high-ReID contradictions="
            f"{summary['high_reid_contradictions']}"
        )


    print()
    print(
        "Supported + ReID >=",
        INTERESTING_REID_MIN,
        ":",
        total_joint,
    )

    print(
        "High-ReID contradictions:",
        total_high_reid_contradictions,
    )


    print()
    print(
        "No identities were modified."
    )

    print(
        "Production thresholds were not modified."
    )

    print(
        "This script is diagnostic only."
    )


if __name__ == "__main__":
    main()
