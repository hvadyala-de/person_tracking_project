import argparse
import csv
from pathlib import Path

try:
    from .tracklet import Detection, Tracklet
except ImportError:
    from tracklet import Detection, Tracklet


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACKING_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
)


# ============================================================
# LOAD TRACKLETS FROM CSV
# ============================================================

def load_tracklets(
    csv_path: Path,
):

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Track CSV not found: {csv_path}"
        )

    tracklets = {}

    with csv_path.open(
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            camera_id = row[
                "camera_id"
            ]

            local_track_id = int(
                row[
                    "local_track_id"
                ]
            )

            frame = int(
                row[
                    "frame"
                ]
            )

            timestamp = float(
                row[
                    "timestamp"
                ]
            )

            x1 = float(
                row[
                    "x1"
                ]
            )

            y1 = float(
                row[
                    "y1"
                ]
            )

            x2 = float(
                row[
                    "x2"
                ]
            )

            y2 = float(
                row[
                    "y2"
                ]
            )

            confidence = float(
                row[
                    "confidence"
                ]
            )


            # ------------------------------------------------
            # Use camera + local ID as the unique key.
            #
            # Example:
            #
            # ("c0", 701)
            # ------------------------------------------------

            key = (
                camera_id,
                local_track_id,
            )


            if key not in tracklets:

                tracklets[
                    key
                ] = Tracklet(
                    camera_id=camera_id,
                    local_track_id=local_track_id,
                )


            detection = Detection(
                frame=frame,
                timestamp=timestamp,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
            )


            tracklets[
                key
            ].add_detection(
                detection
            )


    # --------------------------------------------------------
    # Ensure detections are ordered by frame.
    # --------------------------------------------------------

    for tracklet in tracklets.values():

        tracklet.sort_detections()


    return tracklets


# ============================================================
# FILTER USEFUL TRACKLETS
# ============================================================

def filter_tracklets(
    tracklets,
    min_detections=25,
):

    return {
        key: tracklet
        for key, tracklet
        in tracklets.items()

        if (
            tracklet.num_detections
            >= min_detections
        )
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    tracklets,
    min_detections,
):

    useful = filter_tracklets(
        tracklets,
        min_detections=min_detections,
    )

    total_count = len(
        tracklets
    )

    useful_count = len(
        useful
    )

    short_count = (
        total_count
        - useful_count
    )


    print()
    print("TRACKLET SUMMARY")
    print("================")

    print(
        f"Total tracklets: "
        f"{total_count}"
    )

    print(
        f"Useful tracklets "
        f"(>= {min_detections} detections): "
        f"{useful_count}"
    )

    print(
        f"Short tracklets: "
        f"{short_count}"
    )


    print()
    print("TOP TRACKLETS")
    print("=============")


    sorted_tracklets = sorted(
        useful.values(),
        key=lambda tracklet:
            tracklet.num_detections,
        reverse=True,
    )


    for tracklet in sorted_tracklets[:20]:

        print(
            tracklet.summary()
        )


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build Terrace person tracklets "
            "from ByteTrack CSV output."
        )
    )


    parser.add_argument(
        "--camera",
        type=int,
        required=True,
        choices=[
            0,
            1,
            2,
            3,
        ],
        help=(
            "Terrace camera number "
            "(0, 1, 2, or 3)"
        ),
    )


    parser.add_argument(
        "--min-detections",
        type=int,
        default=25,
        help=(
            "Minimum number of detections "
            "for a useful tracklet."
        ),
    )


    args = parser.parse_args()


    csv_path = (
        TRACKING_DIR
        / f"tracks_c{args.camera}.csv"
    )


    print()
    print(
        f"Loading: {csv_path}"
    )


    tracklets = load_tracklets(
        csv_path
    )


    print_summary(
        tracklets,
        min_detections=args.min_detections,
    )


if __name__ == "__main__":
    main()
