from pathlib import Path
import argparse
import csv

from tracklet import Detection, Tracklet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
)


def load_tracklets(csv_path: Path):

    tracklets = {}

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            camera_id = row["camera_id"]
            local_track_id = int(
                row["local_track_id"]
            )

            key = (
                camera_id,
                local_track_id,
            )

            if key not in tracklets:

                tracklets[key] = Tracklet(
                    camera_id=camera_id,
                    local_track_id=local_track_id,
                )

            detection = Detection(
                frame=int(row["frame"]),
                timestamp=float(row["timestamp"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                confidence=float(row["confidence"]),
            )

            tracklets[key].add_detection(
                detection
            )

    for tracklet in tracklets.values():

        tracklet.sort_detections()

    return tracklets


def main():

    parser = argparse.ArgumentParser(
        description="Build Terrace tracklets from ByteTrack CSV"
    )

    parser.add_argument(
        "--camera",
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help="Terrace camera number",
    )

    parser.add_argument(
        "--min-detections",
        type=int,
        default=25,
        help="Minimum detections for a useful tracklet",
    )

    args = parser.parse_args()

    camera_id = args.camera

    track_csv = (
        TRACK_OUTPUT_DIR
        / f"tracks_c{camera_id}.csv"
    )

    if not track_csv.exists():

        raise FileNotFoundError(
            f"CSV not found: {track_csv}"
        )

    tracklets = load_tracklets(
        track_csv
    )

    useful_tracklets = [
        tracklet
        for tracklet in tracklets.values()
        if tracklet.num_detections
        >= args.min_detections
    ]

    short_tracklets = [
        tracklet
        for tracklet in tracklets.values()
        if tracklet.num_detections
        < args.min_detections
    ]

    useful_tracklets.sort(
        key=lambda t: t.num_detections,
        reverse=True,
    )

    print()
    print("TRACKLET BUILD")
    print("==============")
    print()

    print(
        f"Camera: c{camera_id}"
    )

    print(
        f"Input CSV: {track_csv}"
    )

    print(
        f"Total tracklets: {len(tracklets)}"
    )

    print(
        f"Useful tracklets >="
        f"{args.min_detections} detections: "
        f"{len(useful_tracklets)}"
    )

    print(
        f"Short tracklets <"
        f"{args.min_detections} detections: "
        f"{len(short_tracklets)}"
    )

    print()
    print("Top 20 tracklets")
    print("----------------")

    for tracklet in useful_tracklets[:20]:

        print(
            tracklet.summary()
        )


if __name__ == "__main__":
    main()
