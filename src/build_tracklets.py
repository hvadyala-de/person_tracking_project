from pathlib import Path
from collections import defaultdict
import csv

from tracklet import Detection, Tracklet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_CSV = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
    / "tracks_c0.csv"
)


def load_tracklets(csv_path: Path):

    tracklets = {}

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            camera_id = row["camera_id"]
            local_track_id = int(row["local_track_id"])

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

    if not TRACK_CSV.exists():

        raise FileNotFoundError(
            f"CSV not found: {TRACK_CSV}"
        )

    tracklets = load_tracklets(
        TRACK_CSV
    )

    print()
    print("TRACKLET BUILD")
    print("==============")
    print()

    print(
        f"Input CSV: {TRACK_CSV}"
    )

    print(
        f"Total tracklets: {len(tracklets)}"
    )

    print()

    useful_tracklets = [
        tracklet
        for tracklet in tracklets.values()
        if tracklet.num_detections >= 25
    ]

    short_tracklets = [
        tracklet
        for tracklet in tracklets.values()
        if tracklet.num_detections < 25
    ]

    print(
        f"Useful tracklets >=25 detections: "
        f"{len(useful_tracklets)}"
    )

    print(
        f"Short tracklets <25 detections: "
        f"{len(short_tracklets)}"
    )

    print()

    useful_tracklets.sort(
        key=lambda t: t.num_detections,
        reverse=True,
    )

    print("Top 20 tracklets")
    print("----------------")

    for tracklet in useful_tracklets[:20]:

        print(
            tracklet.summary()
        )


if __name__ == "__main__":
    main()
