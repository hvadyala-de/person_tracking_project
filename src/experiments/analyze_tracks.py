from pathlib import Path
from collections import defaultdict
import argparse
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


parser = argparse.ArgumentParser(
    description="Analyze ByteTrack local track fragmentation"
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to tracks CSV, relative to project root",
)

args = parser.parse_args()


TRACK_CSV = PROJECT_ROOT / args.input


if not TRACK_CSV.exists():
    raise FileNotFoundError(
        f"Track CSV not found: {TRACK_CSV}"
    )


tracks = defaultdict(list)


# ============================================================
# READ TRACK OUTPUT
# ============================================================

with open(TRACK_CSV, "r", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        key = (
            row["camera_id"],
            int(row["local_track_id"]),
        )

        tracks[key].append(
            {
                "frame": int(row["frame"]),
                "timestamp": float(row["timestamp"]),
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            }
        )


# ============================================================
# ANALYZE TRACKS
# ============================================================

summaries = []


for (camera_id, track_id), detections in tracks.items():

    detections.sort(
        key=lambda x: x["frame"]
    )

    frames = [
        d["frame"]
        for d in detections
    ]

    start_frame = frames[0]
    end_frame = frames[-1]

    start_time = detections[0]["timestamp"]
    end_time = detections[-1]["timestamp"]

    num_detections = len(detections)

    mean_confidence = (
        sum(
            d["confidence"]
            for d in detections
        )
        / num_detections
    )

    gaps = [
        frames[i] - frames[i - 1]
        for i in range(
            1,
            len(frames),
        )
    ]

    max_gap = (
        max(gaps)
        if gaps
        else 0
    )

    duration_seconds = (
        end_time - start_time
    )

    summaries.append(
        {
            "camera_id": camera_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "detections": num_detections,
            "duration": duration_seconds,
            "mean_conf": mean_confidence,
            "max_gap": max_gap,
        }
    )


summaries.sort(
    key=lambda x: x["detections"],
    reverse=True,
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

short_tracks = [
    t
    for t in summaries
    if t["detections"] < 25
]

medium_tracks = [
    t
    for t in summaries
    if 25 <= t["detections"] < 100
]

long_tracks = [
    t
    for t in summaries
    if t["detections"] >= 100
]


print()
print("BYTE TRACK ANALYSIS")
print("===================")
print()

print(f"CSV: {TRACK_CSV}")
print()

print(
    f"Total local IDs:          {len(summaries)}"
)

print(
    f"Tracks < 25 detections:   {len(short_tracks)}"
)

print(
    f"Tracks 25-99 detections:  {len(medium_tracks)}"
)

print(
    f"Tracks >=100 detections:  {len(long_tracks)}"
)

print()

print("Longest tracks")
print("--------------")

print(
    f"{'ID':>6} "
    f"{'Start':>7} "
    f"{'End':>7} "
    f"{'Detections':>11} "
    f"{'Duration':>10} "
    f"{'Conf':>8} "
    f"{'MaxGap':>8}"
)

for track in summaries[:30]:

    print(
        f"{track['track_id']:>6} "
        f"{track['start_frame']:>7} "
        f"{track['end_frame']:>7} "
        f"{track['detections']:>11} "
        f"{track['duration']:>9.2f}s "
        f"{track['mean_conf']:>8.3f} "
        f"{track['max_gap']:>8}"
    )


print()
print("Analysis complete.")
