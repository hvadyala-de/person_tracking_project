from pathlib import Path
from collections import defaultdict
import argparse

import cv2

from build_tracklets import load_tracklets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# SETTINGS
# ============================================================

MIN_DETECTIONS = 25
CROPS_PER_TRACKLET = 8


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Extract representative Terrace person crops"
)

parser.add_argument(
    "--camera",
    type=int,
    required=True,
    choices=[0, 1, 2, 3],
    help="Terrace camera number",
)

args = parser.parse_args()

camera_id = args.camera


# ============================================================
# PATHS
# ============================================================

TRACK_CSV = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
    / f"tracks_c{camera_id}.csv"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / f"terrace1-c{camera_id}.avi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_crops"
    / f"c{camera_id}"
)


# ============================================================
# REPRESENTATIVE DETECTION SELECTION
# ============================================================

def select_detections(tracklet, count):

    detections = tracklet.detections

    if len(detections) <= count:
        return detections

    selected = []

    for i in range(count):

        index = round(
            i
            * (len(detections) - 1)
            / (count - 1)
        )

        selected.append(
            detections[index]
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    if not TRACK_CSV.exists():
        raise FileNotFoundError(
            f"Track CSV not found: {TRACK_CSV}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracklets = load_tracklets(
        TRACK_CSV
    )

    useful_tracklets = [
        tracklet
        for tracklet in tracklets.values()
        if tracklet.num_detections >= MIN_DETECTIONS
    ]

    print()
    print("FAST CROP EXTRACTION")
    print("====================")
    print()

    print(f"Camera:             c{camera_id}")
    print(f"Useful tracklets:   {len(useful_tracklets)}")
    print(f"Crops per tracklet: {CROPS_PER_TRACKLET}")
    print()

    # ========================================================
    # FRAME REQUEST LOOKUP
    # ========================================================

    frame_requests = defaultdict(list)

    for tracklet in useful_tracklets:

        selected = select_detections(
            tracklet,
            CROPS_PER_TRACKLET,
        )

        for detection in selected:

            frame_requests[
                detection.frame
            ].append(
                (
                    tracklet,
                    detection,
                )
            )

    print(
        f"Unique frames requested: "
        f"{len(frame_requests)}"
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open: {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # ========================================================
    # SEQUENTIAL VIDEO PASS
    # ========================================================

    frame_index = 0
    total_saved = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_index in frame_requests:

            height, width = frame.shape[:2]

            for tracklet, detection in frame_requests[
                frame_index
            ]:

                x1 = max(
                    0,
                    int(detection.x1),
                )

                y1 = max(
                    0,
                    int(detection.y1),
                )

                x2 = min(
                    width,
                    int(detection.x2),
                )

                y2 = min(
                    height,
                    int(detection.y2),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:
                    continue

                track_dir = (
                    OUTPUT_DIR
                    / f"track_{tracklet.local_track_id}"
                )

                track_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                crop_path = (
                    track_dir
                    / f"frame_{frame_index:06d}.jpg"
                )

                if cv2.imwrite(
                    str(crop_path),
                    crop,
                ):
                    total_saved += 1

        frame_index += 1

        if frame_index % 500 == 0:

            print(
                f"Read {frame_index}/"
                f"{total_frames} frames | "
                f"saved {total_saved}"
            )

    cap.release()

    print()
    print("Crop extraction completed")
    print("=========================")
    print()

    print(f"Camera:            c{camera_id}")
    print(f"Frames read:       {frame_index}")
    print(f"Tracklets:         {len(useful_tracklets)}")
    print(f"Crops saved:       {total_saved}")
    print(f"Output directory:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
