from pathlib import Path
import argparse
import csv

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
DATA_DIR = PROJECT_ROOT / "data" / "videos"

# Important:
# Keep this experiment separate from our original baseline.
OUTPUT_DIR = PROJECT_ROOT / "output" / "terrace_bytetrack_tuned"

TRACKER_CONFIG = (
    PROJECT_ROOT
    / "src"
    / "bytetrack_terrace.yaml"
)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Terrace YOLO + ByteTrack local tracking"
)

parser.add_argument(
    "--camera",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help="Terrace camera number: 0, 1, 2, or 3",
)

args = parser.parse_args()

camera_id = args.camera


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

video_path = (
    DATA_DIR
    / f"terrace1-c{camera_id}.avi"
)

output_video_path = (
    OUTPUT_DIR
    / f"tracked_c{camera_id}.mp4"
)

output_csv_path = (
    OUTPUT_DIR
    / f"tracks_c{camera_id}.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

if not video_path.exists():
    raise FileNotFoundError(
        f"Video not found: {video_path}"
    )

if not TRACKER_CONFIG.exists():
    raise FileNotFoundError(
        f"Tracker config not found: {TRACKER_CONFIG}"
    )


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading model:")
print(MODEL_PATH)

print()
print("Using tracker config:")
print(TRACKER_CONFIG)

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(video_path)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {video_path}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if fps <= 0:
    raise RuntimeError(
        "Invalid video FPS"
    )


print()
print("Video information")
print("-----------------")
print(f"Camera:       c{camera_id}")
print(f"Resolution:   {width} x {height}")
print(f"FPS:          {fps:.2f}")
print(f"Total frames: {total_frames}")
print()


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(output_video_path),
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video"
    )


# ============================================================
# CSV OUTPUT
# ============================================================

csv_file = open(
    output_csv_path,
    "w",
    newline="",
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow(
    [
        "camera_id",
        "frame",
        "timestamp",
        "local_track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    ]
)


# ============================================================
# TRACKING LOOP
# ============================================================

frame_index = 0
total_track_rows = 0


print("Starting tracking...")
print()


while True:

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp = frame_index / fps


    # --------------------------------------------------------
    # YOLO + BYTETRACK
    # --------------------------------------------------------
    #
    # Important:
    #
    # conf = 0.10 lets lower-confidence person detections
    # reach ByteTrack.
    #
    # Our tracker YAML decides:
    #
    # >= 0.25:
    #     strong detection
    #
    # 0.10 - 0.25:
    #     can help recover an existing track
    #
    # < 0.25:
    #     cannot initialize a brand-new track
    #
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,

        tracker=str(
            TRACKER_CONFIG
        ),

        classes=[0],

        conf=0.10,

        device=0,

        verbose=False,
    )


    result = results[0]


    # ========================================================
    # DRAW TRACKS
    # ========================================================

    annotated_frame = (
        result.plot()
    )


    # ========================================================
    # SAVE TRACK INFORMATION
    # ========================================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        for (
            box,
            track_id,
            confidence,
        ) in zip(
            boxes,
            track_ids,
            confidences,
        ):

            x1, y1, x2, y2 = box


            csv_writer.writerow(
                [
                    f"c{camera_id}",
                    frame_index,
                    f"{timestamp:.3f}",
                    track_id,
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                    f"{confidence:.4f}",
                ]
            )


            total_track_rows += 1


    # ========================================================
    # SAVE VIDEO FRAME
    # ========================================================

    writer.write(
        annotated_frame
    )


    frame_index += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_index % 250 == 0:

        print(
            f"Processed "
            f"{frame_index}/"
            f"{total_frames} frames"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

csv_file.close()


# ============================================================
# SUMMARY
# ============================================================

print()
print("Tracking completed")
print("==================")
print()

print(
    f"Camera:          c{camera_id}"
)

print(
    f"Frames:          {frame_index}"
)

print(
    f"Track rows:      {total_track_rows}"
)

print()

print(
    f"Video: {output_video_path}"
)

print(
    f"CSV:   {output_csv_path}"
)

print()
print("Experiment:")
print("YOLO conf        = 0.10")
print("track_high       = 0.25")
print("track_low        = 0.10")
print("new_track        = 0.25")
print("track_buffer     = 75")
print("match_thresh     = 0.85")
