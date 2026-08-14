import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "/home/jetson_agx_orin/person_tracking_project/models/yolov8n.pt"

VIDEO_PATH = "/home/jetson_agx_orin/person_tracking_project/data/videos/terrace1-c0.avi"

OUTPUT_PATH = "/home/jetson_agx_orin/person_tracking_project/output/botsort_c0_120.mp4"

TRACKER_PATH = "/home/jetson_agx_orin/person_tracking_project/src/botsort_c0.yaml"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.40


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO ready.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print("Video opened.")
print("Resolution:", width, "x", height)
print("FPS:", fps)


# ============================================================
# OUTPUT
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output: {OUTPUT_PATH}"
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

total_detections = 0

frames_with_detections = 0

max_active_tracks = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_PATH,
        classes=[0],
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # DRAW TRACKS
    # --------------------------------------------------------

    active_tracks = 0

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        active_tracks = len(track_ids)

        total_detections += active_tracks

        if active_tracks > 0:
            frames_with_detections += 1

        for box, track_id in zip(
            boxes,
            track_ids
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DRAW LOCAL TRACK ID
            # ------------------------------------------------

            label = f"TID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # TRACK COUNT
    # --------------------------------------------------------

    max_active_tracks = max(
        max_active_tracks,
        active_tracks
    )


    cv2.putText(
        frame,
        f"Active tracks: {active_tracks}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )


    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    writer.write(frame)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 100 == 0:

        print(
            f"Processed frame {frame_number} | "
            f"Active tracks: {active_tracks} | "
            f"Max active tracks: {max_active_tracks}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=======================================================")
print("BoT-SORT C0 120-buffer test completed")
print("Frames processed:", frame_number)
print("Frames with detections:", frames_with_detections)
print("Maximum active tracks:", max_active_tracks)
print("Total track detections:", total_detections)
print("Output:", OUTPUT_PATH)
print("=======================================================")
