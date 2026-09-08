from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "/home/jetson_agx_orin/person_tracking_project/models/yolov8n.pt"

VIDEO_PATH = "/home/jetson_agx_orin/person_tracking_project/data/videos/terrace1-c0.avi"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# PERSON TRACKING
# ============================================================

model.track(
    source=VIDEO_PATH,

    # Keep track IDs across consecutive frames
    persist=True,

    # Save the output video
    save=True,

    # Lower confidence to help detect smaller/distant people
    conf=0.25,

    # Output directory
    project="/home/jetson_agx_orin/person_tracking_project/output",

    # Output folder name
    name="track_results_conf025",

    # ByteTrack tracker
    tracker="bytetrack.yaml",

    # Use GPU
    device=0,

    # COCO class 0 = person
    classes=[0]
)


# ============================================================
# DONE
# ============================================================

print("Tracking completed!")
print("Confidence threshold: 0.25")
print("Output folder:")
print(
    "/home/jetson_agx_orin/person_tracking_project/output/"
    "track_results_conf025"
)
