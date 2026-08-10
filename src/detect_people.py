from ultralytics import YOLO

MODEL_PATH = "/home/jetson_agx_orin/person_tracking_project/models/yolov8n.pt"
VIDEO_PATH = "/home/jetson_agx_orin/person_tracking_project/data/videos/terrace1-c0.avi"

model = YOLO(MODEL_PATH)

model.track(
    source=VIDEO_PATH,
    persist=True,
    save=True,
    project="/home/jetson_agx_orin/person_tracking_project/output",
    name="track_results",
    tracker="bytetrack.yaml",
    device=0,
    classes=[0]
)

print("Tracking completed!")
