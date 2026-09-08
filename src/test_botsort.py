from ultralytics import YOLO

MODEL_PATH = "/home/jetson_agx_orin/person_tracking_project/models/yolov8n.pt"
VIDEO_PATH = "/home/jetson_agx_orin/person_tracking_project/data/videos/terrace1-c0.avi"

OUTPUT_DIR = "/home/jetson_agx_orin/person_tracking_project/output"


print("Loading YOLO...")
model = YOLO(MODEL_PATH)
print("YOLO ready.")

print("Starting BoT-SORT tracking...")

model.track(
    source=VIDEO_PATH,
    persist=True,
    tracker="botsort.yaml",
    conf=0.25,
    classes=[0],
    device=0,
    save=True,
    project=OUTPUT_DIR,
    name="botsort_c0",
    verbose=True,
)

print()
print("======================================")
print("BoT-SORT test completed")
print("======================================")
