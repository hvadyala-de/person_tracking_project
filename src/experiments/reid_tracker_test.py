import cv2
import torch
import torchreid
import numpy as np
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "/home/jetson_agx_orin/person_tracking_project/models/yolov8n.pt"

VIDEO_PATH = "/home/jetson_agx_orin/person_tracking_project/data/videos/terrace1-c0.avi"

OUTPUT_PATH = "/home/jetson_agx_orin/person_tracking_project/output/reid_test.mp4"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.40

# Appearance similarity threshold.
# We will tune this using your actual video.
APPEARANCE_THRESHOLD = 0.65

# How many previous embeddings to keep for each global identity.
MAX_EMBEDDINGS_PER_ID = 20


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

yolo = YOLO(MODEL_PATH)

print("YOLO ready.")


# ============================================================
# LOAD OSNET
# ============================================================

print("Loading OSNet...")

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

reid_model = reid_model.to(DEVICE)
reid_model.eval()

print("OSNet ready on:", DEVICE)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def get_embedding(crop):

    if crop is None or crop.size == 0:
        return None

    # OSNet input size
    crop = cv2.resize(crop, (128, 256))

    # BGR -> RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # HWC -> CHW
    crop = np.transpose(crop, (2, 0, 1))

    # Convert to float
    crop = crop.astype(np.float32) / 255.0

    # Add batch dimension
    tensor = torch.from_numpy(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = reid_model(tensor)

    # L2 normalization
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(embedding1, embedding2):

    return torch.sum(
        embedding1 * embedding2,
        dim=1
    ).item()


# ============================================================
# GLOBAL IDENTITY DATABASE
# ============================================================

identity_gallery = {}

next_global_id = 1


# ============================================================
# FIND GLOBAL ID
# ============================================================

def find_global_identity(embedding):

    global next_global_id

    best_id = None
    best_score = -1.0

    for global_id, data in identity_gallery.items():

        stored_embeddings = data["embeddings"]

        for stored_embedding in stored_embeddings:

            score = cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > best_score:
                best_score = score
                best_id = global_id

    # Existing person
    if best_score >= APPEARANCE_THRESHOLD:

        return best_id, best_score

    # New person
    global_id = next_global_id

    next_global_id += 1

    identity_gallery[global_id] = {
        "embeddings": [embedding.detach()]
    }

    return global_id, best_score


# ============================================================
# UPDATE IDENTITY GALLERY
# ============================================================

def update_identity(global_id, embedding):

    identity_gallery[global_id]["embeddings"].append(
        embedding.detach()
    )

    # Keep gallery bounded
    if len(identity_gallery[global_id]["embeddings"]) > MAX_EMBEDDINGS_PER_ID:

        identity_gallery[global_id]["embeddings"].pop(0)


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
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO + BYTETRACK
    # --------------------------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Clamp coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            # Ignore tiny detections
            if (x2 - x1) < 30 or (y2 - y1) < 60:
                continue

            # ------------------------------------------------
            # PERSON CROP
            # ------------------------------------------------

            crop = frame[y1:y2, x1:x2]

            # ------------------------------------------------
            # OSNET EMBEDDING
            # ------------------------------------------------

            embedding = get_embedding(crop)

            if embedding is None:
                continue

            # ------------------------------------------------
            # GLOBAL ID
            # ------------------------------------------------

            global_id, similarity = find_global_identity(
                embedding
            )

            # ------------------------------------------------
            # UPDATE GALLERY
            # ------------------------------------------------

            update_identity(
                global_id,
                embedding
            )

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

            # Display both IDs
            label = (
                f"GID:{global_id} "
                f"TID:{track_id} "
                f"S:{similarity:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    writer.write(frame)

    # Progress
    if frame_number % 100 == 0:

        print(
            f"Processed frame {frame_number} | "
            f"Global identities: {len(identity_gallery)}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

print()
print("======================================")
print("Re-ID tracking test completed")
print("Frames processed:", frame_number)
print("Global identities:", len(identity_gallery))
print("Output:", OUTPUT_PATH)
print("======================================")
