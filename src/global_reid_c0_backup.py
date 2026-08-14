import cv2
import torch
import torchreid
import numpy as np
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/home/jetson_agx_orin/person_tracking_project"

MODEL_PATH = f"{BASE_DIR}/models/yolov8n.pt"
VIDEO_PATH = f"{BASE_DIR}/data/videos/terrace1-c0.avi"

OUTPUT_PATH = f"{BASE_DIR}/output/global_reid_c0.mp4"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.40

# Minimum OSNet cosine similarity required for a match.
APPEARANCE_THRESHOLD = 0.65

# Number of embeddings retained for each GID.
MAX_EMBEDDINGS_PER_ID = 20

# Number of frames for which a GID remains active
# after its local track disappears.
MAX_MISSING_FRAMES = 125

# Minimum useful person crop.
MIN_WIDTH = 30
MIN_HEIGHT = 60


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

    crop = cv2.resize(crop, (128, 256))

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop = np.transpose(
        crop,
        (2, 0, 1)
    )

    crop = crop.astype(
        np.float32
    ) / 255.0

    tensor = (
        torch.from_numpy(crop)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        embedding = reid_model(tensor)

    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding1,
    embedding2
):

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
# CREATE NEW GLOBAL ID
# ============================================================

def create_global_identity(
    embedding,
    frame_number,
    track_id,
    center_x,
    center_y
):

    global next_global_id

    global_id = next_global_id

    next_global_id += 1

    identity_gallery[global_id] = {

        "embeddings": [
            embedding.detach()
        ],

        "last_seen_frame": frame_number,

        "last_track_id": track_id,

        "last_center": (
            center_x,
            center_y
        )
    }

    return global_id


# ============================================================
# FIND BEST GLOBAL ID
# ============================================================

def find_global_identity(
    embedding,
    frame_number
):

    best_id = None
    best_score = -1.0

    for global_id, data in identity_gallery.items():

        # ----------------------------------------------------
        # Ignore identities that have been missing too long.
        # ----------------------------------------------------

        frames_missing = (
            frame_number
            - data["last_seen_frame"]
        )

        if frames_missing > MAX_MISSING_FRAMES:
            continue

        # ----------------------------------------------------
        # Compare against stored appearance embeddings.
        # ----------------------------------------------------

        for stored_embedding in data["embeddings"]:

            score = cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > best_score:

                best_score = score
                best_id = global_id

    # --------------------------------------------------------
    # Existing identity
    # --------------------------------------------------------

    if (
        best_id is not None
        and best_score >= APPEARANCE_THRESHOLD
    ):

        return best_id, best_score

    # --------------------------------------------------------
    # No reliable match.
    # Create a new identity.
    # --------------------------------------------------------

    return None, best_score


# ============================================================
# UPDATE GLOBAL IDENTITY
# ============================================================

def update_identity(
    global_id,
    embedding,
    frame_number,
    track_id,
    center_x,
    center_y
):

    identity_gallery[
        global_id
    ]["embeddings"].append(
        embedding.detach()
    )

    if (
        len(
            identity_gallery[
                global_id
            ]["embeddings"]
        )
        > MAX_EMBEDDINGS_PER_ID
    ):

        identity_gallery[
            global_id
        ]["embeddings"].pop(0)

    identity_gallery[
        global_id
    ]["last_seen_frame"] = frame_number

    identity_gallery[
        global_id
    ]["last_track_id"] = track_id

    identity_gallery[
        global_id
    ]["last_center"] = (
        center_x,
        center_y
    )


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
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


print("Video opened.")
print(
    "Resolution:",
    width,
    "x",
    height
)
print(
    "FPS:",
    fps
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

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


    # ========================================================
    # YOLO + BOTSORT
    # ========================================================

    results = yolo.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # DETECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes
            .xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes
            .conf
            .cpu()
            .numpy()
        )


        for box, track_id, confidence in zip(
            boxes,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )


            person_width = x2 - x1
            person_height = y2 - y1


            # ------------------------------------------------
            # Ignore tiny detections
            # ------------------------------------------------

            if (
                person_width < MIN_WIDTH
                or person_height < MIN_HEIGHT
            ):
                continue


            # ------------------------------------------------
            # Person center
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0


            # ------------------------------------------------
            # Person crop
            # ------------------------------------------------

            crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # OSNET EMBEDDING
            # ------------------------------------------------

            embedding = get_embedding(
                crop
            )

            if embedding is None:
                continue


            # ------------------------------------------------
            # GLOBAL ID MATCH
            # ------------------------------------------------

            global_id, similarity = (
                find_global_identity(
                    embedding,
                    frame_number
                )
            )


            # ------------------------------------------------
            # CREATE NEW GID IF NECESSARY
            # ------------------------------------------------

            if global_id is None:

                global_id = create_global_identity(
                    embedding,
                    frame_number,
                    track_id,
                    center_x,
                    center_y
                )

                similarity_text = "NEW"

            else:

                update_identity(
                    global_id,
                    embedding,
                    frame_number,
                    track_id,
                    center_x,
                    center_y
                )

                similarity_text = (
                    f"{similarity:.2f}"
                )


            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"GID:{global_id} "
                f"S:{similarity_text}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


    # ========================================================
    # PEOPLE / GID COUNT
    # ========================================================

    active_ids = set()

    for global_id, data in identity_gallery.items():

        frames_missing = (
            frame_number
            - data["last_seen_frame"]
        )

        if frames_missing <= MAX_MISSING_FRAMES:

            active_ids.add(
                global_id
            )


    # --------------------------------------------------------
    # Top-right information panel
    # --------------------------------------------------------

    panel_text = (
        f"Active GIDs: {len(active_ids)}"
    )

    cv2.putText(
        frame,
        panel_text,
        (
            max(10, width - 180),
            25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE FRAME
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 100 == 0:

        print(
            f"Processed frame "
            f"{frame_number} | "
            f"Active GIDs: "
            f"{len(active_ids)} | "
            f"Total GIDs: "
            f"{len(identity_gallery)}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


print()
print("=" * 50)
print("Global Re-ID C0 test completed")
print("Frames processed:", frame_number)
print("Total GIDs:", len(identity_gallery))
print("Output:", OUTPUT_PATH)
print("=" * 50)
