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

OUTPUT_PATH = f"{BASE_DIR}/output/global_reid_c0_v3.mp4"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.40

# Normal appearance matching.
APPEARANCE_THRESHOLD = 0.65

# Stronger threshold for recovering a person after disappearance.
RECOVERY_THRESHOLD = 0.72

# Same BoT-SORT track gets a strong preference.
SAME_TRACK_BONUS = 0.10

# Maximum number of embeddings remembered per GID.
MAX_EMBEDDINGS_PER_ID = 20

# How long a disappeared GID remains available for recovery.
MAX_MISSING_FRAMES = 125

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
# EMBEDDING
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
# GID DATABASE
# ============================================================

identity_gallery = {}

next_global_id = 1


# ============================================================
# CREATE NEW GID
# ============================================================

def create_identity(
    embedding,
    frame_number,
    track_id,
    center
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

        "last_center": center
    }

    return global_id


# ============================================================
# UPDATE GID
# ============================================================

def update_identity(
    global_id,
    embedding,
    frame_number,
    track_id,
    center
):

    data = identity_gallery[global_id]

    data["embeddings"].append(
        embedding.detach()
    )

    if len(data["embeddings"]) > MAX_EMBEDDINGS_PER_ID:

        data["embeddings"].pop(0)

    data["last_seen_frame"] = frame_number

    data["last_track_id"] = track_id

    data["last_center"] = center


# ============================================================
# BEST APPEARANCE SCORE
# ============================================================

def get_best_similarity(
    embedding,
    stored_embeddings
):

    best_score = -1.0

    for stored_embedding in stored_embeddings:

        score = cosine_similarity(
            embedding,
            stored_embedding
        )

        if score > best_score:

            best_score = score

    return best_score


# ============================================================
# SPATIAL CONSISTENCY
#
# This is NOT a hard pixel-speed gate.
#
# It is only used as a soft bonus/penalty.
# ============================================================

def spatial_score(
    data,
    center,
    frame_number
):

    last_frame = data["last_seen_frame"]

    last_x, last_y = data["last_center"]

    current_x, current_y = center

    frame_gap = max(
        1,
        frame_number - last_frame
    )

    distance = np.sqrt(
        (current_x - last_x) ** 2
        +
        (current_y - last_y) ** 2
    )

    # Normalize by time gap.
    movement_per_frame = (
        distance / frame_gap
    )

    # Soft spatial score.
    #
    # This does NOT reject a person.
    # It only makes implausible jumps less attractive.

    if movement_per_frame < 4:

        return 0.05

    elif movement_per_frame < 8:

        return 0.02

    elif movement_per_frame < 16:

        return 0.0

    elif movement_per_frame < 30:

        return -0.03

    else:

        return -0.08


# ============================================================
# BUILD MATCH CANDIDATES
# ============================================================

def build_candidates(
    detections,
    frame_number
):

    candidates = []

    for detection_index, detection in enumerate(
        detections
    ):

        embedding = detection["embedding"]

        track_id = detection["track_id"]

        center = detection["center"]

        for global_id, data in identity_gallery.items():

            frames_missing = (
                frame_number
                - data["last_seen_frame"]
            )

            if frames_missing > MAX_MISSING_FRAMES:

                continue

            appearance = get_best_similarity(
                embedding,
                data["embeddings"]
            )

            same_track = (
                track_id
                == data["last_track_id"]
            )

            # ------------------------------------------------
            # Determine minimum appearance requirement.
            # ------------------------------------------------

            if same_track:

                required_threshold = (
                    APPEARANCE_THRESHOLD - 0.05
                )

            elif frames_missing > 0:

                required_threshold = (
                    RECOVERY_THRESHOLD
                )

            else:

                required_threshold = (
                    APPEARANCE_THRESHOLD
                )

            if appearance < required_threshold:

                continue

            # ------------------------------------------------
            # Combined score.
            # ------------------------------------------------

            score = appearance

            if same_track:

                score += SAME_TRACK_BONUS

            score += spatial_score(
                data,
                center,
                frame_number
            )

            candidates.append({

                "detection_index": detection_index,

                "global_id": global_id,

                "appearance": appearance,

                "score": score,

                "same_track": same_track
            })

    return candidates


# ============================================================
# ONE-TO-ONE ASSIGNMENT
#
# IMPORTANT:
# One detection can get only one GID.
# One GID can be assigned to only one detection
# in the current frame.
# ============================================================

def assign_gids(
    detections,
    frame_number
):

    candidates = build_candidates(
        detections,
        frame_number
    )

    # Sort strongest matches first.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    assigned_detections = set()

    assigned_gids = set()

    assignments = {}

    # --------------------------------------------------------
    # Greedy one-to-one matching.
    # --------------------------------------------------------

    for candidate in candidates:

        detection_index = (
            candidate["detection_index"]
        )

        global_id = (
            candidate["global_id"]
        )

        if detection_index in assigned_detections:

            continue

        if global_id in assigned_gids:

            continue

        # ----------------------------------------------------
        # Assign.
        # ----------------------------------------------------

        assignments[detection_index] = {
            "global_id": global_id,
            "score": candidate["score"]
        }

        assigned_detections.add(
            detection_index
        )

        assigned_gids.add(
            global_id
        )

    # --------------------------------------------------------
    # Every unmatched detection gets a NEW GID.
    # --------------------------------------------------------

    for detection_index, detection in enumerate(
        detections
    ):

        if detection_index in assignments:

            continue

        global_id = create_identity(
            detection["embedding"],
            frame_number,
            detection["track_id"],
            detection["center"]
        )

        assignments[detection_index] = {
            "global_id": global_id,
            "score": None
        }

    return assignments


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
# OUTPUT
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
    # BOTSORT
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


    detections = []


    # ========================================================
    # EXTRACT DETECTIONS
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


        for box, track_id in zip(
            boxes,
            track_ids
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # Clamp
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


            # ------------------------------------------------
            # Size filter
            # ------------------------------------------------

            if (
                x2 - x1 < MIN_WIDTH
                or
                y2 - y1 < MIN_HEIGHT
            ):

                continue


            # ------------------------------------------------
            # Center
            # ------------------------------------------------

            center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0
            )


            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # OSNet
            # ------------------------------------------------

            embedding = get_embedding(
                crop
            )

            if embedding is None:

                continue


            detections.append({

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "track_id": track_id,

                "center": center,

                "embedding": embedding
            })


    # ========================================================
    # ONE-TO-ONE GID MATCHING
    # ========================================================

    assignments = assign_gids(
        detections,
        frame_number
    )


    # ========================================================
    # UPDATE + DRAW
    # ========================================================

    for detection_index, detection in enumerate(
        detections
    ):

        assignment = assignments[
            detection_index
        ]

        global_id = assignment[
            "global_id"
        ]


        x1, y1, x2, y2 = detection[
            "box"
        ]


        # ----------------------------------------------------
        # Update identity memory.
        # ----------------------------------------------------

        update_identity(
            global_id,
            detection["embedding"],
            frame_number,
            detection["track_id"],
            detection["center"]
        )


        # ----------------------------------------------------
        # Green bounding box.
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # GID ONLY
        # ----------------------------------------------------

        label = f"GID:{global_id}"


        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
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
            f"Detections: "
            f"{len(detections)} | "
            f"Total GIDs: "
            f"{len(identity_gallery)}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()


print()
print("=" * 60)
print("Global Re-ID C0 v3 completed")
print("Frames processed:", frame_number)
print(
    "Total GIDs created:",
    len(identity_gallery)
)
print("Output:", OUTPUT_PATH)
print("=" * 60)
