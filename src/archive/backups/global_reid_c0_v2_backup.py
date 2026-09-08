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

OUTPUT_PATH = f"{BASE_DIR}/output/global_reid_c0_v2.mp4"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.40

# OSNet appearance threshold.
APPEARANCE_THRESHOLD = 0.65

# Stronger threshold used when recovering a GID
# after the local track disappeared.
RECOVERY_APPEARANCE_THRESHOLD = 0.72

# Maximum number of embeddings stored for each GID.
MAX_EMBEDDINGS_PER_ID = 20

# A GID can remain recoverable for this many frames.
# At 25 FPS, 125 frames = approximately 5 seconds.
MAX_MISSING_FRAMES = 125

# Maximum allowed movement in pixels per frame.
# This is intentionally conservative for this low-resolution video.
MAX_SPEED_PIXELS_PER_FRAME = 12.0

# Minimum detection size.
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
# GLOBAL IDENTITY DATABASE
# ============================================================

identity_gallery = {}

next_global_id = 1


# ============================================================
# CREATE GID
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
# UPDATE GID
# ============================================================

def update_identity(
    global_id,
    embedding,
    frame_number,
    track_id,
    center_x,
    center_y
):

    data = identity_gallery[global_id]

    data["embeddings"].append(
        embedding.detach()
    )

    if len(data["embeddings"]) > MAX_EMBEDDINGS_PER_ID:
        data["embeddings"].pop(0)

    data["last_seen_frame"] = frame_number
    data["last_track_id"] = track_id

    data["last_center"] = (
        center_x,
        center_y
    )


# ============================================================
# SPATIAL DISTANCE CHECK
# ============================================================

def spatially_plausible(
    data,
    center_x,
    center_y,
    frame_number
):

    last_frame = data["last_seen_frame"]

    last_x, last_y = data["last_center"]

    frame_gap = frame_number - last_frame

    if frame_gap <= 0:
        return True

    dx = center_x - last_x
    dy = center_y - last_y

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    maximum_distance = (
        MAX_SPEED_PIXELS_PER_FRAME
        * frame_gap
    )

    return distance <= maximum_distance


# ============================================================
# APPEARANCE SCORE
# ============================================================

def best_appearance_score(
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
# FIND GID
# ============================================================

def find_global_identity(
    embedding,
    frame_number,
    center_x,
    center_y,
    track_id
):

    best_id = None
    best_score = -1.0

    for global_id, data in identity_gallery.items():

        # ----------------------------------------------------
        # Time gate
        # ----------------------------------------------------

        frames_missing = (
            frame_number
            - data["last_seen_frame"]
        )

        if frames_missing > MAX_MISSING_FRAMES:
            continue

        # ----------------------------------------------------
        # Same local track
        #
        # If the tracker is still giving us the same TID,
        # strongly prefer its existing GID.
        # ----------------------------------------------------

        if track_id == data["last_track_id"]:

            score = best_appearance_score(
                embedding,
                data["embeddings"]
            )

            if score >= APPEARANCE_THRESHOLD:

                return global_id, score, "same_track"

        # ----------------------------------------------------
        # Spatial gate
        #
        # Do not allow an identity to jump an unrealistic
        # distance during a short time interval.
        # ----------------------------------------------------

        if not spatially_plausible(
            data,
            center_x,
            center_y,
            frame_number
        ):
            continue

        # ----------------------------------------------------
        # Appearance matching
        # ----------------------------------------------------

        score = best_appearance_score(
            embedding,
            data["embeddings"]
        )

        # ----------------------------------------------------
        # Recovery after missing frames
        #
        # We require stronger appearance similarity when
        # recovering a previously missing identity.
        # ----------------------------------------------------

        if frames_missing > 0:

            required_threshold = (
                RECOVERY_APPEARANCE_THRESHOLD
            )

        else:

            required_threshold = (
                APPEARANCE_THRESHOLD
            )

        if score < required_threshold:
            continue

        # ----------------------------------------------------
        # Keep best candidate
        # ----------------------------------------------------

        if score > best_score:

            best_score = score
            best_id = global_id

    # --------------------------------------------------------
    # Match found
    # --------------------------------------------------------

    if best_id is not None:

        return (
            best_id,
            best_score,
            "appearance_spatial_temporal"
        )

    # --------------------------------------------------------
    # No reliable match
    # --------------------------------------------------------

    return (
        None,
        best_score,
        "new_identity"
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
    # CURRENT FRAME PEOPLE
    # ========================================================

    current_visible_gids = set()


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
            # Clamp box
            # ------------------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)


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
            # Center
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0


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


            # ------------------------------------------------
            # FIND GID
            # ------------------------------------------------

            (
                global_id,
                similarity,
                match_reason
            ) = find_global_identity(
                embedding,
                frame_number,
                center_x,
                center_y,
                track_id
            )


            # ------------------------------------------------
            # CREATE NEW GID
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
            # CURRENT VISIBLE GID
            # ------------------------------------------------

            current_visible_gids.add(
                global_id
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


            # ------------------------------------------------
            # LABEL
            #
            # TID is intentionally not displayed.
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
                0.50,
                (0, 255, 0),
                2
            )


    # ========================================================
    # CURRENT PEOPLE COUNT
    # ========================================================

    visible_people = len(
        current_visible_gids
    )


    # ========================================================
    # TOP-RIGHT INFORMATION
    # ========================================================

    panel = (
        f"People visible: "
        f"{visible_people}"
    )

    cv2.putText(
        frame,
        panel,
        (
            max(
                10,
                width - 205
            ),
            25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE
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
            f"Visible people: "
            f"{visible_people} | "
            f"Total GIDs: "
            f"{len(identity_gallery)}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


print()
print("=" * 55)
print("Global Re-ID C0 v2 completed")
print("Frames processed:", frame_number)
print(
    "Total GIDs created:",
    len(identity_gallery)
)
print("Output:", OUTPUT_PATH)
print("=" * 55)
