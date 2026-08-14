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

OUTPUT_PATH = f"{BASE_DIR}/output/global_reid_c0_v4.mp4"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.40

# General appearance matching threshold.
APPEARANCE_THRESHOLD = 0.68

# Stronger threshold when recovering a disappeared person.
RECOVERY_THRESHOLD = 0.72

# Threshold when the same BoT-SORT track continues.
SAME_TRACK_THRESHOLD = 0.62

# Number of appearance embeddings stored per GID.
MAX_EMBEDDINGS_PER_ID = 30

# How long a lost identity can remain recoverable.
MAX_MISSING_FRAMES = 150

# New identity must be observed several times before
# becoming a permanent GID.
NEW_ID_CONFIRM_FRAMES = 3

# Temporary new-person candidate lifetime.
NEW_ID_MAX_AGE = 10

# Minimum detection size.
MIN_WIDTH = 30
MIN_HEIGHT = 60

# Appearance prototype update speed.
EMA_ALPHA = 0.10


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

    crop = cv2.resize(
        crop,
        (128, 256)
    )

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
# TEMPORARY NEW PERSON CANDIDATES
# ============================================================

new_person_candidates = []


# ============================================================
# CREATE NEW GLOBAL ID
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

    clean_embedding = (
        embedding.detach().clone()
    )

    identity_gallery[global_id] = {

        "embeddings": [
            clean_embedding
        ],

        "prototype": clean_embedding,

        "last_seen_frame": frame_number,

        "last_track_id": track_id,

        "last_center": center
    }

    return global_id


# ============================================================
# UPDATE EXISTING GLOBAL ID
# ============================================================

def update_identity(
    global_id,
    embedding,
    frame_number,
    track_id,
    center
):

    data = identity_gallery[global_id]

    clean_embedding = (
        embedding.detach().clone()
    )

    # --------------------------------------------------------
    # Keep recent appearance embeddings.
    # --------------------------------------------------------

    data["embeddings"].append(
        clean_embedding
    )

    if (
        len(data["embeddings"])
        > MAX_EMBEDDINGS_PER_ID
    ):

        data["embeddings"].pop(0)

    # --------------------------------------------------------
    # Update stable appearance prototype.
    # --------------------------------------------------------

    old_prototype = data["prototype"]

    new_prototype = (
        (1.0 - EMA_ALPHA) * old_prototype
        +
        EMA_ALPHA * clean_embedding
    )

    new_prototype = (
        torch.nn.functional.normalize(
            new_prototype,
            p=2,
            dim=1
        )
    )

    data["prototype"] = (
        new_prototype.detach()
    )

    # --------------------------------------------------------
    # Update tracking information.
    # --------------------------------------------------------

    data["last_seen_frame"] = (
        frame_number
    )

    data["last_track_id"] = (
        track_id
    )

    data["last_center"] = (
        center
    )


# ============================================================
# APPEARANCE SCORE
# ============================================================

def appearance_score(
    embedding,
    data
):

    scores = []

    # Stable prototype.
    scores.append(
        cosine_similarity(
            embedding,
            data["prototype"]
        )
    )

    # Recent embeddings.
    for stored_embedding in data["embeddings"]:

        scores.append(
            cosine_similarity(
                embedding,
                stored_embedding
            )
        )

    return max(scores)


# ============================================================
# SOFT SPATIAL CONSISTENCY
#
# This is NOT a hard movement gate.
# ============================================================

def spatial_consistency(
    data,
    center,
    frame_number
):

    last_frame = (
        data["last_seen_frame"]
    )

    last_x, last_y = (
        data["last_center"]
    )

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

    movement_per_frame = (
        distance / frame_gap
    )

    if movement_per_frame < 5:
        return 0.04

    if movement_per_frame < 12:
        return 0.02

    if movement_per_frame < 25:
        return 0.00

    if movement_per_frame < 45:
        return -0.03

    return -0.06


# ============================================================
# CALCULATE MATCH
# ============================================================

def calculate_match(
    embedding,
    detection,
    global_id,
    frame_number
):

    data = identity_gallery[global_id]

    track_id = (
        detection["track_id"]
    )

    center = (
        detection["center"]
    )

    frames_missing = (
        frame_number
        - data["last_seen_frame"]
    )

    # Identity is too old to recover.
    if frames_missing > MAX_MISSING_FRAMES:

        return None

    appearance = appearance_score(
        embedding,
        data
    )

    same_track = (
        track_id
        == data["last_track_id"]
    )

    # --------------------------------------------------------
    # Select threshold.
    # --------------------------------------------------------

    if same_track:

        threshold = (
            SAME_TRACK_THRESHOLD
        )

    elif frames_missing > 0:

        threshold = (
            RECOVERY_THRESHOLD
        )

    else:

        threshold = (
            APPEARANCE_THRESHOLD
        )

    if appearance < threshold:

        return None

    # --------------------------------------------------------
    # Combined score.
    # --------------------------------------------------------

    score = appearance

    if same_track:

        score += 0.08

    score += spatial_consistency(
        data,
        center,
        frame_number
    )

    # --------------------------------------------------------
    # Penalty for long disappearance.
    # --------------------------------------------------------

    if frames_missing > 0:

        recovery_penalty = min(
            0.08,
            (
                frames_missing
                / MAX_MISSING_FRAMES
            ) * 0.08
        )

        score -= recovery_penalty

    return {

        "global_id": global_id,

        "score": score,

        "appearance": appearance,

        "same_track": same_track,

        "frames_missing": frames_missing
    }


# ============================================================
# BUILD ALL POSSIBLE MATCHES
# ============================================================

def build_matches(
    detections,
    frame_number
):

    matches = []

    for detection_index, detection in enumerate(
        detections
    ):

        embedding = (
            detection["embedding"]
        )

        for global_id in identity_gallery:

            result = calculate_match(
                embedding,
                detection,
                global_id,
                frame_number
            )

            if result is None:

                continue

            result["detection_index"] = (
                detection_index
            )

            matches.append(
                result
            )

    # Strongest matches first.
    matches.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return matches


# ============================================================
# TEMPORARY NEW PERSON CANDIDATE
# ============================================================

def update_new_person_candidates(
    detection,
    frame_number
):

    embedding = (
        detection["embedding"]
    )

    center = (
        detection["center"]
    )

    track_id = (
        detection["track_id"]
    )

    best_candidate = None

    best_candidate_index = -1

    best_score = -1.0

    # --------------------------------------------------------
    # Search existing temporary candidates.
    #
    # IMPORTANT:
    # We store the candidate INDEX.
    # We do NOT use list.remove() because the dictionary
    # contains PyTorch tensors.
    # --------------------------------------------------------

    for candidate_index, candidate in enumerate(
        new_person_candidates
    ):

        age = (
            frame_number
            - candidate["last_frame"]
        )

        if age > NEW_ID_MAX_AGE:

            continue

        similarity = cosine_similarity(
            embedding,
            candidate["prototype"]
        )

        if similarity > best_score:

            best_score = similarity

            best_candidate = candidate

            best_candidate_index = (
                candidate_index
            )

    # --------------------------------------------------------
    # Continue an existing candidate.
    # --------------------------------------------------------

    if (
        best_candidate is not None
        and best_score >= 0.72
    ):

        best_candidate["count"] += 1

        best_candidate["last_frame"] = (
            frame_number
        )

        best_candidate["last_track_id"] = (
            track_id
        )

        best_candidate["last_center"] = (
            center
        )

        # ----------------------------------------------------
        # Update candidate prototype.
        # ----------------------------------------------------

        prototype = (
            best_candidate["prototype"]
        )

        updated = (
            (1.0 - EMA_ALPHA) * prototype
            +
            EMA_ALPHA * embedding.detach()
        )

        updated = (
            torch.nn.functional.normalize(
                updated,
                p=2,
                dim=1
            )
        )

        best_candidate["prototype"] = (
            updated.detach()
        )

        # ----------------------------------------------------
        # Confirm candidate.
        # ----------------------------------------------------

        if (
            best_candidate["count"]
            >= NEW_ID_CONFIRM_FRAMES
        ):

            global_id = create_identity(
                embedding,
                frame_number,
                track_id,
                center
            )

            # Delete by known INDEX.
            del new_person_candidates[
                best_candidate_index
            ]

            return global_id

        return None

    # --------------------------------------------------------
    # No existing candidate matched.
    # Create a temporary candidate.
    # --------------------------------------------------------

    new_person_candidates.append({

        "prototype": (
            embedding.detach().clone()
        ),

        "count": 1,

        "last_frame": frame_number,

        "last_track_id": track_id,

        "last_center": center
    })

    return None


# ============================================================
# CLEAN OLD TEMPORARY CANDIDATES
# ============================================================

def cleanup_new_person_candidates(
    frame_number
):

    alive = []

    for candidate in new_person_candidates:

        age = (
            frame_number
            - candidate["last_frame"]
        )

        if age <= NEW_ID_MAX_AGE:

            alive.append(
                candidate
            )

    new_person_candidates[:] = alive


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
            # Clamp coordinates.
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
            # Ignore very small detections.
            # ------------------------------------------------

            if (
                x2 - x1 < MIN_WIDTH
                or
                y2 - y1 < MIN_HEIGHT
            ):

                continue


            # ------------------------------------------------
            # Center.
            # ------------------------------------------------

            center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0
            )


            # ------------------------------------------------
            # Crop.
            # ------------------------------------------------

            crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # OSNet.
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

    matches = build_matches(
        detections,
        frame_number
    )

    assignments = {}

    used_detections = set()

    used_gids = set()


    # --------------------------------------------------------
    # Strongest valid matches first.
    # --------------------------------------------------------

    for match in matches:

        detection_index = (
            match["detection_index"]
        )

        global_id = (
            match["global_id"]
        )

        if detection_index in used_detections:

            continue

        if global_id in used_gids:

            continue


        assignments[detection_index] = {

            "global_id": global_id,

            "score": match["score"]
        }

        used_detections.add(
            detection_index
        )

        used_gids.add(
            global_id
        )


    # ========================================================
    # HANDLE UNMATCHED DETECTIONS
    # ========================================================

    for detection_index, detection in enumerate(
        detections
    ):

        if detection_index in assignments:

            continue


        # ----------------------------------------------------
        # Do not immediately create a permanent GID.
        # ----------------------------------------------------

        confirmed_gid = (
            update_new_person_candidates(
                detection,
                frame_number
            )
        )


        if confirmed_gid is not None:

            # New identity confirmed.
            assignments[detection_index] = {

                "global_id": confirmed_gid,

                "score": None
            }

            used_gids.add(
                confirmed_gid
            )

        else:

            # Still unconfirmed.
            assignments[detection_index] = {

                "global_id": None,

                "score": None
            }


    # ========================================================
    # CLEAN OLD TEMPORARY CANDIDATES
    # ========================================================

    cleanup_new_person_candidates(
        frame_number
    )


    # ========================================================
    # DRAW
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
        # CONFIRMED IDENTITY
        # ----------------------------------------------------

        if global_id is not None:

            update_identity(
                global_id,
                detection["embedding"],
                frame_number,
                detection["track_id"],
                detection["center"]
            )

            color = (
                0,
                255,
                0
            )

            label = (
                f"GID:{global_id}"
            )


        # ----------------------------------------------------
        # UNCONFIRMED PERSON
        # ----------------------------------------------------

        else:

            color = (
                255,
                180,
                0
            )

            label = "NEW?"


        # ----------------------------------------------------
        # Bounding box.
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ----------------------------------------------------
        # Label.
        # ----------------------------------------------------

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
            0.60,
            color,
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
            f"Confirmed GIDs: "
            f"{len(identity_gallery)}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()


print()
print("=" * 60)
print("Global Re-ID C0 v4 completed")
print(
    "Frames processed:",
    frame_number
)
print(
    "Total confirmed GIDs:",
    len(identity_gallery)
)
print(
    "Output:",
    OUTPUT_PATH
)
print("=" * 60)
