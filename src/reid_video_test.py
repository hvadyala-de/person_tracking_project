import cv2
import torch
import torchreid
import numpy as np

VIDEO_PATH = "data/videos/terrace1-c0.avi"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading OSNet...")

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model = model.to(device)
model.eval()

print("OSNet ready on:", device)


def get_embedding(crop):
    """
    Convert a person crop into a 512-D OSNet appearance embedding.
    """

    if crop is None or crop.size == 0:
        return None

    crop = cv2.resize(crop, (128, 256))

    # BGR -> RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # HWC -> CHW
    crop = np.transpose(crop, (2, 0, 1))

    # uint8 -> float
    crop = crop.astype(np.float32) / 255.0

    tensor = torch.from_numpy(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    # Normalize embedding
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

print("Video opened successfully.")

frame_number = 0
embeddings_found = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Test every 30th frame
    if frame_number % 30 != 0:
        continue

    height, width = frame.shape[:2]

    # Temporary test crop.
    # This is NOT our final detector.
    #
    # We take the central part of the frame simply
    # to verify that OSNet can process a real video crop.

    x1 = int(width * 0.20)
    y1 = int(height * 0.10)
    x2 = int(width * 0.80)
    y2 = int(height * 0.95)

    crop = frame[y1:y2, x1:x2]

    embedding = get_embedding(crop)

    if embedding is not None:

        embeddings_found += 1

        print(
            f"Frame {frame_number}: "
            f"embedding shape = {tuple(embedding.shape)}"
        )

    # Stop after 10 successful tests
    if embeddings_found >= 10:
        break


cap.release()

print()
print("===================================")
print("Real-video Re-ID test completed")
print("Frames tested:", embeddings_found)
print("Device:", device)
print("===================================")
