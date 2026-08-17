import cv2
import numpy as np

from reid_extractor import ReIDExtractor


# ============================================================
# CHANGE ONLY THESE THREE PATHS
# ============================================================

SAME_PERSON_A = (
    "output/terrace_crops/c0/track_634/frame_003165.jpg"
)

SAME_PERSON_B = (
    "output/terrace_crops/c0/track_634/frame_003172.jpg"
)

DIFFERENT_PERSON = (
    "output/terrace_crops/c0/track_538/frame_003169.jpg"
)

def cosine_similarity(a, b):
    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return image


def main():

    extractor = ReIDExtractor()

    image_a = load_image(
        SAME_PERSON_A
    )

    image_b = load_image(
        SAME_PERSON_B
    )

    image_c = load_image(
        DIFFERENT_PERSON
    )


    feature_a = extractor.extract(
        image_a
    )

    feature_b = extractor.extract(
        image_b
    )

    feature_c = extractor.extract(
        image_c
    )


    same_similarity = cosine_similarity(
        feature_a,
        feature_b,
    )

    different_similarity = cosine_similarity(
        feature_a,
        feature_c,
    )


    print()
    print("REID SANITY TEST")
    print("================")
    print()

    print(
        f"Same track similarity:      "
        f"{same_similarity:.4f}"
    )

    print(
        f"Different track similarity: "
        f"{different_similarity:.4f}"
    )

    print()

    print(
        f"Separation: "
        f"{same_similarity - different_similarity:.4f}"
    )


if __name__ == "__main__":
    main()
