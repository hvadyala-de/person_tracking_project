import numpy as np

from src.global_identity_manager import GlobalIdentityManager


# ------------------------------------------------------------
# VERIFIED CROSS-CAMERA PAIRS
#
# c0 track -> c1 track
# ------------------------------------------------------------

VERIFIED_PAIRS = [
    (112, 41),
    (651, 560),
    (399, 503),
    (701, 639),
    (707, 536),
    (492, 511),
    (209, 227),
    (1, 1),
]


def load_embedding(
    camera_id,
    track_id,
):

    path = (
        f"output/terrace_embeddings/"
        f"c{camera_id}/"
        f"track_{track_id}.npy"
    )

    return np.load(path)


def main():

    manager = GlobalIdentityManager()

    expected = {}

    print()
    print("CREATING GLOBAL IDENTITIES")
    print("==========================")

    # --------------------------------------------------------
    # Create one GID from each Camera 0 tracklet.
    # --------------------------------------------------------

    for c0_id, c1_id in VERIFIED_PAIRS:

        embedding = load_embedding(
            camera_id=0,
            track_id=c0_id,
        )

        gid = manager.create_identity(
            camera_id=0,
            local_track_id=c0_id,
            start_frame=0,
            end_frame=0,
            embedding=embedding,
        )

        expected[c1_id] = gid

        print(
            f"c0:{c0_id:<4} "
            f"-> GID_{gid:04d}"
        )

    print()
    print("TESTING CAMERA 1 TRACKLETS")
    print("==========================")

    correct = 0

    total = len(
        VERIFIED_PAIRS
    )

    # --------------------------------------------------------
    # Evaluate each corresponding Camera 1 tracklet.
    # --------------------------------------------------------

    for c0_id, c1_id in VERIFIED_PAIRS:

        embedding = load_embedding(
            camera_id=1,
            track_id=c1_id,
        )

        result = manager.evaluate(
            camera_id=1,
            embedding=embedding,
        )

        expected_gid = expected[
            c1_id
        ]

        is_correct = (
            result.global_id
            == expected_gid
        )

        if is_correct:
            correct += 1

        predicted_text = (
            f"GID_{result.global_id:04d}"
            if result.global_id is not None
            else "None"
        )

        expected_text = (
            f"GID_{expected_gid:04d}"
        )

        mark = (
            "OK"
            if is_correct
            else "WRONG"
        )

        print(
            f"c1:{c1_id:<4} | "
            f"expected={expected_text} | "
            f"predicted={predicted_text} | "
            f"{result.status:<7} | "
            f"max={result.max_similarity:.4f} | "
            f"topk={result.topk_similarity:.4f} | "
            f"{mark}"
        )

    print()
    print("RESULT")
    print("======")
    print(
        f"Correct GID: "
        f"{correct}/{total}"
    )

    accuracy = (
        correct
        / total
        * 100.0
    )

    print(
        f"Accuracy: "
        f"{accuracy:.1f}%"
    )


if __name__ == "__main__":
    main()
