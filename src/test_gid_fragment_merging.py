import numpy as np

from src.global_identity_manager import GlobalIdentityManager


BASE_PAIRS = [
    (112, 41),
    (651, 560),
    (399, 503),
    (701, 639),
    (707, 536),
    (492, 511),
    (209, 227),
    (1, 1),
]


# Additional verified fragments.
#
# c1:382 should belong to the GID seeded by c0:399
# c1:66  should belong to the GID seeded by c0:1

EXTRA_FRAGMENTS = [
    (382, 399),
    (66, 1),
]


def load_embedding(
    camera_id,
    track_id,
):

    return np.load(
        f"output/terrace_embeddings/"
        f"c{camera_id}/"
        f"track_{track_id}.npy"
    )


def main():

    manager = GlobalIdentityManager()

    gid_for_c0 = {}

    print()
    print("CREATING GIDS")
    print("=============")

    # ----------------------------------------------------
    # First create 8 competing GIDs using Camera 0.
    # ----------------------------------------------------

    for c0_id, c1_id in BASE_PAIRS:

        embedding = load_embedding(
            0,
            c0_id,
        )

        gid = manager.create_identity(
            camera_id=0,
            local_track_id=c0_id,
            start_frame=0,
            end_frame=0,
            embedding=embedding,
        )

        gid_for_c0[c0_id] = gid

        print(
            f"c0:{c0_id:<4} "
            f"-> GID_{gid:04d}"
        )


    # ----------------------------------------------------
    # Add one VERIFIED Camera 1 member to every GID.
    #
    # Now each identity has:
    #
    # c0 embedding
    # +
    # c1 embedding
    # ----------------------------------------------------

    print()
    print("ADDING VERIFIED C1 MEMBERS")
    print("==========================")

    for c0_id, c1_id in BASE_PAIRS:

        gid = gid_for_c0[
            c0_id
        ]

        embedding = load_embedding(
            1,
            c1_id,
        )

        manager.add_to_identity(
            global_id=gid,
            camera_id=1,
            local_track_id=c1_id,
            start_frame=0,
            end_frame=0,
            embedding=embedding,
        )

        print(
            f"c1:{c1_id:<4} "
            f"-> GID_{gid:04d}"
        )


    # ----------------------------------------------------
    # Now test additional fragmented C1 tracks.
    # ----------------------------------------------------

    print()
    print("TESTING EXTRA FRAGMENTS")
    print("=======================")

    correct = 0

    for c1_id, expected_c0_id in EXTRA_FRAGMENTS:

        expected_gid = gid_for_c0[
            expected_c0_id
        ]

        embedding = load_embedding(
            1,
            c1_id,
        )

        result = manager.evaluate(
            camera_id=1,
            embedding=embedding,
        )

        predicted = (
            f"GID_{result.global_id:04d}"
            if result.global_id is not None
            else "None"
        )

        expected = (
            f"GID_{expected_gid:04d}"
        )

        is_correct = (
            result.global_id
            == expected_gid
        )

        if is_correct:
            correct += 1

        mark = (
            "OK"
            if is_correct
            else "WRONG"
        )

        print()
        print(
            f"Query: c1:{c1_id}"
        )

        print(
            f"Expected: {expected}"
        )

        print(
            f"Predicted: {predicted}"
        )

        print(
            f"Status: {result.status}"
        )

        print(
            f"Max: {result.max_similarity:.4f}"
        )

        print(
            f"Top-k: {result.topk_similarity:.4f}"
        )

        print(
            f"Mean: {result.mean_similarity:.4f}"
        )

        print(
            f"Result: {mark}"
        )


    print()
    print("FINAL RESULT")
    print("============")

    print(
        f"Correct: "
        f"{correct}/{len(EXTRA_FRAGMENTS)}"
    )


if __name__ == "__main__":
    main()

