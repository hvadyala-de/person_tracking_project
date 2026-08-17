import numpy as np

from src.global_identity_manager import (
    GlobalIdentityManager,
)


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

    return np.load(
        f"output/terrace_embeddings/"
        f"c{camera_id}/"
        f"track_{track_id}.npy"
    )


def main():

    manager = GlobalIdentityManager()

    expected = {}

    # Create competing identities from Camera 0.
    for c0_id, c1_id in VERIFIED_PAIRS:

        gid = manager.create_identity(
            camera_id=0,
            local_track_id=c0_id,
            start_frame=0,
            end_frame=0,
            embedding=load_embedding(
                0,
                c0_id,
            ),
        )

        expected[c1_id] = gid


    print()
    print("GID COMPETITION TEST")
    print("====================")
    print()


    for c0_id, c1_id in VERIFIED_PAIRS:

        result = manager.evaluate(
            camera_id=1,
            embedding=load_embedding(
                1,
                c1_id,
            ),
        )

        expected_gid = expected[
            c1_id
        ]

        best_text = (
            f"GID_{result.global_id:04d}"
            if result.global_id is not None
            else "None"
        )

        second_text = (
            f"GID_{result.second_global_id:04d}"
            if result.second_global_id is not None
            else "None"
        )

        second_score_text = (
            f"{result.second_topk_similarity:.4f}"
            if result.second_topk_similarity is not None
            else "None"
        )

        margin_text = (
            f"{result.score_margin:.4f}"
            if result.score_margin is not None
            else "None"
        )

        correct = (
            result.global_id
            == expected_gid
        )

        print(
            f"c1:{c1_id:<4} | "
            f"best={best_text} "
            f"{result.topk_similarity:.4f} | "
            f"second={second_text} "
            f"{second_score_text} | "
            f"margin={margin_text} | "
            f"{result.status:<7} | "
            f"{'OK' if correct else 'WRONG'}"
        )


if __name__ == "__main__":
    main()
