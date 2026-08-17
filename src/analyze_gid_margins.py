from pathlib import Path

import numpy as np

from src.global_identity_manager import (
    GlobalIdentityManager,
)


# ------------------------------------------------------------
# Camera-0 seeds for our currently verified GIDs
# ------------------------------------------------------------

C0_GID_SEEDS = [
    112,
    651,
    399,
    701,
    707,
    492,
    209,
    1,
]


EMBEDDING_DIR = Path(
    "output/terrace_embeddings/c1"
)


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

    # --------------------------------------------------------
    # Build our 8 known GIDs using Camera 0.
    # --------------------------------------------------------

    print()
    print("CREATING KNOWN GIDS")
    print("===================")

    for c0_id in C0_GID_SEEDS:

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

        print(
            f"GID_{gid:04d} <- c0:{c0_id}"
        )


    # --------------------------------------------------------
    # Evaluate every available Camera-1 embedding.
    # --------------------------------------------------------

    rows = []

    files = sorted(
        EMBEDDING_DIR.glob(
            "track_*.npy"
        )
    )

    for path in files:

        c1_id = int(
            path.stem.split("_")[1]
        )

        embedding = np.load(
            path
        )

        # Raw ranking gives us best + second-best
        # even if evaluate() later classifies it as NEW.
        best = manager.find_best_identity(
            embedding
        )

        if best is None:
            continue

        result = manager.evaluate(
            camera_id=1,
            embedding=embedding,
        )

        # We only care about appearance-plausible
        # candidates. Ignore very weak matches.
        if (
            best["topk"]
            < manager.cross_camera_pending
        ):
            continue

        rows.append(
            {
                "c1_id": c1_id,

                "best_gid": best[
                    "global_id"
                ],

                "best_score": best[
                    "topk"
                ],

                "second_gid": best[
                    "second_global_id"
                ],

                "second_score": best[
                    "second_topk"
                ],

                "margin": best[
                    "margin"
                ],

                "status": result.status,
            }
        )


    # --------------------------------------------------------
    # Smallest margin = hardest competition.
    # --------------------------------------------------------

    rows.sort(
        key=lambda row: row[
            "margin"
        ]
    )


    print()
    print("HARDEST GID COMPETITION CASES")
    print("=============================")

    print()
    print(
        f"Camera-1 embeddings: "
        f"{len(files)}"
    )

    print(
        f"Candidates with best score >= "
        f"{manager.cross_camera_pending:.2f}: "
        f"{len(rows)}"
    )

    print()

    print(
        f"{'C1 ID':>7} "
        f"{'Best GID':>9} "
        f"{'Best':>7} "
        f"{'2nd GID':>9} "
        f"{'Second':>7} "
        f"{'Margin':>8} "
        f"{'Status':>8}"
    )

    print("-" * 70)


    for row in rows[:30]:

        print(
            f"{row['c1_id']:>7} "
            f"{row['best_gid']:>9} "
            f"{row['best_score']:>7.4f} "
            f"{row['second_gid']:>9} "
            f"{row['second_score']:>7.4f} "
            f"{row['margin']:>8.4f} "
            f"{row['status']:>8}"
        )


if __name__ == "__main__":
    main()
