from pathlib import Path

import numpy as np

from src.global_identity_manager import (
    GlobalIdentityManager,
)


# ============================================================
# VERIFIED GLOBAL IDENTITIES
#
# Each dictionary becomes one GID.
# ============================================================

KNOWN_IDENTITIES = [
    {
        "c0": [112],
        "c1": [41],
    },
    {
        "c0": [651],
        "c1": [560, 423],
    },
    {
        "c0": [399],
        "c1": [503, 382, 319],
    },
    {
        "c0": [701, 740],
        "c1": [639],
    },
    {
        "c0": [707, 746],
        "c1": [536],
    },
    {
        "c0": [492],
        "c1": [511],
    },
    {
        "c0": [209],
        "c1": [227],
    },
    {
        "c0": [1],
        "c1": [1, 66],
    },
]


EMBEDDING_DIR = Path(
    "output/terrace_embeddings/c1"
)


# We still inspect appearance candidates down to 0.84.
#
# This is an ANALYSIS threshold only.
# It is NOT an automatic matching threshold.

ANALYSIS_MIN_SCORE = 0.84


def load_embedding(
    camera_id,
    track_id,
):

    return np.load(
        f"output/terrace_embeddings/"
        f"c{camera_id}/"
        f"track_{track_id}.npy"
    )


def add_member(
    manager,
    gid,
    camera_id,
    track_id,
):

    manager.add_to_identity(
        global_id=gid,
        camera_id=camera_id,
        local_track_id=track_id,
        start_frame=0,
        end_frame=0,
        embedding=load_embedding(
            camera_id,
            track_id,
        ),
    )


def main():

    manager = GlobalIdentityManager()

    known_c1_tracks = set()

    print()
    print("BUILDING VERIFIED GID GALLERY")
    print("=============================")

    # ========================================================
    # Build each verified GID.
    # ========================================================

    for identity_data in KNOWN_IDENTITIES:

        c0_tracks = identity_data[
            "c0"
        ]

        c1_tracks = identity_data[
            "c1"
        ]

        # First C0 fragment creates the GID.
        first_c0 = c0_tracks[0]

        gid = manager.create_identity(
            camera_id=0,
            local_track_id=first_c0,
            start_frame=0,
            end_frame=0,
            embedding=load_embedding(
                0,
                first_c0,
            ),
        )

        # Add remaining C0 fragments.
        for track_id in c0_tracks[1:]:

            add_member(
                manager,
                gid,
                0,
                track_id,
            )

        # Add verified C1 fragments.
        for track_id in c1_tracks:

            add_member(
                manager,
                gid,
                1,
                track_id,
            )

            known_c1_tracks.add(
                track_id
            )

        print(
            manager.identities[
                gid
            ].summary()
        )


    # ========================================================
    # Evaluate unknown C1 tracklets.
    # ========================================================

    rows = []

    files = sorted(
        EMBEDDING_DIR.glob(
            "track_*.npy"
        )
    )

    skipped_known = 0

    for path in files:

        c1_id = int(
            path.stem.split("_")[1]
        )

        # Do not test a track that is already
        # part of our verified gallery.
        if c1_id in known_c1_tracks:

            skipped_known += 1
            continue

        embedding = np.load(
            path
        )

        best = manager.find_best_identity(
            embedding
        )

        if best is None:
            continue

        # Ignore obviously weak appearance candidates.
        if (
            best["topk"]
            < ANALYSIS_MIN_SCORE
        ):
            continue

        result = manager.evaluate(
            camera_id=1,
            embedding=embedding,
        )

        rows.append(
            {
                "c1_id": c1_id,

                "best_gid": best[
                    "global_id"
                ],

                "best_score": best[
                    "topk"
                ],

                "best_max": best[
                    "max"
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


    # ========================================================
    # Hardest competition first.
    # ========================================================

    rows.sort(
        key=lambda row: row[
            "margin"
        ]
    )


    print()
    print("HARD CASES WITH RICHER GALLERY")
    print("==============================")

    print()
    print(
        f"Total Camera-1 embeddings: "
        f"{len(files)}"
    )

    print(
        f"Known C1 tracks skipped: "
        f"{skipped_known}"
    )

    print(
        f"Unknown candidates >= "
        f"{ANALYSIS_MIN_SCORE:.2f}: "
        f"{len(rows)}"
    )

    print()

    print(
        f"{'C1 ID':>7} "
        f"{'Best GID':>9} "
        f"{'TopK':>7} "
        f"{'Max':>7} "
        f"{'2nd GID':>9} "
        f"{'Second':>7} "
        f"{'Margin':>8} "
        f"{'Status':>8}"
    )

    print("-" * 80)


    for row in rows[:30]:

        print(
            f"{row['c1_id']:>7} "
            f"{row['best_gid']:>9} "
            f"{row['best_score']:>7.4f} "
            f"{row['best_max']:>7.4f} "
            f"{row['second_gid']:>9} "
            f"{row['second_score']:>7.4f} "
            f"{row['margin']:>8.4f} "
            f"{row['status']:>8}"
        )


if __name__ == "__main__":
    main()
