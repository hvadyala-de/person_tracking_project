import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
)

RESULT_CSV = (
    PROJECT_ROOT
    / "output"
    / "global_id_analysis_c0_c1.csv"
)


# ============================================================
# MANUALLY VERIFIED SAME-PERSON GROUPS
# ============================================================

VERIFIED_GROUPS = {
    "A_112": [
        (0, 112),
        (1, 41),
    ],

    "B_651": [
        (0, 651),
        (1, 560),
        (1, 423),
    ],

    "C_399": [
        (0, 399),
        (1, 503),
        (1, 382),
        (1, 319),
    ],

    "D_701": [
        (0, 701),
        (0, 740),
        (1, 639),
    ],

    "E_707": [
        (0, 707),
        (0, 746),
        (1, 536),
    ],

    "F_492": [
        (0, 492),
        (1, 511),
        (1, 721),
    ],

    "G_209": [
        (0, 209),
        (1, 227),
    ],

    "H_1": [
        (0, 1),
        (1, 1),
        (1, 66),
    ],

    "I_255": [
        (0, 255),
        (1, 298),
    ],
}


# Suspicious final GIDs from the offline run.
SUSPICIOUS_GIDS = [
    1,
    2,
    21,
    31,
    43,
    44,
]


def load_embedding(
    camera_id,
    track_id,
):

    path = (
        EMBEDDING_DIR
        / f"c{camera_id}"
        / f"track_{track_id}.npy"
    )

    embedding = np.load(path).astype(
        np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    return embedding / norm


def group_embeddings(
    members,
):

    embeddings = []

    for camera_id, track_id in members:

        embeddings.append(
            load_embedding(
                camera_id,
                track_id,
            )
        )

    return np.stack(
        embeddings,
        axis=0,
    )


def coherence_statistics(
    members,
):

    embeddings = group_embeddings(
        members
    )

    n = len(
        embeddings
    )

    pair_scores = []

    for i in range(n):

        for j in range(i + 1, n):

            pair_scores.append(
                float(
                    embeddings[i]
                    @ embeddings[j]
                )
            )

    centroid = np.mean(
        embeddings,
        axis=0,
    )

    centroid /= np.linalg.norm(
        centroid
    )

    centroid_scores = (
        embeddings
        @ centroid
    )

    if pair_scores:

        pair_min = min(
            pair_scores
        )

        pair_mean = float(
            np.mean(pair_scores)
        )

        pair_median = float(
            np.median(pair_scores)
        )

    else:

        pair_min = 1.0
        pair_mean = 1.0
        pair_median = 1.0

    return {
        "members": n,

        "pair_min":
            pair_min,

        "pair_mean":
            pair_mean,

        "pair_median":
            pair_median,

        "centroid_min":
            float(
                np.min(
                    centroid_scores
                )
            ),

        "centroid_mean":
            float(
                np.mean(
                    centroid_scores
                )
            ),
    }


def cross_group_statistics(
    members_a,
    members_b,
):

    a = group_embeddings(
        members_a
    )

    b = group_embeddings(
        members_b
    )

    scores = (
        a
        @ b.T
    ).reshape(-1)

    sorted_scores = np.sort(
        scores
    )[::-1]

    top_count = min(
        3,
        len(sorted_scores),
    )

    return {
        "max":
            float(
                sorted_scores[0]
            ),

        "top3":
            float(
                np.mean(
                    sorted_scores[
                        :top_count
                    ]
                )
            ),

        "mean":
            float(
                np.mean(
                    sorted_scores
                )
            ),
    }


def load_final_gids():

    gids = {}

    with RESULT_CSV.open() as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            gid_text = row[
                "global_id"
            ]

            if not gid_text:
                continue

            gid = int(
                gid_text
            )

            camera_id = int(
                row[
                    "camera_id"
                ]
            )

            track_id = int(
                row[
                    "local_track_id"
                ]
            )

            gids.setdefault(
                gid,
                [],
            ).append(
                (
                    camera_id,
                    track_id,
                )
            )

    return gids


def print_stats(
    name,
    stats,
):

    print(
        f"{name:<12} | "
        f"n={stats['members']:<2} | "
        f"pair_min={stats['pair_min']:.4f} | "
        f"pair_mean={stats['pair_mean']:.4f} | "
        f"pair_med={stats['pair_median']:.4f} | "
        f"cent_min={stats['centroid_min']:.4f} | "
        f"cent_mean={stats['centroid_mean']:.4f}"
    )


def main():

    print()
    print("VERIFIED SAME-PERSON COHERENCE")
    print("==============================")

    for name, members in (
        VERIFIED_GROUPS.items()
    ):

        stats = coherence_statistics(
            members
        )

        print_stats(
            name,
            stats,
        )


    print()
    print("KNOWN DIFFERENT-GROUP COMPARISON")
    print("================================")

    comparisons = [
        (
            "D_701",
            "F_492",
        ),
        (
            "C_399",
            "I_255",
        ),
        (
            "D_701",
            "E_707",
        ),
    ]

    for name_a, name_b in comparisons:

        stats = cross_group_statistics(
            VERIFIED_GROUPS[
                name_a
            ],
            VERIFIED_GROUPS[
                name_b
            ],
        )

        print(
            f"{name_a:<8} vs "
            f"{name_b:<8} | "
            f"max={stats['max']:.4f} | "
            f"top3={stats['top3']:.4f} | "
            f"mean={stats['mean']:.4f}"
        )


    print()
    print("SUSPICIOUS FINAL GID COHERENCE")
    print("==============================")

    final_gids = load_final_gids()

    for gid in SUSPICIOUS_GIDS:

        members = final_gids.get(
            gid
        )

        if not members:

            print(
                f"GID_{gid:04d}: missing"
            )

            continue

        stats = coherence_statistics(
            members
        )

        print_stats(
            f"GID_{gid:04d}",
            stats,
        )


if __name__ == "__main__":
    main()
