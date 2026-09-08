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
# REFERENCE SAME-PERSON / CANDIDATE GROUPS
# ============================================================

REFERENCE_GROUPS = {

    "A_112": [
        (0, 112),
        (1, 41),
    ],

    "B_651": [
        (0, 651),
        (1, 560),
        (1, 423),
    ],

    # c1:503 was REMOVED from this group.
    #
    # Geometry:
    # c0:399 <-> c1:382
    # median distance ~= 16
    #
    # c0:399 <-> c1:503
    # median distance ~= 347
    #
    # Therefore c1:503 is physically incompatible
    # with c0:399 during synchronized observations.
    "C_399": [
        (0, 399),
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

    # --------------------------------------------------------
    # Geometry-supported cross-camera bridge candidate.
    #
    # c0:469 <-> c1:503
    # median ~= 15.87
    #
    # c0:600 <-> c1:503
    # median ~= 15.69
    #
    # c0:469 and c0:600 do not overlap in time.
    # --------------------------------------------------------

    "J_503": [
        (0, 469),
        (0, 600),
        (1, 503),
    ],
}


# ============================================================
# SUSPICIOUS GIDS FROM FIRST OFFLINE RUN
# ============================================================

SUSPICIOUS_GIDS = [
    1,
    2,
    21,
    31,
    43,
    44,
]


# ============================================================
# LOAD EMBEDDING
# ============================================================

def load_embedding(
    camera_id,
    track_id,
):

    path = (
        EMBEDDING_DIR
        / f"c{camera_id}"
        / f"track_{track_id}.npy"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Embedding not found: {path}"
        )

    embedding = np.load(
        path
    ).astype(
        np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        raise ValueError(
            f"Zero-norm embedding: {path}"
        )

    return (
        embedding
        / norm
    )


# ============================================================
# LOAD EMBEDDINGS FOR GROUP
# ============================================================

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


# ============================================================
# INTERNAL GROUP COHERENCE
# ============================================================

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


    # --------------------------------------------------------
    # All pairwise cosine similarities inside the group.
    # --------------------------------------------------------

    for i in range(n):

        for j in range(
            i + 1,
            n,
        ):

            similarity = float(
                embeddings[i]
                @ embeddings[j]
            )

            pair_scores.append(
                similarity
            )


    # --------------------------------------------------------
    # Group centroid
    # --------------------------------------------------------

    centroid = np.mean(
        embeddings,
        axis=0,
    )

    centroid_norm = np.linalg.norm(
        centroid
    )

    if centroid_norm == 0:

        raise ValueError(
            "Group centroid has zero norm"
        )

    centroid = (
        centroid
        / centroid_norm
    )


    centroid_scores = (
        embeddings
        @ centroid
    )


    # --------------------------------------------------------
    # Pairwise statistics
    # --------------------------------------------------------

    if pair_scores:

        pair_min = float(
            np.min(
                pair_scores
            )
        )

        pair_mean = float(
            np.mean(
                pair_scores
            )
        )

        pair_median = float(
            np.median(
                pair_scores
            )
        )

    else:

        # One-member identity.
        pair_min = 1.0
        pair_mean = 1.0
        pair_median = 1.0


    return {
        "members":
            n,

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


# ============================================================
# COMPARE TWO DIFFERENT GROUPS
# ============================================================

def cross_group_statistics(
    members_a,
    members_b,
):

    embeddings_a = group_embeddings(
        members_a
    )

    embeddings_b = group_embeddings(
        members_b
    )


    scores = (
        embeddings_a
        @ embeddings_b.T
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
                    scores
                )
            ),
    }


# ============================================================
# LOAD FINAL GIDS FROM OFFLINE EXPERIMENT
# ============================================================

def load_final_gids():

    gids = {}


    if not RESULT_CSV.exists():

        raise FileNotFoundError(
            f"Offline GID CSV not found: "
            f"{RESULT_CSV}"
        )


    with RESULT_CSV.open(
        "r"
    ) as file:

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


# ============================================================
# PRINT COHERENCE STATS
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("REFERENCE GROUP COHERENCE")
    print("=========================")


    for name, members in (
        REFERENCE_GROUPS.items()
    ):

        stats = coherence_statistics(
            members
        )

        print_stats(
            name,
            stats,
        )


    # ========================================================
    # KNOWN / SUSPECTED DIFFERENT GROUPS
    # ========================================================

    print()
    print("DIFFERENT-GROUP APPEARANCE COMPARISON")
    print("=====================================")


    comparisons = [

        # Look-alike groups that caused the first
        # offline GID contamination.
        (
            "D_701",
            "F_492",
        ),

        (
            "C_399",
            "I_255",
        ),

        # Important corrected relationship:
        #
        # c1:503 was previously associated with C_399,
        # but geometry strongly contradicts that.
        (
            "C_399",
            "J_503",
        ),

        (
            "D_701",
            "E_707",
        ),
    ]


    for name_a, name_b in comparisons:

        stats = cross_group_statistics(
            REFERENCE_GROUPS[
                name_a
            ],
            REFERENCE_GROUPS[
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


    # ========================================================
    # COHERENCE OF BAD/OFFLINE GIDS
    # ========================================================

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
