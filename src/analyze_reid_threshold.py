from pathlib import Path

import numpy as np

from build_tracklets import load_tracklets


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_CSV = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
    / "tracks_c0.csv"
)

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
    / "c0"
)


# ============================================================
# MANUALLY VERIFIED SAME-PERSON PAIRS
# ============================================================

VERIFIED_POSITIVE_PAIRS = [
    (701, 740),
    (56, 149),
    (707, 746),
    (188, 221),
]


# Require both tracks to actually have detections
# at the same time for at least this many frames.
MIN_SHARED_FRAMES_FOR_NEGATIVE = 10


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a, b):

    # Embeddings are already L2-normalized.
    return float(
        np.dot(a, b)
    )


def load_embeddings():

    embeddings = {}

    for path in EMBEDDING_DIR.glob("track_*.npy"):

        track_id = int(
            path.stem.replace(
                "track_",
                ""
            )
        )

        embedding = np.load(
            path
        ).astype(np.float32)

        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:
            embedding = (
                embedding / norm
            )

        embeddings[
            track_id
        ] = embedding

    return embeddings


# ============================================================
# MAIN
# ============================================================

def main():

    tracklets_raw = load_tracklets(
        TRACK_CSV
    )

    tracklets = {
        track.local_track_id: track
        for track in tracklets_raw.values()
        if track.num_detections >= 25
    }

    embeddings = load_embeddings()

    print()
    print("REID THRESHOLD ANALYSIS")
    print("=======================")
    print()

    print(
        f"Useful tracklets: {len(tracklets)}"
    )

    print(
        f"Embeddings:       {len(embeddings)}"
    )

    print()

    # ========================================================
    # VERIFIED POSITIVES
    # ========================================================

    positive_scores = []

    print("VERIFIED POSITIVE PAIRS")
    print("-----------------------")

    for id_a, id_b in VERIFIED_POSITIVE_PAIRS:

        if (
            id_a not in embeddings
            or id_b not in embeddings
        ):
            print(
                f"{id_a}/{id_b}: missing embedding"
            )
            continue

        similarity = cosine_similarity(
            embeddings[id_a],
            embeddings[id_b],
        )

        positive_scores.append(
            similarity
        )

        print(
            f"{id_a:>5} <-> {id_b:<5} "
            f"similarity={similarity:.4f}"
        )

    print()

    # ========================================================
    # STRONG NEGATIVE PAIRS
    # ========================================================

    # Build actual frame sets.
    frame_sets = {}

    for track_id, tracklet in tracklets.items():

        frame_sets[track_id] = {
            detection.frame
            for detection in tracklet.detections
        }

    track_ids = sorted(
        embeddings.keys()
    )

    negative_pairs = []

    for i in range(len(track_ids)):

        id_a = track_ids[i]

        if id_a not in tracklets:
            continue

        for j in range(
            i + 1,
            len(track_ids),
        ):

            id_b = track_ids[j]

            if id_b not in tracklets:
                continue

            shared_frames = (
                frame_sets[id_a]
                & frame_sets[id_b]
            )

            num_shared = len(
                shared_frames
            )

            if (
                num_shared
                < MIN_SHARED_FRAMES_FOR_NEGATIVE
            ):
                continue

            similarity = cosine_similarity(
                embeddings[id_a],
                embeddings[id_b],
            )

            negative_pairs.append(
                (
                    similarity,
                    num_shared,
                    id_a,
                    id_b,
                )
            )

    negative_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    negative_scores = np.array(
        [
            item[0]
            for item in negative_pairs
        ],
        dtype=np.float32,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print("NEGATIVE PAIR STATISTICS")
    print("------------------------")

    print(
        f"Strong negative pairs: "
        f"{len(negative_pairs)}"
    )

    if len(negative_scores) > 0:

        print(
            f"Mean:    "
            f"{negative_scores.mean():.4f}"
        )

        print(
            f"Median:  "
            f"{np.median(negative_scores):.4f}"
        )

        print(
            f"90th %%:  "
            f"{np.percentile(negative_scores, 90):.4f}"
        )

        print(
            f"95th %%:  "
            f"{np.percentile(negative_scores, 95):.4f}"
        )

        print(
            f"99th %%:  "
            f"{np.percentile(negative_scores, 99):.4f}"
        )

        print(
            f"Maximum: "
            f"{negative_scores.max():.4f}"
        )

    print()

    # ========================================================
    # POSITIVE STATISTICS
    # ========================================================

    if positive_scores:

        positive_scores_np = np.array(
            positive_scores,
            dtype=np.float32,
        )

        print("POSITIVE PAIR STATISTICS")
        print("------------------------")

        print(
            f"Minimum verified positive: "
            f"{positive_scores_np.min():.4f}"
        )

        print(
            f"Mean verified positive:    "
            f"{positive_scores_np.mean():.4f}"
        )

        print(
            f"Maximum verified positive: "
            f"{positive_scores_np.max():.4f}"
        )

        print()

    # ========================================================
    # HARDEST NEGATIVES
    # ========================================================

    print("TOP 20 HARDEST NEGATIVES")
    print("------------------------")

    print(
        f"{'Track A':>8} "
        f"{'Track B':>8} "
        f"{'Similarity':>11} "
        f"{'Shared':>8}"
    )

    print("-" * 42)

    for (
        similarity,
        shared,
        id_a,
        id_b,
    ) in negative_pairs[:20]:

        print(
            f"{id_a:>8} "
            f"{id_b:>8} "
            f"{similarity:>11.4f} "
            f"{shared:>8}"
        )

    print()
    print("IMPORTANT")
    print("---------")
    print(
        "Do not choose a final threshold from "
        "these numbers alone."
    )
    print(
        "We will visually inspect the hardest "
        "negative pairs next."
    )


if __name__ == "__main__":
    main()
