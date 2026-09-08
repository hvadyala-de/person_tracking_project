from pathlib import Path

import numpy as np

from src.build_tracklets import load_tracklets
from src.experiments.analyze_cross_camera_geometry import geometry_stats
from src.terrace_geometry import load_ground_homographies


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
)

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
)


TARGET_C1 = 503
MIN_SHARED_FRAMES = 10


def load_by_id(camera_id):

    tracklets = load_tracklets(
        TRACK_DIR
        / f"tracks_c{camera_id}.csv"
    )

    if isinstance(tracklets, dict):
        values = tracklets.values()
    else:
        values = tracklets

    return {
        t.local_track_id: t
        for t in values
    }


def load_embedding(camera_id, track_id):

    path = (
        EMBEDDING_DIR
        / f"c{camera_id}"
        / f"track_{track_id}.npy"
    )

    if not path.exists():
        return None

    embedding = np.load(
        path
    ).astype(np.float32)

    embedding /= np.linalg.norm(
        embedding
    )

    return embedding


def main():

    c0 = load_by_id(0)
    c1 = load_by_id(1)

    H = load_ground_homographies()

    target = c1[TARGET_C1]

    target_embedding = load_embedding(
        1,
        TARGET_C1,
    )

    rows = []

    for c0_id, tracklet in c0.items():

        embedding = load_embedding(
            0,
            c0_id,
        )

        if embedding is None:
            continue

        stats = geometry_stats(
            tracklet,
            target,
            H,
        )

        if stats is None:
            continue

        if (
            stats["shared"]
            < MIN_SHARED_FRAMES
        ):
            continue

        similarity = float(
            embedding
            @ target_embedding
        )

        rows.append(
            (
                stats["median"],
                -similarity,
                c0_id,
                stats,
                similarity,
            )
        )

    rows.sort()

    print()
    print(
        f"GEOMETRY CANDIDATES FOR c1:{TARGET_C1}"
    )
    print("=" * 56)

    print(
        f"{'C0 ID':>7} "
        f"{'Shared':>7} "
        f"{'Median':>9} "
        f"{'P90':>9} "
        f"{'ReID':>8}"
    )

    print("-" * 48)

    for (
        median,
        neg_similarity,
        c0_id,
        stats,
        similarity,
    ) in rows[:25]:

        print(
            f"{c0_id:>7} "
            f"{stats['shared']:>7} "
            f"{stats['median']:>9.2f} "
            f"{stats['p90']:>9.2f} "
            f"{similarity:>8.4f}"
        )


if __name__ == "__main__":
    main()

