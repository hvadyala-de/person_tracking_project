from pathlib import Path
import numpy as np

from build_tracklets import load_tracklets


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

MAX_GAP_FRAMES = 250


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def temporal_gap(track_a, track_b):

    if track_a.end_frame < track_b.start_frame:
        return track_b.start_frame - track_a.end_frame

    if track_b.end_frame < track_a.start_frame:
        return track_a.start_frame - track_b.end_frame

    return -1


def main():

    tracklets = load_tracklets(TRACK_CSV)

    useful = {
        track.local_track_id: track
        for track in tracklets.values()
        if track.num_detections >= 25
    }

    embeddings = {}

    for path in EMBEDDING_DIR.glob("track_*.npy"):

        track_id = int(
            path.stem.replace("track_", "")
        )

        embeddings[track_id] = np.load(path)

    candidates = []

    track_ids = sorted(embeddings.keys())

    for i in range(len(track_ids)):

        id_a = track_ids[i]

        if id_a not in useful:
            continue

        for j in range(i + 1, len(track_ids)):

            id_b = track_ids[j]

            if id_b not in useful:
                continue

            track_a = useful[id_a]
            track_b = useful[id_b]

            gap = temporal_gap(
                track_a,
                track_b,
            )

            # -1 means both tracks overlap in time.
            # They cannot be the same person on the same camera.
            if gap < 0:
                continue

            # For this first test we only inspect nearby fragments.
            if gap > MAX_GAP_FRAMES:
                continue

            similarity = cosine_similarity(
                embeddings[id_a],
                embeddings[id_b],
            )

            candidates.append(
                (
                    similarity,
                    gap,
                    id_a,
                    id_b,
                    track_a.end_frame,
                    track_b.start_frame,
                )
            )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    print()
    print("TOP SAME-CAMERA REID CANDIDATES")
    print("===============================")
    print()

    print(
        f"{'Track A':>8} "
        f"{'Track B':>8} "
        f"{'Similarity':>11} "
        f"{'Gap':>7}"
    )

    print("-" * 42)

    for (
        similarity,
        gap,
        id_a,
        id_b,
        _,
        _,
    ) in candidates[:30]:

        print(
            f"{id_a:>8} "
            f"{id_b:>8} "
            f"{similarity:>11.4f} "
            f"{gap:>7}"
        )


if __name__ == "__main__":
    main()
