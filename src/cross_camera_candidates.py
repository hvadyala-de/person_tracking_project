from pathlib import Path

import numpy as np

from build_tracklets import load_tracklets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CAM_A = 0
CAM_B = 1

MIN_SHARED_FRAMES = 10


# ============================================================
# PATH HELPERS
# ============================================================

def track_csv(camera_id):

    return (
        PROJECT_ROOT
        / "output"
        / "terrace_bytetrack_tuned"
        / f"tracks_c{camera_id}.csv"
    )


def embedding_dir(camera_id):

    return (
        PROJECT_ROOT
        / "output"
        / "terrace_embeddings"
        / f"c{camera_id}"
    )


# ============================================================
# LOAD TRACKLETS
# ============================================================

def load_useful_tracklets(camera_id):

    raw = load_tracklets(
        track_csv(camera_id)
    )

    return {
        track.local_track_id: track
        for track in raw.values()
        if track.num_detections >= 25
    }


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(camera_id):

    embeddings = {}

    directory = embedding_dir(
        camera_id
    )

    for path in directory.glob(
        "track_*.npy"
    ):

        track_id = int(
            path.stem.replace(
                "track_",
                "",
            )
        )

        feature = np.load(
            path
        ).astype(
            np.float32
        )

        norm = np.linalg.norm(
            feature
        )

        if norm > 0:
            feature = (
                feature / norm
            )

        embeddings[
            track_id
        ] = feature

    return embeddings


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("CROSS-CAMERA REID")
    print("=================")
    print()

    tracks_a = load_useful_tracklets(
        CAM_A
    )

    tracks_b = load_useful_tracklets(
        CAM_B
    )

    embeddings_a = load_embeddings(
        CAM_A
    )

    embeddings_b = load_embeddings(
        CAM_B
    )

    print(
        f"Camera c{CAM_A}: "
        f"{len(tracks_a)} tracklets / "
        f"{len(embeddings_a)} embeddings"
    )

    print(
        f"Camera c{CAM_B}: "
        f"{len(tracks_b)} tracklets / "
        f"{len(embeddings_b)} embeddings"
    )

    print()

    # --------------------------------------------------------
    # FRAME SETS
    # --------------------------------------------------------

    frame_sets_a = {
        track_id: {
            detection.frame
            for detection
            in tracklet.detections
        }
        for track_id, tracklet
        in tracks_a.items()
    }

    frame_sets_b = {
        track_id: {
            detection.frame
            for detection
            in tracklet.detections
        }
        for track_id, tracklet
        in tracks_b.items()
    }

    candidates = []

    # --------------------------------------------------------
    # CROSS-CAMERA PAIRING
    # --------------------------------------------------------

    for id_a, embedding_a in embeddings_a.items():

        if id_a not in tracks_a:
            continue

        for id_b, embedding_b in embeddings_b.items():

            if id_b not in tracks_b:
                continue

            shared_frames = (
                frame_sets_a[id_a]
                & frame_sets_b[id_b]
            )

            shared_count = len(
                shared_frames
            )

            if (
                shared_count
                < MIN_SHARED_FRAMES
            ):
                continue

            similarity = cosine_similarity(
                embedding_a,
                embedding_b,
            )

            candidates.append(
                (
                    similarity,
                    shared_count,
                    id_a,
                    id_b,
                )
            )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    print(
        f"Synchronized candidate pairs: "
        f"{len(candidates)}"
    )

    print()
    print("TOP CROSS-CAMERA MATCHES")
    print("------------------------")

    print(
        f"{'C0 ID':>8} "
        f"{'C1 ID':>8} "
        f"{'Similarity':>11} "
        f"{'Shared':>8}"
    )

    print("-" * 42)

    for (
        similarity,
        shared,
        id_a,
        id_b,
    ) in candidates[:40]:

        print(
            f"{id_a:>8} "
            f"{id_b:>8} "
            f"{similarity:>11.4f} "
            f"{shared:>8}"
        )


if __name__ == "__main__":
    main()
