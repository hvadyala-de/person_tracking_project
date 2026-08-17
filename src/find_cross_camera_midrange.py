from cross_camera_candidates import (
    load_useful_tracklets,
    load_embeddings,
    cosine_similarity,
)


CAM_A = 0
CAM_B = 1

SIM_LOW = 0.84
SIM_HIGH = 0.90

MIN_SHARED_FRAMES = 20


def main():

    tracks_a = load_useful_tracklets(CAM_A)
    tracks_b = load_useful_tracklets(CAM_B)

    embeddings_a = load_embeddings(CAM_A)
    embeddings_b = load_embeddings(CAM_B)

    frame_sets_a = {
        track_id: {
            detection.frame
            for detection in tracklet.detections
        }
        for track_id, tracklet in tracks_a.items()
    }

    frame_sets_b = {
        track_id: {
            detection.frame
            for detection in tracklet.detections
        }
        for track_id, tracklet in tracks_b.items()
    }

    candidates = []

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

            shared_count = len(shared_frames)

            if shared_count < MIN_SHARED_FRAMES:
                continue

            similarity = cosine_similarity(
                embedding_a,
                embedding_b,
            )

            if SIM_LOW <= similarity < SIM_HIGH:

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

    print()
    print("CROSS-CAMERA MID-RANGE CANDIDATES")
    print("=================================")
    print()

    print(
        f"Similarity range: "
        f"{SIM_LOW:.2f} <= sim < {SIM_HIGH:.2f}"
    )

    print(
        f"Minimum shared frames: "
        f"{MIN_SHARED_FRAMES}"
    )

    print(
        f"Candidate pairs: "
        f"{len(candidates)}"
    )

    print()

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
    ) in candidates[:30]:

        print(
            f"{id_a:>8} "
            f"{id_b:>8} "
            f"{similarity:>11.4f} "
            f"{shared:>8}"
        )


if __name__ == "__main__":
    main()
