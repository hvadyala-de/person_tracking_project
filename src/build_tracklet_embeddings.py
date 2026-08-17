from pathlib import Path
import csv

import cv2
import numpy as np

from reid_extractor import ReIDExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CROPS_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_crops"
    / "c0"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
    / "c0"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "tracklet_embeddings_summary.csv"
)


def normalize(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def main():

    if not CROPS_DIR.exists():
        raise FileNotFoundError(
            f"Crops directory not found: {CROPS_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    extractor = ReIDExtractor()

    track_dirs = sorted(
        [
            path
            for path in CROPS_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print()
    print("TRACKLET EMBEDDING BUILD")
    print("========================")
    print()

    print(f"Crop directory: {CROPS_DIR}")
    print(f"Track folders:  {len(track_dirs)}")
    print()

    summary_rows = []

    total_tracks_saved = 0

    for track_dir in track_dirs:

        crop_paths = sorted(
            track_dir.glob("*.jpg")
        )

        if len(crop_paths) == 0:
            continue

        embeddings = []

        for crop_path in crop_paths:

            image = cv2.imread(
                str(crop_path)
            )

            if image is None:
                continue

            feature = extractor.extract(
                image
            )

            if feature is None:
                continue

            embeddings.append(
                feature
            )

        if len(embeddings) == 0:
            continue

        embeddings = np.stack(
            embeddings,
            axis=0,
        )

        mean_embedding = np.mean(
            embeddings,
            axis=0,
        )

        mean_embedding = normalize(
            mean_embedding
        ).astype(np.float32)

        output_path = (
            OUTPUT_DIR
            / f"{track_dir.name}.npy"
        )

        np.save(
            output_path,
            mean_embedding,
        )

        summary_rows.append(
            {
                "track_id": track_dir.name,
                "num_crops": len(crop_paths),
                "feature_dim": mean_embedding.shape[0],
                "output_path": str(output_path),
            }
        )

        total_tracks_saved += 1

        if total_tracks_saved % 20 == 0:
            print(
                f"Saved embeddings for "
                f"{total_tracks_saved} tracklets"
            )

    with open(SUMMARY_CSV, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "num_crops",
                "feature_dim",
                "output_path",
            ],
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print()
    print("Embedding build completed")
    print("=========================")
    print()

    print(
        f"Tracklet embeddings saved: "
        f"{total_tracks_saved}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        f"Summary CSV:      {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
