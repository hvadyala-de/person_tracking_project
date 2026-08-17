from pathlib import Path
import argparse
import csv

import cv2
import numpy as np

from reid_extractor import ReIDExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Build OSNet embeddings for Terrace tracklets"
)

parser.add_argument(
    "--camera",
    type=int,
    required=True,
    choices=[0, 1, 2, 3],
    help="Terrace camera number",
)

args = parser.parse_args()

camera_id = args.camera


# ============================================================
# PATHS
# ============================================================

CROPS_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_crops"
    / f"c{camera_id}"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_embeddings"
    / f"c{camera_id}"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "tracklet_embeddings_summary.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# ============================================================
# MAIN
# ============================================================

def main():

    if not CROPS_DIR.exists():

        raise FileNotFoundError(
            f"Crops directory not found: {CROPS_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD OSNET ONCE
    # --------------------------------------------------------

    extractor = ReIDExtractor()

    # --------------------------------------------------------
    # FIND TRACKLET DIRECTORIES
    # --------------------------------------------------------

    track_dirs = sorted(
        [
            path
            for path in CROPS_DIR.iterdir()
            if path.is_dir()
            and path.name.startswith("track_")
        ]
    )

    print()
    print("TRACKLET EMBEDDING BUILD")
    print("========================")
    print()

    print(f"Camera:          c{camera_id}")
    print(f"Crop directory: {CROPS_DIR}")
    print(f"Track folders:  {len(track_dirs)}")
    print()

    summary_rows = []

    total_tracks_saved = 0
    total_crops_processed = 0

    # ========================================================
    # BUILD EACH TRACKLET DESCRIPTOR
    # ========================================================

    for track_dir in track_dirs:

        crop_paths = sorted(
            track_dir.glob("*.jpg")
        )

        if not crop_paths:
            continue

        embeddings = []

        for crop_path in crop_paths:

            image = cv2.imread(
                str(crop_path)
            )

            if image is None:
                print(
                    f"Warning: could not read "
                    f"{crop_path}"
                )
                continue

            feature = extractor.extract(
                image
            )

            if feature is None:
                continue

            embeddings.append(
                feature
            )

            total_crops_processed += 1

        if not embeddings:
            continue

        # ----------------------------------------------------
        # MULTI-CROP TRACKLET DESCRIPTOR
        # ----------------------------------------------------

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
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # SAVE 512-D DESCRIPTOR
        # ----------------------------------------------------

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
                "camera_id": f"c{camera_id}",
                "track_id": track_dir.name,
                "num_crops": len(embeddings),
                "feature_dim": mean_embedding.shape[0],
                "output_path": str(output_path),
            }
        )

        total_tracks_saved += 1

        if total_tracks_saved % 20 == 0:

            print(
                f"Saved embeddings for "
                f"{total_tracks_saved}/"
                f"{len(track_dirs)} tracklets"
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "camera_id",
                "track_id",
                "num_crops",
                "feature_dim",
                "output_path",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("Embedding build completed")
    print("=========================")
    print()

    print(
        f"Camera:                   c{camera_id}"
    )

    print(
        f"Tracklets processed:      "
        f"{len(track_dirs)}"
    )

    print(
        f"Tracklet embeddings saved:"
        f" {total_tracks_saved}"
    )

    print(
        f"Person crops processed:   "
        f"{total_crops_processed}"
    )

    print(
        f"Feature dimension:        512"
    )

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        f"Summary CSV:      {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
