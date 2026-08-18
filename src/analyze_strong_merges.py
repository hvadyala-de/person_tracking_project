from pathlib import Path
import csv

import numpy as np

from src.build_tracklets import (
    load_tracklets,
    filter_tracklets,
)

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.terrace_geometry import (
    load_ground_homographies,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


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


OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "strong_merge_analysis.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CAMERAS = [
    0,
    1,
]


MIN_DETECTIONS = 25


# ============================================================
# LOAD ONE TRACKLET EMBEDDING
# ============================================================

def load_embedding(
    camera_id,
    local_track_id,
):

    path = (
        EMBEDDING_DIR
        / f"c{camera_id}"
        / f"track_{local_track_id}.npy"
    )

    if not path.exists():
        return None

    embedding = np.load(
        path
    ).astype(
        np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        return None

    return embedding / norm


# ============================================================
# LOAD TRACKLETS + EMBEDDINGS
# ============================================================

def load_data():

    items = []

    tracklet_lookup = {}


    for camera_id in CAMERAS:

        csv_path = (
            TRACK_DIR
            / f"tracks_c{camera_id}.csv"
        )

        raw_tracklets = load_tracklets(
            csv_path
        )

        useful_tracklets = filter_tracklets(
            raw_tracklets,
            min_detections=MIN_DETECTIONS,
        )

        for tracklet in useful_tracklets.values():

            normalized_camera = normalize_camera_id(
                tracklet.camera_id
            )

            embedding = load_embedding(
                normalized_camera,
                tracklet.local_track_id,
            )

            if embedding is None:
                continue

            items.append(
                (
                    tracklet,
                    embedding,
                )
            )

            # Integer camera lookup.
            tracklet_lookup[
                (
                    normalized_camera,
                    tracklet.local_track_id,
                )
            ] = tracklet

            # String camera lookup.
            tracklet_lookup[
                (
                    f"c{normalized_camera}",
                    tracklet.local_track_id,
                )
            ] = tracklet


    # Same chronological processing order as the
    # normal offline GID experiment.
    items.sort(
        key=lambda item: (
            item[0].start_frame,
            normalize_camera_id(
                item[0].camera_id
            ),
            item[0].local_track_id,
        )
    )


    return (
        items,
        tracklet_lookup,
    )


# ============================================================
# SIMILARITY TO EVERY EXISTING GALLERY MEMBER
# ============================================================

def gallery_similarity_stats(
    identity,
    embedding,
):

    if not identity.embedding_gallery:

        return {
            "gallery_size": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }


    similarities = np.asarray(
        identity.similarities(
            embedding
        ),
        dtype=np.float32,
    )


    if similarities.size == 0:

        return {
            "gallery_size": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }


    return {
        "gallery_size":
            int(
                similarities.size
            ),

        "min":
            float(
                np.min(
                    similarities
                )
            ),

        "max":
            float(
                np.max(
                    similarities
                )
            ),

        "mean":
            float(
                np.mean(
                    similarities
                )
            ),

        "median":
            float(
                np.median(
                    similarities
                )
            ),

        "std":
            float(
                np.std(
                    similarities
                )
            ),
    }


# ============================================================
# MEMBER LABELS
# ============================================================

def identity_member_string(
    identity,
):

    parts = []


    for member in identity.members:

        camera_id = normalize_camera_id(
            member.camera_id
        )

        parts.append(
            f"c{camera_id}:"
            f"{member.local_track_id}"
        )


    return ",".join(
        parts
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("STRONG MERGE COHERENCE ANALYSIS")
    print("===============================")


    (
        items,
        tracklet_lookup,
    ) = load_data()


    print(
        "Tracklets loaded:",
        len(items),
    )


    # ========================================================
    # LOAD TERRACE GEOMETRY
    # ========================================================

    homographies = load_ground_homographies()


    print(
        "Homographies loaded:",
        sorted(
            homographies.keys()
        ),
    )


    # ========================================================
    # CURRENT MANAGER
    #
    # Includes:
    #
    # - same-camera spatial conflict gate
    # - cross-camera geometry contradiction veto
    #
    # No coherence veto is added here.
    # This script is diagnostic only.
    # ========================================================

    manager = GlobalIdentityManager(
        homographies=homographies
    )


    rows = []

    processed = 0


    # ========================================================
    # PROCESS TRACKLETS
    # ========================================================

    for tracklet, embedding in items:

        camera_id = normalize_camera_id(
            tracklet.camera_id
        )

        local_track_id = (
            tracklet.local_track_id
        )


        # ====================================================
        # EVALUATE BEFORE ASSIGNMENT
        # ====================================================

        match = manager.evaluate(
            camera_id=camera_id,
            embedding=embedding,
            candidate_tracklet=tracklet,
            tracklet_lookup=tracklet_lookup,
        )


        gallery_stats = {
            "gallery_size": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }


        member_string = ""


        if (
            match.global_id is not None
            and match.global_id in manager.identities
        ):

            identity = manager.identities[
                match.global_id
            ]

            gallery_stats = gallery_similarity_stats(
                identity,
                embedding,
            )

            member_string = identity_member_string(
                identity
            )


        # ====================================================
        # NORMAL ASSIGNMENT
        # ====================================================

        result = manager.assign_tracklet(
            camera_id=camera_id,
            local_track_id=local_track_id,
            start_frame=tracklet.start_frame,
            end_frame=tracklet.end_frame,
            embedding=embedding,
            candidate_tracklet=tracklet,
            tracklet_lookup=tracklet_lookup,
        )


        # ====================================================
        # RECORD DIRECT STRONG MERGES
        # ====================================================

        if result.status == "STRONG":

            rows.append(
                {
                    "camera_id":
                        camera_id,

                    "local_track_id":
                        local_track_id,

                    "start_frame":
                        tracklet.start_frame,

                    "end_frame":
                        tracklet.end_frame,

                    "global_id":
                        result.global_id,

                    "gallery_size_before":
                        gallery_stats[
                            "gallery_size"
                        ],

                    "gallery_min":
                        gallery_stats[
                            "min"
                        ],

                    "gallery_mean":
                        gallery_stats[
                            "mean"
                        ],

                    "gallery_median":
                        gallery_stats[
                            "median"
                        ],

                    "gallery_max":
                        gallery_stats[
                            "max"
                        ],

                    "gallery_std":
                        gallery_stats[
                            "std"
                        ],

                    "manager_max":
                        match.max_similarity,

                    "manager_mean":
                        match.mean_similarity,

                    "manager_topk":
                        match.topk_similarity,

                    "second_gid":
                        match.second_global_id,

                    "second_topk":
                        match.second_topk_similarity,

                    "score_margin":
                        match.score_margin,

                    "members_before":
                        member_string,
                }
            )


        # ====================================================
        # REEVALUATE PENDING AFTER GALLERY CHANGES
        # ====================================================

        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=tracklet_lookup
            )


        processed += 1


        if processed % 25 == 0:

            print(
                f"Processed "
                f"{processed}/"
                f"{len(items)} | "
                f"GIDs="
                f"{len(manager.identities)} | "
                f"pending="
                f"{len(manager.pending_tracklets)}"
            )


    # ========================================================
    # SORT LOWEST COHERENCE FIRST
    # ========================================================

    rows.sort(
        key=lambda row: (
            row["gallery_min"]
            if row["gallery_min"] is not None
            else 999.0
        )
    )


    # ========================================================
    # WRITE CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if rows:

        with OUTPUT_CSV.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                rows
            )


    # ========================================================
    # PRINT LOWEST-COHERENCE STRONG MERGES
    # ========================================================

    print()
    print("LOWEST-COHERENCE STRONG MERGES")
    print("==============================")
    print()


    header = (
        f"{'Cam':>4} "
        f"{'Track':>7} "
        f"{'GID':>6} "
        f"{'N':>4} "
        f"{'Min':>8} "
        f"{'Mean':>8} "
        f"{'Median':>8} "
        f"{'TopK':>8} "
        f"{'Max':>8} "
        f"{'Margin':>8}"
    )


    print(
        header
    )

    print(
        "-" * len(header)
    )


    for row in rows[:40]:

        minimum = (
            f"{row['gallery_min']:.4f}"
            if row["gallery_min"] is not None
            else "None"
        )

        mean = (
            f"{row['gallery_mean']:.4f}"
            if row["gallery_mean"] is not None
            else "None"
        )

        median = (
            f"{row['gallery_median']:.4f}"
            if row["gallery_median"] is not None
            else "None"
        )

        margin = (
            f"{row['score_margin']:.4f}"
            if row["score_margin"] is not None
            else "None"
        )


        print(
            f"{row['camera_id']:>4} "
            f"{row['local_track_id']:>7} "
            f"{row['global_id']:>6} "
            f"{row['gallery_size_before']:>4} "
            f"{minimum:>8} "
            f"{mean:>8} "
            f"{median:>8} "
            f"{row['manager_topk']:>8.4f} "
            f"{row['manager_max']:>8.4f} "
            f"{margin:>8}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("SUMMARY")
    print("=======")


    print(
        "Direct STRONG merges analyzed:",
        len(rows),
    )


    if rows:

        valid_minimums = [
            row["gallery_min"]
            for row in rows
            if row["gallery_min"] is not None
        ]


        if valid_minimums:

            print(
                "Lowest gallery minimum:",
                f"{min(valid_minimums):.4f}",
            )

            print(
                "Median gallery minimum:",
                f"{np.median(valid_minimums):.4f}",
            )


    print(
        "Final GIDs:",
        len(
            manager.identities
        ),
    )

    print(
        "Final pending:",
        len(
            manager.pending_tracklets
        ),
    )


    print()
    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
