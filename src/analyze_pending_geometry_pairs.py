from pathlib import Path

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

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
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


# ============================================================
# CONFIGURATION
# ============================================================

CAMERAS = [
    0,
    1,
]


MIN_DETECTIONS = 25


# Only print events involving these especially important tracks.
WATCH_TRACKS = {
    (1, 298),
    (0, 399),

    (1, 639),
    (0, 701),
    (0, 740),
    (1, 721),

    (1, 503),
    (0, 469),
    (0, 600),
}


# ============================================================
# LOAD EMBEDDING
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
# LOAD DATA
# ============================================================

def load_data():

    items = []

    lookup = {}


    for camera_id in CAMERAS:

        path = (
            TRACK_DIR
            / f"tracks_c{camera_id}.csv"
        )


        raw = load_tracklets(
            path
        )


        useful = filter_tracklets(
            raw,
            min_detections=MIN_DETECTIONS,
        )


        for tracklet in useful.values():

            camera = normalize_camera_id(
                tracklet.camera_id
            )


            embedding = load_embedding(
                camera,
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


            lookup[
                (
                    camera,
                    tracklet.local_track_id,
                )
            ] = tracklet


            lookup[
                (
                    f"c{camera}",
                    tracklet.local_track_id,
                )
            ] = tracklet


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
        lookup,
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    first,
    second,
):

    first = np.asarray(
        first,
        dtype=np.float32,
    )

    second = np.asarray(
        second,
        dtype=np.float32,
    )


    first_norm = np.linalg.norm(
        first
    )

    second_norm = np.linalg.norm(
        second
    )


    if (
        first_norm == 0
        or second_norm == 0
    ):
        return -1.0


    first = (
        first
        / first_norm
    )

    second = (
        second
        / second_norm
    )


    return float(
        np.dot(
            first,
            second,
        )
    )


# ============================================================
# PRINT PENDING SUPPORT AGAINST CURRENT TRACKLET
# ============================================================

def inspect_pending_pairs(
    manager,
    current_tracklet,
    current_embedding,
    lookup,
    homographies,
):

    current_camera = normalize_camera_id(
        current_tracklet.camera_id
    )

    current_id = (
        current_tracklet.local_track_id
    )


    rows = []


    for (
        pending_key,
        pending,
    ) in manager.pending_tracklets.items():

        pending_camera = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )

        pending_id = (
            pending[
                "local_track_id"
            ]
        )


        # ----------------------------------------------------
        # Geometry evidence is only useful cross-camera.
        # ----------------------------------------------------

        if (
            pending_camera
            == current_camera
        ):
            continue


        pending_tracklet = pending.get(
            "candidate_tracklet"
        )


        if pending_tracklet is None:

            pending_tracklet = lookup.get(
                (
                    pending_camera,
                    pending_id,
                )
            )


        if pending_tracklet is None:
            continue


        evidence = evaluate_cross_camera_geometry(
            current_tracklet,
            pending_tracklet,
            homographies,
        )


        appearance = cosine_similarity(
            current_embedding,
            pending[
                "embedding"
            ],
        )


        rows.append(
            {
                "pending_camera":
                    pending_camera,

                "pending_id":
                    pending_id,

                "appearance":
                    appearance,

                "status":
                    evidence.status,

                "shared":
                    evidence.shared_frames,

                "median":
                    evidence.median_distance,
            }
        )


    # --------------------------------------------------------
    # Interesting pairs first:
    #
    # SUPPORTED geometry,
    # then higher appearance.
    # --------------------------------------------------------

    rows.sort(
        key=lambda row: (
            1
            if row["status"] == "SUPPORTED"
            else 0,

            row["appearance"],
        ),
        reverse=True,
    )


    current_key = (
        current_camera,
        current_id,
    )


    watched = (
        current_key
        in WATCH_TRACKS
    )


    strong_supported = [
        row
        for row in rows
        if (
            row["status"]
            == "SUPPORTED"

            and

            row["appearance"]
            >= 0.84
        )
    ]


    if (
        not watched
        and not strong_supported
    ):
        return


    print()
    print("=" * 100)

    print(
        f"CURRENT c{current_camera}:"
        f"{current_id} "
        f"frames="
        f"{current_tracklet.start_frame}-"
        f"{current_tracklet.end_frame}"
    )

    print(
        "Pending pool size:",
        len(
            manager.pending_tracklets
        ),
    )

    print("=" * 100)


    if not rows:

        print(
            "No cross-camera pending candidates."
        )

        return


    print()

    print(
        f"{'Pending':>14} "
        f"{'Appearance':>11} "
        f"{'Geometry':>13} "
        f"{'Shared':>8} "
        f"{'Median':>10}"
    )

    print("-" * 64)


    for row in rows[:20]:

        median = (
            f"{row['median']:.2f}"
            if row["median"] is not None
            else "None"
        )


        print(
            f"c{row['pending_camera']}:"
            f"{row['pending_id']:>10} "
            f"{row['appearance']:>11.4f} "
            f"{row['status']:>13} "
            f"{row['shared']:>8} "
            f"{median:>10}"
        )


    print()
    print(
        "SUPPORTED + appearance >= 0.84:"
    )


    if not strong_supported:

        print(
            "None"
        )


    else:

        for row in strong_supported:

            median = (
                f"{row['median']:.2f}"
                if row["median"] is not None
                else "None"
            )


            print(
                f"  c{current_camera}:"
                f"{current_id}"
                f" <-> "
                f"c{row['pending_camera']}:"
                f"{row['pending_id']} | "
                f"appearance="
                f"{row['appearance']:.4f} | "
                f"shared="
                f"{row['shared']} | "
                f"median="
                f"{median}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        items,
        lookup,
    ) = load_data()


    homographies = (
        load_ground_homographies()
    )


    manager = GlobalIdentityManager(
        homographies=
            homographies
    )


    print()
    print(
        "PENDING-TRACKLET GEOMETRY PAIR ANALYSIS"
    )

    print(
        "======================================="
    )


    supported_pairs = 0


    for (
        tracklet,
        embedding,
    ) in items:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Inspect the pending pool BEFORE this tracklet is
        # assigned.
        # ----------------------------------------------------

        inspect_pending_pairs(
            manager,
            tracklet,
            embedding,
            lookup,
            homographies,
        )


        camera = normalize_camera_id(
            tracklet.camera_id
        )


        result = manager.assign_tracklet(
            camera_id=
                camera,

            local_track_id=
                tracklet.local_track_id,

            start_frame=
                tracklet.start_frame,

            end_frame=
                tracklet.end_frame,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )


        # ----------------------------------------------------
        # Keep behavior identical to current offline run.
        # ----------------------------------------------------

        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup
            )


    print()
    print("FINAL CURRENT-MANAGER STATE")
    print("===========================")

    print(
        "GIDs:",
        len(
            manager.identities
        ),
    )

    print(
        "Pending:",
        len(
            manager.pending_tracklets
        ),
    )


if __name__ == "__main__":
    main()
