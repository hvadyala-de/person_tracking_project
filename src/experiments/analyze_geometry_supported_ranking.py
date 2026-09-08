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
    identity_has_conflict,
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


# ============================================================
# TARGET TRACKLETS
#
# These are inspected immediately BEFORE normal assignment.
# ============================================================

TARGETS = {
    (1, 298),
    (0, 399),
    (0, 701),
    (1, 639),
    (0, 740),
    (1, 721),
    (1, 503),
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
# LOAD TRACKLETS
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


    # Same chronological order used by the
    # offline global-ID runner.
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
# GET TRACKLET FROM MEMBER
# ============================================================

def get_member_tracklet(
    member,
    lookup,
):

    camera = normalize_camera_id(
        member.camera_id
    )


    tracklet = lookup.get(
        (
            camera,
            member.local_track_id,
        )
    )


    if tracklet is not None:
        return tracklet


    return lookup.get(
        (
            f"c{camera}",
            member.local_track_id,
        )
    )


# ============================================================
# IDENTITY MEMBER STRING
# ============================================================

def member_string(
    identity,
):

    parts = []


    for member in identity.members:

        camera = normalize_camera_id(
            member.camera_id
        )


        parts.append(
            f"c{camera}:"
            f"{member.local_track_id}"
        )


    return ",".join(
        parts
    )


# ============================================================
# GEOMETRY EVIDENCE AGAINST ONE GID
# ============================================================

def geometry_summary(
    candidate,
    identity,
    lookup,
    homographies,
):

    candidate_camera = normalize_camera_id(
        candidate.camera_id
    )


    supported = []

    contradicted = []

    unknown = []


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        # Cross-camera geometry only.
        if member_camera == candidate_camera:
            continue


        member_tracklet = get_member_tracklet(
            member,
            lookup,
        )


        if member_tracklet is None:
            continue


        evidence = evaluate_cross_camera_geometry(
            candidate,
            member_tracklet,
            homographies,
        )


        label = (
            f"c{member_camera}:"
            f"{member.local_track_id}"
        )


        if evidence.median_distance is None:

            distance_text = "None"

        else:

            distance_text = (
                f"{evidence.median_distance:.2f}"
            )


        item = (
            f"{label}"
            f"(shared={evidence.shared_frames},"
            f"med={distance_text})"
        )


        if evidence.status == "SUPPORTED":

            supported.append(
                item
            )


        elif evidence.status == "CONTRADICTED":

            contradicted.append(
                item
            )


        else:

            unknown.append(
                item
            )


    return {
        "supported":
            supported,

        "contradicted":
            contradicted,

        "unknown":
            unknown,
    }


# ============================================================
# INSPECT TARGET BEFORE ASSIGNMENT
# ============================================================

def inspect_target(
    manager,
    candidate,
    embedding,
    lookup,
    homographies,
):

    camera = normalize_camera_id(
        candidate.camera_id
    )


    track_id = candidate.local_track_id


    print()
    print("=" * 110)

    print(
        f"BEFORE ASSIGNMENT: "
        f"c{camera}:{track_id} "
        f"frames="
        f"{candidate.start_frame}-"
        f"{candidate.end_frame}"
    )

    print("=" * 110)


    rows = []


    for gid, identity in manager.identities.items():

        # ----------------------------------------------------
        # SAME-CAMERA SPATIAL CONFLICT
        # ----------------------------------------------------

        spatial_conflict = identity_has_conflict(
            candidate,
            identity,
            lookup,
        )


        if spatial_conflict:
            continue


        # ----------------------------------------------------
        # APPEARANCE
        # ----------------------------------------------------

        scores = manager.score_identity(
            identity,
            embedding,
        )


        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        geometry = geometry_summary(
            candidate,
            identity,
            lookup,
            homographies,
        )


        rows.append(
            {
                "gid":
                    gid,

                "max":
                    scores["max"],

                "mean":
                    scores["mean"],

                "topk":
                    scores["topk"],

                "support_count":
                    len(
                        geometry[
                            "supported"
                        ]
                    ),

                "contradiction_count":
                    len(
                        geometry[
                            "contradicted"
                        ]
                    ),

                "unknown_count":
                    len(
                        geometry[
                            "unknown"
                        ]
                    ),

                "supported":
                    geometry[
                        "supported"
                    ],

                "contradicted":
                    geometry[
                        "contradicted"
                    ],

                "unknown":
                    geometry[
                        "unknown"
                    ],

                "members":
                    member_string(
                        identity
                    ),
            }
        )


    # Current appearance ranking.
    rows.sort(
        key=lambda row:
            row["topk"],
        reverse=True,
    )


    print()
    print(
        f"{'Rank':>4} "
        f"{'GID':>5} "
        f"{'SUP':>4} "
        f"{'CON':>4} "
        f"{'UNK':>4} "
        f"{'Mean':>8} "
        f"{'TopK':>8} "
        f"{'Max':>8} "
        f"Members"
    )

    print("-" * 110)


    for rank, row in enumerate(
        rows[:12],
        start=1,
    ):

        print(
            f"{rank:>4} "
            f"{row['gid']:>5} "
            f"{row['support_count']:>4} "
            f"{row['contradiction_count']:>4} "
            f"{row['unknown_count']:>4} "
            f"{row['mean']:>8.4f} "
            f"{row['topk']:>8.4f} "
            f"{row['max']:>8.4f} "
            f"{row['members']}"
        )


        if row["supported"]:

            print(
                "      SUPPORTED:",
                "; ".join(
                    row["supported"]
                ),
            )


        if row["contradicted"]:

            print(
                "      CONTRADICTED:",
                "; ".join(
                    row["contradicted"]
                ),
            )


    # ========================================================
    # SECOND VIEW:
    # geometry-supported candidates first
    # ========================================================

    supported_rows = [
        row
        for row in rows
        if (
            row["support_count"] > 0
            and row["contradiction_count"] == 0
        )
    ]


    supported_rows.sort(
        key=lambda row:
            row["topk"],
        reverse=True,
    )


    print()
    print("GEOMETRY-SUPPORTED CANDIDATES")
    print("-----------------------------")


    if not supported_rows:

        print(
            "None"
        )


    else:

        for row in supported_rows:

            print(
                f"GID {row['gid']} | "
                f"support="
                f"{row['support_count']} | "
                f"topk="
                f"{row['topk']:.4f} | "
                f"max="
                f"{row['max']:.4f} | "
                f"members="
                f"{row['members']}"
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
        homographies=homographies
    )


    print()
    print(
        "GEOMETRY-SUPPORTED GID RANKING ANALYSIS"
    )

    print(
        "======================================="
    )


    for tracklet, embedding in items:

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        key = (
            camera,
            tracklet.local_track_id,
        )


        # ----------------------------------------------------
        # Inspect BEFORE current manager modifies its gallery.
        # ----------------------------------------------------

        if key in TARGETS:

            inspect_target(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
            )


        # ----------------------------------------------------
        # Continue normal current assignment so that each
        # target sees exactly the same historical gallery
        # state as the current offline experiment.
        # ----------------------------------------------------

        result = manager.assign_tracklet(
            camera_id=camera,

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


        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup
            )


if __name__ == "__main__":
    main()
