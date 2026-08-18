import csv
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
    / "global_id_analysis_c0_c1.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_DETECTIONS = 25


CAMERAS = [
    0,
    1,
]


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


    return (
        embedding
        / norm
    )


# ============================================================
# LOAD ALL USEFUL TRACKLETS
# ============================================================

def load_all_tracklets():

    all_tracklets = []

    tracklet_lookup = {}


    for camera_id in CAMERAS:

        csv_path = (
            TRACK_DIR
            / f"tracks_c{camera_id}.csv"
        )


        tracklets = load_tracklets(
            csv_path
        )


        useful = filter_tracklets(
            tracklets,
            min_detections=MIN_DETECTIONS,
        )


        for tracklet in useful.values():

            normalized_camera = (
                normalize_camera_id(
                    tracklet.camera_id
                )
            )


            embedding = load_embedding(
                normalized_camera,
                tracklet.local_track_id,
            )


            if embedding is None:

                continue


            all_tracklets.append(
                (
                    tracklet,
                    embedding,
                )
            )


            tracklet_lookup[
                (
                    normalized_camera,
                    tracklet.local_track_id,
                )
            ] = tracklet


            tracklet_lookup[
                (
                    f"c{normalized_camera}",
                    tracklet.local_track_id,
                )
            ] = tracklet


    return (
        all_tracklets,
        tracklet_lookup,
    )


# ============================================================
# BUILD FINAL MEMBER -> GID LOOKUP
# ============================================================

def build_member_to_gid(
    manager,
):

    member_to_gid = {}


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            camera_id = (
                normalize_camera_id(
                    member.camera_id
                )
            )


            key = (
                camera_id,
                member.local_track_id,
            )


            member_to_gid[
                key
            ] = gid


    return member_to_gid


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("OFFLINE GLOBAL ID ANALYSIS")
    print("==========================")


    # ========================================================
    # LOAD TRACKLETS + EMBEDDINGS
    # ========================================================

    (
        tracklets,
        tracklet_lookup,
    ) = load_all_tracklets()


    tracklets.sort(
        key=lambda item: (
            item[0].start_frame,

            normalize_camera_id(
                item[0].camera_id
            ),

            item[0].local_track_id,
        )
    )


    print(
        "Tracklets loaded:",
        len(tracklets),
    )


    # ========================================================
    # LOAD TERRACE GEOMETRY
    # ========================================================

    print(
        "Loading Terrace homographies..."
    )


    homographies = (
        load_ground_homographies()
    )


    print(
        "Homographies loaded:",
        sorted(
            homographies.keys()
        ),
    )


    # ========================================================
    # CREATE MANAGER
    # ========================================================

    manager = GlobalIdentityManager(
        homographies=homographies
    )


    initial_status = {}


    processed = 0

    strong_count = 0
    new_count = 0
    pending_count = 0


    # ========================================================
    # CHRONOLOGICAL PROCESSING
    #
    # Same-camera continuation is deliberately disabled here.
    #
    # During chronological construction we only allow the
    # validated cross-camera CORE-supported pending sweep.
    # ========================================================

    for tracklet, embedding in tracklets:

        camera_id = (
            normalize_camera_id(
                tracklet.camera_id
            )
        )


        local_track_id = (
            tracklet.local_track_id
        )


        key = (
            camera_id,
            local_track_id,
        )


        result = manager.assign_tracklet(
            camera_id=camera_id,

            local_track_id=
                local_track_id,

            start_frame=
                tracklet.start_frame,

            end_frame=
                tracklet.end_frame,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )


        initial_status[
            key
        ] = result.status


        processed += 1


        if result.status == "STRONG":

            strong_count += 1


        elif result.status == "NEW":

            new_count += 1


        elif result.status == "PENDING":

            pending_count += 1


        # ----------------------------------------------------
        # Gallery changed.
        #
        # IMPORTANT:
        # Same-camera continuation remains disabled here.
        # ----------------------------------------------------

        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    tracklet_lookup,

                include_same_camera=
                    False,
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed % 25
            == 0
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{len(tracklets)} | "
                f"GIDs="
                f"{len(manager.identities)} | "
                f"pending="
                f"{len(manager.pending_tracklets)}"
            )


    # ========================================================
    # PRE-FINAL BASELINE
    #
    # Expected validated state:
    #
    # GIDs    = 76
    # pending = 96
    # ========================================================

    pre_final_gids = len(
        manager.identities
    )


    pre_final_pending = len(
        manager.pending_tracklets
    )


    print()
    print(
        "PRE-FINAL CONTINUATION STATE"
    )

    print(
        "============================"
    )


    print(
        "GIDs:",
        pre_final_gids,
    )


    print(
        "Pending:",
        pre_final_pending,
    )


    # ========================================================
    # FINAL PENDING PASS
    #
    # This is the ONLY place where same-camera continuation
    # is enabled.
    #
    # Order inside manager:
    #
    #   1. cross-camera CORE-supported pending sweep
    #   2. one same-camera CORE continuation pass
    #   3. stop -- no extra cross-camera cascade afterward
    # ========================================================

    final_pending_results = (
        manager.reevaluate_pending(
            tracklet_lookup=
                tracklet_lookup,

            include_same_camera=
                True,
        )
    )


    same_camera_results = [
        result

        for result
        in final_pending_results

        if result.get(
            "reason"
        )
        == "SAME_CAMERA_CORE_CONTINUATION"
    ]


    print()
    print(
        "FINAL SAME-CAMERA CONTINUATIONS"
    )

    print(
        "==============================="
    )


    if not same_camera_results:

        print(
            "NONE"
        )


    else:

        for result in same_camera_results:

            print(
                f"c{result['camera_id']}:"
                f"{result['local_track_id']}"
                f" -> "
                f"GID {result['global_id']}"
                f" | CORE support="
                f"c{result['support_camera_id']}:"
                f"{result['support_local_track_id']}"
                f" | gap="
                f"{result['gap']}"
                f" | ReID="
                f"{result['direct_similarity']:.4f}"
                f" | center="
                f"{result['center_distance']:.2f}"
                f" | bottom="
                f"{result['bottom_distance']:.2f}"
            )


    # ========================================================
    # FINAL MEMBER -> GID LOOKUP
    # ========================================================

    member_to_gid = (
        build_member_to_gid(
            manager
        )
    )


    # ========================================================
    # BUILD REPORT ROWS
    # ========================================================

    rows = []


    for tracklet, embedding in tracklets:

        camera_id = (
            normalize_camera_id(
                tracklet.camera_id
            )
        )


        key = (
            camera_id,
            tracklet.local_track_id,
        )


        gid = member_to_gid.get(
            key
        )


        is_pending = (
            key
            in manager.pending_tracklets
        )


        if gid is not None:

            final_state = "ASSIGNED"


        elif is_pending:

            final_state = "PENDING"


        else:

            final_state = "UNRESOLVED"


        suggested_gid = None

        max_similarity = None
        mean_similarity = None
        topk_similarity = None

        second_gid = None
        second_score = None
        margin = None


        if is_pending:

            match = manager.evaluate(
                camera_id=camera_id,

                embedding=embedding,

                candidate_tracklet=
                    tracklet,

                tracklet_lookup=
                    tracklet_lookup,
            )


            suggested_gid = (
                match.global_id
            )


            max_similarity = (
                match.max_similarity
            )


            mean_similarity = (
                match.mean_similarity
            )


            topk_similarity = (
                match.topk_similarity
            )


            second_gid = (
                match.second_global_id
            )


            second_score = (
                match.second_topk_similarity
            )


            margin = (
                match.score_margin
            )


        rows.append(
            {
                "camera_id":
                    camera_id,

                "local_track_id":
                    tracklet.local_track_id,

                "start_frame":
                    tracklet.start_frame,

                "end_frame":
                    tracklet.end_frame,

                "num_detections":
                    tracklet.num_detections,

                "initial_status":
                    initial_status.get(
                        key
                    ),

                "final_state":
                    final_state,

                "global_id":
                    gid,

                "suggested_gid":
                    suggested_gid,

                "max_similarity":
                    max_similarity,

                "mean_similarity":
                    mean_similarity,

                "topk_similarity":
                    topk_similarity,

                "second_gid":
                    second_gid,

                "second_score":
                    second_score,

                "margin":
                    margin,
            }
        )


    # ========================================================
    # WRITE CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


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
    # FINAL STATISTICS
    # ========================================================

    print()
    print("FINAL RESULT")
    print("============")


    print(
        "Processed tracklets:",
        len(tracklets),
    )


    print(
        "Initial STRONG:",
        strong_count,
    )


    print(
        "Initial NEW:",
        new_count,
    )


    print(
        "Initial PENDING:",
        pending_count,
    )


    print(
        "Pre-final GIDs:",
        pre_final_gids,
    )


    print(
        "Pre-final pending:",
        pre_final_pending,
    )


    print(
        "Final same-camera continuations:",
        len(
            same_camera_results
        ),
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


    # ========================================================
    # LARGEST GLOBAL IDENTITIES
    # ========================================================

    largest = sorted(
        manager.identities.values(),

        key=lambda identity:
            len(identity.members),

        reverse=True,
    )


    print()
    print("LARGEST GLOBAL IDENTITIES")
    print("=========================")


    for identity in largest[:20]:

        print(
            identity.summary()
        )


    print()

    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
