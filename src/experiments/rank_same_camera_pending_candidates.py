import numpy as np

from src.identity_compatibility import (
    identity_has_conflict,
    normalize_camera_id,
)

from src.run_global_id_offline import (
    load_all_tracklets,
)

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.terrace_geometry import (
    load_ground_homographies,
)


# ============================================================
# DIAGNOSTIC LIMITS
#
# These are NOT production thresholds.
# ============================================================

MAX_GAP = 60

MIN_REID = 0.84

FIT_DETECTIONS = 8


# ============================================================
# REPLAY PRODUCTION
# ============================================================

def replay_production():

    (
        items,
        lookup,
    ) = load_all_tracklets()


    items.sort(
        key=lambda item: (
            item[0].start_frame,
            normalize_camera_id(
                item[0].camera_id
            ),
            item[0].local_track_id,
        )
    )


    homographies = (
        load_ground_homographies()
    )


    manager = GlobalIdentityManager(
        homographies=homographies
    )


    for tracklet, embedding in items:

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


        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup,

                include_same_camera=
                    False,
            )


    manager.reevaluate_pending(
        tracklet_lookup=
            lookup,

        include_same_camera=
            True,
    )


    return (
        manager,
        lookup,
    )


# ============================================================
# BASIC DETECTION HELPERS
# ============================================================

def sorted_detections(
    tracklet,
):

    return sorted(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def bottom_center(
    detection,
):

    return np.asarray(
        [
            (
                detection.x1
                + detection.x2
            ) / 2.0,

            detection.y2,
        ],
        dtype=np.float64,
    )


def bbox_height(
    detection,
):

    return float(
        detection.y2
        - detection.y1
    )


# ============================================================
# MOTION FIT
# ============================================================

def fit_motion(
    detections,
):

    if len(detections) < 2:

        return None


    frames = np.asarray(
        [
            detection.frame
            for detection in detections
        ],
        dtype=np.float64,
    )


    points = np.asarray(
        [
            bottom_center(
                detection
            )
            for detection in detections
        ],
        dtype=np.float64,
    )


    x_fit = np.polyfit(
        frames,
        points[:, 0],
        1,
    )


    y_fit = np.polyfit(
        frames,
        points[:, 1],
        1,
    )


    velocity = np.asarray(
        [
            x_fit[0],
            y_fit[0],
        ],
        dtype=np.float64,
    )


    return {
        "x_fit":
            x_fit,

        "y_fit":
            y_fit,

        "velocity":
            velocity,
    }


def predict(
    fit,
    frame,
):

    return np.asarray(
        [
            (
                fit["x_fit"][0]
                * frame
                + fit["x_fit"][1]
            ),

            (
                fit["y_fit"][0]
                * frame
                + fit["y_fit"][1]
            ),
        ],
        dtype=np.float64,
    )


# ============================================================
# PAIR METRICS
# ============================================================

def pair_metrics(
    first,
    second,
):

    # --------------------------------------------------------
    # Put pair in chronological order.
    # --------------------------------------------------------

    if (
        second.start_frame
        < first.start_frame
    ):

        first, second = (
            second,
            first,
        )


    gap = (
        second.start_frame
        - first.end_frame
        - 1
    )


    overlap_start = max(
        first.start_frame,
        second.start_frame,
    )


    overlap_end = min(
        first.end_frame,
        second.end_frame,
    )


    overlap = max(
        0,
        overlap_end
        - overlap_start
        + 1,
    )


    if overlap != 0:

        return None


    if gap < 0:

        return None


    if gap > MAX_GAP:

        return None


    first_detections = (
        sorted_detections(
            first
        )
    )


    second_detections = (
        sorted_detections(
            second
        )
    )


    if (
        len(first_detections) < 2
        or len(second_detections) < 2
    ):

        return None


    first_tail = (
        first_detections[
            -FIT_DETECTIONS:
        ]
    )


    second_head = (
        second_detections[
            :FIT_DETECTIONS
        ]
    )


    first_fit = fit_motion(
        first_tail
    )


    second_fit = fit_motion(
        second_head
    )


    if (
        first_fit is None
        or second_fit is None
    ):

        return None


    first_last = (
        first_detections[-1]
    )


    second_first = (
        second_detections[0]
    )


    first_bottom = bottom_center(
        first_last
    )


    second_bottom = bottom_center(
        second_first
    )


    endpoint_distance = float(
        np.linalg.norm(
            second_bottom
            - first_bottom
        )
    )


    predicted_second = predict(
        first_fit,
        second_first.frame,
    )


    predicted_first = predict(
        second_fit,
        first_last.frame,
    )


    forward_error = float(
        np.linalg.norm(
            predicted_second
            - second_bottom
        )
    )


    backward_error = float(
        np.linalg.norm(
            predicted_first
            - first_bottom
        )
    )


    first_height = bbox_height(
        first_last
    )


    second_height = bbox_height(
        second_first
    )


    mean_height = (
        first_height
        + second_height
    ) / 2.0


    if mean_height <= 0:

        return None


    endpoint_h = (
        endpoint_distance
        / mean_height
    )


    forward_h = (
        forward_error
        / mean_height
    )


    backward_h = (
        backward_error
        / mean_height
    )


    velocity_difference = float(
        np.linalg.norm(
            first_fit[
                "velocity"
            ]
            - second_fit[
                "velocity"
            ]
        )
    )


    if (
        first_height <= 0
        or second_height <= 0
    ):

        height_ratio = None


    else:

        height_ratio = (
            max(
                first_height,
                second_height,
            )
            /
            min(
                first_height,
                second_height,
            )
        )


    return {
        "gap":
            gap,

        "endpoint":
            endpoint_distance,

        "endpoint_h":
            endpoint_h,

        "forward_h":
            forward_h,

        "backward_h":
            backward_h,

        "velocity_difference":
            velocity_difference,

        "height_ratio":
            height_ratio,
    }


# ============================================================
# COMBINED DIAGNOSTIC SCORE
#
# LOWER = MORE CONTINUOUS.
#
# This score is ONLY for ranking.
# It is NOT a production acceptance rule.
# ============================================================

def motion_score(
    metrics,
):

    return (
        metrics[
            "endpoint_h"
        ]

        +

        metrics[
            "forward_h"
        ]

        +

        metrics[
            "backward_h"
        ]
    )


# ============================================================
# FIND ALL PENDING -> SAME-CAMERA CORE PAIRS
# ============================================================

def collect_candidates(
    manager,
    lookup,
):

    rows = []


    for pending_key, pending in (
        manager.pending_tracklets.items()
    ):

        pending_camera = (
            normalize_camera_id(
                pending[
                    "camera_id"
                ]
            )
        )


        pending_tracklet = pending.get(
            "candidate_tracklet"
        )


        if pending_tracklet is None:

            pending_tracklet = (
                manager.get_tracklet(
                    pending_camera,
                    pending[
                        "local_track_id"
                    ],
                    lookup,
                )
            )


        if pending_tracklet is None:

            continue


        for gid in sorted(
            manager.anchored_gids
        ):

            identity = (
                manager.identities.get(
                    gid
                )
            )


            if identity is None:

                continue


            # ------------------------------------------------
            # Keep the production conflict veto.
            # ------------------------------------------------

            if identity_has_conflict(
                pending_tracklet,
                identity,
                lookup,
            ):

                continue


            # ------------------------------------------------
            # Keep the production cross-camera contradiction
            # veto.
            # ------------------------------------------------

            if manager.identity_geometry_contradicted(
                pending_tracklet,
                identity,
                lookup,
            ):

                continue


            for member in identity.members:

                member_camera = (
                    normalize_camera_id(
                        member.camera_id
                    )
                )


                if (
                    member_camera
                    != pending_camera
                ):

                    continue


                member_key = (
                    member_camera,
                    member.local_track_id,
                )


                # --------------------------------------------
                # Positive evidence still comes from CORE only.
                # --------------------------------------------

                if (
                    member_key
                    not in manager.core_members[
                        gid
                    ]
                ):

                    continue


                member_tracklet = (
                    manager.get_member_tracklet(
                        member,
                        lookup,
                    )
                )


                if member_tracklet is None:

                    continue


                member_embedding = (
                    manager.member_embeddings.get(
                        member_key
                    )
                )


                if member_embedding is None:

                    continue


                similarity = (
                    manager.cosine_similarity(
                        pending[
                            "embedding"
                        ],
                        member_embedding,
                    )
                )


                if similarity < MIN_REID:

                    continue


                metrics = pair_metrics(
                    member_tracklet,
                    pending_tracklet,
                )


                if metrics is None:

                    continue


                rows.append(
                    {
                        "camera_id":
                            pending_camera,

                        "pending_id":
                            pending[
                                "local_track_id"
                            ],

                        "gid":
                            gid,

                        "core_id":
                            member.local_track_id,

                        "reid":
                            similarity,

                        **metrics,
                    }
                )


    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "RANK SAME-CAMERA PENDING CANDIDATES"
    )

    print(
        "==================================="
    )


    print(
        "Diagnostic gap <=",
        MAX_GAP,
    )


    print(
        "Diagnostic ReID >=",
        MIN_REID,
    )


    (
        manager,
        lookup,
    ) = replay_production()


    rows = collect_candidates(
        manager,
        lookup,
    )


    for row in rows:

        row[
            "motion_score"
        ] = motion_score(
            row
        )


    rows.sort(
        key=lambda row: (
            row[
                "motion_score"
            ],
            -row[
                "reid"
            ],
            row[
                "gap"
            ],
        )
    )


    print()
    print(
        "Production GIDs:",
        len(
            manager.identities
        ),
    )


    print(
        "Production pending:",
        len(
            manager.pending_tracklets
        ),
    )


    print(
        "Candidate pairs:",
        len(
            rows
        ),
    )


    print()
    print(
        "TOP SAME-CAMERA PENDING -> CORE PAIRS"
    )

    print(
        "====================================="
    )


    header = (
        f"{'Rank':>4} "
        f"{'Pending':>10} "
        f"{'GID':>5} "
        f"{'CORE':>10} "
        f"{'Gap':>5} "
        f"{'ReID':>7} "
        f"{'End/H':>7} "
        f"{'Fwd/H':>7} "
        f"{'Back/H':>7} "
        f"{'Score':>7} "
        f"{'dV':>7} "
        f"{'Hrat':>7}"
    )


    print(
        header
    )


    print(
        "-" * len(
            header
        )
    )


    for rank, row in enumerate(
        rows[:50],
        start=1,
    ):

        height_ratio = (
            f"{row['height_ratio']:.2f}"
            if row[
                "height_ratio"
            ]
            is not None
            else "None"
        )


        print(
            f"{rank:>4} "
            f"c{row['camera_id']}:"
            f"{row['pending_id']:<6} "
            f"{row['gid']:>5} "
            f"c{row['camera_id']}:"
            f"{row['core_id']:<6} "
            f"{row['gap']:>5} "
            f"{row['reid']:>7.4f} "
            f"{row['endpoint_h']:>7.2f} "
            f"{row['forward_h']:>7.2f} "
            f"{row['backward_h']:>7.2f} "
            f"{row['motion_score']:>7.2f} "
            f"{row['velocity_difference']:>7.2f} "
            f"{height_ratio:>7}"
        )


    print()
    print(
        "KNOWN NEAR-MISS LOCATIONS"
    )

    print(
        "========================="
    )


    targets = {
        (
            0,
            264,
            243,
        ),

        (
            0,
            331,
            431,
        ),

        (
            1,
            565,
            503,
        ),
    }


    found = False


    for rank, row in enumerate(
        rows,
        start=1,
    ):

        key = (
            row[
                "camera_id"
            ],
            row[
                "pending_id"
            ],
            row[
                "core_id"
            ],
        )


        reverse_key = (
            row[
                "camera_id"
            ],
            row[
                "core_id"
            ],
            row[
                "pending_id"
            ],
        )


        if (
            key not in targets
            and reverse_key not in targets
        ):

            continue


        found = True


        print(
            f"rank={rank}"
            f" | c{row['camera_id']}:"
            f"{row['pending_id']}"
            f" -> GID "
            f"{row['gid']}"
            f" via c{row['camera_id']}:"
            f"{row['core_id']}"
            f" | gap="
            f"{row['gap']}"
            f" | ReID="
            f"{row['reid']:.4f}"
            f" | score="
            f"{row['motion_score']:.2f}"
            f" | End/H="
            f"{row['endpoint_h']:.2f}"
            f" | Fwd/H="
            f"{row['forward_h']:.2f}"
            f" | Back/H="
            f"{row['backward_h']:.2f}"
        )


    if not found:

        print(
            "NONE"
        )


if __name__ == "__main__":
    main()
