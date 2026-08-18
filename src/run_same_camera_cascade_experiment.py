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
# DATA CONFIGURATION
# ============================================================

CAMERAS = [
    0,
    1,
]


MIN_DETECTIONS = 25


# ============================================================
# SAME-CAMERA CONTINUATION THRESHOLDS
#
# Same conservative values as the previous experiment.
# ============================================================

MAX_GAP_FRAMES = 15

MIN_DIRECT_REID = 0.92

MAX_CENTER_DISTANCE = 50.0

MAX_BOTTOM_DISTANCE = 50.0


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
# LOAD TRACKLETS + EMBEDDINGS
# ============================================================

def load_data():

    items = []

    lookup = {}


    for camera_id in CAMERAS:

        csv_path = (
            TRACK_DIR
            / f"tracks_c{camera_id}.csv"
        )


        raw = load_tracklets(
            csv_path
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
# DETECTION HELPERS
# ============================================================

def first_detection(
    tracklet,
):

    if not tracklet.detections:
        return None


    return min(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def last_detection(
    tracklet,
):

    if not tracklet.detections:
        return None


    return max(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def bbox_center(
    detection,
):

    x = (
        detection.x1
        + detection.x2
    ) / 2.0


    y = (
        detection.y1
        + detection.y2
    ) / 2.0


    return np.asarray(
        [
            x,
            y,
        ],
        dtype=np.float32,
    )


def bbox_bottom_center(
    detection,
):

    x = (
        detection.x1
        + detection.x2
    ) / 2.0


    y = detection.y2


    return np.asarray(
        [
            x,
            y,
        ],
        dtype=np.float32,
    )


# ============================================================
# CHRONOLOGICAL ORDER
# ============================================================

def chronological_pair(
    first,
    second,
):

    if (
        first.start_frame
        <= second.start_frame
    ):

        return (
            first,
            second,
        )


    return (
        second,
        first,
    )


# ============================================================
# TEMPORAL STATS
# ============================================================

def temporal_stats(
    first,
    second,
):

    first, second = (
        chronological_pair(
            first,
            second,
        )
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


    return (
        gap,
        overlap,
    )


# ============================================================
# END -> START DISTANCE
# ============================================================

def endpoint_distance(
    first,
    second,
):

    first, second = (
        chronological_pair(
            first,
            second,
        )
    )


    end_detection = last_detection(
        first
    )


    start_detection = first_detection(
        second
    )


    if (
        end_detection is None
        or start_detection is None
    ):

        return (
            None,
            None,
        )


    center_distance = float(
        np.linalg.norm(
            bbox_center(
                end_detection
            )
            - bbox_center(
                start_detection
            )
        )
    )


    bottom_distance = float(
        np.linalg.norm(
            bbox_bottom_center(
                end_detection
            )
            - bbox_bottom_center(
                start_detection
            )
        )
    )


    return (
        center_distance,
        bottom_distance,
    )


# ============================================================
# TRACKLET LOOKUP
# ============================================================

def get_tracklet(
    camera_id,
    local_track_id,
    lookup,
):

    camera = normalize_camera_id(
        camera_id
    )


    result = lookup.get(
        (
            camera,
            local_track_id,
        )
    )


    if result is not None:
        return result


    return lookup.get(
        (
            f"c{camera}",
            local_track_id,
        )
    )


# ============================================================
# REPLAY PRODUCTION MANAGER
# ============================================================

def replay_production_manager(
    items,
    lookup,
    homographies,
):

    manager = GlobalIdentityManager(
        homographies=
            homographies
    )


    for (
        tracklet,
        embedding,
    ) in items:

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
                    lookup
            )


    manager.reevaluate_pending(
        tracklet_lookup=
            lookup
    )


    return manager


# ============================================================
# TEST ONE CORE -> PENDING SAME-CAMERA TRANSITION
# ============================================================

def evaluate_core_transition(
    manager,
    pending_tracklet,
    pending_embedding,
    core_member,
    lookup,
):

    pending_camera = normalize_camera_id(
        pending_tracklet.camera_id
    )


    core_camera = normalize_camera_id(
        core_member.camera_id
    )


    if core_camera != pending_camera:
        return None


    core_key = (
        core_camera,
        core_member.local_track_id,
    )


    core_tracklet = get_tracklet(
        core_camera,
        core_member.local_track_id,
        lookup,
    )


    if core_tracklet is None:
        return None


    core_embedding = (
        manager.member_embeddings.get(
            core_key
        )
    )


    if core_embedding is None:
        return None


    (
        gap,
        overlap,
    ) = temporal_stats(
        core_tracklet,
        pending_tracklet,
    )


    if overlap != 0:
        return None


    if gap < 0:
        return None


    if gap > MAX_GAP_FRAMES:
        return None


    (
        center_distance,
        bottom_distance,
    ) = endpoint_distance(
        core_tracklet,
        pending_tracklet,
    )


    if (
        center_distance is None
        or bottom_distance is None
    ):

        return None


    if (
        center_distance
        > MAX_CENTER_DISTANCE
    ):

        return None


    if (
        bottom_distance
        > MAX_BOTTOM_DISTANCE
    ):

        return None


    similarity = (
        manager.cosine_similarity(
            pending_embedding,
            core_embedding,
        )
    )


    if similarity < MIN_DIRECT_REID:
        return None


    if (
        core_tracklet.start_frame
        <= pending_tracklet.start_frame
    ):

        first_id = (
            core_member.local_track_id
        )

        second_id = (
            pending_tracklet.local_track_id
        )


    else:

        first_id = (
            pending_tracklet.local_track_id
        )

        second_id = (
            core_member.local_track_id
        )


    return {
        "core_camera":
            core_camera,

        "core_track_id":
            core_member.local_track_id,

        "first_track_id":
            first_id,

        "second_track_id":
            second_id,

        "gap":
            gap,

        "overlap":
            overlap,

        "center_distance":
            center_distance,

        "bottom_distance":
            bottom_distance,

        "similarity":
            similarity,
    }


# ============================================================
# FIND SAME-CAMERA CORE CONTINUATION GIDS
# ============================================================

def find_same_camera_continuation_candidates(
    manager,
    pending_tracklet,
    pending_embedding,
    lookup,
):

    candidates = []


    for gid in sorted(
        manager.anchored_gids
    ):

        identity = manager.identities.get(
            gid
        )


        if identity is None:
            continue


        # ----------------------------------------------------
        # Existing same-camera spatial conflict remains a veto.
        # ----------------------------------------------------

        if identity_has_conflict(
            pending_tracklet,
            identity,
            lookup,
        ):

            continue


        # ----------------------------------------------------
        # Existing cross-camera geometry contradiction remains
        # a hard veto.
        # ----------------------------------------------------

        if manager.identity_geometry_contradicted(
            pending_tracklet,
            identity,
            lookup,
        ):

            continue


        support_rows = []


        for member in identity.members:

            member_key = (
                normalize_camera_id(
                    member.camera_id
                ),
                member.local_track_id,
            )


            # ------------------------------------------------
            # Positive same-camera continuation support can
            # come from CORE only.
            # ------------------------------------------------

            if (
                member_key
                not in manager.core_members[
                    gid
                ]
            ):

                continue


            row = evaluate_core_transition(
                manager=
                    manager,

                pending_tracklet=
                    pending_tracklet,

                pending_embedding=
                    pending_embedding,

                core_member=
                    member,

                lookup=
                    lookup,
            )


            if row is not None:

                support_rows.append(
                    row
                )


        if not support_rows:
            continue


        support_rows.sort(
            key=lambda row:
                row["similarity"],
            reverse=True,
        )


        candidates.append(
            {
                "gid":
                    gid,

                "supports":
                    support_rows,

                "best":
                    support_rows[0],
            }
        )


    candidates.sort(
        key=lambda row:
            (
                row["best"]["similarity"],
                -row["best"]["gap"],
            ),
        reverse=True,
    )


    return candidates


# ============================================================
# MEMBER -> GID LOOKUP
# ============================================================

def build_member_gid_lookup(
    manager,
):

    result = {}


    for gid, identity in (
        manager.identities.items()
    ):

        for member in identity.members:

            camera = normalize_camera_id(
                member.camera_id
            )


            result[
                (
                    camera,
                    member.local_track_id,
                )
            ] = gid


    return result


# ============================================================
# REGRESSION CHECK HELPERS
# ============================================================

def same_gid(
    mapping,
    keys,
):

    gids = [
        mapping.get(
            key
        )
        for key in keys
    ]


    return (
        all(
            gid is not None
            for gid in gids
        )
        and
        len(
            set(
                gids
            )
        ) == 1
    )


def different_gid(
    mapping,
    first,
    second,
):

    first_gid = mapping.get(
        first
    )


    second_gid = mapping.get(
        second
    )


    return (
        first_gid is not None
        and second_gid is not None
        and first_gid != second_gid
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "SAME-CAMERA CASCADE EXPERIMENT"
    )

    print(
        "=============================="
    )


    print(
        "Maximum gap:",
        MAX_GAP_FRAMES,
    )


    print(
        "Minimum direct ReID:",
        MIN_DIRECT_REID,
    )


    print(
        "Maximum center distance:",
        MAX_CENTER_DISTANCE,
    )


    print(
        "Maximum bottom distance:",
        MAX_BOTTOM_DISTANCE,
    )


    print()
    print(
        "PURPOSE:"
    )

    print(
        "Add conservative same-camera RELAXED members,"
    )

    print(
        "then call the real manager.reevaluate_pending()"
    )

    print(
        "to test whether gallery changes cause extra cascades."
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    items, lookup = load_data()


    homographies = (
        load_ground_homographies()
    )


    # ========================================================
    # BASELINE PRODUCTION STATE
    # ========================================================

    manager = replay_production_manager(
        items,
        lookup,
        homographies,
    )


    baseline_gids = len(
        manager.identities
    )


    baseline_pending = len(
        manager.pending_tracklets
    )


    baseline_core_members = sum(
        len(
            members
        )
        for members in (
            manager.core_members.values()
        )
    )


    print()
    print(
        "BASELINE"
    )

    print(
        "========"
    )


    print(
        "GIDs:",
        baseline_gids,
    )


    print(
        "Pending:",
        baseline_pending,
    )


    print(
        "Anchored GIDs:",
        len(
            manager.anchored_gids
        ),
    )


    print(
        "CORE members:",
        baseline_core_members,
    )


    # ========================================================
    # PHASE 1:
    # SAME-CAMERA CONTINUATION
    # ========================================================

    same_camera_resolved = []

    same_camera_ambiguous = []


    pending_items = list(
        manager.pending_tracklets.items()
    )


    for pending_key, pending in pending_items:

        if (
            pending_key
            not in manager.pending_tracklets
        ):

            continue


        camera = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )


        track_id = (
            pending[
                "local_track_id"
            ]
        )


        pending_tracklet = pending.get(
            "candidate_tracklet"
        )


        if pending_tracklet is None:

            pending_tracklet = get_tracklet(
                camera,
                track_id,
                lookup,
            )


        if pending_tracklet is None:
            continue


        candidates = (
            find_same_camera_continuation_candidates(
                manager=
                    manager,

                pending_tracklet=
                    pending_tracklet,

                pending_embedding=
                    pending[
                        "embedding"
                    ],

                lookup=
                    lookup,
            )
        )


        # ----------------------------------------------------
        # Exactly one qualifying GID.
        # ----------------------------------------------------

        if len(candidates) == 1:

            candidate = (
                candidates[0]
            )


            gid = (
                candidate[
                    "gid"
                ]
            )


            best = (
                candidate[
                    "best"
                ]
            )


            manager.add_to_identity(
                global_id=
                    gid,

                camera_id=
                    camera,

                local_track_id=
                    track_id,

                start_frame=
                    pending[
                        "start_frame"
                    ],

                end_frame=
                    pending[
                        "end_frame"
                    ],

                embedding=
                    pending[
                        "embedding"
                    ],

                trust_level=
                    "RELAXED",
            )


            del manager.pending_tracklets[
                pending_key
            ]


            same_camera_resolved.append(
                {
                    "camera_id":
                        camera,

                    "local_track_id":
                        track_id,

                    "gid":
                        gid,

                    "core_track_id":
                        best[
                            "core_track_id"
                        ],

                    "first_track_id":
                        best[
                            "first_track_id"
                        ],

                    "second_track_id":
                        best[
                            "second_track_id"
                        ],

                    "gap":
                        best[
                            "gap"
                        ],

                    "similarity":
                        best[
                            "similarity"
                        ],

                    "center_distance":
                        best[
                            "center_distance"
                        ],

                    "bottom_distance":
                        best[
                            "bottom_distance"
                        ],
                }
            )


        elif len(candidates) > 1:

            same_camera_ambiguous.append(
                {
                    "camera_id":
                        camera,

                    "local_track_id":
                        track_id,

                    "candidates":
                        candidates,
                }
            )


    pending_after_same_camera = set(
        manager.pending_tracklets.keys()
    )


    pending_count_after_same_camera = len(
        pending_after_same_camera
    )


    # ========================================================
    # PHASE 2:
    # CALL REAL MANAGER REEVALUATE_PENDING()
    #
    # This is the cascade test.
    # ========================================================

    cascade_results = (
        manager.reevaluate_pending(
            tracklet_lookup=
                lookup
        )
    )


    pending_after_cascade = set(
        manager.pending_tracklets.keys()
    )


    # --------------------------------------------------------
    # Tracklets present immediately before reevaluation but
    # gone afterward are actual cascade resolutions.
    # --------------------------------------------------------

    cascade_resolved_keys = (
        pending_after_same_camera
        - pending_after_cascade
    )


    cascade_resolved = []


    for row in cascade_results:

        key = (
            normalize_camera_id(
                row[
                    "camera_id"
                ]
            ),
            row[
                "local_track_id"
            ],
        )


        if key not in cascade_resolved_keys:
            continue


        cascade_resolved.append(
            {
                "camera_id":
                    key[0],

                "local_track_id":
                    key[1],

                "global_id":
                    row.get(
                        "global_id"
                    ),

                "status":
                    row.get(
                        "status"
                    ),

                "trust_level":
                    row.get(
                        "trust_level"
                    ),

                "reason":
                    row.get(
                        "reason"
                    ),
            }
        )


    # ========================================================
    # CORE COUNT AFTER BOTH PHASES
    # ========================================================

    final_core_members = sum(
        len(
            members
        )
        for members in (
            manager.core_members.values()
        )
    )


    # ========================================================
    # FINAL MAPPING
    # ========================================================

    mapping = build_member_gid_lookup(
        manager
    )


    # ========================================================
    # PRINT SAME-CAMERA RESOLUTIONS
    # ========================================================

    print()
    print(
        "SAME-CAMERA RESOLUTIONS"
    )

    print(
        "======================="
    )


    if not same_camera_resolved:

        print(
            "NONE"
        )


    else:

        for row in same_camera_resolved:

            print(
                f"c{row['camera_id']}:"
                f"{row['first_track_id']}"
                f" -> "
                f"c{row['camera_id']}:"
                f"{row['second_track_id']}"
                f" | GID {row['gid']}"
                f" | CORE support="
                f"c{row['camera_id']}:"
                f"{row['core_track_id']}"
                f" | gap="
                f"{row['gap']}"
                f" | ReID="
                f"{row['similarity']:.4f}"
                f" | center="
                f"{row['center_distance']:.2f}"
                f" | bottom="
                f"{row['bottom_distance']:.2f}"
            )


    # ========================================================
    # PRINT SAME-CAMERA AMBIGUITIES
    # ========================================================

    print()
    print(
        "AMBIGUOUS SAME-CAMERA"
    )

    print(
        "====================="
    )


    if not same_camera_ambiguous:

        print(
            "NONE"
        )


    else:

        for row in same_camera_ambiguous:

            print(
                f"c{row['camera_id']}:"
                f"{row['local_track_id']}"
                f" | "
                f"{len(row['candidates'])}"
                f" qualifying GIDs"
            )


    # ========================================================
    # CASCADE RESULTS
    # ========================================================

    print()
    print(
        "POST-CONTINUATION CASCADE RESOLUTIONS"
    )

    print(
        "====================================="
    )


    if not cascade_resolved:

        print(
            "NONE"
        )


    else:

        for row in cascade_resolved:

            print(
                f"c{row['camera_id']}:"
                f"{row['local_track_id']}"
                f" -> GID "
                f"{row['global_id']}"
                f" | status="
                f"{row['status']}"
                f" | trust="
                f"{row['trust_level']}"
                f" | reason="
                f"{row['reason']}"
            )


    # ========================================================
    # REGRESSION CHECKS
    # ========================================================

    checks = [
        (
            "255 family together",
            same_gid(
                mapping,
                [
                    (0, 255),
                    (1, 298),
                    (1, 319),
                ],
            ),
        ),

        (
            "399 family together",
            same_gid(
                mapping,
                [
                    (0, 399),
                    (1, 369),
                    (1, 382),
                ],
            ),
        ),

        (
            "469/503/600 together",
            same_gid(
                mapping,
                [
                    (0, 469),
                    (1, 503),
                    (0, 600),
                ],
            ),
        ),

        (
            "492/511 together",
            same_gid(
                mapping,
                [
                    (0, 492),
                    (1, 511),
                ],
            ),
        ),

        (
            "651/560 together",
            same_gid(
                mapping,
                [
                    (0, 651),
                    (1, 560),
                ],
            ),
        ),

        (
            "701 family together",
            same_gid(
                mapping,
                [
                    (1, 639),
                    (0, 701),
                    (0, 740),
                    (1, 721),
                ],
            ),
        ),

        (
            "255 separated from 399",
            different_gid(
                mapping,
                (0, 255),
                (0, 399),
            ),
        ),

        (
            "492 separated from 701",
            different_gid(
                mapping,
                (0, 492),
                (0, 701),
            ),
        ),

        (
            "651 separated from 701",
            different_gid(
                mapping,
                (0, 651),
                (0, 701),
            ),
        ),
    ]


    print()
    print(
        "REGRESSION CHECKS"
    )

    print(
        "================="
    )


    passed = 0


    for name, result in checks:

        label = (
            "PASS"
            if result
            else "FAIL"
        )


        if result:

            passed += 1


        print(
            f"{label:<4} | "
            f"{name}"
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print(
        "FINAL RESULT"
    )

    print(
        "============"
    )


    print(
        "Baseline GIDs:",
        baseline_gids,
    )


    print(
        "Baseline pending:",
        baseline_pending,
    )


    print(
        "Same-camera resolutions:",
        len(
            same_camera_resolved
        ),
    )


    print(
        "Same-camera ambiguities:",
        len(
            same_camera_ambiguous
        ),
    )


    print(
        "Pending immediately after same-camera:",
        pending_count_after_same_camera,
    )


    print(
        "Cascade resolutions:",
        len(
            cascade_resolved
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


    print(
        "CORE members before:",
        baseline_core_members,
    )


    print(
        "CORE members after:",
        final_core_members,
    )


    print(
        "CORE unchanged:",
        (
            baseline_core_members
            == final_core_members
        ),
    )


    print(
        "Regression checks passed:",
        f"{passed}/{len(checks)}",
    )


if __name__ == "__main__":
    main()
