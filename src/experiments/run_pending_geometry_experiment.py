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


# Temporary experimental threshold.
#
# We are NOT making this a production threshold yet.
PENDING_GEOMETRY_APPEARANCE_MIN = 0.84


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


    return float(
        np.dot(
            first / first_norm,
            second / second_norm,
        )
    )


# ============================================================
# FIND GEOMETRY-SUPPORTED PENDING MATCHES
# ============================================================

def find_supported_pending_matches(
    manager,
    current_tracklet,
    current_embedding,
    lookup,
    homographies,
):

    current_camera = normalize_camera_id(
        current_tracklet.camera_id
    )


    candidates = []


    for (
        pending_key,
        pending,
    ) in manager.pending_tracklets.items():

        pending_camera = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )


        # Cross-camera only.
        if pending_camera == current_camera:
            continue


        pending_tracklet = pending.get(
            "candidate_tracklet"
        )


        if pending_tracklet is None:

            pending_tracklet = lookup.get(
                (
                    pending_camera,
                    pending[
                        "local_track_id"
                    ],
                )
            )


        if pending_tracklet is None:
            continue


        appearance = cosine_similarity(
            current_embedding,
            pending[
                "embedding"
            ],
        )


        if (
            appearance
            < PENDING_GEOMETRY_APPEARANCE_MIN
        ):
            continue


        evidence = evaluate_cross_camera_geometry(
            current_tracklet,
            pending_tracklet,
            homographies,
        )


        if evidence.status != "SUPPORTED":
            continue


        candidates.append(
            {
                "key":
                    pending_key,

                "camera_id":
                    pending_camera,

                "local_track_id":
                    pending[
                        "local_track_id"
                    ],

                "appearance":
                    appearance,

                "evidence":
                    evidence,

                "pending":
                    pending,
            }
        )


    candidates.sort(
        key=lambda item:
            item["appearance"],
        reverse=True,
    )


    return candidates


# ============================================================
# CREATE NEW GID FROM CURRENT + PENDING
# ============================================================

def resolve_current_with_pending(
    manager,
    current_tracklet,
    current_embedding,
    pending_candidate,
):

    current_camera = normalize_camera_id(
        current_tracklet.camera_id
    )


    pending = pending_candidate[
        "pending"
    ]


    pending_camera = normalize_camera_id(
        pending[
            "camera_id"
        ]
    )


    # --------------------------------------------------------
    # Start a clean GID using the pending tracklet.
    # --------------------------------------------------------

    gid = manager.create_identity(
        camera_id=
            pending_camera,

        local_track_id=
            pending[
                "local_track_id"
            ],

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
    )


    # --------------------------------------------------------
    # Add the new/current tracklet.
    # --------------------------------------------------------

    manager.add_to_identity(
        global_id=
            gid,

        camera_id=
            current_camera,

        local_track_id=
            current_tracklet.local_track_id,

        start_frame=
            current_tracklet.start_frame,

        end_frame=
            current_tracklet.end_frame,

        embedding=
            current_embedding,
    )


    # --------------------------------------------------------
    # Pending tracklet has now been resolved.
    # --------------------------------------------------------

    manager.pending_tracklets.pop(
        pending_candidate[
            "key"
        ],
        None,
    )


    return gid


# ============================================================
# MEMBER -> GID LOOKUP
# ============================================================

def member_gid_lookup(
    manager,
):

    result = {}


    for (
        gid,
        identity,
    ) in manager.identities.items():

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
# PRINT IMPORTANT TRACKS
# ============================================================

def print_track_gid(
    mapping,
    camera_id,
    track_id,
):

    gid = mapping.get(
        (
            camera_id,
            track_id,
        )
    )


    if gid is None:

        print(
            f"c{camera_id}:{track_id}"
            f" -> not assigned"
        )

    else:

        print(
            f"c{camera_id}:{track_id}"
            f" -> GID {gid}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "PENDING GEOMETRY RESOLUTION EXPERIMENT"
    )

    print(
        "======================================"
    )


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


    geometry_resolutions = []


    # ========================================================
    # PROCESS CHRONOLOGICALLY
    # ========================================================

    for (
        tracklet,
        embedding,
    ) in items:

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        # ====================================================
        # BEFORE NORMAL GID ASSIGNMENT:
        #
        # Check whether this tracklet has a uniquely strong
        # geometry-supported pending partner.
        # ====================================================

        pending_matches = (
            find_supported_pending_matches(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
            )
        )


        # ----------------------------------------------------
        # Conservative rule:
        #
        # Resolve automatically ONLY when exactly one
        # pending candidate satisfies:
        #
        # appearance >= 0.84
        # geometry   == SUPPORTED
        #
        # If multiple candidates qualify, do not guess.
        # ----------------------------------------------------

        if len(
            pending_matches
        ) == 1:

            candidate = (
                pending_matches[0]
            )


            gid = resolve_current_with_pending(
                manager,
                tracklet,
                embedding,
                candidate,
            )


            evidence = candidate[
                "evidence"
            ]


            geometry_resolutions.append(
                {
                    "gid":
                        gid,

                    "current_camera":
                        camera,

                    "current_track":
                        tracklet.local_track_id,

                    "pending_camera":
                        candidate[
                            "camera_id"
                        ],

                    "pending_track":
                        candidate[
                            "local_track_id"
                        ],

                    "appearance":
                        candidate[
                            "appearance"
                        ],

                    "shared":
                        evidence.shared_frames,

                    "median":
                        evidence.median_distance,
                }
            )


            print(
                "GEOMETRY RESOLVE | "
                f"GID {gid} | "
                f"c{camera}:"
                f"{tracklet.local_track_id}"
                f" <-> "
                f"c{candidate['camera_id']}:"
                f"{candidate['local_track_id']} | "
                f"appearance="
                f"{candidate['appearance']:.4f} | "
                f"shared="
                f"{evidence.shared_frames} | "
                f"median="
                f"{evidence.median_distance:.2f}"
            )


            # Gallery changed, so reevaluate old pending tracks.
            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup
            )


            continue


        # ====================================================
        # NO UNIQUE SUPPORTED PENDING MATCH
        #
        # Use current normal manager behavior.
        # ====================================================

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


    # ========================================================
    # FINAL PASS
    # ========================================================

    manager.reevaluate_pending(
        tracklet_lookup=
            lookup
    )


    mapping = member_gid_lookup(
        manager
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("FINAL RESULT")
    print("============")


    print(
        "Tracklets:",
        len(items),
    )


    print(
        "Geometry pending resolutions:",
        len(
            geometry_resolutions
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
    # IMPORTANT REGRESSION TRACKS
    # ========================================================

    print()
    print("IMPORTANT TRACK MAPPING")
    print("=======================")


    important = [
        (0, 255),
        (1, 298),

        (0, 399),
        (1, 369),
        (1, 382),
        (1, 503),

        (0, 469),
        (0, 600),

        (0, 492),
        (1, 511),

        (1, 639),
        (0, 701),
        (0, 740),
        (1, 721),

        (0, 651),
        (1, 560),
        (1, 423),

        (0, 1),
        (1, 1),
        (1, 66),
    ]


    for (
        camera,
        track_id,
    ) in important:

        print_track_gid(
            mapping,
            camera,
            track_id,
        )


    # ========================================================
    # LARGEST GIDS
    # ========================================================

    print()
    print("LARGEST GLOBAL IDENTITIES")
    print("=========================")


    identities = sorted(
        manager.identities.values(),

        key=lambda identity:
            len(
                identity.members
            ),

        reverse=True,
    )


    for identity in identities[:15]:

        print(
            identity.summary()
        )


if __name__ == "__main__":
    main()
