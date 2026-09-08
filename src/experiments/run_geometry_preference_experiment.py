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


# ------------------------------------------------------------
# TEMPORARY EXPERIMENTAL THRESHOLD
#
# Geometry must be SUPPORTED and appearance must still be
# reasonably strong.
#
# This is NOT a final production threshold.
# ------------------------------------------------------------

GEOMETRY_APPEARANCE_MIN = 0.84


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


    # --------------------------------------------------------
    # Same chronological ordering as our normal offline run.
    # --------------------------------------------------------

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
# NORMALIZED COSINE SIMILARITY
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


    first = first / first_norm
    second = second / second_norm


    return float(
        np.dot(
            first,
            second,
        )
    )


# ============================================================
# LOOK UP A TRACKLET FROM AN IDENTITY MEMBER
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
# GEOMETRY SUMMARY AGAINST AN EXISTING GID
# ============================================================

def identity_geometry_evidence(
    candidate_tracklet,
    identity,
    lookup,
    homographies,
):

    candidate_camera = normalize_camera_id(
        candidate_tracklet.camera_id
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
            candidate_tracklet,
            member_tracklet,
            homographies,
        )


        record = {
            "camera_id":
                member_camera,

            "local_track_id":
                member.local_track_id,

            "shared":
                evidence.shared_frames,

            "median":
                evidence.median_distance,

            "status":
                evidence.status,
        }


        if evidence.status == "SUPPORTED":

            supported.append(
                record
            )


        elif evidence.status == "CONTRADICTED":

            contradicted.append(
                record
            )


        else:

            unknown.append(
                record
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
# FIND POSITIVE-GEOMETRY EXISTING GID CANDIDATES
# ============================================================

def find_supported_existing_gids(
    manager,
    current_tracklet,
    current_embedding,
    lookup,
    homographies,
):

    candidates = []


    for (
        gid,
        identity,
    ) in manager.identities.items():

        # ----------------------------------------------------
        # Existing same-camera spatial-conflict veto remains
        # mandatory.
        # ----------------------------------------------------

        if identity_has_conflict(
            current_tracklet,
            identity,
            lookup,
        ):
            continue


        geometry = identity_geometry_evidence(
            current_tracklet,
            identity,
            lookup,
            homographies,
        )


        # ----------------------------------------------------
        # Any synchronized contradiction means this GID is
        # forbidden, even if another member gives support.
        #
        # Example:
        #
        # c1:503
        #   supported by c0:469
        #   contradicted by c0:399
        #
        # Such a GID is already contaminated and must NOT
        # receive another member.
        # ----------------------------------------------------

        if geometry["contradicted"]:
            continue


        # At least one member must have actual synchronized
        # geometry support.
        if not geometry["supported"]:
            continue


        scores = manager.score_identity(
            identity,
            current_embedding,
        )


        # Geometry support alone is not enough.
        if (
            scores["max"]
            < GEOMETRY_APPEARANCE_MIN
        ):
            continue


        if (
            scores["topk"]
            < GEOMETRY_APPEARANCE_MIN
        ):
            continue


        candidates.append(
            {
                "gid":
                    gid,

                "identity":
                    identity,

                "max":
                    scores["max"],

                "mean":
                    scores["mean"],

                "topk":
                    scores["topk"],

                "supported":
                    geometry["supported"],
            }
        )


    candidates.sort(
        key=lambda item:
            item["topk"],
        reverse=True,
    )


    return candidates


# ============================================================
# FIND GEOMETRY-SUPPORTED PENDING CANDIDATES
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
            < GEOMETRY_APPEARANCE_MIN
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
# ADD CURRENT TRACK TO EXISTING GEOMETRY-SUPPORTED GID
# ============================================================

def resolve_with_existing_gid(
    manager,
    current_tracklet,
    current_embedding,
    gid,
):

    camera = normalize_camera_id(
        current_tracklet.camera_id
    )


    manager.add_to_identity(
        global_id=
            gid,

        camera_id=
            camera,

        local_track_id=
            current_tracklet.local_track_id,

        start_frame=
            current_tracklet.start_frame,

        end_frame=
            current_tracklet.end_frame,

        embedding=
            current_embedding,
    )


    # Remove stale pending copy if one exists.
    manager.pending_tracklets.pop(
        (
            camera,
            current_tracklet.local_track_id,
        ),
        None,
    )


# ============================================================
# CREATE CLEAN GID FROM CURRENT + PENDING TRACKLET
# ============================================================

def resolve_with_pending(
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
    # Create the identity from the older pending tracklet.
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
    # Add current tracklet.
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

def build_member_gid_lookup(
    manager,
):

    mapping = {}


    for (
        gid,
        identity,
    ) in manager.identities.items():

        for member in identity.members:

            camera = normalize_camera_id(
                member.camera_id
            )


            mapping[
                (
                    camera,
                    member.local_track_id,
                )
            ] = gid


    return mapping


# ============================================================
# PRINT ONE IMPORTANT TRACK
# ============================================================

def print_track_gid(
    mapping,
    manager,
    camera_id,
    local_track_id,
):

    key = (
        camera_id,
        local_track_id,
    )


    gid = mapping.get(
        key
    )


    if gid is not None:

        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> GID {gid}"
        )

        return


    if key in manager.pending_tracklets:

        print(
            f"c{camera_id}:"
            f"{local_track_id}"
            f" -> PENDING"
        )

        return


    print(
        f"c{camera_id}:"
        f"{local_track_id}"
        f" -> not assigned"
    )


# ============================================================
# FORMAT SUPPORT
# ============================================================

def support_text(
    support,
):

    parts = []


    for record in support:

        median = record[
            "median"
        ]


        if median is None:

            median_text = "None"

        else:

            median_text = (
                f"{median:.2f}"
            )


        parts.append(
            f"c{record['camera_id']}:"
            f"{record['local_track_id']}"
            f"(shared={record['shared']},"
            f"med={median_text})"
        )


    return "; ".join(
        parts
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "POSITIVE GEOMETRY PREFERENCE EXPERIMENT"
    )

    print(
        "======================================="
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


    existing_geometry_resolutions = 0

    pending_geometry_resolutions = 0

    ambiguous_existing = 0

    ambiguous_pending = 0


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


        track_id = (
            tracklet.local_track_id
        )


        # ====================================================
        # PRIORITY 1:
        # POSITIVE GEOMETRY TO EXISTING GID
        # ====================================================

        existing_candidates = (
            find_supported_existing_gids(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
            )
        )


        # ----------------------------------------------------
        # Conservative:
        #
        # automatically use positive existing-GID geometry
        # ONLY when exactly one GID qualifies.
        # ----------------------------------------------------

        if len(
            existing_candidates
        ) == 1:

            candidate = (
                existing_candidates[0]
            )


            gid = candidate[
                "gid"
            ]


            resolve_with_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )


            existing_geometry_resolutions += 1


            print()
            print(
                "EXISTING GEOMETRY RESOLVE | "
                f"c{camera}:{track_id}"
                f" -> GID {gid} | "
                f"topk="
                f"{candidate['topk']:.4f} | "
                f"max="
                f"{candidate['max']:.4f}"
            )

            print(
                "  SUPPORTED:",
                support_text(
                    candidate[
                        "supported"
                    ]
                ),
            )


            # Gallery changed.
            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup
            )


            continue


        elif len(
            existing_candidates
        ) > 1:

            ambiguous_existing += 1


            print()
            print(
                "AMBIGUOUS EXISTING GEOMETRY | "
                f"c{camera}:{track_id} | "
                f"{len(existing_candidates)} "
                f"qualifying GIDs"
            )


            for candidate in (
                existing_candidates
            ):

                print(
                    f"  GID "
                    f"{candidate['gid']} | "
                    f"topk="
                    f"{candidate['topk']:.4f} | "
                    f"max="
                    f"{candidate['max']:.4f} | "
                    f"support="
                    f"{support_text(candidate['supported'])}"
                )


            # Do NOT choose among them.
            # Fall through to pending/normal logic.


        # ====================================================
        # PRIORITY 2:
        # UNIQUE POSITIVE-GEOMETRY PENDING PAIR
        # ====================================================

        pending_candidates = (
            find_supported_pending_matches(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
            )
        )


        if len(
            pending_candidates
        ) == 1:

            candidate = (
                pending_candidates[0]
            )


            gid = resolve_with_pending(
                manager,
                tracklet,
                embedding,
                candidate,
            )


            pending_geometry_resolutions += 1


            evidence = candidate[
                "evidence"
            ]


            print()
            print(
                "PENDING GEOMETRY RESOLVE | "
                f"GID {gid} | "
                f"c{camera}:{track_id}"
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


            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup
            )


            continue


        elif len(
            pending_candidates
        ) > 1:

            ambiguous_pending += 1


            print()
            print(
                "AMBIGUOUS PENDING GEOMETRY | "
                f"c{camera}:{track_id} | "
                f"{len(pending_candidates)} "
                f"qualifying pending tracks"
            )


            for candidate in (
                pending_candidates
            ):

                evidence = candidate[
                    "evidence"
                ]


                print(
                    f"  c{candidate['camera_id']}:"
                    f"{candidate['local_track_id']} | "
                    f"appearance="
                    f"{candidate['appearance']:.4f} | "
                    f"shared="
                    f"{evidence.shared_frames} | "
                    f"median="
                    f"{evidence.median_distance:.2f}"
                )


            # Do not guess.
            # Fall through to normal manager.


        # ====================================================
        # PRIORITY 3:
        # CURRENT NORMAL MANAGER
        # ====================================================

        result = manager.assign_tracklet(
            camera_id=
                camera,

            local_track_id=
                track_id,

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
    # FINAL PENDING PASS
    # ========================================================

    manager.reevaluate_pending(
        tracklet_lookup=
            lookup
    )


    mapping = build_member_gid_lookup(
        manager
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("FINAL RESULT")
    print("============")


    print(
        "Tracklets:",
        len(items),
    )


    print(
        "Existing-GID geometry resolutions:",
        existing_geometry_resolutions,
    )


    print(
        "Pending geometry resolutions:",
        pending_geometry_resolutions,
    )


    print(
        "Ambiguous existing geometry cases:",
        ambiguous_existing,
    )


    print(
        "Ambiguous pending geometry cases:",
        ambiguous_pending,
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

        # ----------------------------------------
        # 255 / 298 / 399 issue
        # ----------------------------------------

        (0, 255),
        (1, 298),

        (0, 399),
        (1, 369),
        (1, 382),

        # ----------------------------------------
        # 503 bridge issue
        # ----------------------------------------

        (0, 469),
        (1, 503),
        (0, 600),

        # ----------------------------------------
        # 492 / 701 contamination issue
        # ----------------------------------------

        (0, 492),
        (1, 511),

        (1, 639),
        (0, 701),
        (0, 740),
        (1, 721),

        # ----------------------------------------
        # 651 family
        # ----------------------------------------

        (0, 651),
        (1, 560),
        (1, 423),

        # ----------------------------------------
        # Known 1 family
        # ----------------------------------------

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
            manager,
            camera,
            track_id,
        )


    # ========================================================
    # LARGEST GLOBAL IDENTITIES
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
