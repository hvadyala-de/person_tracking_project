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
# STRICT TRUSTED GEOMETRY
#
# Experimental only.
#
# Positive cross-camera geometry can directly resolve an
# identity only when:
#
#   appearance >= 0.88
#   geometry == SUPPORTED
#   geometry median <= 20
#
# Appearance-only STRONG matches are NOT merged.
# They are stored as PENDING.
# ============================================================

TRUSTED_APPEARANCE_MIN = 0.88

TRUSTED_GEOMETRY_MEDIAN_MAX = 20.0


# ============================================================
# IMPORTANT TRACKS FOR DEBUG OUTPUT
# ============================================================

WATCH_TRACKS = {
    (0, 255),
    (1, 298),

    (0, 399),
    (1, 369),
    (1, 382),

    (0, 469),
    (1, 503),
    (0, 600),

    (0, 492),
    (1, 511),

    (0, 651),
    (1, 560),
    (1, 423),

    (1, 639),
    (0, 701),
    (0, 740),
    (1, 721),

    (0, 1),
    (1, 1),
    (1, 66),
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
    # Same chronological ordering as our offline experiment.
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


    first = first / first_norm

    second = second / second_norm


    return float(
        np.dot(
            first,
            second,
        )
    )


# ============================================================
# MEMBER TRACKLET LOOKUP
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
# TRUSTED GEOMETRY CHECK
# ============================================================

def is_trusted_geometry(
    evidence,
):

    if evidence.status != "SUPPORTED":
        return False


    if evidence.median_distance is None:
        return False


    if (
        evidence.median_distance
        > TRUSTED_GEOMETRY_MEDIAN_MAX
    ):
        return False


    return True


# ============================================================
# GEOMETRY EVIDENCE AGAINST AN EXISTING GID
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


    trusted_supported = []

    weak_supported = []

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


        if evidence.status == "CONTRADICTED":

            contradicted.append(
                record
            )


        elif is_trusted_geometry(
            evidence
        ):

            trusted_supported.append(
                record
            )


        elif evidence.status == "SUPPORTED":

            weak_supported.append(
                record
            )


        else:

            unknown.append(
                record
            )


    return {
        "trusted_supported":
            trusted_supported,

        "weak_supported":
            weak_supported,

        "contradicted":
            contradicted,

        "unknown":
            unknown,
    }


# ============================================================
# FIND TRUSTED EXISTING GID CANDIDATES
# ============================================================

def find_trusted_existing_gids(
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
        # Same-camera spatial conflict remains mandatory.
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
        # Any synchronized contradiction forbids this GID.
        # ----------------------------------------------------

        if geometry[
            "contradicted"
        ]:
            continue


        # ----------------------------------------------------
        # Require at least one trusted positive geometry link.
        # ----------------------------------------------------

        if not geometry[
            "trusted_supported"
        ]:
            continue


        scores = manager.score_identity(
            identity,
            current_embedding,
        )


        # ----------------------------------------------------
        # Geometry alone is not enough.
        # Require adequate appearance support.
        # ----------------------------------------------------

        if (
            scores["max"]
            < TRUSTED_APPEARANCE_MIN
        ):
            continue


        if (
            scores["topk"]
            < TRUSTED_APPEARANCE_MIN
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

                "support":
                    geometry[
                        "trusted_supported"
                    ],
            }
        )


    candidates.sort(
        key=lambda item:
            item["topk"],
        reverse=True,
    )


    return candidates


# ============================================================
# FIND TRUSTED PENDING PAIRS
# ============================================================

def find_trusted_pending_matches(
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
            < TRUSTED_APPEARANCE_MIN
        ):
            continue


        evidence = evaluate_cross_camera_geometry(
            current_tracklet,
            pending_tracklet,
            homographies,
        )


        if not is_trusted_geometry(
            evidence
        ):
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
# ADD CURRENT TRACK TO EXISTING GID
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


    # Remove stale pending copy if present.
    manager.pending_tracklets.pop(
        (
            camera,
            current_tracklet.local_track_id,
        ),
        None,
    )


# ============================================================
# CREATE CLEAN GID FROM CURRENT + PENDING
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
    # Create new clean identity from the older pending track.
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
    # Add current track.
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
# ADD CURRENT TRACK TO PENDING
# ============================================================

def hold_as_pending(
    manager,
    tracklet,
    embedding,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )


    manager.add_pending(
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
    )


# ============================================================
# FORMAT SUPPORT TEXT
# ============================================================

def support_text(
    records,
):

    parts = []


    for record in records:

        median = record[
            "median"
        ]


        median_text = (
            f"{median:.2f}"
            if median is not None
            else "None"
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
# PRINT IMPORTANT TRACK
# ============================================================

def print_track_state(
    mapping,
    manager,
    camera_id,
    track_id,
):

    key = (
        camera_id,
        track_id,
    )


    gid = mapping.get(
        key
    )


    if gid is not None:

        print(
            f"c{camera_id}:"
            f"{track_id}"
            f" -> GID {gid}"
        )

        return


    if key in manager.pending_tracklets:

        print(
            f"c{camera_id}:"
            f"{track_id}"
            f" -> PENDING"
        )

        return


    print(
        f"c{camera_id}:"
        f"{track_id}"
        f" -> not assigned"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "EVIDENCE-GATED GLOBAL ID EXPERIMENT"
    )

    print(
        "==================================="
    )


    print(
        "Trusted appearance minimum:",
        TRUSTED_APPEARANCE_MIN,
    )


    print(
        "Trusted geometry median maximum:",
        TRUSTED_GEOMETRY_MEDIAN_MAX,
    )


    print()

    print(
        "IMPORTANT POLICY:"
    )

    print(
        "Appearance-only STRONG -> PENDING"
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


    # ========================================================
    # COUNTERS
    # ========================================================

    existing_geometry_resolutions = 0

    pending_geometry_resolutions = 0

    appearance_strong_held = 0

    appearance_pending_held = 0

    new_identity_count = 0

    ambiguous_existing = 0

    ambiguous_pending = 0


    # ========================================================
    # PROCESS TRACKLETS CHRONOLOGICALLY
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


        key = (
            camera,
            track_id,
        )


        # ====================================================
        # FIRST TRACKLET
        #
        # There is no identity hypothesis yet.
        # ====================================================

        if not manager.identities:

            gid = manager.create_identity(
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
            )


            new_identity_count += 1


            if key in WATCH_TRACKS:

                print(
                    "WATCH NEW | "
                    f"c{camera}:{track_id}"
                    f" -> GID {gid}"
                )


            continue


        # ====================================================
        # PRIORITY 1:
        # TRUSTED POSITIVE GEOMETRY TO EXISTING GID
        # ====================================================

        existing_candidates = (
            find_trusted_existing_gids(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
            )
        )


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


            print(
                "TRUSTED EXISTING | "
                f"c{camera}:{track_id}"
                f" -> GID {gid} | "
                f"topk="
                f"{candidate['topk']:.4f} | "
                f"max="
                f"{candidate['max']:.4f} | "
                f"support="
                f"{support_text(candidate['support'])}"
            )


            # IMPORTANT:
            #
            # We intentionally DO NOT call
            # manager.reevaluate_pending().
            #
            # That function currently permits appearance-only
            # STRONG merges, which would defeat this experiment.

            continue


        elif len(
            existing_candidates
        ) > 1:

            ambiguous_existing += 1


            print()
            print(
                "AMBIGUOUS TRUSTED EXISTING | "
                f"c{camera}:{track_id} | "
                f"{len(existing_candidates)} GIDs"
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
                    f"{support_text(candidate['support'])}"
                )


            # Do not guess.
            # Continue to pending evidence.


        # ====================================================
        # PRIORITY 2:
        # TRUSTED CURRENT <-> PENDING GEOMETRY PAIR
        # ====================================================

        pending_candidates = (
            find_trusted_pending_matches(
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


            print(
                "TRUSTED PENDING | "
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


            # Again:
            # NO normal reevaluate_pending() here.

            continue


        elif len(
            pending_candidates
        ) > 1:

            ambiguous_pending += 1


            print()
            print(
                "AMBIGUOUS TRUSTED PENDING | "
                f"c{camera}:{track_id} | "
                f"{len(pending_candidates)} tracks"
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


        # ====================================================
        # PRIORITY 3:
        # ASK CURRENT MANAGER FOR APPEARANCE DECISION
        #
        # But DO NOT call assign_tracklet().
        #
        # evaluate() is read-only.
        # ====================================================

        match = manager.evaluate(
            camera_id=
                camera,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )


        # ====================================================
        # APPEARANCE-ONLY STRONG
        #
        # OLD:
        #     merge immediately
        #
        # NEW EXPERIMENT:
        #     hold as pending
        # ====================================================

        if match.status == "STRONG":

            hold_as_pending(
                manager,
                tracklet,
                embedding,
            )


            appearance_strong_held += 1


            if key in WATCH_TRACKS:

                print(
                    "WATCH HOLD STRONG | "
                    f"c{camera}:{track_id} | "
                    f"suggested_gid="
                    f"{match.global_id} | "
                    f"max="
                    f"{match.max_similarity:.4f} | "
                    f"topk="
                    f"{match.topk_similarity:.4f}"
                )


            continue


        # ====================================================
        # NORMAL PENDING
        #
        # Keep pending.
        # ====================================================

        if match.status == "PENDING":

            hold_as_pending(
                manager,
                tracklet,
                embedding,
            )


            appearance_pending_held += 1


            if key in WATCH_TRACKS:

                print(
                    "WATCH HOLD PENDING | "
                    f"c{camera}:{track_id} | "
                    f"suggested_gid="
                    f"{match.global_id} | "
                    f"max="
                    f"{match.max_similarity:.4f} | "
                    f"topk="
                    f"{match.topk_similarity:.4f}"
                )


            continue


        # ====================================================
        # CLEAR NEW
        #
        # No existing GID has sufficient compatible
        # appearance evidence.
        #
        # Safe to create a fresh singleton GID.
        # ====================================================

        gid = manager.create_identity(
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
        )


        manager.pending_tracklets.pop(
            key,
            None,
        )


        new_identity_count += 1


        if key in WATCH_TRACKS:

            print(
                "WATCH NEW | "
                f"c{camera}:{track_id}"
                f" -> GID {gid}"
            )


    # ========================================================
    # IMPORTANT:
    #
    # No final manager.reevaluate_pending() call.
    #
    # Pending tracks remain pending unless trusted geometry
    # explicitly resolved them during chronological replay.
    # ========================================================


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
        "Trusted existing-GID resolutions:",
        existing_geometry_resolutions,
    )


    print(
        "Trusted pending-pair resolutions:",
        pending_geometry_resolutions,
    )


    print(
        "Appearance STRONG held pending:",
        appearance_strong_held,
    )


    print(
        "Appearance PENDING held pending:",
        appearance_pending_held,
    )


    print(
        "New singleton GIDs created:",
        new_identity_count,
    )


    print(
        "Ambiguous trusted existing cases:",
        ambiguous_existing,
    )


    print(
        "Ambiguous trusted pending cases:",
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
    # IMPORTANT TRACK MAPPING
    # ========================================================

    print()
    print("IMPORTANT TRACK MAPPING")
    print("=======================")


    important = [

        # 255 family
        (0, 255),
        (1, 298),
        (1, 319),

        # 399 family
        (0, 399),
        (1, 369),
        (1, 382),

        # 503 bridge
        (0, 469),
        (1, 503),
        (0, 600),

        # 492 family
        (0, 492),
        (1, 511),

        # 651 family
        (0, 431),
        (1, 423),
        (1, 560),
        (0, 651),

        # 701 family
        (1, 639),
        (0, 701),
        (0, 740),
        (1, 721),

        # 1 family
        (0, 1),
        (1, 1),
        (1, 66),
    ]


    for (
        camera,
        track_id,
    ) in important:

        print_track_state(
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


    for identity in identities[:20]:

        print(
            identity.summary()
        )


if __name__ == "__main__":
    main()
