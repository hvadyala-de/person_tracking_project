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
# PATHS
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
# CONFIG
# ============================================================

CAMERAS = [0, 1]

MIN_DETECTIONS = 25


# Strict threshold used to CREATE a trusted geometry anchor.
STRICT_APPEARANCE_MIN = 0.88


# Relaxed threshold allowed ONLY after the GID has already
# been geometry anchored.
ANCHORED_EXPANSION_APPEARANCE_MIN = 0.84


# Keep the stricter Terrace analysis geometry boundary.
TRUSTED_GEOMETRY_MEDIAN_MAX = 20.0


WATCH_TRACKS = {
    (0, 255),
    (1, 298),
    (1, 319),

    (0, 399),
    (1, 369),
    (1, 382),

    (0, 469),
    (1, 503),
    (0, 600),

    (0, 492),
    (1, 511),

    (0, 431),
    (1, 423),

    (1, 560),
    (0, 651),

    (1, 639),
    (0, 701),
    (0, 740),
    (1, 721),

    (0, 1),
    (1, 1),
    (1, 66),
}


# ============================================================
# EMBEDDINGS
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
# DATA
# ============================================================

def load_data():

    items = []
    lookup = {}

    for camera_id in CAMERAS:

        raw = load_tracklets(
            TRACK_DIR
            / f"tracks_c{camera_id}.csv"
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

    return items, lookup


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
# TRACKLET LOOKUP
# ============================================================

def get_member_tracklet(
    member,
    lookup,
):

    camera = normalize_camera_id(
        member.camera_id
    )

    result = lookup.get(
        (
            camera,
            member.local_track_id,
        )
    )

    if result is not None:
        return result

    return lookup.get(
        (
            f"c{camera}",
            member.local_track_id,
        )
    )


# ============================================================
# TRUSTED GEOMETRY
# ============================================================

def is_trusted_geometry(
    evidence,
):

    if evidence.status != "SUPPORTED":
        return False

    if evidence.median_distance is None:
        return False

    return (
        evidence.median_distance
        <= TRUSTED_GEOMETRY_MEDIAN_MAX
    )


# ============================================================
# GID GEOMETRY
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

    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )

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
        }

        if evidence.status == "CONTRADICTED":

            contradicted.append(
                record
            )

        elif is_trusted_geometry(
            evidence
        ):

            supported.append(
                record
            )

    return {
        "supported":
            supported,

        "contradicted":
            contradicted,
    }


# ============================================================
# FORMAT SUPPORT
# ============================================================

def support_text(
    records,
):

    parts = []

    for record in records:

        median = record["median"]

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
# FIND EXISTING GID CANDIDATES
# ============================================================

def find_existing_geometry_candidates(
    manager,
    tracklet,
    embedding,
    lookup,
    homographies,
    anchored_gids,
    relaxed_anchored=False,
):

    candidates = []

    for gid, identity in manager.identities.items():

        if (
            relaxed_anchored
            and gid not in anchored_gids
        ):
            continue

        if identity_has_conflict(
            tracklet,
            identity,
            lookup,
        ):
            continue

        geometry = identity_geometry_evidence(
            tracklet,
            identity,
            lookup,
            homographies,
        )

        # Any contradiction kills the candidate.
        if geometry["contradicted"]:
            continue

        if not geometry["supported"]:
            continue

        scores = manager.score_identity(
            identity,
            embedding,
        )

        if relaxed_anchored:

            appearance_min = (
                ANCHORED_EXPANSION_APPEARANCE_MIN
            )

        else:

            appearance_min = (
                STRICT_APPEARANCE_MIN
            )

        if scores["max"] < appearance_min:
            continue

        if scores["topk"] < appearance_min:
            continue

        candidates.append(
            {
                "gid":
                    gid,

                "max":
                    scores["max"],

                "mean":
                    scores["mean"],

                "topk":
                    scores["topk"],

                "support":
                    geometry["supported"],
            }
        )

    candidates.sort(
        key=lambda row:
            row["topk"],
        reverse=True,
    )

    return candidates


# ============================================================
# FIND STRICT PENDING PAIRS
# ============================================================

def find_strict_pending_candidates(
    manager,
    tracklet,
    embedding,
    lookup,
    homographies,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )

    candidates = []

    for key, pending in (
        manager.pending_tracklets.items()
    ):

        pending_camera = normalize_camera_id(
            pending["camera_id"]
        )

        if pending_camera == camera:
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
            embedding,
            pending["embedding"],
        )

        if appearance < STRICT_APPEARANCE_MIN:
            continue

        evidence = evaluate_cross_camera_geometry(
            tracklet,
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
                    key,

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
        key=lambda row:
            row["appearance"],
        reverse=True,
    )

    return candidates


# ============================================================
# ADD CURRENT TO EXISTING GID
# ============================================================

def add_to_existing_gid(
    manager,
    tracklet,
    embedding,
    gid,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )

    manager.add_to_identity(
        global_id=
            gid,

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
    )

    manager.pending_tracklets.pop(
        (
            camera,
            tracklet.local_track_id,
        ),
        None,
    )


# ============================================================
# CREATE ANCHOR FROM PENDING PAIR
# ============================================================

def create_anchor_from_pending(
    manager,
    tracklet,
    embedding,
    pending_candidate,
):

    camera = normalize_camera_id(
        tracklet.camera_id
    )

    pending = pending_candidate[
        "pending"
    ]

    pending_camera = normalize_camera_id(
        pending["camera_id"]
    )

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

    manager.add_to_identity(
        global_id=
            gid,

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
    )

    manager.pending_tracklets.pop(
        pending_candidate["key"],
        None,
    )

    return gid


# ============================================================
# HOLD TRACKLET
# ============================================================

def hold_pending(
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
# RELAXED PENDING -> ANCHORED GID SWEEP
#
# This is the key new part.
#
# Only GIDs already proven by trusted geometry are eligible.
# Appearance threshold is reduced from 0.88 to 0.84.
# ============================================================

def sweep_pending_into_anchored_gids(
    manager,
    lookup,
    homographies,
    anchored_gids,
):

    resolved = []

    changed = True

    while changed:

        changed = False

        pending_items = list(
            manager.pending_tracklets.items()
        )

        for key, pending in pending_items:

            tracklet = pending.get(
                "candidate_tracklet"
            )

            if tracklet is None:

                camera = normalize_camera_id(
                    pending["camera_id"]
                )

                tracklet = lookup.get(
                    (
                        camera,
                        pending[
                            "local_track_id"
                        ],
                    )
                )

            if tracklet is None:
                continue

            candidates = (
                find_existing_geometry_candidates(
                    manager,
                    tracklet,
                    pending["embedding"],
                    lookup,
                    homographies,
                    anchored_gids,
                    relaxed_anchored=True,
                )
            )

            # Conservative:
            # exactly one anchored GID must qualify.
            if len(candidates) != 1:
                continue

            candidate = candidates[0]

            gid = candidate["gid"]

            add_to_existing_gid(
                manager,
                tracklet,
                pending["embedding"],
                gid,
            )

            resolved.append(
                {
                    "camera_id":
                        normalize_camera_id(
                            pending[
                                "camera_id"
                            ]
                        ),

                    "local_track_id":
                        pending[
                            "local_track_id"
                        ],

                    "gid":
                        gid,

                    "topk":
                        candidate[
                            "topk"
                        ],

                    "max":
                        candidate[
                            "max"
                        ],

                    "support":
                        candidate[
                            "support"
                        ],
                }
            )

            changed = True

    return resolved


# ============================================================
# MAPPING
# ============================================================

def build_member_gid_lookup(
    manager,
):

    mapping = {}

    for gid, identity in (
        manager.identities.items()
    ):

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
# PRINT TRACK STATE
# ============================================================

def print_track_state(
    mapping,
    manager,
    camera,
    track_id,
):

    key = (
        camera,
        track_id,
    )

    gid = mapping.get(
        key
    )

    if gid is not None:

        print(
            f"c{camera}:{track_id}"
            f" -> GID {gid}"
        )

    elif key in manager.pending_tracklets:

        print(
            f"c{camera}:{track_id}"
            f" -> PENDING"
        )

    else:

        print(
            f"c{camera}:{track_id}"
            f" -> not assigned"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "GEOMETRY-ANCHORED EXPANSION EXPERIMENT"
    )
    print(
        "======================================"
    )

    print(
        "Bootstrap appearance minimum:",
        STRICT_APPEARANCE_MIN,
    )

    print(
        "Anchored expansion appearance minimum:",
        ANCHORED_EXPANSION_APPEARANCE_MIN,
    )

    print(
        "Geometry median maximum:",
        TRUSTED_GEOMETRY_MEDIAN_MAX,
    )


    items, lookup = load_data()

    homographies = (
        load_ground_homographies()
    )

    manager = GlobalIdentityManager(
        homographies=
            homographies
    )


    # GIDs with at least one trusted cross-camera geometry
    # anchor.
    anchored_gids = set()


    strict_existing_count = 0
    strict_pending_count = 0
    relaxed_existing_count = 0
    relaxed_sweep_count = 0

    appearance_strong_held = 0
    appearance_pending_held = 0
    new_gid_count = 0

    ambiguous_strict_existing = 0
    ambiguous_relaxed_existing = 0
    ambiguous_strict_pending = 0


    for tracklet, embedding in items:

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
        # FIRST TRACK
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

            new_gid_count += 1

            continue


        # ====================================================
        # PRIORITY 1:
        # STRICT GEOMETRY -> ANY EXISTING GID
        # ====================================================

        strict_existing = (
            find_existing_geometry_candidates(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
                anchored_gids,
                relaxed_anchored=False,
            )
        )

        if len(strict_existing) == 1:

            candidate = strict_existing[0]

            gid = candidate["gid"]

            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )

            anchored_gids.add(
                gid
            )

            strict_existing_count += 1

            print(
                "STRICT EXISTING | "
                f"c{camera}:{track_id}"
                f" -> GID {gid} | "
                f"topk={candidate['topk']:.4f} | "
                f"support="
                f"{support_text(candidate['support'])}"
            )

            swept = (
                sweep_pending_into_anchored_gids(
                    manager,
                    lookup,
                    homographies,
                    anchored_gids,
                )
            )

            for row in swept:

                relaxed_sweep_count += 1

                print(
                    "ANCHOR SWEEP | "
                    f"c{row['camera_id']}:"
                    f"{row['local_track_id']}"
                    f" -> GID {row['gid']} | "
                    f"topk={row['topk']:.4f} | "
                    f"support="
                    f"{support_text(row['support'])}"
                )

            continue


        elif len(strict_existing) > 1:

            ambiguous_strict_existing += 1


        # ====================================================
        # PRIORITY 2:
        # RELAXED GEOMETRY -> ALREADY ANCHORED GID
        # ====================================================

        relaxed_existing = (
            find_existing_geometry_candidates(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
                anchored_gids,
                relaxed_anchored=True,
            )
        )

        if len(relaxed_existing) == 1:

            candidate = relaxed_existing[0]

            gid = candidate["gid"]

            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )

            relaxed_existing_count += 1

            print(
                "ANCHORED EXPAND | "
                f"c{camera}:{track_id}"
                f" -> GID {gid} | "
                f"topk={candidate['topk']:.4f} | "
                f"support="
                f"{support_text(candidate['support'])}"
            )

            swept = (
                sweep_pending_into_anchored_gids(
                    manager,
                    lookup,
                    homographies,
                    anchored_gids,
                )
            )

            for row in swept:

                relaxed_sweep_count += 1

                print(
                    "ANCHOR SWEEP | "
                    f"c{row['camera_id']}:"
                    f"{row['local_track_id']}"
                    f" -> GID {row['gid']} | "
                    f"topk={row['topk']:.4f} | "
                    f"support="
                    f"{support_text(row['support'])}"
                )

            continue


        elif len(relaxed_existing) > 1:

            ambiguous_relaxed_existing += 1


        # ====================================================
        # PRIORITY 3:
        # STRICT CURRENT <-> PENDING PAIR
        # ====================================================

        pending_candidates = (
            find_strict_pending_candidates(
                manager,
                tracklet,
                embedding,
                lookup,
                homographies,
            )
        )

        if len(pending_candidates) == 1:

            pending_candidate = (
                pending_candidates[0]
            )

            gid = create_anchor_from_pending(
                manager,
                tracklet,
                embedding,
                pending_candidate,
            )

            anchored_gids.add(
                gid
            )

            strict_pending_count += 1

            evidence = pending_candidate[
                "evidence"
            ]

            print(
                "STRICT PENDING ANCHOR | "
                f"GID {gid} | "
                f"c{camera}:{track_id}"
                f" <-> "
                f"c{pending_candidate['camera_id']}:"
                f"{pending_candidate['local_track_id']} | "
                f"appearance="
                f"{pending_candidate['appearance']:.4f} | "
                f"median="
                f"{evidence.median_distance:.2f}"
            )

            swept = (
                sweep_pending_into_anchored_gids(
                    manager,
                    lookup,
                    homographies,
                    anchored_gids,
                )
            )

            for row in swept:

                relaxed_sweep_count += 1

                print(
                    "ANCHOR SWEEP | "
                    f"c{row['camera_id']}:"
                    f"{row['local_track_id']}"
                    f" -> GID {row['gid']} | "
                    f"topk={row['topk']:.4f} | "
                    f"support="
                    f"{support_text(row['support'])}"
                )

            continue


        elif len(pending_candidates) > 1:

            ambiguous_strict_pending += 1


        # ====================================================
        # PRIORITY 4:
        # APPEARANCE-ONLY DECISION
        #
        # Still NEVER merges.
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


        if match.status == "STRONG":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )

            appearance_strong_held += 1

            if key in WATCH_TRACKS:

                print(
                    "WATCH HOLD STRONG | "
                    f"c{camera}:{track_id}"
                )

            continue


        if match.status == "PENDING":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )

            appearance_pending_held += 1

            if key in WATCH_TRACKS:

                print(
                    "WATCH HOLD PENDING | "
                    f"c{camera}:{track_id}"
                )

            continue


        # ====================================================
        # CLEAR NEW
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

        new_gid_count += 1


    # ========================================================
    # FINAL ANCHOR SWEEP
    # ========================================================

    swept = sweep_pending_into_anchored_gids(
        manager,
        lookup,
        homographies,
        anchored_gids,
    )

    for row in swept:

        relaxed_sweep_count += 1

        print(
            "FINAL ANCHOR SWEEP | "
            f"c{row['camera_id']}:"
            f"{row['local_track_id']}"
            f" -> GID {row['gid']} | "
            f"topk={row['topk']:.4f}"
        )


    mapping = build_member_gid_lookup(
        manager
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("FINAL RESULT")
    print("============")

    print(
        "Tracklets:",
        len(items),
    )

    print(
        "Anchored GIDs:",
        len(
            anchored_gids
        ),
    )

    print(
        "Strict existing resolutions:",
        strict_existing_count,
    )

    print(
        "Strict pending anchors:",
        strict_pending_count,
    )

    print(
        "Relaxed anchored expansions:",
        relaxed_existing_count,
    )

    print(
        "Relaxed pending sweep resolutions:",
        relaxed_sweep_count,
    )

    print(
        "Appearance STRONG held:",
        appearance_strong_held,
    )

    print(
        "Appearance PENDING held:",
        appearance_pending_held,
    )

    print(
        "New singleton GIDs:",
        new_gid_count,
    )

    print(
        "Ambiguous strict existing:",
        ambiguous_strict_existing,
    )

    print(
        "Ambiguous relaxed existing:",
        ambiguous_relaxed_existing,
    )

    print(
        "Ambiguous strict pending:",
        ambiguous_strict_pending,
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
        (0, 255),
        (1, 298),
        (1, 319),

        (0, 399),
        (1, 369),
        (1, 382),

        (0, 469),
        (1, 503),
        (0, 600),

        (0, 492),
        (1, 511),

        (0, 431),
        (1, 423),

        (1, 560),
        (0, 651),

        (1, 639),
        (0, 701),
        (0, 740),
        (1, 721),

        (0, 1),
        (1, 1),
        (1, 66),
    ]

    for camera, track_id in important:

        print_track_state(
            mapping,
            manager,
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

    for identity in identities[:20]:

        print(
            identity.summary()
        )


if __name__ == "__main__":
    main()
