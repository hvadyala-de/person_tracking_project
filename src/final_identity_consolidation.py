import argparse
import csv
from pathlib import Path

import numpy as np

import src.experiments.analyze_c3_gap_identity_changes as reference
import src.global_identity_manager as gim
import src.experiments.run_c3_production_policy_experiment as c3prod

from src.identity_compatibility import (
    same_camera_conflict,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "output"
    / "presentation"
)


# ============================================================
# CURRENT VALIDATED FOUR-CAMERA REPLAY EXPECTATIONS
# ============================================================

EXPECTED_C0_C1_GIDS = 77
EXPECTED_C0_C1_PENDING = 91
EXPECTED_C0_C1_ANCHORED = 22

EXPECTED_C012_GIDS = 109
EXPECTED_C012_PENDING = 162
EXPECTED_C012_ANCHORED = 22

EXPECTED_C2_ASSOCIATIONS = {
    "c2:9a": 1,
    "c2:9b": 11,
    "c2:508": 66,
    "c2:601b": 74,
}


# ============================================================
# MODE A:
# STRICT APPEARANCE + GEOMETRY SIMULTANEOUS CONSOLIDATION
# ============================================================

SIMULTANEOUS_REID_MIN = 0.84

SIMULTANEOUS_MEAN_MIN = 0.88

SIMULTANEOUS_PRIMARY_MIN = 0.90

SIMULTANEOUS_MIN_SUPPORTS = 2

SIMULTANEOUS_MIN_CAMERA_PAIRS = 2


# ============================================================
# MODE B:
# STRONG MULTI-CAMERA GEOMETRY CONSENSUS
#
# Designed for viewpoint-heavy cases where ReID is weaker,
# but multiple cameras independently place both GIDs at the
# same physical ground-plane location for a long time.
#
# Full 134-GID global scan produced exactly one candidate:
#
#   GID 1 <-> GID 110
#
# Known visual-positive case.
# ============================================================

GEOMETRY_CONSENSUS_MIN_SHARED = 100

GEOMETRY_CONSENSUS_MEDIAN_MAX = 20.0

GEOMETRY_CONSENSUS_MEAN_REID_MIN = 0.80

GEOMETRY_CONSENSUS_PRIMARY_REID_MIN = 0.83

GEOMETRY_CONSENSUS_MIN_PRIMARY = 2

GEOMETRY_CONSENSUS_MIN_SUPPORTS = 4

GEOMETRY_CONSENSUS_MIN_CAMERAS_A = 2

GEOMETRY_CONSENSUS_MIN_CAMERAS_B = 2


# ============================================================
# MODE C:
# MULTI-CAMERA TEMPORAL / REAPPEARANCE CONSOLIDATION
#
# Known visual-positive chain:
#
#   GID 78 -> GID 111 -> GID 113
#
# Same person fragmented after tracker loss / occlusion.
# ============================================================

SEQUENTIAL_REID_MIN = 0.86

SEQUENTIAL_MEAN_MIN = 0.885

SEQUENTIAL_PRIMARY_MIN = 0.90

SEQUENTIAL_MAX_GAP = 250

SEQUENTIAL_MIN_CAMERAS = 2


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_camera_id(
    camera_id,
):

    if isinstance(
        camera_id,
        str,
    ):

        camera_id = (
            camera_id
            .strip()
            .lower()
        )

        if camera_id.startswith(
            "c"
        ):

            camera_id = (
                camera_id[1:]
            )

    return int(
        camera_id
    )


def member_key(
    member,
):

    return (
        normalize_camera_id(
            member.camera_id
        ),
        int(
            member.local_track_id
        ),
    )


def member_label(
    member,
):

    return (
        f"c{normalize_camera_id(member.camera_id)}:"
        f"{int(member.local_track_id)}"
    )


def normalize_embedding(
    embedding,
):

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    norm = float(
        np.linalg.norm(
            embedding
        )
    )

    if norm <= 0.0:

        raise ValueError(
            "Zero-norm embedding"
        )

    return (
        embedding
        / norm
    )


def cosine_similarity(
    a,
    b,
):

    a = normalize_embedding(
        a
    )

    b = normalize_embedding(
        b
    )

    return float(
        np.dot(
            a,
            b,
        )
    )


def get_member_embedding(
    manager,
    member,
):

    return (
        manager.member_embeddings.get(
            member_key(
                member
            )
        )
    )


def get_member_tracklet(
    manager,
    member,
    tracklet_lookup,
):

    return (
        manager.get_member_tracklet(
            member,
            tracklet_lookup,
        )
    )


def member_is_core(
    manager,
    gid,
    member,
):

    if (
        gid
        not in manager.anchored_gids
    ):

        return False


    return (
        member_key(
            member
        )
        in manager.core_members[
            gid
        ]
    )


# ============================================================
# BUILD CURRENT VALIDATED FOUR-CAMERA STATE
# ============================================================

def build_current_state():

    reference.EXPECTED_C0_C1_GIDS = (
        EXPECTED_C0_C1_GIDS
    )

    reference.EXPECTED_C0_C1_PENDING = (
        EXPECTED_C0_C1_PENDING
    )

    reference.EXPECTED_C0_C1_ANCHORED = (
        EXPECTED_C0_C1_ANCHORED
    )


    reference.EXPECTED_FINAL_GIDS = (
        EXPECTED_C012_GIDS
    )

    reference.EXPECTED_FINAL_PENDING = (
        EXPECTED_C012_PENDING
    )

    reference.EXPECTED_FINAL_ANCHORED = (
        EXPECTED_C012_ANCHORED
    )


    reference.EXPECTED_C2_ASSOCIATIONS = dict(
        EXPECTED_C2_ASSOCIATIONS
    )


    (
        manager,
        tracklet_lookup,
    ) = (
        reference
        .build_validated_reference_state()
    )


    c3_rows = (
        c3prod.load_c3_rows()
    )


    population = (
        c3prod.build_c3_population(
            c3_rows
        )
    )


    collisions = (
        c3prod.register_population_tracklets(
            population=
                population,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if collisions:

        raise RuntimeError(
            "Unexpected c3 lookup collisions: "
            f"{collisions}"
        )


    c3prod.process_c3_population(
        manager=
            manager,

        population=
            population,

        tracklet_lookup=
            tracklet_lookup,
    )


    manager.reevaluate_pending(
        tracklet_lookup=
            tracklet_lookup,

        include_same_camera=
            True,

        include_dual_evidence=
            True,

        include_core_gallery=
            True,
    )


    return (
        manager,
        tracklet_lookup,
    )


# ============================================================
# SAME-CAMERA CONFLICT BETWEEN TWO GIDS
#
# ALL members participate in negative evidence.
# ============================================================

def identity_pair_same_camera_conflict(
    manager,
    identity_a,
    identity_b,
    tracklet_lookup,
):

    rows = []


    for member_a in identity_a.members:

        camera_a = (
            normalize_camera_id(
                member_a.camera_id
            )
        )


        for member_b in identity_b.members:

            camera_b = (
                normalize_camera_id(
                    member_b.camera_id
                )
            )


            if camera_a != camera_b:

                continue


            tracklet_a = (
                get_member_tracklet(
                    manager,
                    member_a,
                    tracklet_lookup,
                )
            )

            tracklet_b = (
                get_member_tracklet(
                    manager,
                    member_b,
                    tracklet_lookup,
                )
            )


            if (
                tracklet_a is None
                or tracklet_b is None
            ):

                continue


            if same_camera_conflict(
                tracklet_a,
                tracklet_b,
            ):

                rows.append(
                    {
                        "member_a":
                            member_label(
                                member_a
                            ),

                        "member_b":
                            member_label(
                                member_b
                            ),
                    }
                )


    return rows


# ============================================================
# CROSS-CAMERA GEOMETRY CONTRADICTIONS
#
# ALL members participate in negative evidence.
# ============================================================

def identity_pair_geometry_contradictions(
    manager,
    identity_a,
    identity_b,
    tracklet_lookup,
):

    contradictions = []

    seen = set()


    for member_a in identity_a.members:

        tracklet_a = (
            get_member_tracklet(
                manager,
                member_a,
                tracklet_lookup,
            )
        )


        if tracklet_a is None:

            continue


        rows = (
            manager
            .identity_geometry_contradictions(
                candidate_tracklet=
                    tracklet_a,

                identity=
                    identity_b,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


        for row in rows:

            pair = (
                member_label(
                    member_a
                ),
                (
                    f"c{int(row['camera_id'])}:"
                    f"{int(row['local_track_id'])}"
                ),
            )


            canonical = tuple(
                sorted(
                    pair
                )
            )


            if canonical in seen:

                continue


            seen.add(
                canonical
            )


            contradictions.append(
                {
                    "member_a":
                        pair[0],

                    "member_b":
                        pair[1],

                    "shared_frames":
                        row[
                            "shared_frames"
                        ],

                    "median_distance":
                        row[
                            "median_distance"
                        ],
                }
            )


    return contradictions


# ============================================================
# MODE A SUPPORT:
# STRICT APPEARANCE + STRICT GEOMETRY
# ============================================================

def simultaneous_support_rows(
    manager,
    gid_a,
    gid_b,
    tracklet_lookup,
):

    identity_a = (
        manager.identities[
            gid_a
        ]
    )

    identity_b = (
        manager.identities[
            gid_b
        ]
    )


    rows = []

    seen = set()


    for member_a in identity_a.members:

        if not member_is_core(
            manager,
            gid_a,
            member_a,
        ):

            continue


        tracklet_a = (
            get_member_tracklet(
                manager,
                member_a,
                tracklet_lookup,
            )
        )

        embedding_a = (
            get_member_embedding(
                manager,
                member_a,
            )
        )


        if (
            tracklet_a is None
            or embedding_a is None
        ):

            continue


        support_rows = (
            manager.strict_support_rows(
                global_id=
                    gid_b,

                identity=
                    identity_b,

                candidate_tracklet=
                    tracklet_a,

                embedding=
                    embedding_a,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


        for row in support_rows:

            similarity = (
                row[
                    "direct_similarity"
                ]
            )


            if similarity is None:

                continue


            if (
                similarity
                < SIMULTANEOUS_REID_MIN
            ):

                continue


            member_b_key = (
                int(
                    row[
                        "camera_id"
                    ]
                ),
                int(
                    row[
                        "local_track_id"
                    ]
                ),
            )


            if (
                member_b_key
                not in manager.core_members[
                    gid_b
                ]
            ):

                continue


            key_a = (
                member_key(
                    member_a
                )
            )


            pair_key = (
                key_a,
                member_b_key,
            )


            if pair_key in seen:

                continue


            seen.add(
                pair_key
            )


            rows.append(
                {
                    "camera_a":
                        key_a[0],

                    "track_a":
                        key_a[1],

                    "camera_b":
                        member_b_key[0],

                    "track_b":
                        member_b_key[1],

                    "direct_similarity":
                        float(
                            similarity
                        ),

                    "shared_frames":
                        int(
                            row[
                                "shared_frames"
                            ]
                        ),

                    "median_distance":
                        float(
                            row[
                                "median_distance"
                            ]
                        ),
                }
            )


    return rows


def simultaneous_mode(
    manager,
    gid_a,
    gid_b,
    tracklet_lookup,
):

    rows = (
        simultaneous_support_rows(
            manager=
                manager,

            gid_a=
                gid_a,

            gid_b=
                gid_b,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if not rows:

        return {
            "accepted":
                False,

            "rows":
                [],

            "mean_similarity":
                None,

            "max_similarity":
                None,

            "camera_pairs":
                set(),
        }


    similarities = [
        row[
            "direct_similarity"
        ]

        for row in rows
    ]


    mean_similarity = float(
        np.mean(
            similarities
        )
    )

    max_similarity = float(
        np.max(
            similarities
        )
    )


    camera_pairs = {
        (
            row[
                "camera_a"
            ],
            row[
                "camera_b"
            ],
        )

        for row in rows
    }


    accepted = (
        len(
            rows
        )
        >= SIMULTANEOUS_MIN_SUPPORTS

        and

        len(
            camera_pairs
        )
        >= SIMULTANEOUS_MIN_CAMERA_PAIRS

        and

        mean_similarity
        >= SIMULTANEOUS_MEAN_MIN

        and

        max_similarity
        >= SIMULTANEOUS_PRIMARY_MIN
    )


    return {
        "accepted":
            accepted,

        "rows":
            rows,

        "mean_similarity":
            mean_similarity,

        "max_similarity":
            max_similarity,

        "camera_pairs":
            camera_pairs,
    }


# ============================================================
# MODE B SUPPORT:
# STRONG MULTI-CAMERA GEOMETRY CONSENSUS
# ============================================================

def geometry_consensus_support_rows(
    manager,
    gid_a,
    gid_b,
    tracklet_lookup,
):

    identity_a = (
        manager.identities[
            gid_a
        ]
    )

    identity_b = (
        manager.identities[
            gid_b
        ]
    )


    rows = []


    for member_a in identity_a.members:

        if not member_is_core(
            manager,
            gid_a,
            member_a,
        ):

            continue


        key_a = (
            member_key(
                member_a
            )
        )


        tracklet_a = (
            get_member_tracklet(
                manager,
                member_a,
                tracklet_lookup,
            )
        )

        embedding_a = (
            get_member_embedding(
                manager,
                member_a,
            )
        )


        if (
            tracklet_a is None
            or embedding_a is None
        ):

            continue


        for member_b in identity_b.members:

            if not member_is_core(
                manager,
                gid_b,
                member_b,
            ):

                continue


            key_b = (
                member_key(
                    member_b
                )
            )


            if (
                key_a[0]
                == key_b[0]
            ):

                continue


            tracklet_b = (
                get_member_tracklet(
                    manager,
                    member_b,
                    tracklet_lookup,
                )
            )

            embedding_b = (
                get_member_embedding(
                    manager,
                    member_b,
                )
            )


            if (
                tracklet_b is None
                or embedding_b is None
            ):

                continue


            evidence = (
                gim.evaluate_cross_camera_geometry(
                    tracklet_a,
                    tracklet_b,
                    manager.homographies,
                )
            )


            if (
                evidence.status
                != "SUPPORTED"
            ):

                continue


            if (
                evidence.shared_frames
                < GEOMETRY_CONSENSUS_MIN_SHARED
            ):

                continue


            if (
                evidence.median_distance
                is None
            ):

                continue


            if (
                evidence.median_distance
                > GEOMETRY_CONSENSUS_MEDIAN_MAX
            ):

                continue


            similarity = (
                manager.cosine_similarity(
                    embedding_a,
                    embedding_b,
                )
            )


            rows.append(
                {
                    "camera_a":
                        key_a[0],

                    "track_a":
                        key_a[1],

                    "camera_b":
                        key_b[0],

                    "track_b":
                        key_b[1],

                    "direct_similarity":
                        float(
                            similarity
                        ),

                    "shared_frames":
                        int(
                            evidence.shared_frames
                        ),

                    "median_distance":
                        float(
                            evidence.median_distance
                        ),
                }
            )


    return rows


def geometry_consensus_mode(
    manager,
    gid_a,
    gid_b,
    tracklet_lookup,
):

    rows = (
        geometry_consensus_support_rows(
            manager=
                manager,

            gid_a=
                gid_a,

            gid_b=
                gid_b,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if not rows:

        return {
            "accepted":
                False,

            "rows":
                [],

            "mean_similarity":
                None,

            "primary_count":
                0,

            "cameras_a":
                set(),

            "cameras_b":
                set(),
        }


    similarities = [
        row[
            "direct_similarity"
        ]

        for row in rows
    ]


    mean_similarity = float(
        np.mean(
            similarities
        )
    )


    primary_count = sum(
        similarity
        >= GEOMETRY_CONSENSUS_PRIMARY_REID_MIN

        for similarity in similarities
    )


    cameras_a = {
        row[
            "camera_a"
        ]

        for row in rows
    }


    cameras_b = {
        row[
            "camera_b"
        ]

        for row in rows
    }


    accepted = (
        len(
            rows
        )
        >= GEOMETRY_CONSENSUS_MIN_SUPPORTS

        and

        len(
            cameras_a
        )
        >= GEOMETRY_CONSENSUS_MIN_CAMERAS_A

        and

        len(
            cameras_b
        )
        >= GEOMETRY_CONSENSUS_MIN_CAMERAS_B

        and

        mean_similarity
        >= GEOMETRY_CONSENSUS_MEAN_REID_MIN

        and

        primary_count
        >= GEOMETRY_CONSENSUS_MIN_PRIMARY
    )


    return {
        "accepted":
            accepted,

        "rows":
            rows,

        "mean_similarity":
            mean_similarity,

        "primary_count":
            primary_count,

        "cameras_a":
            cameras_a,

        "cameras_b":
            cameras_b,
    }


# ============================================================
# MODE C:
# MULTI-CAMERA TEMPORAL CONTINUATION
# ============================================================

def temporal_relation(
    member_a,
    member_b,
):

    a_start = int(
        member_a.start_frame
    )

    a_end = int(
        member_a.end_frame
    )

    b_start = int(
        member_b.start_frame
    )

    b_end = int(
        member_b.end_frame
    )


    if a_end < b_start:

        return {
            "overlap":
                False,

            "gap":
                (
                    b_start
                    - a_end
                    - 1
                ),

            "direction":
                "A_TO_B",
        }


    if b_end < a_start:

        return {
            "overlap":
                False,

            "gap":
                (
                    a_start
                    - b_end
                    - 1
                ),

            "direction":
                "B_TO_A",
        }


    return {
        "overlap":
            True,

        "gap":
            0,

        "direction":
            "OVERLAP",
    }


def sequential_support_rows(
    manager,
    gid_a,
    gid_b,
):

    identity_a = (
        manager.identities[
            gid_a
        ]
    )

    identity_b = (
        manager.identities[
            gid_b
        ]
    )


    best_by_camera = {}


    for member_a in identity_a.members:

        if not member_is_core(
            manager,
            gid_a,
            member_a,
        ):

            continue


        camera_a = (
            normalize_camera_id(
                member_a.camera_id
            )
        )


        embedding_a = (
            get_member_embedding(
                manager,
                member_a,
            )
        )


        if embedding_a is None:

            continue


        for member_b in identity_b.members:

            if not member_is_core(
                manager,
                gid_b,
                member_b,
            ):

                continue


            camera_b = (
                normalize_camera_id(
                    member_b.camera_id
                )
            )


            if camera_a != camera_b:

                continue


            embedding_b = (
                get_member_embedding(
                    manager,
                    member_b,
                )
            )


            if embedding_b is None:

                continue


            relation = (
                temporal_relation(
                    member_a,
                    member_b,
                )
            )


            if relation[
                "overlap"
            ]:

                continue


            gap = int(
                relation[
                    "gap"
                ]
            )


            if (
                gap
                > SEQUENTIAL_MAX_GAP
            ):

                continue


            similarity = (
                cosine_similarity(
                    embedding_a,
                    embedding_b,
                )
            )


            if (
                similarity
                < SEQUENTIAL_REID_MIN
            ):

                continue


            row = {
                "camera_id":
                    camera_a,

                "member_a":
                    member_label(
                        member_a
                    ),

                "member_b":
                    member_label(
                        member_b
                    ),

                "similarity":
                    float(
                        similarity
                    ),

                "gap":
                    gap,

                "direction":
                    relation[
                        "direction"
                    ],
            }


            current = (
                best_by_camera.get(
                    camera_a
                )
            )


            if current is None:

                best_by_camera[
                    camera_a
                ] = row

                continue


            candidate_key = (
                row[
                    "similarity"
                ],
                -row[
                    "gap"
                ],
            )

            current_key = (
                current[
                    "similarity"
                ],
                -current[
                    "gap"
                ],
            )


            if candidate_key > current_key:

                best_by_camera[
                    camera_a
                ] = row


    return [
        best_by_camera[
            camera_id
        ]

        for camera_id in sorted(
            best_by_camera
        )
    ]


def sequential_mode(
    manager,
    gid_a,
    gid_b,
):

    rows = (
        sequential_support_rows(
            manager=
                manager,

            gid_a=
                gid_a,

            gid_b=
                gid_b,
        )
    )


    if not rows:

        return {
            "accepted":
                False,

            "rows":
                [],

            "mean_similarity":
                None,

            "max_similarity":
                None,

            "direction":
                None,
        }


    directions = {
        row[
            "direction"
        ]

        for row in rows
    }


    similarities = [
        row[
            "similarity"
        ]

        for row in rows
    ]


    mean_similarity = float(
        np.mean(
            similarities
        )
    )


    max_similarity = float(
        np.max(
            similarities
        )
    )


    consistent_direction = (
        len(
            directions
        )
        == 1
    )


    accepted = (
        len(
            rows
        )
        >= SEQUENTIAL_MIN_CAMERAS

        and

        consistent_direction

        and

        mean_similarity
        >= SEQUENTIAL_MEAN_MIN

        and

        max_similarity
        >= SEQUENTIAL_PRIMARY_MIN
    )


    direction = None


    if consistent_direction:

        direction = next(
            iter(
                directions
            )
        )


    return {
        "accepted":
            accepted,

        "rows":
            rows,

        "mean_similarity":
            mean_similarity,

        "max_similarity":
            max_similarity,

        "direction":
            direction,
    }


# ============================================================
# COMPLETE GID-PAIR EVALUATION
# ============================================================

def evaluate_gid_pair(
    manager,
    gid_a,
    gid_b,
    tracklet_lookup,
):

    identity_a = (
        manager.identities[
            gid_a
        ]
    )

    identity_b = (
        manager.identities[
            gid_b
        ]
    )


    anchored = (
        gid_a
        in manager.anchored_gids

        and

        gid_b
        in manager.anchored_gids
    )


    if not anchored:

        return {
            "accepted":
                False,

            "gid_a":
                gid_a,

            "gid_b":
                gid_b,

            "reason":
                "UNANCHORED",
        }


    same_camera_conflicts = (
        identity_pair_same_camera_conflict(
            manager=
                manager,

            identity_a=
                identity_a,

            identity_b=
                identity_b,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if same_camera_conflicts:

        return {
            "accepted":
                False,

            "gid_a":
                gid_a,

            "gid_b":
                gid_b,

            "reason":
                "SAME_CAMERA_CONFLICT",

            "same_camera_conflicts":
                same_camera_conflicts,
        }


    geometry_contradictions = (
        identity_pair_geometry_contradictions(
            manager=
                manager,

            identity_a=
                identity_a,

            identity_b=
                identity_b,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    if geometry_contradictions:

        return {
            "accepted":
                False,

            "gid_a":
                gid_a,

            "gid_b":
                gid_b,

            "reason":
                "GEOMETRY_CONTRADICTION",

            "geometry_contradictions":
                geometry_contradictions,
        }


    simultaneous = (
        simultaneous_mode(
            manager=
                manager,

            gid_a=
                gid_a,

            gid_b=
                gid_b,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    geometry_consensus = (
        geometry_consensus_mode(
            manager=
                manager,

            gid_a=
                gid_a,

            gid_b=
                gid_b,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    sequential = (
        sequential_mode(
            manager=
                manager,

            gid_a=
                gid_a,

            gid_b=
                gid_b,
        )
    )


    modes = []


    if simultaneous[
        "accepted"
    ]:

        modes.append(
            "SIMULTANEOUS"
        )


    if geometry_consensus[
        "accepted"
    ]:

        modes.append(
            "GEOMETRY_CONSENSUS"
        )


    if sequential[
        "accepted"
    ]:

        modes.append(
            "SEQUENTIAL"
        )


    accepted = bool(
        modes
    )


    return {
        "accepted":
            accepted,

        "gid_a":
            gid_a,

        "gid_b":
            gid_b,

        "reason":
            (
                "ACCEPTED"
                if accepted
                else
                "NO_SUFFICIENT_POSITIVE_EVIDENCE"
            ),

        "mode":
            (
                "+".join(
                    modes
                )
                if modes
                else None
            ),

        "simultaneous":
            simultaneous,

        "geometry_consensus":
            geometry_consensus,

        "sequential":
            sequential,

        "same_camera_conflicts":
            same_camera_conflicts,

        "geometry_contradictions":
            geometry_contradictions,
    }


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(
        self,
        values,
    ):

        self.parent = {
            value:
                value

            for value in values
        }


    def find(
        self,
        value,
    ):

        parent = (
            self.parent[
                value
            ]
        )


        if parent != value:

            self.parent[
                value
            ] = self.find(
                parent
            )


        return (
            self.parent[
                value
            ]
        )


    def union(
        self,
        a,
        b,
    ):

        root_a = self.find(
            a
        )

        root_b = self.find(
            b
        )


        if root_a == root_b:

            return


        if root_a < root_b:

            self.parent[
                root_b
            ] = root_a

        else:

            self.parent[
                root_a
            ] = root_b


    def groups(
        self,
    ):

        result = {}


        for value in self.parent:

            root = self.find(
                value
            )


            result.setdefault(
                root,
                [],
            ).append(
                value
            )


        for values in (
            result.values()
        ):

            values.sort()


        return result


# ============================================================
# COMPONENT SAFETY
# ============================================================

def components_compatible(
    manager,
    component_a,
    component_b,
    tracklet_lookup,
):

    for gid_a in component_a:

        identity_a = (
            manager.identities[
                gid_a
            ]
        )


        for gid_b in component_b:

            identity_b = (
                manager.identities[
                    gid_b
                ]
            )


            conflicts = (
                identity_pair_same_camera_conflict(
                    manager=
                        manager,

                    identity_a=
                        identity_a,

                    identity_b=
                        identity_b,

                    tracklet_lookup=
                        tracklet_lookup,
                )
            )


            if conflicts:

                return (
                    False,
                    (
                        "SAME_CAMERA_CONFLICT "
                        f"GID {gid_a}<->{gid_b}"
                    ),
                )


            contradictions = (
                identity_pair_geometry_contradictions(
                    manager=
                        manager,

                    identity_a=
                        identity_a,

                    identity_b=
                        identity_b,

                    tracklet_lookup=
                        tracklet_lookup,
                )
            )


            if contradictions:

                return (
                    False,
                    (
                        "GEOMETRY_CONTRADICTION "
                        f"GID {gid_a}<->{gid_b}"
                    ),
                )


    return (
        True,
        None,
    )


# ============================================================
# EDGE STRENGTH
# ============================================================

def edge_sort_key(
    row,
):

    modes = (
        row[
            "mode"
        ]
        or ""
    )


    if (
        "GEOMETRY_CONSENSUS"
        in modes
    ):

        mode_priority = 3

    elif (
        "SIMULTANEOUS"
        in modes
    ):

        mode_priority = 2

    else:

        mode_priority = 1


    values = []


    for support in (
        row[
            "simultaneous"
        ][
            "rows"
        ]
    ):

        values.append(
            support[
                "direct_similarity"
            ]
        )


    for support in (
        row[
            "geometry_consensus"
        ][
            "rows"
        ]
    ):

        values.append(
            support[
                "direct_similarity"
            ]
        )


    for support in (
        row[
            "sequential"
        ][
            "rows"
        ]
    ):

        values.append(
            support[
                "similarity"
            ]
        )


    mean_score = (
        float(
            np.mean(
                values
            )
        )

        if values

        else -1.0
    )


    return (
        -mode_priority,
        -len(
            values
        ),
        -mean_score,
        row[
            "gid_a"
        ],
        row[
            "gid_b"
        ],
    )


# ============================================================
# ALL-GID CONSOLIDATION SCAN
# ============================================================

def build_consolidation(
    manager,
    tracklet_lookup,
):

    gids = sorted(
        manager.identities
    )


    print()
    print(
        "SCANNING ALL FINAL GID PAIRS"
    )

    print(
        "============================"
    )


    total_pairs = (
        len(
            gids
        )
        * (
            len(
                gids
            )
            - 1
        )
        // 2
    )


    print(
        "GIDs:",
        len(
            gids
        ),
    )

    print(
        "Pairs:",
        total_pairs,
    )


    accepted_edges = []

    checked = 0


    for index, gid_a in enumerate(
        gids
    ):

        for gid_b in gids[
            index + 1:
        ]:

            checked += 1


            result = evaluate_gid_pair(
                manager=
                    manager,

                gid_a=
                    gid_a,

                gid_b=
                    gid_b,

                tracklet_lookup=
                    tracklet_lookup,
            )


            if result[
                "accepted"
            ]:

                accepted_edges.append(
                    result
                )


        if (
            index % 20 == 0

            or

            index
            == len(
                gids
            )
            - 1
        ):

            print(
                f"Pair scan: "
                f"{checked}/"
                f"{total_pairs}"
                f" | candidates="
                f"{len(accepted_edges)}"
            )


    accepted_edges.sort(
        key=edge_sort_key
    )


    print()
    print(
        "RAW ACCEPTED PAIR EDGES"
    )

    print(
        "======================="
    )

    print(
        "Count:",
        len(
            accepted_edges
        ),
    )


    for row in accepted_edges:

        marker = ""


        pair = {
            row[
                "gid_a"
            ],
            row[
                "gid_b"
            ],
        }


        if pair == {
            1,
            110,
        }:

            marker = (
                "  <== KNOWN SAME PERSON"
            )


        if pair in (
            {
                78,
                111,
            },

            {
                111,
                113,
            },
        ):

            marker = (
                "  <== KNOWN SAME PERSON"
            )


        geo = (
            row[
                "geometry_consensus"
            ]
        )


        seq = (
            row[
                "sequential"
            ]
        )


        geo_text = "-"


        if geo[
            "rows"
        ]:

            geo_text = (
                f"{len(geo['rows'])}"
                f"/"
                f"{geo['mean_similarity']:.3f}"
                f"/primary"
                f"{geo['primary_count']}"
            )


        seq_text = "-"


        if seq[
            "rows"
        ]:

            seq_text = (
                f"{len(seq['rows'])}"
                f"/"
                f"{seq['mean_similarity']:.3f}"
                f"/"
                f"{seq['direction']}"
            )


        print(
            f"GID "
            f"{row['gid_a']:3d}"
            f" <-> "
            f"{row['gid_b']:3d}"
            f" | mode="
            f"{row['mode']}"
            f" | geo="
            f"{geo_text}"
            f" | seq="
            f"{seq_text}"
            f"{marker}"
        )


    union_find = (
        UnionFind(
            gids
        )
    )


    applied_edges = []

    blocked_component_edges = []


    for edge in accepted_edges:

        gid_a = (
            edge[
                "gid_a"
            ]
        )

        gid_b = (
            edge[
                "gid_b"
            ]
        )


        groups = (
            union_find.groups()
        )


        root_a = (
            union_find.find(
                gid_a
            )
        )

        root_b = (
            union_find.find(
                gid_b
            )
        )


        if root_a == root_b:

            continue


        component_a = (
            groups[
                root_a
            ]
        )

        component_b = (
            groups[
                root_b
            ]
        )


        (
            compatible,
            block_reason,
        ) = (
            components_compatible(
                manager=
                    manager,

                component_a=
                    component_a,

                component_b=
                    component_b,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


        if not compatible:

            blocked_component_edges.append(
                {
                    "gid_a":
                        gid_a,

                    "gid_b":
                        gid_b,

                    "reason":
                        block_reason,
                }
            )

            continue


        union_find.union(
            gid_a,
            gid_b,
        )


        applied_edges.append(
            edge
        )


    return {
        "accepted_edges":
            accepted_edges,

        "applied_edges":
            applied_edges,

        "blocked_component_edges":
            blocked_component_edges,

        "groups":
            union_find.groups(),
    }


# ============================================================
# FINAL COMPONENT VALIDATION
# ============================================================

def validate_groups(
    manager,
    groups,
    tracklet_lookup,
):

    failures = []


    for gids in (
        groups.values()
    ):

        if len(
            gids
        ) < 2:

            continue


        for index, gid_a in enumerate(
            gids
        ):

            identity_a = (
                manager.identities[
                    gid_a
                ]
            )


            for gid_b in gids[
                index + 1:
            ]:

                identity_b = (
                    manager.identities[
                        gid_b
                    ]
                )


                conflicts = (
                    identity_pair_same_camera_conflict(
                        manager=
                            manager,

                        identity_a=
                            identity_a,

                        identity_b=
                            identity_b,

                        tracklet_lookup=
                            tracklet_lookup,
                    )
                )


                contradictions = (
                    identity_pair_geometry_contradictions(
                        manager=
                            manager,

                        identity_a=
                            identity_a,

                        identity_b=
                            identity_b,

                        tracklet_lookup=
                            tracklet_lookup,
                    )
                )


                if conflicts:

                    failures.append(
                        (
                            gid_a,
                            gid_b,
                            "SAME_CAMERA_CONFLICT",
                        )
                    )


                if contradictions:

                    failures.append(
                        (
                            gid_a,
                            gid_b,
                            "GEOMETRY_CONTRADICTION",
                        )
                    )


    return failures


# ============================================================
# COMPACT PERSON IDS
# ============================================================

def group_first_frame(
    manager,
    gids,
):

    frames = []


    for gid in gids:

        identity = (
            manager.identities[
                gid
            ]
        )


        for member in (
            identity.members
        ):

            frames.append(
                int(
                    member.start_frame
                )
            )


    if not frames:

        return 10**9


    return min(
        frames
    )


def build_person_mapping(
    manager,
    groups,
):

    rows = []


    for gids in (
        groups.values()
    ):

        rows.append(
            {
                "gids":
                    sorted(
                        gids
                    ),

                "first_frame":
                    group_first_frame(
                        manager,
                        gids,
                    ),
            }
        )


    rows.sort(
        key=lambda row: (
            row[
                "first_frame"
            ],

            row[
                "gids"
            ][0],
        )
    )


    gid_to_person = {}

    person_groups = {}


    for person_id, row in enumerate(
        rows,
        start=1,
    ):

        gids = (
            row[
                "gids"
            ]
        )


        person_groups[
            person_id
        ] = gids


        for gid in gids:

            gid_to_person[
                gid
            ] = person_id


    return (
        gid_to_person,
        person_groups,
    )


# ============================================================
# OUTPUT CSV
# ============================================================

def write_gid_mapping_csv(
    output_path,
    gid_to_person,
):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )


        writer.writerow(
            [
                "internal_gid",
                "person_id",
            ]
        )


        for gid in sorted(
            gid_to_person
        ):

            writer.writerow(
                [
                    gid,
                    gid_to_person[
                        gid
                    ],
                ]
            )


def write_member_mapping_csv(
    output_path,
    manager,
    gid_to_person,
):

    rows = []


    for gid, identity in (
        manager.identities.items()
    ):

        person_id = (
            gid_to_person[
                gid
            ]
        )


        for member in (
            identity.members
        ):

            rows.append(
                {
                    "person_id":
                        person_id,

                    "internal_gid":
                        gid,

                    "camera_id":
                        normalize_camera_id(
                            member.camera_id
                        ),

                    "local_track_id":
                        int(
                            member.local_track_id
                        ),

                    "start_frame":
                        int(
                            member.start_frame
                        ),

                    "end_frame":
                        int(
                            member.end_frame
                        ),

                    "trust":
                        manager.get_member_trust(
                            gid,
                            member.camera_id,
                            member.local_track_id,
                        ),
                }
            )


    rows.sort(
        key=lambda row: (
            row[
                "person_id"
            ],
            row[
                "camera_id"
            ],
            row[
                "start_frame"
            ],
            row[
                "local_track_id"
            ],
        )
    )


    fieldnames = [
        "person_id",
        "internal_gid",
        "camera_id",
        "local_track_id",
        "start_frame",
        "end_frame",
        "trust",
    ]


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = (
            csv.DictWriter(
                handle,
                fieldnames=
                    fieldnames,
            )
        )


        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# REGRESSION HELPERS
# ============================================================

def same_group(
    gid_to_person,
    gids,
):

    values = {
        gid_to_person.get(
            gid
        )

        for gid in gids
    }


    return (
        None not in values

        and

        len(
            values
        )
        == 1
    )


def pending_key_exists(
    manager,
    camera_id,
    local_track_id,
):

    key = (
        int(
            camera_id
        ),
        int(
            local_track_id
        ),
    )


    if (
        key
        in manager.pending_tracklets
    ):

        return True


    alt_key = (
        f"c{int(camera_id)}",
        int(
            local_track_id
        ),
    )


    return (
        alt_key
        in manager.pending_tracklets
    )


# ============================================================
# REPORT
# ============================================================

def print_final_groups(
    manager,
    person_groups,
):

    merged_groups = [
        (
            person_id,
            gids,
        )

        for person_id, gids
        in person_groups.items()

        if len(
            gids
        ) > 1
    ]


    print()
    print(
        "FINAL PHYSICAL-PERSON CONSOLIDATION"
    )

    print(
        "==================================="
    )

    print(
        "Original internal GIDs:",
        len(
            manager.identities
        ),
    )

    print(
        "Final Person IDs:",
        len(
            person_groups
        ),
    )

    print(
        "Merged groups:",
        len(
            merged_groups
        ),
    )


    print()
    print(
        "MERGED PERSON GROUPS"
    )

    print(
        "===================="
    )


    for person_id, gids in (
        merged_groups
    ):

        marker = ""


        if (
            1 in gids
            and
            110 in gids
        ):

            marker += (
                "  <== SAME PERSON "
                "SIMULTANEOUS FOUR-CAMERA"
            )


        if (
            78 in gids
            and
            111 in gids
            and
            113 in gids
        ):

            marker += (
                "  <== SAME PERSON REAPPEARANCE"
            )


        print(
            f"Person "
            f"{person_id:3d}"
            f" <- GIDs "
            f"{gids}"
            f"{marker}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Final all-camera physical-person "
            "identity consolidation"
        )
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=
            DEFAULT_OUTPUT_DIR,
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print(
        "FINAL IDENTITY CONSOLIDATION"
    )

    print(
        "============================"
    )

    print()
    print(
        "Scanning ALL final GID pairs."
    )

    print(
        "No production identity is modified."
    )

    print(
        "No hard-coded merge is performed."
    )


    (
        manager,
        tracklet_lookup,
    ) = build_current_state()


    print()
    print(
        "VALIDATED INPUT STATE"
    )

    print(
        "====================="
    )

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

    print(
        "Anchored:",
        len(
            manager.anchored_gids
        ),
    )


    consolidation = (
        build_consolidation(
            manager=
                manager,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    groups = (
        consolidation[
            "groups"
        ]
    )


    validation_failures = (
        validate_groups(
            manager=
                manager,

            groups=
                groups,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    (
        gid_to_person,
        person_groups,
    ) = (
        build_person_mapping(
            manager=
                manager,

            groups=
                groups,
        )
    )


    print_final_groups(
        manager=
            manager,

        person_groups=
            person_groups,
    )


    print()
    print(
        "COMPONENT-LEVEL BLOCKED EDGES"
    )

    print(
        "============================="
    )


    blocked = (
        consolidation[
            "blocked_component_edges"
        ]
    )


    print(
        "Count:",
        len(
            blocked
        ),
    )


    for row in blocked[
        :40
    ]:

        print(
            f"GID "
            f"{row['gid_a']}"
            f" <-> "
            f"{row['gid_b']}"
            f" | "
            f"{row['reason']}"
        )


    print()
    print(
        "FINAL GROUP SAFETY"
    )

    print(
        "=================="
    )


    if validation_failures:

        print(
            "FAIL"
        )


        for failure in (
            validation_failures
        ):

            print(
                " ",
                failure,
            )

    else:

        print(
            "PASS | no same-camera conflict "
            "inside any consolidated person"
        )

        print(
            "PASS | no geometry contradiction "
            "inside any consolidated person"
        )


    checks = []


    checks.append(
        (
            same_group(
                gid_to_person,
                [
                    1,
                    110,
                ],
            ),

            (
                "GID 1 and GID 110 consolidate "
                "to one physical person"
            ),
        )
    )


    checks.append(
        (
            same_group(
                gid_to_person,
                [
                    78,
                    111,
                    113,
                ],
            ),

            (
                "GID 78/111/113 consolidate "
                "to one physical person"
            ),
        )
    )


    checks.append(
        (
            pending_key_exists(
                manager,
                3,
                15,
            ),

            (
                "c3:15 remains unresolved/pending "
                "rather than merging with c3:12"
            ),
        )
    )


    print()
    print(
        "KNOWN REGRESSION CHECKS"
    )

    print(
        "======================="
    )


    passed_count = 0


    for passed, label in checks:

        if passed:

            passed_count += 1


        print(
            (
                "PASS"
                if passed
                else "FAIL"
            ),
            "|",
            label,
        )


    print()
    print(
        f"{passed_count}/"
        f"{len(checks)} "
        "known checks passed"
    )


    gid_mapping_path = (
        output_dir
        /
        "terrace_gid_to_person.csv"
    )


    member_mapping_path = (
        output_dir
        /
        "terrace_person_members.csv"
    )


    write_gid_mapping_csv(
        output_path=
            gid_mapping_path,

        gid_to_person=
            gid_to_person,
    )


    write_member_mapping_csv(
        output_path=
            member_mapping_path,

        manager=
            manager,

        gid_to_person=
            gid_to_person,
    )


    print()
    print(
        "OUTPUT"
    )

    print(
        "======"
    )

    print(
        gid_mapping_path
    )

    print(
        member_mapping_path
    )


    if validation_failures:

        raise RuntimeError(
            "Consolidated groups failed "
            "safety validation"
        )


    if (
        passed_count
        != len(
            checks
        )
    ):

        raise RuntimeError(
            "Known identity-consolidation "
            "regression checks failed"
        )


    print()
    print(
        "FINAL IDENTITY CONSOLIDATION: PASS"
    )


if __name__ == "__main__":
    main()
