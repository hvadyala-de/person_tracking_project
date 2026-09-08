from dataclasses import dataclass

import numpy as np

import src.global_identity_manager as gim

from src.final_identity_consolidation import (
    build_current_state,
    normalize_camera_id,
)

from src.identity_compatibility import (
    same_camera_conflict,
)


# ============================================================
# CONSERVATIVE GLOBAL PHYSICAL-PERSON GRAPH
#
# IMPORTANT:
#
# This is DIAGNOSTIC ONLY.
#
# It does not modify GlobalIdentityManager.
# It does not force a target number of people.
# It includes:
#
#   * every assigned GID member
#   * every remaining pending tracklet
#
# Existing GIDs are treated as seed groups.
# Pending tracklets begin as singleton seed groups.
# ============================================================


# ============================================================
# CROSS-CAMERA MULTI-SUPPORT
# ============================================================

MULTIXCAM_REID_MIN = 0.84
MULTIXCAM_MEAN_MIN = 0.88
MULTIXCAM_PRIMARY_MIN = 0.90

MULTIXCAM_SHARED_MIN = 20
MULTIXCAM_MEDIAN_MAX = 18.0

MULTIXCAM_MIN_SUPPORTS = 2
MULTIXCAM_MIN_CAMERA_PAIRS = 2


# ============================================================
# LONG SIMULTANEOUS GEOMETRY CONSENSUS
#
# Already globally validated:
# GID 1 <-> GID 110 was the only pair that passed this
# pattern in the earlier 134-GID scan.
# ============================================================

GEOM_LONG_SHARED_MIN = 100
GEOM_LONG_MEDIAN_MAX = 20.0
GEOM_LONG_MEAN_REID_MIN = 0.80
GEOM_LONG_PRIMARY_REID_MIN = 0.83
GEOM_LONG_MIN_PRIMARY = 2
GEOM_LONG_MIN_SUPPORTS = 4
GEOM_LONG_MIN_CAMERAS_EACH_SIDE = 2


# ============================================================
# MULTI-CAMERA TEMPORAL CONTINUATION
#
# Handles cases like:
#
#   GID 78 -> 111 -> 113
# ============================================================

SEQ_REID_MIN = 0.86
SEQ_MEAN_MIN = 0.885
SEQ_PRIMARY_MIN = 0.90
SEQ_MAX_GAP = 250
SEQ_MIN_CAMERAS = 2


# ============================================================
# VERY STRONG SINGLE-CAMERA CONTINUATION
#
# This is deliberately much stricter than multi-camera
# temporal continuation.
#
# It is intended only for short tracker fragmentation.
# ============================================================

SINGLE_SAMECAM_REID_MIN = 0.94
SINGLE_SAMECAM_MAX_GAP = 75
SINGLE_SAMECAM_ENDPOINT_MAX = 60.0


# ============================================================
# VERY STRONG SINGLE CROSS-CAMERA SUPPORT
#
# Used especially for pending singleton fragments.
# ============================================================

SINGLE_XCAM_REID_MIN = 0.90
SINGLE_XCAM_SHARED_MIN = 30
SINGLE_XCAM_MEDIAN_MAX = 15.0


@dataclass
class Fragment:
    fragment_id: str

    owner_id: str

    camera_id: int

    local_track_id: int

    start_frame: int

    end_frame: int

    embedding: np.ndarray

    tracklet: object

    source: str

    global_id: int | None


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(
        self,
        values,
    ):

        self.parent = {
            value: value
            for value in values
        }


    def find(
        self,
        value,
    ):

        parent = self.parent[
            value
        ]

        if parent != value:

            self.parent[
                value
            ] = self.find(
                parent
            )

        return self.parent[
            value
        ]


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

        for values in result.values():

            values.sort()

        return result


# ============================================================
# BASIC HELPERS
# ============================================================

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

        return None

    return embedding / norm


def cosine_similarity(
    first,
    second,
):

    first = normalize_embedding(
        first
    )

    second = normalize_embedding(
        second
    )

    if (
        first is None
        or second is None
    ):

        return None

    return float(
        np.dot(
            first,
            second,
        )
    )


def bbox_diagonal(
    detection,
):

    width = float(
        detection.x2
        - detection.x1
    )

    height = float(
        detection.y2
        - detection.y1
    )

    return float(
        np.hypot(
            width,
            height,
        )
    )


def first_detection(
    tracklet,
):

    if (
        tracklet is None
        or not tracklet.detections
    ):

        return None

    return min(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def last_detection(
    tracklet,
):

    if (
        tracklet is None
        or not tracklet.detections
    ):

        return None

    return max(
        tracklet.detections,
        key=lambda detection:
            detection.frame,
    )


def center(
    detection,
):

    return np.asarray(
        [
            (
                detection.x1
                + detection.x2
            ) / 2.0,
            (
                detection.y1
                + detection.y2
            ) / 2.0,
        ],
        dtype=np.float32,
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
        dtype=np.float32,
    )


# ============================================================
# COLLECT EVERY ASSIGNED + PENDING FRAGMENT
# ============================================================

def collect_fragments(
    manager,
    tracklet_lookup,
):

    fragments = []

    owner_to_fragments = {}

    assigned_count = 0
    pending_count = 0

    missing_tracklet = 0
    missing_embedding = 0


    # ========================================================
    # ASSIGNED GID MEMBERS
    # ========================================================

    for gid in sorted(
        manager.identities
    ):

        identity = manager.identities[
            gid
        ]

        owner_id = (
            f"G{gid}"
        )

        owner_to_fragments.setdefault(
            owner_id,
            [],
        )

        for member in identity.members:

            camera_id = normalize_camera_id(
                member.camera_id
            )

            local_track_id = int(
                member.local_track_id
            )

            key = (
                camera_id,
                local_track_id,
            )

            embedding = (
                manager.member_embeddings.get(
                    key
                )
            )

            tracklet = (
                manager.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )

            if tracklet is None:

                missing_tracklet += 1
                continue

            if embedding is None:

                missing_embedding += 1
                continue

            fragment = Fragment(
                fragment_id=(
                    f"G{gid}:"
                    f"c{camera_id}:"
                    f"{local_track_id}"
                ),

                owner_id=
                    owner_id,

                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                start_frame=
                    int(
                        member.start_frame
                    ),

                end_frame=
                    int(
                        member.end_frame
                    ),

                embedding=
                    np.asarray(
                        embedding,
                        dtype=np.float32,
                    ),

                tracklet=
                    tracklet,

                source=
                    "ASSIGNED",

                global_id=
                    gid,
            )

            fragments.append(
                fragment
            )

            owner_to_fragments[
                owner_id
            ].append(
                fragment
            )

            assigned_count += 1


    # ========================================================
    # PENDING TRACKLETS
    # ========================================================

    for key, pending in sorted(
        manager.pending_tracklets.items(),
        key=lambda item: (
            normalize_camera_id(
                item[1][
                    "camera_id"
                ]
            ),
            int(
                item[1][
                    "local_track_id"
                ]
            ),
        ),
    ):

        camera_id = normalize_camera_id(
            pending[
                "camera_id"
            ]
        )

        local_track_id = int(
            pending[
                "local_track_id"
            ]
        )

        owner_id = (
            f"P:c{camera_id}:"
            f"{local_track_id}"
        )

        tracklet = pending.get(
            "candidate_tracklet"
        )

        if tracklet is None:

            tracklet = (
                manager.get_tracklet(
                    camera_id,
                    local_track_id,
                    tracklet_lookup,
                )
            )

        embedding = pending.get(
            "embedding"
        )

        if tracklet is None:

            missing_tracklet += 1
            continue

        if embedding is None:

            missing_embedding += 1
            continue

        fragment = Fragment(
            fragment_id=
                owner_id,

            owner_id=
                owner_id,

            camera_id=
                camera_id,

            local_track_id=
                local_track_id,

            start_frame=
                int(
                    pending[
                        "start_frame"
                    ]
                ),

            end_frame=
                int(
                    pending[
                        "end_frame"
                    ]
                ),

            embedding=
                np.asarray(
                    embedding,
                    dtype=np.float32,
                ),

            tracklet=
                tracklet,

            source=
                "PENDING",

            global_id=
                None,
        )

        fragments.append(
            fragment
        )

        owner_to_fragments[
            owner_id
        ] = [
            fragment
        ]

        pending_count += 1


    return {
        "fragments":
            fragments,

        "owner_to_fragments":
            owner_to_fragments,

        "assigned_count":
            assigned_count,

        "pending_count":
            pending_count,

        "missing_tracklet":
            missing_tracklet,

        "missing_embedding":
            missing_embedding,
    }


# ============================================================
# FRAGMENT-PAIR EVIDENCE
# ============================================================

def same_camera_evidence(
    first,
    second,
):

    conflict = same_camera_conflict(
        first.tracklet,
        second.tracklet,
    )

    if (
        first.start_frame
        <= second.start_frame
    ):

        earlier = first
        later = second

    else:

        earlier = second
        later = first


    gap = (
        later.start_frame
        - earlier.end_frame
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


    similarity = cosine_similarity(
        first.embedding,
        second.embedding,
    )


    end_detection = last_detection(
        earlier.tracklet
    )

    start_detection = first_detection(
        later.tracklet
    )


    center_distance = None
    bottom_distance = None
    endpoint_scale = None


    if (
        end_detection is not None
        and start_detection is not None
    ):

        center_distance = float(
            np.linalg.norm(
                center(
                    end_detection
                )
                -
                center(
                    start_detection
                )
            )
        )

        bottom_distance = float(
            np.linalg.norm(
                bottom_center(
                    end_detection
                )
                -
                bottom_center(
                    start_detection
                )
            )
        )

        endpoint_scale = max(
            bbox_diagonal(
                end_detection
            ),
            bbox_diagonal(
                start_detection
            ),
        )


    direction = (
        "FIRST_TO_SECOND"
        if first is earlier
        else
        "SECOND_TO_FIRST"
    )


    return {
        "type":
            "SAME_CAMERA",

        "conflict":
            bool(
                conflict
            ),

        "similarity":
            similarity,

        "gap":
            int(
                gap
            ),

        "overlap":
            int(
                overlap
            ),

        "center_distance":
            center_distance,

        "bottom_distance":
            bottom_distance,

        "endpoint_scale":
            endpoint_scale,

        "direction":
            direction,
    }


def cross_camera_evidence(
    manager,
    first,
    second,
):

    similarity = cosine_similarity(
        first.embedding,
        second.embedding,
    )

    geometry = (
        gim.evaluate_cross_camera_geometry(
            first.tracklet,
            second.tracklet,
            manager.homographies,
        )
    )


    return {
        "type":
            "CROSS_CAMERA",

        "similarity":
            similarity,

        "geometry_status":
            geometry.status,

        "shared_frames":
            int(
                geometry.shared_frames
            ),

        "median_distance":
            (
                None
                if geometry.median_distance
                is None
                else float(
                    geometry.median_distance
                )
            ),
    }


def fragment_pair_evidence(
    manager,
    first,
    second,
):

    if (
        first.camera_id
        == second.camera_id
    ):

        return same_camera_evidence(
            first,
            second,
        )


    return cross_camera_evidence(
        manager,
        first,
        second,
    )


# ============================================================
# OWNER-PAIR EVIDENCE
# ============================================================

def evaluate_owner_pair(
    manager,
    fragments_a,
    fragments_b,
):

    same_rows = []
    cross_rows = []

    hard_same_camera_conflicts = []
    hard_geometry_contradictions = []


    for first in fragments_a:

        for second in fragments_b:

            evidence = (
                fragment_pair_evidence(
                    manager,
                    first,
                    second,
                )
            )

            row = {
                "first":
                    first,

                "second":
                    second,

                "evidence":
                    evidence,
            }


            if (
                evidence[
                    "type"
                ]
                ==
                "SAME_CAMERA"
            ):

                same_rows.append(
                    row
                )

                if evidence[
                    "conflict"
                ]:

                    hard_same_camera_conflicts.append(
                        row
                    )

            else:

                cross_rows.append(
                    row
                )

                if (
                    evidence[
                        "geometry_status"
                    ]
                    ==
                    "CONTRADICTED"
                ):

                    hard_geometry_contradictions.append(
                        row
                    )


    if hard_same_camera_conflicts:

        return {
            "accepted":
                False,

            "hard_negative":
                True,

            "reason":
                "SAME_CAMERA_CONFLICT",

            "modes":
                [],
        }


    if hard_geometry_contradictions:

        return {
            "accepted":
                False,

            "hard_negative":
                True,

            "reason":
                "GEOMETRY_CONTRADICTION",

            "modes":
                [],
        }


    modes = []

    mode_details = {}


    # ========================================================
    # MODE 1:
    # LONG MULTI-CAMERA GEOMETRY CONSENSUS
    # ========================================================

    long_geo_rows = []


    for row in cross_rows:

        evidence = row[
            "evidence"
        ]

        if (
            evidence[
                "geometry_status"
            ]
            !=
            "SUPPORTED"
        ):

            continue

        if (
            evidence[
                "shared_frames"
            ]
            <
            GEOM_LONG_SHARED_MIN
        ):

            continue

        if (
            evidence[
                "median_distance"
            ]
            is None
        ):

            continue

        if (
            evidence[
                "median_distance"
            ]
            >
            GEOM_LONG_MEDIAN_MAX
        ):

            continue

        long_geo_rows.append(
            row
        )


    if long_geo_rows:

        similarities = [
            row[
                "evidence"
            ][
                "similarity"
            ]

            for row in long_geo_rows

            if (
                row[
                    "evidence"
                ][
                    "similarity"
                ]
                is not None
            )
        ]

        cameras_a = {
            row[
                "first"
            ].camera_id

            for row in long_geo_rows
        }

        cameras_b = {
            row[
                "second"
            ].camera_id

            for row in long_geo_rows
        }

        primary_count = sum(
            similarity
            >=
            GEOM_LONG_PRIMARY_REID_MIN

            for similarity in similarities
        )

        mean_similarity = (
            float(
                np.mean(
                    similarities
                )
            )

            if similarities

            else 0.0
        )


        if (
            len(
                long_geo_rows
            )
            >=
            GEOM_LONG_MIN_SUPPORTS

            and

            len(
                cameras_a
            )
            >=
            GEOM_LONG_MIN_CAMERAS_EACH_SIDE

            and

            len(
                cameras_b
            )
            >=
            GEOM_LONG_MIN_CAMERAS_EACH_SIDE

            and

            mean_similarity
            >=
            GEOM_LONG_MEAN_REID_MIN

            and

            primary_count
            >=
            GEOM_LONG_MIN_PRIMARY
        ):

            modes.append(
                "LONG_GEOMETRY_CONSENSUS"
            )

            mode_details[
                "LONG_GEOMETRY_CONSENSUS"
            ] = {
                "supports":
                    len(
                        long_geo_rows
                    ),

                "mean_similarity":
                    mean_similarity,

                "primary_count":
                    primary_count,
            }


    # ========================================================
    # MODE 2:
    # MULTI-CAMERA CROSS-CAMERA CONSENSUS
    # ========================================================

    multixcam_rows = []


    for row in cross_rows:

        evidence = row[
            "evidence"
        ]

        similarity = evidence[
            "similarity"
        ]

        if similarity is None:

            continue

        if (
            evidence[
                "geometry_status"
            ]
            !=
            "SUPPORTED"
        ):

            continue

        if (
            evidence[
                "shared_frames"
            ]
            <
            MULTIXCAM_SHARED_MIN
        ):

            continue

        if (
            evidence[
                "median_distance"
            ]
            is None
        ):

            continue

        if (
            evidence[
                "median_distance"
            ]
            >
            MULTIXCAM_MEDIAN_MAX
        ):

            continue

        if (
            similarity
            <
            MULTIXCAM_REID_MIN
        ):

            continue

        multixcam_rows.append(
            row
        )


    if multixcam_rows:

        similarities = [
            row[
                "evidence"
            ][
                "similarity"
            ]

            for row in multixcam_rows
        ]

        camera_pairs = {
            (
                row[
                    "first"
                ].camera_id,
                row[
                    "second"
                ].camera_id,
            )

            for row in multixcam_rows
        }

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


        if (
            len(
                multixcam_rows
            )
            >=
            MULTIXCAM_MIN_SUPPORTS

            and

            len(
                camera_pairs
            )
            >=
            MULTIXCAM_MIN_CAMERA_PAIRS

            and

            mean_similarity
            >=
            MULTIXCAM_MEAN_MIN

            and

            max_similarity
            >=
            MULTIXCAM_PRIMARY_MIN
        ):

            modes.append(
                "MULTICAM_XCAM"
            )

            mode_details[
                "MULTICAM_XCAM"
            ] = {
                "supports":
                    len(
                        multixcam_rows
                    ),

                "mean_similarity":
                    mean_similarity,

                "max_similarity":
                    max_similarity,
            }


    # ========================================================
    # MODE 3:
    # MULTI-CAMERA SEQUENTIAL CONTINUATION
    # ========================================================

    best_by_camera = {}


    for row in same_rows:

        evidence = row[
            "evidence"
        ]

        similarity = evidence[
            "similarity"
        ]

        if similarity is None:

            continue

        if (
            evidence[
                "overlap"
            ]
            > 0
        ):

            continue

        if (
            evidence[
                "gap"
            ]
            < 0
        ):

            continue

        if (
            evidence[
                "gap"
            ]
            >
            SEQ_MAX_GAP
        ):

            continue

        if (
            similarity
            <
            SEQ_REID_MIN
        ):

            continue


        camera_id = row[
            "first"
        ].camera_id


        if (
            row[
                "first"
            ].end_frame
            <
            row[
                "second"
            ].start_frame
        ):

            direction = (
                "A_TO_B"
            )

        elif (
            row[
                "second"
            ].end_frame
            <
            row[
                "first"
            ].start_frame
        ):

            direction = (
                "B_TO_A"
            )

        else:

            continue


        candidate = {
            "camera_id":
                camera_id,

            "similarity":
                similarity,

            "gap":
                evidence[
                    "gap"
                ],

            "direction":
                direction,

            "row":
                row,
        }


        current = (
            best_by_camera.get(
                camera_id
            )
        )


        if current is None:

            best_by_camera[
                camera_id
            ] = candidate

        else:

            candidate_key = (
                candidate[
                    "similarity"
                ],
                -candidate[
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
                    camera_id
                ] = candidate


    seq_rows = list(
        best_by_camera.values()
    )


    if seq_rows:

        directions = {
            row[
                "direction"
            ]

            for row in seq_rows
        }

        similarities = [
            row[
                "similarity"
            ]

            for row in seq_rows
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


        if (
            len(
                seq_rows
            )
            >=
            SEQ_MIN_CAMERAS

            and

            len(
                directions
            )
            == 1

            and

            mean_similarity
            >=
            SEQ_MEAN_MIN

            and

            max_similarity
            >=
            SEQ_PRIMARY_MIN
        ):

            modes.append(
                "MULTICAM_SEQUENTIAL"
            )

            mode_details[
                "MULTICAM_SEQUENTIAL"
            ] = {
                "supports":
                    len(
                        seq_rows
                    ),

                "mean_similarity":
                    mean_similarity,

                "max_similarity":
                    max_similarity,

                "direction":
                    next(
                        iter(
                            directions
                        )
                    ),
            }


    # ========================================================
    # MODE 4:
    # VERY STRONG SINGLE-CAMERA SHORT CONTINUATION
    # ========================================================

    strong_samecam = []


    for row in same_rows:

        evidence = row[
            "evidence"
        ]

        similarity = evidence[
            "similarity"
        ]

        if similarity is None:

            continue

        if (
            evidence[
                "overlap"
            ]
            > 0
        ):

            continue

        if (
            evidence[
                "gap"
            ]
            < 0
            or
            evidence[
                "gap"
            ]
            >
            SINGLE_SAMECAM_MAX_GAP
        ):

            continue

        if (
            similarity
            <
            SINGLE_SAMECAM_REID_MIN
        ):

            continue


        distances = [
            value

            for value in (
                evidence[
                    "center_distance"
                ],
                evidence[
                    "bottom_distance"
                ],
            )

            if value is not None
        ]


        if not distances:

            continue


        if (
            min(
                distances
            )
            >
            SINGLE_SAMECAM_ENDPOINT_MAX
        ):

            continue


        strong_samecam.append(
            row
        )


    if strong_samecam:

        best = max(
            strong_samecam,
            key=lambda row: (
                row[
                    "evidence"
                ][
                    "similarity"
                ],
                -row[
                    "evidence"
                ][
                    "gap"
                ],
            ),
        )

        modes.append(
            "STRONG_SAMECAM"
        )

        mode_details[
            "STRONG_SAMECAM"
        ] = {
            "similarity":
                best[
                    "evidence"
                ][
                    "similarity"
                ],

            "gap":
                best[
                    "evidence"
                ][
                    "gap"
                ],

            "endpoint_distance":
                min(
                    value

                    for value in (
                        best[
                            "evidence"
                        ][
                            "center_distance"
                        ],
                        best[
                            "evidence"
                        ][
                            "bottom_distance"
                        ],
                    )

                    if value is not None
                ),
        }


    # ========================================================
    # MODE 5:
    # VERY STRONG SINGLE CROSS-CAMERA ANCHOR
    # ========================================================

    strong_xcam = []


    for row in cross_rows:

        evidence = row[
            "evidence"
        ]

        similarity = evidence[
            "similarity"
        ]

        if similarity is None:

            continue

        if (
            evidence[
                "geometry_status"
            ]
            !=
            "SUPPORTED"
        ):

            continue

        if (
            evidence[
                "shared_frames"
            ]
            <
            SINGLE_XCAM_SHARED_MIN
        ):

            continue

        if (
            evidence[
                "median_distance"
            ]
            is None
            or
            evidence[
                "median_distance"
            ]
            >
            SINGLE_XCAM_MEDIAN_MAX
        ):

            continue

        if (
            similarity
            <
            SINGLE_XCAM_REID_MIN
        ):

            continue

        strong_xcam.append(
            row
        )


    if strong_xcam:

        best = max(
            strong_xcam,
            key=lambda row:
                row[
                    "evidence"
                ][
                    "similarity"
                ],
        )

        modes.append(
            "STRONG_XCAM"
        )

        mode_details[
            "STRONG_XCAM"
        ] = {
            "similarity":
                best[
                    "evidence"
                ][
                    "similarity"
                ],

            "shared_frames":
                best[
                    "evidence"
                ][
                    "shared_frames"
                ],

            "median_distance":
                best[
                    "evidence"
                ][
                    "median_distance"
                ],
        }


    return {
        "accepted":
            bool(
                modes
            ),

        "hard_negative":
            False,

        "reason":
            (
                "ACCEPTED"
                if modes
                else
                "NO_POSITIVE_EVIDENCE"
            ),

        "modes":
            modes,

        "details":
            mode_details,
    }


# ============================================================
# COMPONENT SAFETY
# ============================================================

def component_compatible(
    component_a,
    component_b,
    pair_results,
):

    for owner_a in component_a:

        for owner_b in component_b:

            pair_key = tuple(
                sorted(
                    (
                        owner_a,
                        owner_b,
                    )
                )
            )

            result = pair_results.get(
                pair_key
            )

            if result is None:

                continue

            if result.get(
                "hard_negative",
                False,
            ):

                return (
                    False,
                    (
                        f"{owner_a}<->{owner_b} "
                        f"{result['reason']}"
                    ),
                )


    return (
        True,
        None,
    )


# ============================================================
# EDGE PRIORITY
# ============================================================

def edge_priority(
    edge,
):

    modes = edge[
        "result"
    ][
        "modes"
    ]


    if (
        "LONG_GEOMETRY_CONSENSUS"
        in modes
    ):

        priority = 5

    elif (
        "MULTICAM_XCAM"
        in modes
        or
        "MULTICAM_SEQUENTIAL"
        in modes
    ):

        priority = 4

    elif (
        "STRONG_XCAM"
        in modes
    ):

        priority = 3

    elif (
        "STRONG_SAMECAM"
        in modes
    ):

        priority = 2

    else:

        priority = 1


    return (
        -priority,
        edge[
            "owner_a"
        ],
        edge[
            "owner_b"
        ],
    )


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(
    manager,
    owner_to_fragments,
):

    owners = sorted(
        owner_to_fragments
    )

    pair_results = {}

    accepted_edges = []


    total_pairs = (
        len(
            owners
        )
        *
        (
            len(
                owners
            )
            - 1
        )
        //
        2
    )


    checked = 0


    print()
    print(
        "SCANNING ALL SEED-OWNER PAIRS"
    )

    print(
        "============================="
    )

    print(
        "Seed owners:",
        len(
            owners
        ),
    )

    print(
        "Owner pairs:",
        total_pairs,
    )


    for index, owner_a in enumerate(
        owners
    ):

        fragments_a = (
            owner_to_fragments[
                owner_a
            ]
        )

        for owner_b in owners[
            index + 1:
        ]:

            checked += 1

            fragments_b = (
                owner_to_fragments[
                    owner_b
                ]
            )

            result = evaluate_owner_pair(
                manager,
                fragments_a,
                fragments_b,
            )

            pair_key = (
                owner_a,
                owner_b,
            )

            pair_results[
                pair_key
            ] = result


            if result[
                "accepted"
            ]:

                accepted_edges.append(
                    {
                        "owner_a":
                            owner_a,

                        "owner_b":
                            owner_b,

                        "result":
                            result,
                    }
                )


        if (
            index % 25 == 0
            or
            index
            ==
            len(
                owners
            )
            - 1
        ):

            print(
                f"pair scan "
                f"{checked}/"
                f"{total_pairs}"
                f" | positive edges="
                f"{len(accepted_edges)}"
            )


    accepted_edges.sort(
        key=edge_priority
    )


    return (
        owners,
        pair_results,
        accepted_edges,
    )


# ============================================================
# SAFE CLUSTERING
# ============================================================

def cluster_owners(
    owners,
    pair_results,
    accepted_edges,
):

    union_find = UnionFind(
        owners
    )

    applied_edges = []

    blocked_edges = []


    for edge in accepted_edges:

        owner_a = edge[
            "owner_a"
        ]

        owner_b = edge[
            "owner_b"
        ]


        root_a = union_find.find(
            owner_a
        )

        root_b = union_find.find(
            owner_b
        )


        if root_a == root_b:

            continue


        groups = union_find.groups()


        component_a = groups[
            root_a
        ]

        component_b = groups[
            root_b
        ]


        compatible, reason = (
            component_compatible(
                component_a,
                component_b,
                pair_results,
            )
        )


        if not compatible:

            blocked_edges.append(
                {
                    "owner_a":
                        owner_a,

                    "owner_b":
                        owner_b,

                    "reason":
                        reason,
                }
            )

            continue


        union_find.union(
            owner_a,
            owner_b,
        )

        applied_edges.append(
            edge
        )


    return {
        "groups":
            union_find.groups(),

        "applied_edges":
            applied_edges,

        "blocked_edges":
            blocked_edges,
    }


# ============================================================
# REGRESSION HELPERS
# ============================================================

def owner_component_map(
    groups,
):

    result = {}

    for root, owners in (
        groups.items()
    ):

        for owner in owners:

            result[
                owner
            ] = root

    return result


def same_component(
    component_map,
    owners,
):

    roots = {
        component_map.get(
            owner
        )

        for owner in owners
    }

    return (
        None not in roots
        and
        len(
            roots
        )
        == 1
    )


# ============================================================
# REPORT
# ============================================================

def main():

    print()
    print(
        "FULL PHYSICAL-IDENTITY GRAPH DIAGNOSTIC"
    )

    print(
        "======================================="
    )

    print(
        "This includes assigned AND pending fragments."
    )

    print(
        "No production identities are modified."
    )


    (
        manager,
        tracklet_lookup,
    ) = build_current_state()


    population = collect_fragments(
        manager,
        tracklet_lookup,
    )


    owner_to_fragments = (
        population[
            "owner_to_fragments"
        ]
    )


    print()
    print(
        "FRAGMENT POPULATION"
    )

    print(
        "==================="
    )

    print(
        "Existing GIDs:",
        len(
            manager.identities
        ),
    )

    print(
        "Remaining pending:",
        len(
            manager.pending_tracklets
        ),
    )

    print(
        "Assigned fragments:",
        population[
            "assigned_count"
        ],
    )

    print(
        "Pending fragments:",
        population[
            "pending_count"
        ],
    )

    print(
        "Total usable fragments:",
        len(
            population[
                "fragments"
            ]
        ),
    )

    print(
        "Seed owners:",
        len(
            owner_to_fragments
        ),
    )

    print(
        "Missing tracklets:",
        population[
            "missing_tracklet"
        ],
    )

    print(
        "Missing embeddings:",
        population[
            "missing_embedding"
        ],
    )


    camera_counts = {}

    for fragment in population[
        "fragments"
    ]:

        camera_counts.setdefault(
            fragment.camera_id,
            0,
        )

        camera_counts[
            fragment.camera_id
        ] += 1


    print()
    print(
        "FRAGMENTS BY CAMERA"
    )

    print(
        "==================="
    )

    for camera_id in sorted(
        camera_counts
    ):

        print(
            f"c{camera_id}: "
            f"{camera_counts[camera_id]}"
        )


    (
        owners,
        pair_results,
        accepted_edges,
    ) = build_graph(
        manager,
        owner_to_fragments,
    )


    print()
    print(
        "POSITIVE GRAPH EDGES"
    )

    print(
        "===================="
    )

    print(
        "Count:",
        len(
            accepted_edges
        ),
    )


    mode_counts = {}

    for edge in accepted_edges:

        for mode in edge[
            "result"
        ][
            "modes"
        ]:

            mode_counts.setdefault(
                mode,
                0,
            )

            mode_counts[
                mode
            ] += 1


    for mode in sorted(
        mode_counts
    ):

        print(
            f"{mode}: "
            f"{mode_counts[mode]}"
        )


    print()
    print(
        "FIRST 80 POSITIVE EDGES"
    )

    print(
        "======================="
    )


    for edge in accepted_edges[
        :80
    ]:

        marker = ""


        pair = {
            edge[
                "owner_a"
            ],
            edge[
                "owner_b"
            ],
        }


        if pair == {
            "G1",
            "G110",
        }:

            marker = (
                "  <== KNOWN SAME PERSON"
            )


        if pair in (
            {
                "G78",
                "G111",
            },

            {
                "G111",
                "G113",
            },
        ):

            marker = (
                "  <== KNOWN SAME PERSON"
            )


        print(
            f"{edge['owner_a']:14s}"
            f" <-> "
            f"{edge['owner_b']:14s}"
            f" | "
            f"{'+'.join(edge['result']['modes'])}"
            f"{marker}"
        )


    clustering = cluster_owners(
        owners,
        pair_results,
        accepted_edges,
    )


    groups = clustering[
        "groups"
    ]


    component_map = owner_component_map(
        groups
    )


    merged_groups = [
        owners_in_group

        for owners_in_group in groups.values()

        if len(
            owners_in_group
        )
        > 1
    ]


    gid_only_components = set()

    pending_absorbed = 0
    pending_only_groups = 0


    for root, owners_in_group in (
        groups.items()
    ):

        has_gid = any(
            owner.startswith(
                "G"
            )

            for owner in owners_in_group
        )

        has_pending = any(
            owner.startswith(
                "P:"
            )

            for owner in owners_in_group
        )


        if has_gid:

            gid_only_components.add(
                root
            )


        if (
            has_gid
            and has_pending
        ):

            pending_absorbed += sum(
                owner.startswith(
                    "P:"
                )

                for owner in owners_in_group
            )


        if (
            has_pending
            and not has_gid
        ):

            pending_only_groups += 1


    print()
    print(
        "CONSERVATIVE CLUSTER RESULT"
    )

    print(
        "==========================="
    )

    print(
        "Initial seed owners:",
        len(
            owners
        ),
    )

    print(
        "Final physical-person components:",
        len(
            groups
        ),
    )

    print(
        "Merged components:",
        len(
            merged_groups
        ),
    )

    print(
        "GID-containing components:",
        len(
            gid_only_components
        ),
    )

    print(
        "Pending fragments absorbed into GID components:",
        pending_absorbed,
    )

    print(
        "Pending-only components:",
        pending_only_groups,
    )

    print(
        "Applied edges:",
        len(
            clustering[
                "applied_edges"
            ]
        ),
    )

    print(
        "Blocked by component safety:",
        len(
            clustering[
                "blocked_edges"
            ]
        ),
    )


    print()
    print(
        "LARGEST MERGED COMPONENTS"
    )

    print(
        "========================="
    )


    largest = sorted(
        merged_groups,
        key=lambda values: (
            -len(
                values
            ),
            values,
        ),
    )


    for owners_in_group in largest[
        :30
    ]:

        print(
            f"size={len(owners_in_group):2d} | "
            +
            ", ".join(
                owners_in_group
            )
        )


    print()
    print(
        "KNOWN REGRESSION CHECKS"
    )

    print(
        "======================="
    )


    checks = [
        (
            same_component(
                component_map,
                [
                    "G1",
                    "G110",
                ],
            ),

            "GID 1 + GID 110 become one person",
        ),

        (
            same_component(
                component_map,
                [
                    "G78",
                    "G111",
                    "G113",
                ],
            ),

            "GID 78 + 111 + 113 become one person",
        ),

        (
            not same_component(
                component_map,
                [
                    "G111",
                    "P:c3:15",
                ],
            ),

            "c3:15 remains different from GID 111",
        ),
    ]


    passed = 0


    for success, label in checks:

        if success:

            passed += 1

        print(
            (
                "PASS"
                if success
                else
                "FAIL"
            ),
            "|",
            label,
        )


    print()
    print(
        f"{passed}/{len(checks)} "
        "known checks passed"
    )


    if passed != len(
        checks
    ):

        raise RuntimeError(
            "Known physical-identity "
            "regression check failed"
        )


    print()
    print(
        "PHYSICAL-IDENTITY GRAPH DIAGNOSTIC: PASS"
    )


if __name__ == "__main__":
    main()
