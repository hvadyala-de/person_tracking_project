from pathlib import Path

import cv2

from src.experiments.analyze_final_pending_population import (
    replay_final_production,
)
from src.experiments.analyze_c2_601_segment_gid_ranking import (
    build_segment_embedding,
    build_segment_tracklet,
)
from src.experiments.analyze_c2_gid_associations import (
    load_c2_tracklets,
)
from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)
from src.identity_compatibility import (
    normalize_camera_id,
)
from src.reid_extractor import (
    ReIDExtractor,
)


# ============================================================
# PURPOSE
#
# Audit why promising c2 -> existing GID associations pass or
# fail the CURRENT production STRICT / RELAXED policy.
#
# Production thresholds are NOT modified.
#
# We use the two already-validated splits directly:
#
#   c2:9
#       455-1321  -> c2:9a
#       1357-1997 -> c2:9b
#
#   c2:601
#       3796-4016 -> c2:601a
#       4049-4571 -> c2:601b
#
# No gap-discovery analysis is repeated.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c2.avi"
)


# ============================================================
# VALIDATED SPLITS
# ============================================================

SPLIT_SPECS = {
    9: [
        {
            "label": "c2:9a",
            "start": 455,
            "end": 1321,
            "synthetic_id": 990091,
        },
        {
            "label": "c2:9b",
            "start": 1357,
            "end": 1997,
            "synthetic_id": 990092,
        },
    ],

    601: [
        {
            "label": "c2:601a",
            "start": 3796,
            "end": 4016,
            "synthetic_id": 9906011,
        },
        {
            "label": "c2:601b",
            "start": 4049,
            "end": 4571,
            "synthetic_id": 9906012,
        },
    ],
}


# ============================================================
# PROMISING ASSOCIATIONS FOUND BY THE BROADER DIAGNOSTIC
#
# These are candidates to audit, not ground truth.
# ============================================================

TARGET_GIDS = {
    "c2:1": 1,
    "c2:9a": 1,
    "c2:53": 4,
    "c2:9b": 11,
    "c2:153": 11,
    "c2:371": 52,
    "c2:513": 63,
    "c2:508": 65,
    "c2:601b": 73,
    "c2:701": 73,
}


# ============================================================
# HELPERS
# ============================================================

def tracklet_key(tracklet):

    return (
        normalize_camera_id(
            tracklet.camera_id
        ),
        tracklet.local_track_id,
    )


def member_key(member):

    return (
        normalize_camera_id(
            member.camera_id
        ),
        member.local_track_id,
    )


def fmt(value, digits=4):

    if value is None:
        return "None"

    return f"{value:.{digits}f}"


def status_text(value):

    if value is None:
        return "None"

    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


def find_member_gid(
    manager,
    key,
):

    for gid, identity in manager.identities.items():

        for member in identity.members:

            if member_key(member) == key:
                return gid

    return None


# ============================================================
# SNAPSHOT EXISTING c0+c1 MEMBERS
# ============================================================

def snapshot_members(manager):

    snapshot = {}

    for gid, identity in manager.identities.items():

        for member in identity.members:

            key = member_key(member)

            snapshot[key] = (
                gid,
                manager.get_member_trust(
                    gid,
                    key[0],
                    key[1],
                ),
            )

    return snapshot


def audit_snapshot(
    manager,
    snapshot,
):

    changes = []

    for key, old_value in snapshot.items():

        gid = find_member_gid(
            manager,
            key,
        )

        trust = None

        if gid is not None:

            trust = manager.get_member_trust(
                gid,
                key[0],
                key[1],
            )

        new_value = (
            gid,
            trust,
        )

        if new_value != old_value:

            changes.append(
                (
                    key,
                    old_value,
                    new_value,
                )
            )

    return changes


# ============================================================
# BUILD THE 107-ITEM IDENTITY-AWARE c2 POPULATION
#
# Only four segment embeddings are recomputed.
# ============================================================

def build_population(
    c2_rows,
):

    source_by_id = {
        tracklet.local_track_id: (
            tracklet,
            embedding,
        )
        for tracklet, embedding
        in c2_rows
    }

    split_items = {}

    extractor = ReIDExtractor(
        device="cuda"
    )

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Could not open {VIDEO_PATH}"
        )

    try:

        for source_id, specs in SPLIT_SPECS.items():

            source_tracklet, _ = (
                source_by_id[source_id]
            )

            for spec in specs:

                segment = build_segment_tracklet(
                    source_tracklet=
                        source_tracklet,

                    start_frame=
                        spec["start"],

                    end_frame=
                        spec["end"],

                    synthetic_track_id=
                        spec["synthetic_id"],
                )

                (
                    embedding,
                    embedding_frames,
                ) = build_segment_embedding(
                    extractor=
                        extractor,

                    video=
                        video,

                    tracklet=
                        segment,
                )

                split_items[
                    spec["label"]
                ] = {
                    "label":
                        spec["label"],

                    "tracklet":
                        segment,

                    "embedding":
                        embedding,

                    "embedding_frames":
                        embedding_frames,
                }

    finally:

        video.release()


    population = []

    for tracklet, embedding in c2_rows:

        track_id = (
            tracklet.local_track_id
        )

        if track_id in SPLIT_SPECS:

            for spec in SPLIT_SPECS[
                track_id
            ]:

                population.append(
                    split_items[
                        spec["label"]
                    ]
                )

            continue

        population.append(
            {
                "label":
                    f"c2:{track_id}",

                "tracklet":
                    tracklet,

                "embedding":
                    embedding,

                "embedding_frames":
                    None,
            }
        )


    population.sort(
        key=lambda item: (
            item["tracklet"].start_frame,
            item["tracklet"].end_frame,
            item["label"],
        )
    )

    return population


# ============================================================
# REGISTER SPLIT TRACKLETS IN LOOKUP
# ============================================================

def register_population(
    population,
    lookup,
):

    for item in population:

        tracklet = item[
            "tracklet"
        ]

        lookup[
            tracklet_key(tracklet)
        ] = tracklet


# ============================================================
# RAW CORE-MEMBER EVIDENCE
#
# This lets us see why relaxed_core_support_rows() is empty.
# ============================================================

def core_member_evidence(
    manager,
    gid,
    candidate_tracklet,
    candidate_embedding,
    lookup,
):

    identity = manager.identities[
        gid
    ]

    candidate_camera = (
        normalize_camera_id(
            candidate_tracklet.camera_id
        )
    )

    rows = []

    for member in identity.members:

        key = member_key(
            member
        )

        if (
            key
            not in manager.core_members[
                gid
            ]
        ):
            continue

        camera_id = key[0]

        member_embedding = (
            manager.member_embeddings.get(
                key
            )
        )

        direct_similarity = None

        if member_embedding is not None:

            direct_similarity = (
                manager.cosine_similarity(
                    candidate_embedding,
                    member_embedding,
                )
            )

        if camera_id == candidate_camera:

            rows.append(
                {
                    "camera_id":
                        camera_id,

                    "track_id":
                        member.local_track_id,

                    "reid":
                        direct_similarity,

                    "same_camera":
                        True,

                    "status":
                        None,

                    "shared":
                        None,

                    "median":
                        None,
                }
            )

            continue


        member_tracklet = (
            manager.get_member_tracklet(
                member,
                lookup,
            )
        )

        if member_tracklet is None:

            continue


        geometry = (
            evaluate_cross_camera_geometry(
                candidate_tracklet,
                member_tracklet,
                manager.homographies,
            )
        )


        rows.append(
            {
                "camera_id":
                    camera_id,

                "track_id":
                    member.local_track_id,

                "reid":
                    direct_similarity,

                "same_camera":
                    False,

                "status":
                    status_text(
                        geometry.status
                    ),

                "shared":
                    geometry.shared_frames,

                "median":
                    geometry.median_distance,
            }
        )

    return rows


# ============================================================
# AUDIT ONE EXPECTED GID
# ============================================================

def audit_expected_gid(
    manager,
    item,
    gid,
    lookup,
):

    tracklet = item[
        "tracklet"
    ]

    embedding = item[
        "embedding"
    ]

    identity = manager.identities[
        gid
    ]


    scores = manager.score_identity(
        identity,
        embedding,
    )


    strict_support = (
        manager.strict_support_rows(
            gid,
            identity,
            tracklet,
            embedding,
            lookup,
        )
    )


    relaxed_support = (
        manager.relaxed_core_support_rows(
            gid,
            identity,
            tracklet,
            embedding,
            lookup,
        )
    )


    contradicted = (
        manager.identity_geometry_contradicted(
            tracklet,
            identity,
            lookup,
        )
    )


    strict_candidates = (
        manager.find_strict_existing_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )
    )


    relaxed_candidates = (
        manager.find_relaxed_core_candidates(
            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                lookup,
        )
    )


    fallback = manager.evaluate(
        camera_id=
            normalize_camera_id(
                tracklet.camera_id
            ),

        embedding=
            embedding,

        candidate_tracklet=
            tracklet,

        tracklet_lookup=
            lookup,
    )


    return {
        "gid":
            gid,

        "anchored":
            gid in manager.anchored_gids,

        "scores":
            scores,

        "strict_support":
            strict_support,

        "relaxed_support":
            relaxed_support,

        "contradicted":
            contradicted,

        "strict_candidate_gids": [
            row["global_id"]
            for row
            in strict_candidates
        ],

        "relaxed_candidate_gids": [
            row["global_id"]
            for row
            in relaxed_candidates
        ],

        "fallback":
            fallback,

        "core_rows":
            core_member_evidence(
                manager=
                    manager,

                gid=
                    gid,

                candidate_tracklet=
                    tracklet,

                candidate_embedding=
                    embedding,

                lookup=
                    lookup,
            ),
    }


# ============================================================
# FAILURE REASONS
# ============================================================

def strict_failure_reasons(
    manager,
    audit,
):

    reasons = []

    scores = audit[
        "scores"
    ]


    if (
        scores["max"]
        < manager.strict_appearance_min
    ):

        reasons.append(
            f"max {scores['max']:.4f}"
            f" < "
            f"{manager.strict_appearance_min:.2f}"
        )


    if (
        scores["topk"]
        < manager.strict_appearance_min
    ):

        reasons.append(
            f"topk {scores['topk']:.4f}"
            f" < "
            f"{manager.strict_appearance_min:.2f}"
        )


    if not audit[
        "strict_support"
    ]:

        reasons.append(
            "no strict geometry support"
        )


    if audit[
        "contradicted"
    ]:

        reasons.append(
            "identity geometry contradiction"
        )


    if (
        audit["gid"]
        not in audit[
            "strict_candidate_gids"
        ]
    ):

        reasons.append(
            "expected GID absent from "
            "strict candidate set"
        )


    if not reasons:

        if (
            len(
                audit[
                    "strict_candidate_gids"
                ]
            )
            != 1
        ):

            reasons.append(
                "strict candidate set "
                "is not unique"
            )


    return reasons


def relaxed_failure_reasons(
    manager,
    audit,
):

    reasons = []

    scores = audit[
        "scores"
    ]


    if not audit[
        "anchored"
    ]:

        reasons.append(
            "GID not anchored"
        )


    if (
        scores["max"]
        < manager.relaxed_gid_appearance_min
    ):

        reasons.append(
            f"max {scores['max']:.4f}"
            f" < "
            f"{manager.relaxed_gid_appearance_min:.2f}"
        )


    if (
        scores["topk"]
        < manager.relaxed_gid_appearance_min
    ):

        reasons.append(
            f"topk {scores['topk']:.4f}"
            f" < "
            f"{manager.relaxed_gid_appearance_min:.2f}"
        )


    if not audit[
        "relaxed_support"
    ]:

        reasons.append(
            "no CORE member passes "
            "ReID>=.86 + shared>=15 "
            "+ geometry<=18"
        )


    if audit[
        "contradicted"
    ]:

        reasons.append(
            "identity geometry contradiction"
        )


    if (
        audit["gid"]
        not in audit[
            "relaxed_candidate_gids"
        ]
    ):

        reasons.append(
            "expected GID absent from "
            "relaxed candidate set"
        )


    if not reasons:

        if (
            len(
                audit[
                    "relaxed_candidate_gids"
                ]
            )
            != 1
        ):

            reasons.append(
                "relaxed candidate set "
                "is not unique"
            )


    return reasons


# ============================================================
# PRINT TARGET AUDIT
# ============================================================

def print_audit(
    manager,
    item,
    audit,
):

    label = item[
        "label"
    ]

    gid = audit[
        "gid"
    ]

    scores = audit[
        "scores"
    ]


    print()
    print(
        "=" * 72
    )

    print(
        f"{label} -> DIAGNOSTIC GID {gid}"
    )

    print(
        "=" * 72
    )


    tracklet = item[
        "tracklet"
    ]


    print(
        "Frames:",
        f"{tracklet.start_frame}-"
        f"{tracklet.end_frame}",
    )


    if (
        item[
            "embedding_frames"
        ]
        is not None
    ):

        print(
            "Embedding frames:",
            item[
                "embedding_frames"
            ],
        )


    print()
    print(
        "GID SCORES"
    )

    print(
        "----------"
    )

    print(
        "anchored:",
        audit[
            "anchored"
        ],
    )

    print(
        "max :",
        fmt(
            scores["max"]
        ),
    )

    print(
        "topk:",
        fmt(
            scores["topk"]
        ),
    )

    print(
        "mean:",
        fmt(
            scores["mean"]
        ),
    )

    print(
        "identity geometry contradicted:",
        audit[
            "contradicted"
        ],
    )


    print()
    print(
        "CORE MEMBER EVIDENCE"
    )

    print(
        "--------------------"
    )


    for row in audit[
        "core_rows"
    ]:

        prefix = (
            f"c{row['camera_id']}:"
            f"{row['track_id']}"
        )


        if row[
            "same_camera"
        ]:

            print(
                f"{prefix:<10}"
                f" | ReID="
                f"{fmt(row['reid'])}"
                f" | SAME_CAMERA"
            )

        else:

            print(
                f"{prefix:<10}"
                f" | ReID="
                f"{fmt(row['reid'])}"
                f" | geom="
                f"{row['status']}"
                f" | shared="
                f"{row['shared']}"
                f" | median="
                f"{fmt(row['median'], 2)}"
            )


    print()
    print(
        "STRICT GATE"
    )

    print(
        "-----------"
    )

    print(
        "required max/topk >=",
        manager.strict_appearance_min,
    )

    print(
        "geometry median max:",
        manager.strict_geometry_median_max,
    )

    print(
        "strict support rows:",
        len(
            audit[
                "strict_support"
            ]
        ),
    )

    print(
        "all strict candidate GIDs:",
        audit[
            "strict_candidate_gids"
        ],
    )


    strict_fail = (
        strict_failure_reasons(
            manager,
            audit,
        )
    )


    if strict_fail:

        print(
            "RESULT: FAIL"
        )

        for reason in strict_fail:

            print(
                "  -",
                reason,
            )

    else:

        print(
            "RESULT: PASS"
        )


    print()
    print(
        "RELAXED GATE"
    )

    print(
        "------------"
    )

    print(
        "GID max/topk >=",
        manager.relaxed_gid_appearance_min,
    )

    print(
        "CORE direct ReID >=",
        manager.relaxed_core_direct_appearance_min,
    )

    print(
        "shared frames >=",
        manager.relaxed_min_shared_frames,
    )

    print(
        "geometry median <=",
        manager.relaxed_geometry_median_max,
    )

    print(
        "relaxed support rows:",
        len(
            audit[
                "relaxed_support"
            ]
        ),
    )

    print(
        "all relaxed candidate GIDs:",
        audit[
            "relaxed_candidate_gids"
        ],
    )


    relaxed_fail = (
        relaxed_failure_reasons(
            manager,
            audit,
        )
    )


    if relaxed_fail:

        print(
            "RESULT: FAIL"
        )

        for reason in relaxed_fail:

            print(
                "  -",
                reason,
            )

    else:

        print(
            "RESULT: PASS"
        )


    fallback = audit[
        "fallback"
    ]


    print()
    print(
        "APPEARANCE FALLBACK"
    )

    print(
        "-------------------"
    )

    print(
        "status:",
        fallback.status,
    )

    print(
        "suggested GID:",
        fallback.global_id,
    )

    print(
        "max:",
        fmt(
            fallback.max_similarity
        ),
    )

    print(
        "topk:",
        fmt(
            fallback.topk_similarity
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2 PRODUCTION GATE FAILURE AUDIT"
    )

    print(
        "================================"
    )


    (
        manager,
        baseline_tracklets,
        lookup,
        _,
    ) = replay_final_production()


    baseline_snapshot = (
        snapshot_members(
            manager
        )
    )


    print()
    print(
        "VALIDATED c0+c1 START STATE"
    )

    print(
        "==========================="
    )

    print(
        "Tracklets:",
        len(
            baseline_tracklets
        ),
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
        "Anchored GIDs:",
        len(
            manager.anchored_gids
        ),
    )


    print()
    print(
        "EXACT PRODUCTION THRESHOLDS"
    )

    print(
        "==========================="
    )

    print(
        "STRICT max/topk:",
        manager.strict_appearance_min,
    )

    print(
        "STRICT geometry median max:",
        manager.strict_geometry_median_max,
    )

    print(
        "RELAXED GID max/topk:",
        manager.relaxed_gid_appearance_min,
    )

    print(
        "RELAXED CORE direct:",
        manager.relaxed_core_direct_appearance_min,
    )

    print(
        "RELAXED shared frames:",
        manager.relaxed_min_shared_frames,
    )

    print(
        "RELAXED geometry median max:",
        manager.relaxed_geometry_median_max,
    )

    print(
        "Fallback cross-camera strong:",
        manager.cross_camera_strong,
    )

    print(
        "Fallback cross-camera pending:",
        manager.cross_camera_pending,
    )


    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()


    lookup.update(
        c2_lookup
    )


    print()
    print(
        "BUILDING FOUR VALIDATED "
        "SEGMENT EMBEDDINGS..."
    )


    population = build_population(
        c2_rows
    )


    register_population(
        population,
        lookup,
    )


    print(
        "c2 production population:",
        len(
            population
        ),
    )


    if len(
        population
    ) != 107:

        raise RuntimeError(
            "Expected 107 c2 segments"
        )


    target_records = {}


    print()
    print(
        "PROCESSING c2 CHRONOLOGICALLY..."
    )


    for index, item in enumerate(
        population,
        start=1,
    ):

        label = item[
            "label"
        ]

        tracklet = item[
            "tracklet"
        ]

        embedding = item[
            "embedding"
        ]


        audit = None


        if label in TARGET_GIDS:

            audit = audit_expected_gid(
                manager=
                    manager,

                item=
                    item,

                gid=
                    TARGET_GIDS[
                        label
                    ],

                lookup=
                    lookup,
            )


            print_audit(
                manager=
                    manager,

                item=
                    item,

                audit=
                    audit,
            )


        camera_id = normalize_camera_id(
            tracklet.camera_id
        )


        result = manager.assign_tracklet(
            camera_id=
                camera_id,

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


        if label in TARGET_GIDS:

            print()
            print(
                "ACTUAL PRODUCTION RESULT"
            )

            print(
                "------------------------"
            )

            print(
                "status:",
                result.status,
            )

            print(
                "gid:",
                result.global_id,
            )

            print(
                "trust:",
                result.trust_level,
            )

            print(
                "reason:",
                result.reason,
            )


            target_records[
                label
            ] = {
                "key":
                    (
                        camera_id,
                        tracklet.local_track_id,
                    ),

                "audit":
                    audit,

                "result":
                    result,
            }


        if (
            result.merged
            or result.created_new
        ):

            manager.reevaluate_pending(
                tracklet_lookup=
                    lookup,

                include_same_camera=
                    False,

                include_dual_evidence=
                    False,

                include_core_gallery=
                    False,
            )


        if (
            index % 25
            == 0
            or
            index
            == len(
                population
            )
        ):

            print()
            print(
                f"Progress "
                f"{index}/"
                f"{len(population)}"
                f" | GIDs="
                f"{len(manager.identities)}"
                f" | pending="
                f"{len(manager.pending_tracklets)}"
            )


    # ========================================================
    # SAME FINAL PRODUCTION CONTINUATION PASS
    # ========================================================

    print()
    print(
        "RUNNING FINAL CONTINUATION PASS..."
    )


    manager.reevaluate_pending(
        tracklet_lookup=
            lookup,

        include_same_camera=
            True,

        include_dual_evidence=
            True,

        include_core_gallery=
            True,
    )


    # ========================================================
    # FAILURE SUMMARY
    # ========================================================

    print()
    print(
        "TARGET GATE FAILURE SUMMARY"
    )

    print(
        "==========================="
    )


    for label, expected_gid in (
        TARGET_GIDS.items()
    ):

        record = target_records[
            label
        ]

        audit = record[
            "audit"
        ]

        result = record[
            "result"
        ]


        strict_fail = (
            strict_failure_reasons(
                manager,
                audit,
            )
        )


        relaxed_fail = (
            relaxed_failure_reasons(
                manager,
                audit,
            )
        )


        print()
        print(
            f"{label}"
            f" -> diagnostic GID "
            f"{expected_gid}"
        )


        print(
            "  STRICT :",
            (
                "PASS"
                if not strict_fail
                else "; ".join(
                    strict_fail
                )
            ),
        )


        print(
            "  RELAXED:",
            (
                "PASS"
                if not relaxed_fail
                else "; ".join(
                    relaxed_fail
                )
            ),
        )


        print(
            "  ACTUAL :",
            result.status,
            "|",
            result.reason,
            "| gid=",
            result.global_id,
        )


    # ========================================================
    # FINAL TARGET STATES
    # ========================================================

    print()
    print(
        "FINAL TARGET STATES"
    )

    print(
        "==================="
    )


    for label, expected_gid in (
        TARGET_GIDS.items()
    ):

        record = target_records[
            label
        ]

        key = record[
            "key"
        ]

        final_gid = find_member_gid(
            manager,
            key,
        )


        if final_gid is None:

            trust = None

        else:

            trust = (
                manager.get_member_trust(
                    final_gid,
                    key[0],
                    key[1],
                )
            )


        pending = (
            key
            in manager.pending_tracklets
        )


        print(
            f"{label:<10}"
            f" | diagnostic="
            f"{expected_gid:<3}"
            f" | final_gid="
            f"{str(final_gid):<4}"
            f" | trust="
            f"{str(trust):<10}"
            f" | pending="
            f"{pending}"
        )


    # ========================================================
    # REGRESSION AUDIT
    # ========================================================

    changes = audit_snapshot(
        manager=
            manager,

        snapshot=
            baseline_snapshot,
    )


    print()
    print(
        "EXISTING c0+c1 REGRESSION AUDIT"
    )

    print(
        "================================"
    )


    if not changes:

        print(
            "PASS - existing c0+c1 "
            "members unchanged"
        )

    else:

        print(
            "FAIL - changed members:",
            len(
                changes
            ),
        )


        for (
            key,
            old_value,
            new_value,
        ) in changes:

            print(
                key,
                old_value,
                "->",
                new_value,
            )


    print()
    print(
        "SANITY"
    )

    print(
        "======"
    )


    checks = [
        (
            "c2 population = 107",
            len(
                population
            )
            == 107,
        ),

        (
            "all 10 target candidates audited",
            len(
                target_records
            )
            == 10,
        ),

        (
            "existing c0+c1 members unchanged",
            len(
                changes
            )
            == 0,
        ),
    ]


    passed = 0


    for description, ok in checks:

        if ok:
            passed += 1

        print(
            "PASS"
            if ok
            else "FAIL",
            "|",
            description,
        )


    print()
    print(
        f"{passed}/"
        f"{len(checks)} checks passed"
    )


    print()
    print(
        "PRODUCTION THRESHOLDS MODIFIED: NO"
    )

    print(
        "PRODUCTION SOURCE MODIFIED: NO"
    )

    print(
        "SPLITS PERSISTED: NO"
    )

    print(
        "Audit only."
    )


if __name__ == "__main__":
    main()
