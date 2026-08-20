from pathlib import Path

import cv2

from src.analyze_final_pending_population import (
    replay_final_production,
)

from src.analyze_c2_601_segment_gid_ranking import (
    build_segment_embedding,
    build_segment_tracklet,
    cosine_similarity,
    rank_segment,
)

from src.analyze_c2_gid_associations import (
    load_c2_tracklets,
)

from src.reid_extractor import (
    ReIDExtractor,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c2.avi"
)

TARGET_TRACK_ID = 9

EARLY_START = 455
EARLY_END = 1321

LATE_START = 1357
LATE_END = 1997

EXPECTED_EARLY_GID = 1
EXPECTED_LATE_GID = 11


# ============================================================
# STRONG DIAGNOSTIC IDENTITY DECISION
#
# These are diagnostic gates only.
# They do not modify production thresholds.
# ============================================================

MIN_JOINT_SUPPORTS = 2
MIN_CORE_MAX = 0.82
MIN_CORE_TOP3 = 0.80


# ============================================================
# FIND STRONG GIDS
# ============================================================

def strong_candidates(
    rows,
):

    results = []

    for row in rows:

        if row["identity_veto"]:
            continue

        if (
            row["joint_support_count"]
            < MIN_JOINT_SUPPORTS
        ):
            continue

        if (
            row["core_max"]
            < MIN_CORE_MAX
        ):
            continue

        if (
            row["core_top3"]
            < MIN_CORE_TOP3
        ):
            continue

        results.append(
            row
        )

    results.sort(
        key=lambda row: (
            row["joint_support_count"],
            row["core_top3"],
            row["core_max"],
        ),
        reverse=True,
    )

    return results


# ============================================================
# PRINT RESULT
# ============================================================

def print_gid_result(
    row,
):

    print(
        f"GID {row['gid']}"
        f" | max={row['core_max']:.4f}"
        f" | top3={row['core_top3']:.4f}"
        f" | mean={row['core_mean']:.4f}"
        f" | geom+={row['geometry_support_count']}"
        f" | joint+={row['joint_support_count']}"
        f" | CORE contradictions="
        f"{row['core_contradiction_count']}"
        f" | veto={row['identity_veto']}"
    )

    for member in row["core_rows"]:

        median = member[
            "median_distance"
        ]

        median_text = (
            "None"
            if median is None
            else f"{median:.2f}"
        )

        print(
            f"    CORE "
            f"c{member['camera_id']}:"
            f"{member['track_id']}"
            f" | ReID="
            f"{member['direct_similarity']:.4f}"
            f" | geom="
            f"{member['geometry_status']}"
            f" | shared="
            f"{member['shared_frames']}"
            f" | median="
            f"{median_text}"
        )


# ============================================================
# REPORT ONE SEGMENT
# ============================================================

def report_segment(
    name,
    ranked_rows,
    expected_gid,
):

    print()
    print(
        "=" * 72
    )

    print(
        name
    )

    print(
        "=" * 72
    )

    by_appearance = sorted(
        ranked_rows,
        key=lambda row: (
            row["core_top3"],
            row["core_max"],
            row["core_mean"],
        ),
        reverse=True,
    )

    print()
    print(
        "TOP 5 CORE-GALLERY APPEARANCE GIDS"
    )

    print(
        "=================================="
    )

    for row in by_appearance[:5]:

        print(
            f"GID {row['gid']:<3}"
            f" | max={row['core_max']:.4f}"
            f" | top3={row['core_top3']:.4f}"
            f" | mean={row['core_mean']:.4f}"
            f" | geom+={row['geometry_support_count']}"
            f" | joint+={row['joint_support_count']}"
            f" | veto={row['identity_veto']}"
        )


    strong = strong_candidates(
        ranked_rows
    )

    print()
    print(
        "STRONG JOINT IDENTITY CANDIDATES"
    )

    print(
        "================================"
    )

    if not strong:

        print(
            "NONE"
        )

    else:

        for row in strong:

            print_gid_result(
                row
            )


    expected = next(
        (
            row
            for row in ranked_rows
            if row["gid"] == expected_gid
        ),
        None,
    )

    print()
    print(
        f"EXPLICIT GID {expected_gid} CHECK"
    )

    print(
        "=" * 32
    )

    if expected is None:

        print(
            "NOT AVAILABLE"
        )

    else:

        print_gid_result(
            expected
        )


    unique_gid = None

    if len(strong) == 1:

        unique_gid = strong[0][
            "gid"
        ]


    passed = (
        unique_gid
        == expected_gid
    )


    print()
    print(
        "EXPECTED:",
        f"GID {expected_gid}",
    )

    print(
        "ACTUAL UNIQUE STRONG GID:",
        unique_gid,
    )

    print(
        "PASS"
        if passed
        else "REVIEW REQUIRED"
    )


    return (
        passed,
        unique_gid,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "c2:9 FINAL SPLIT VERIFICATION"
    )

    print(
        "============================="
    )


    (
        manager,
        _,
        tracklet_lookup,
        _,
    ) = replay_final_production()


    (
        c2_rows,
        c2_lookup,
    ) = load_c2_tracklets()


    tracklet_lookup.update(
        c2_lookup
    )


    source = {
        tracklet.local_track_id:
            tracklet

        for tracklet, _
        in c2_rows
    }.get(
        TARGET_TRACK_ID
    )


    if source is None:

        raise RuntimeError(
            "c2:9 not found"
        )


    early = build_segment_tracklet(
        source_tracklet=
            source,

        start_frame=
            EARLY_START,

        end_frame=
            EARLY_END,

        synthetic_track_id=
            900001,
    )


    late = build_segment_tracklet(
        source_tracklet=
            source,

        start_frame=
            LATE_START,

        end_frame=
            LATE_END,

        synthetic_track_id=
            900002,
    )


    print()
    print(
        "EARLY fragment:"
    )

    print(
        early.summary()
    )


    print()
    print(
        "LATE fragment:"
    )

    print(
        late.summary()
    )


    extractor = ReIDExtractor(
        device="cuda"
    )


    video = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )


    if not video.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )


    try:

        (
            early_embedding,
            early_frames,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                early,
        )


        (
            late_embedding,
            late_frames,
        ) = build_segment_embedding(
            extractor=
                extractor,

            video=
                video,

            tracklet=
                late,
        )


    finally:

        video.release()


    print()
    print(
        "EARLY embedding frames:",
        early_frames,
    )


    print(
        "LATE embedding frames:",
        late_frames,
    )


    print()
    print(
        "EARLY <-> LATE ReID:",
        f"{cosine_similarity(early_embedding, late_embedding):.4f}",
    )


    early_rows = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            early,

        candidate_embedding=
            early_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    late_rows = rank_segment(
        manager=
            manager,

        candidate_tracklet=
            late,

        candidate_embedding=
            late_embedding,

        tracklet_lookup=
            tracklet_lookup,
    )


    early_pass, early_gid = (
        report_segment(
            name=
                "EARLY c2:9a 455-1321",

            ranked_rows=
                early_rows,

            expected_gid=
                EXPECTED_EARLY_GID,
        )
    )


    late_pass, late_gid = (
        report_segment(
            name=
                "LATE c2:9b 1357-1997",

            ranked_rows=
                late_rows,

            expected_gid=
                EXPECTED_LATE_GID,
        )
    )


    print()
    print()
    print(
        "FINAL SPLIT CHECK"
    )

    print(
        "================="
    )


    print(
        "c2:9a:",
        early_gid,
        "expected",
        EXPECTED_EARLY_GID,
    )


    print(
        "c2:9b:",
        late_gid,
        "expected",
        EXPECTED_LATE_GID,
    )


    final_pass = (
        early_pass
        and late_pass
        and early_gid
        != late_gid
    )


    print()
    print(
        "RESULT:",
        (
            "PASS - SPLIT SUPPORTED"
            if final_pass
            else "REVIEW REQUIRED"
        ),
    )


    print()
    print(
        "IDENTITIES MODIFIED: NO"
    )

    print(
        "Original c2:9 was not modified."
    )

    print(
        "Temporary fragments existed "
        "in memory only."
    )


if __name__ == "__main__":
    main()
