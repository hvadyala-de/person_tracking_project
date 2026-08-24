import src.identity_compatibility as identity_compatibility
import src.run_c3_candidate_same_camera_conflict_experiment as experiment

from src.identity_compatibility import (
    MIN_SEPARATED_RATIO,
    MIN_SHARED_FRAMES,
    detections_by_frame,
    normalize_camera_id,
)


# ============================================================
# AUDIT-ONLY SECONDARY CONFLICT RULE
#
# Keep the original production conflict rule intact.
#
# Add one extra way to declare a conflict:
#
#   - same camera
#   - at least 10 simultaneous detections
#   - candidate spatial separation on >= 80% of shared frames
#   - at least 10 CONSECUTIVE separated frames
#
# This specifically distinguishes:
#
#   c1:369 <-> c1:382
#       longest consecutive separation = 9
#
# from:
#
#   c3:12 <-> c3:15
#       longest consecutive separation = 13
#
# This file does NOT modify production source code.
# ============================================================


MIN_CONSECUTIVE_SEPARATED_FRAMES = 10


PRODUCTION_SAME_CAMERA_CONFLICT = (
    identity_compatibility.same_camera_conflict
)


# ============================================================
# LONGEST CONSECUTIVE RUN
# ============================================================

def longest_consecutive_run(
    frames,
):

    if not frames:
        return 0


    frames = sorted(
        frames
    )


    best = 1
    current = 1


    for previous, current_frame in zip(
        frames,
        frames[1:],
    ):

        if (
            current_frame
            == previous + 1
        ):

            current += 1

            best = max(
                best,
                current,
            )

        else:

            current = 1


    return best


# ============================================================
# PERSISTENT SAME-CAMERA CONFLICT
# ============================================================

def persistent_same_camera_conflict(
    candidate_tracklet,
    existing_tracklet,
):

    # --------------------------------------------------------
    # Preserve every conflict already detected by production.
    # --------------------------------------------------------

    if PRODUCTION_SAME_CAMERA_CONFLICT(
        candidate_tracklet,
        existing_tracklet,
    ):

        return True


    candidate_camera = normalize_camera_id(
        candidate_tracklet.camera_id
    )

    existing_camera = normalize_camera_id(
        existing_tracklet.camera_id
    )


    if (
        candidate_camera
        != existing_camera
    ):

        return False


    candidate_frames = detections_by_frame(
        candidate_tracklet
    )

    existing_frames = detections_by_frame(
        existing_tracklet
    )


    shared_frames = sorted(
        set(
            candidate_frames
        )
        &
        set(
            existing_frames
        )
    )


    if (
        len(
            shared_frames
        )
        < MIN_SHARED_FRAMES
    ):

        return False


    separated_frames = []


    for frame in shared_frames:

        if experiment.candidate_spatially_separated(
            candidate_frames[
                frame
            ],
            existing_frames[
                frame
            ],
        ):

            separated_frames.append(
                frame
            )


    separated_ratio = (
        len(
            separated_frames
        )
        / len(
            shared_frames
        )
    )


    if (
        separated_ratio
        < MIN_SEPARATED_RATIO
    ):

        return False


    consecutive_run = longest_consecutive_run(
        separated_frames
    )


    return (
        consecutive_run
        >= MIN_CONSECUTIVE_SEPARATED_FRAMES
    )


# ============================================================
# TARGET SANITY CHECK
# ============================================================

def print_target_sanity():

    from src.analyze_c3_geometry_candidates import (
        load_camera_tracklets,
    )


    print()
    print(
        "PERSISTENT-CONFLICT TARGET SANITY"
    )

    print(
        "================================="
    )


    targets = (
        (
            1,
            369,
            382,
            False,
        ),

        (
            3,
            12,
            15,
            True,
        ),
    )


    all_passed = True


    for (
        camera_id,
        track_a,
        track_b,
        expected,
    ) in targets:

        tracklets = load_camera_tracklets(
            camera_id
        )


        result = persistent_same_camera_conflict(
            tracklets[
                track_a
            ],
            tracklets[
                track_b
            ],
        )


        passed = (
            result
            == expected
        )


        all_passed = (
            all_passed
            and passed
        )


        print(
            "PASS"
            if passed
            else "FAIL",
            "|",
            f"c{camera_id}:{track_a}"
            f" <-> "
            f"c{camera_id}:{track_b}",
            "| conflict=",
            result,
            "| expected=",
            expected,
        )


    return all_passed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "FOUR-CAMERA PERSISTENT SAME-CAMERA "
        "CONFLICT VALIDATION"
    )

    print(
        "=========================================="
        "================"
    )


    print()
    print(
        "Minimum consecutive separated frames:",
        MIN_CONSECUTIVE_SEPARATED_FRAMES,
    )


    if not print_target_sanity():

        raise RuntimeError(
            "Target sanity check failed"
        )


    # --------------------------------------------------------
    # Replace only the candidate function used by the existing
    # four-camera experiment.
    #
    # identity_compatibility.py remains untouched.
    # --------------------------------------------------------

    experiment.candidate_same_camera_conflict = (
        persistent_same_camera_conflict
    )


    experiment.main()


if __name__ == "__main__":
    main()
