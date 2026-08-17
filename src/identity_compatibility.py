import math


MIN_SHARED_FRAMES = 10
MIN_SEPARATED_FRAMES = 10
MIN_SEPARATED_RATIO = 0.80

MAX_IOU_FOR_DIFFERENT = 0.05
MIN_CENTER_DISTANCE = 45.0


# ============================================================
# CAMERA ID NORMALIZATION
#
# Supports:
#
#   0
#   "0"
#   "c0"
#
# and converts all of them to:
#
#   0
# ============================================================

def normalize_camera_id(camera_id):

    if isinstance(camera_id, str):

        camera_id = camera_id.strip().lower()

        if camera_id.startswith("c"):
            camera_id = camera_id[1:]

    return int(camera_id)


# ============================================================
# BBOX IOU
# ============================================================

def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    acx = (
        ax1 + ax2
    ) / 2.0

    acy = (
        ay1 + ay2
    ) / 2.0

    bcx = (
        bx1 + bx2
    ) / 2.0

    bcy = (
        by1 + by2
    ) / 2.0

    return math.hypot(
        acx - bcx,
        acy - bcy,
    )


# ============================================================
# BBOX DIAGONAL
# ============================================================

def bbox_diagonal(box):

    x1, y1, x2, y2 = box

    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


# ============================================================
# DETECTIONS INDEXED BY FRAME
# ============================================================

def detections_by_frame(
    tracklet,
):

    return {
        detection.frame: detection
        for detection in tracklet.detections
    }


# ============================================================
# ARE TWO DETECTIONS CLEARLY DIFFERENT PEOPLE?
# ============================================================

def spatially_separated(
    detection_a,
    detection_b,
):

    box_a = (
        detection_a.x1,
        detection_a.y1,
        detection_a.x2,
        detection_a.y2,
    )

    box_b = (
        detection_b.x1,
        detection_b.y1,
        detection_b.x2,
        detection_b.y2,
    )

    iou = bbox_iou(
        box_a,
        box_b,
    )

    distance = center_distance(
        box_a,
        box_b,
    )

    size_threshold = (
        0.75
        * max(
            bbox_diagonal(
                box_a
            ),
            bbox_diagonal(
                box_b
            ),
        )
    )

    required_distance = max(
        MIN_CENTER_DISTANCE,
        size_threshold,
    )

    return (
        iou
        <= MAX_IOU_FOR_DIFFERENT

        and

        distance
        >= required_distance
    )


# ============================================================
# SAME-CAMERA CONFLICT
# ============================================================

def same_camera_conflict(
    candidate_tracklet,
    existing_tracklet,
):

    candidate_camera = (
        normalize_camera_id(
            candidate_tracklet.camera_id
        )
    )

    existing_camera = (
        normalize_camera_id(
            existing_tracklet.camera_id
        )
    )

    # Different cameras cannot create this type
    # of same-camera spatial conflict.
    if (
        candidate_camera
        != existing_camera
    ):
        return False


    candidate_frames = (
        detections_by_frame(
            candidate_tracklet
        )
    )

    existing_frames = (
        detections_by_frame(
            existing_tracklet
        )
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


    # Not enough simultaneous observations
    # to confidently declare a conflict.
    if (
        len(shared_frames)
        < MIN_SHARED_FRAMES
    ):
        return False


    separated_count = 0


    for frame in shared_frames:

        if spatially_separated(
            candidate_frames[
                frame
            ],
            existing_frames[
                frame
            ],
        ):

            separated_count += 1


    separated_ratio = (
        separated_count
        / len(shared_frames)
    )


    return (
        separated_count
        >= MIN_SEPARATED_FRAMES

        and

        separated_ratio
        >= MIN_SEPARATED_RATIO
    )


# ============================================================
# DOES A CANDIDATE CONFLICT WITH ANY MEMBER OF A GID?
# ============================================================

def identity_has_conflict(
    candidate_tracklet,
    identity,
    tracklet_lookup,
):

    candidate_camera = (
        normalize_camera_id(
            candidate_tracklet.camera_id
        )
    )


    for member in identity.members:

        member_camera = (
            normalize_camera_id(
                member.camera_id
            )
        )


        # Only members from the same camera
        # can create a same-camera conflict.
        if (
            member_camera
            != candidate_camera
        ):
            continue


        # ------------------------------------------------
        # First try lookup using normalized integer
        # camera IDs:
        #
        #   (0, track_id)
        # ------------------------------------------------

        key = (
            member_camera,
            member.local_track_id,
        )

        existing_tracklet = (
            tracklet_lookup.get(
                key
            )
        )


        # ------------------------------------------------
        # Also support lookup dictionaries using:
        #
        #   ("c0", track_id)
        # ------------------------------------------------

        if existing_tracklet is None:

            key = (
                f"c{member_camera}",
                member.local_track_id,
            )

            existing_tracklet = (
                tracklet_lookup.get(
                    key
                )
            )


        # Tracklet information unavailable.
        if existing_tracklet is None:
            continue


        if same_camera_conflict(
            candidate_tracklet,
            existing_tracklet,
        ):

            return True


    return False
