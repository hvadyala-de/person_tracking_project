import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import src.experiments.analyze_c3_gap_identity_changes as reference
import src.experiments.run_c3_production_policy_experiment as c3prod


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "output"
    / "presentation"
)

VIDEO_PATHS = {
    0: ROOT / "data" / "videos" / "terrace1-c0.avi",
    1: ROOT / "data" / "videos" / "terrace1-c1.avi",
    2: ROOT / "data" / "videos" / "terrace1-c2.avi",
    3: ROOT / "data" / "videos" / "terrace1-c3.avi",
}

OUTPUT_FPS = 25.0

EXPECTED_FINAL_GIDS = 134
EXPECTED_FINAL_PENDING = 175
EXPECTED_FINAL_ANCHORED = 47

EXPECTED_C3_EXISTING = 14
EXPECTED_C3_NEW = 29
EXPECTED_C3_PENDING = 52


# ============================================================
# CAMERA / KEY NORMALIZATION
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


def normalize_key(
    key,
):

    return (
        normalize_camera_id(
            key[0]
        ),
        int(
            key[1]
        ),
    )


# ============================================================
# INSTALL CURRENT VALIDATED REFERENCE EXPECTATIONS
#
# These values correspond to production behavior after:
#
#   ceee0e4
#   Add persistent same-camera identity conflict veto
#
# c0+c1:
#   77 GIDs
#   91 pending
#   22 anchored
#
# c0+c1+c2:
#   109 GIDs
#   162 pending
#   22 anchored
# ============================================================

def install_validated_reference_expectations():

    reference.EXPECTED_C0_C1_GIDS = 77

    reference.EXPECTED_C0_C1_PENDING = 91

    reference.EXPECTED_C0_C1_ANCHORED = 22


    reference.EXPECTED_FINAL_GIDS = 109

    reference.EXPECTED_FINAL_PENDING = 162

    reference.EXPECTED_FINAL_ANCHORED = 22


    reference.EXPECTED_C2_ASSOCIATIONS = {
        "c2:9a": 1,
        "c2:9b": 11,
        "c2:508": 66,
        "c2:601b": 74,
    }


# ============================================================
# REPLAY EXACT FOUR-CAMERA PRODUCTION STATE
# ============================================================

def replay_four_camera_state():

    install_validated_reference_expectations()


    print()
    print(
        "RENDERER: REPLAYING VALIDATED "
        "FOUR-CAMERA PRODUCTION STATE"
    )

    print(
        "=========================================="
    )


    (
        manager,
        tracklet_lookup,
    ) = (
        reference
        .build_validated_reference_state()
    )


    baseline_gid_ids = set(
        manager.identities.keys()
    )


    # ========================================================
    # BUILD VALIDATED c3 POPULATION
    # ========================================================

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
            "Unexpected c3 tracklet "
            "lookup collisions: "
            f"{collisions}"
        )


    # ========================================================
    # PROCESS c3 WITH CURRENT PRODUCTION MANAGER
    # ========================================================

    records = (
        c3prod.process_c3_population(
            manager=
                manager,

            population=
                population,

            tracklet_lookup=
                tracklet_lookup,
        )
    )


    # ========================================================
    # FINAL PRODUCTION CONTINUATION
    #
    # Same final continuation stages used by the validated
    # four-camera production experiment.
    # ========================================================

    final_results = (
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
    )


    print()
    print(
        "Renderer final continuation "
        "result rows:",
        len(
            final_results
        ),
    )


    records = (
        c3prod.finalize_c3_records(
            manager=
                manager,

            records=
                records,

            baseline_gid_ids=
                baseline_gid_ids,
        )
    )


    validate_final_state(
        manager=
            manager,

        records=
            records,
    )


    return (
        manager,
        tracklet_lookup,
        records,
    )


# ============================================================
# FINAL STATE GUARDS
# ============================================================

def validate_final_state(
    manager,
    records,
):

    final_gids = len(
        manager.identities
    )

    final_pending = len(
        manager.pending_tracklets
    )

    final_anchored = len(
        manager.anchored_gids
    )


    c3_existing = sum(
        1
        for record in records
        if (
            record.get(
                "final_gid"
            )
            is not None

            and

            record.get(
                "gid_origin"
            )
            == "EXISTING"
        )
    )


    c3_new = sum(
        1
        for record in records
        if (
            record.get(
                "final_gid"
            )
            is not None

            and

            record.get(
                "gid_origin"
            )
            == "NEW"
        )
    )


    c3_pending = sum(
        1
        for record in records
        if record.get(
            "is_pending"
        )
    )


    print()
    print(
        "RENDERER FINAL STATE"
    )

    print(
        "===================="
    )

    print(
        "GIDs:",
        final_gids,
    )

    print(
        "Pending:",
        final_pending,
    )

    print(
        "Anchored:",
        final_anchored,
    )

    print(
        "c3 existing:",
        c3_existing,
    )

    print(
        "c3 new:",
        c3_new,
    )

    print(
        "c3 pending:",
        c3_pending,
    )


    checks = [
        (
            final_gids
            == EXPECTED_FINAL_GIDS,

            "final GID count",
        ),

        (
            final_pending
            == EXPECTED_FINAL_PENDING,

            "final pending count",
        ),

        (
            final_anchored
            == EXPECTED_FINAL_ANCHORED,

            "final anchored count",
        ),

        (
            c3_existing
            == EXPECTED_C3_EXISTING,

            "c3 existing count",
        ),

        (
            c3_new
            == EXPECTED_C3_NEW,

            "c3 new count",
        ),

        (
            c3_pending
            == EXPECTED_C3_PENDING,

            "c3 pending count",
        ),
    ]


    for passed, label in checks:

        if not passed:

            raise RuntimeError(
                "Renderer state regression: "
                f"{label}"
            )


    by_label = {
        record[
            "label"
        ]:
            record

        for record in records
    }


    c3_12 = by_label.get(
        "c3:12"
    )

    c3_15 = by_label.get(
        "c3:15"
    )

    c3_104a = by_label.get(
        "c3:104a"
    )

    c3_104b = by_label.get(
        "c3:104b"
    )


    if (
        c3_12 is None
        or c3_15 is None
        or c3_104a is None
        or c3_104b is None
    ):

        raise RuntimeError(
            "Renderer cannot find "
            "validated c3 safety records"
        )


    if (
        c3_12.get(
            "final_gid"
        )
        is not None

        and

        c3_12.get(
            "final_gid"
        )
        == c3_15.get(
            "final_gid"
        )
    ):

        raise RuntimeError(
            "Renderer safety regression: "
            "c3:12 and c3:15 share a GID"
        )


    if (
        c3_104b.get(
            "final_gid"
        )
        != 11
    ):

        raise RuntimeError(
            "Renderer safety regression: "
            "c3:104b is not GID 11"
        )


    if (
        c3_104a.get(
            "final_gid"
        )
        == 11
    ):

        raise RuntimeError(
            "Renderer safety regression: "
            "c3:104a incorrectly became GID 11"
        )


    print()
    print(
        "Renderer state guards: PASS"
    )


# ============================================================
# NORMALIZED TRACKLET LOOKUP
# ============================================================

def build_normalized_tracklet_lookup(
    tracklet_lookup,
):

    normalized = {}


    for raw_key, tracklet in (
        tracklet_lookup.items()
    ):

        try:

            key = normalize_key(
                raw_key
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):

            continue


        normalized[
            key
        ] = tracklet


    return normalized


# ============================================================
# FINAL ASSIGNMENT MAP
# ============================================================

def build_assignment_maps(
    manager,
):

    assigned_gid = {}

    trust_by_key = {}


    for gid, identity in (
        manager.identities.items()
    ):

        for member in (
            identity.members
        ):

            key = (
                normalize_camera_id(
                    member.camera_id
                ),
                int(
                    member.local_track_id
                ),
            )


            assigned_gid[
                key
            ] = gid


            trust = (
                c3prod
                .get_member_trust_safe(
                    manager,
                    gid,
                    key,
                )
            )


            trust_by_key[
                key
            ] = trust


    pending = {
        normalize_key(
            key
        )

        for key in (
            c3prod.pending_keys(
                manager
            )
        )
    }


    return (
        assigned_gid,
        trust_by_key,
        pending,
    )


# ============================================================
# HUMAN-READABLE SOURCE TRACK LABELS
# ============================================================

def build_source_label_map(
    c3_records,
):

    source_labels = {}


    for record in c3_records:

        source_labels[
            normalize_key(
                record[
                    "key"
                ]
            )
        ] = record[
            "label"
        ]


    return source_labels


def source_track_label(
    key,
    tracklet,
    c3_source_labels,
):

    camera_id = key[0]


    if (
        key
        in c3_source_labels
    ):

        return (
            c3_source_labels[
                key
            ]
        )


    # ========================================================
    # c2 VALIDATED SPLIT LABELS
    #
    # The manager uses synthetic local IDs internally for
    # split segments. For presentation, show the original
    # Terrace ByteTrack ID plus a/b suffix.
    # ========================================================

    if camera_id == 2:

        start_frame = int(
            tracklet.start_frame
        )

        end_frame = int(
            tracklet.end_frame
        )


        if (
            start_frame == 455
            and end_frame == 1321
        ):

            return "c2:9a"


        if (
            start_frame == 1357
            and end_frame == 1997
        ):

            return "c2:9b"


        if (
            start_frame == 3796
            and end_frame == 4016
        ):

            return "c2:601a"


        if (
            start_frame == 4049
            and end_frame == 4571
        ):

            return "c2:601b"


    return (
        f"c{camera_id}:"
        f"{tracklet.local_track_id}"
    )


# ============================================================
# BUILD FRAME -> OVERLAY INDEX
# ============================================================

def build_frame_overlays(
    manager,
    tracklet_lookup,
    c3_records,
):

    (
        assigned_gid,
        trust_by_key,
        pending,
    ) = build_assignment_maps(
        manager
    )


    normalized_lookup = (
        build_normalized_tracklet_lookup(
            tracklet_lookup
        )
    )


    c3_source_labels = (
        build_source_label_map(
            c3_records
        )
    )


    active_keys = (
        set(
            assigned_gid.keys()
        )
        |
        set(
            pending
        )
    )


    overlays = {
        camera_id:
            defaultdict(
                list
            )

        for camera_id
        in range(
            4
        )
    }


    assignment_rows = []


    missing_tracklets = []


    for key in sorted(
        active_keys
    ):

        camera_id = (
            key[0]
        )


        if (
            camera_id
            not in overlays
        ):

            continue


        tracklet = (
            normalized_lookup.get(
                key
            )
        )


        if tracklet is None:

            missing_tracklets.append(
                key
            )

            continue


        gid = (
            assigned_gid.get(
                key
            )
        )


        is_pending = (
            key
            in pending
        )


        trust = (
            trust_by_key.get(
                key
            )
        )


        display_source = (
            source_track_label(
                key=
                    key,

                tracklet=
                    tracklet,

                c3_source_labels=
                    c3_source_labels,
            )
        )


        assignment_rows.append(
            {
                "camera_id":
                    camera_id,

                "internal_track_id":
                    key[1],

                "display_track":
                    display_source,

                "start_frame":
                    int(
                        tracklet.start_frame
                    ),

                "end_frame":
                    int(
                        tracklet.end_frame
                    ),

                "gid":
                    (
                        ""
                        if gid is None
                        else gid
                    ),

                "trust":
                    (
                        ""
                        if trust is None
                        else trust
                    ),

                "pending":
                    int(
                        is_pending
                    ),
            }
        )


        for detection in (
            tracklet.detections
        ):

            frame_id = int(
                detection.frame
            )


            overlays[
                camera_id
            ][
                frame_id
            ].append(
                {
                    "x1":
                        float(
                            detection.x1
                        ),

                    "y1":
                        float(
                            detection.y1
                        ),

                    "x2":
                        float(
                            detection.x2
                        ),

                    "y2":
                        float(
                            detection.y2
                        ),

                    "gid":
                        gid,

                    "trust":
                        trust,

                    "pending":
                        is_pending,

                    "source":
                        display_source,
                }
            )


    if missing_tracklets:

        print()
        print(
            "WARNING: active identity keys "
            "missing from tracklet lookup:"
        )

        for key in (
            missing_tracklets
        ):

            print(
                " ",
                key,
            )


    print()
    print(
        "RENDER OVERLAY POPULATION"
    )

    print(
        "========================="
    )


    for camera_id in range(
        4
    ):

        camera_keys = [
            row

            for row
            in assignment_rows

            if (
                row[
                    "camera_id"
                ]
                == camera_id
            )
        ]


        assigned_count = sum(
            1
            for row in camera_keys
            if (
                row[
                    "gid"
                ]
                != ""
            )
        )


        pending_count = sum(
            1
            for row in camera_keys
            if (
                row[
                    "pending"
                ]
                == 1
            )
        )


        print(
            f"c{camera_id}: "
            f"{len(camera_keys)} active segments | "
            f"{assigned_count} assigned | "
            f"{pending_count} pending"
        )


    return (
        overlays,
        assignment_rows,
    )


# ============================================================
# COLOR
# ============================================================

def gid_color(
    gid,
):

    hue = int(
        (
            int(
                gid
            )
            * 47
        )
        % 180
    )


    hsv = np.array(
        [
            [
                [
                    hue,
                    220,
                    255,
                ]
            ]
        ],
        dtype=np.uint8,
    )


    bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )[0, 0]


    return (
        int(
            bgr[0]
        ),
        int(
            bgr[1]
        ),
        int(
            bgr[2]
        ),
    )


def overlay_color(
    item,
):

    if (
        item[
            "gid"
        ]
        is not None
    ):

        return gid_color(
            item[
                "gid"
            ]
        )


    if item[
        "pending"
    ]:

        return (
            0,
            215,
            255,
        )


    return (
        180,
        180,
        180,
    )


# ============================================================
# DRAW TEXT WITH BACKGROUND
# ============================================================

def draw_label(
    frame,
    text,
    x,
    y,
    color,
):

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.40

    thickness = 1


    (
        text_width,
        text_height,
    ), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )


    x = max(
        0,
        min(
            int(
                x
            ),
            frame.shape[1]
            - text_width
            - 5,
        ),
    )


    y = int(
        y
    )


    if (
        y
        - text_height
        - baseline
        - 5
        < 0
    ):

        y = (
            text_height
            + baseline
            + 6
        )


    top = max(
        0,
        y
        - text_height
        - baseline
        - 4,
    )


    bottom = min(
        frame.shape[0]
        - 1,
        y
        + 3,
    )


    right = min(
        frame.shape[1]
        - 1,
        x
        + text_width
        + 5,
    )


    cv2.rectangle(
        frame,
        (
            x,
            top,
        ),
        (
            right,
            bottom,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )


    cv2.putText(
        frame,
        text,
        (
            x + 2,
            y - 2,
        ),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# DRAW CAMERA FRAME
# ============================================================

def render_camera_frame(
    frame,
    camera_id,
    frame_id,
    frame_items,
):

    rendered = (
        frame.copy()
    )


    # Draw larger boxes first so smaller overlapping boxes
    # remain visible on top.
    sorted_items = sorted(
        frame_items,
        key=lambda item: (
            -(
                max(
                    0.0,
                    item[
                        "x2"
                    ]
                    - item[
                        "x1"
                    ],
                )
                *
                max(
                    0.0,
                    item[
                        "y2"
                    ]
                    - item[
                        "y1"
                    ],
                )
            )
        ),
    )


    for item in sorted_items:

        color = overlay_color(
            item
        )


        x1 = int(
            round(
                item[
                    "x1"
                ]
            )
        )

        y1 = int(
            round(
                item[
                    "y1"
                ]
            )
        )

        x2 = int(
            round(
                item[
                    "x2"
                ]
            )
        )

        y2 = int(
            round(
                item[
                    "y2"
                ]
            )
        )


        x1 = max(
            0,
            min(
                x1,
                rendered.shape[1]
                - 1,
            ),
        )

        x2 = max(
            0,
            min(
                x2,
                rendered.shape[1]
                - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                rendered.shape[0]
                - 1,
            ),
        )

        y2 = max(
            0,
            min(
                y2,
                rendered.shape[0]
                - 1,
            ),
        )


        cv2.rectangle(
            rendered,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            2,
        )


        if (
            item[
                "gid"
            ]
            is not None
        ):

            text = (
                f"GID "
                f"{item['gid']} "
                f"| "
                f"{item['source']}"
            )

        elif item[
            "pending"
        ]:

            text = (
                "PENDING"
                " | "
                f"{item['source']}"
            )

        else:

            text = (
                "UNASSIGNED"
                " | "
                f"{item['source']}"
            )


        draw_label(
            frame=
                rendered,

            text=
                text,

            x=
                x1,

            y=
                y1 - 2,

            color=
                color,
        )


    # ========================================================
    # CAMERA HEADER
    # ========================================================

    header = (
        f"Camera {camera_id}  "
        f"|  Frame {frame_id:04d}"
    )


    cv2.rectangle(
        rendered,
        (
            0,
            0,
        ),
        (
            190,
            22,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )


    cv2.putText(
        rendered,
        header,
        (
            6,
            15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )


    return rendered


# ============================================================
# VIDEO IO
# ============================================================

def open_captures():

    captures = {}


    for camera_id, path in (
        VIDEO_PATHS.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing input video: "
                f"{path}"
            )


        capture = cv2.VideoCapture(
            str(
                path
            )
        )


        if not capture.isOpened():

            raise RuntimeError(
                f"Unable to open input video: "
                f"{path}"
            )


        captures[
            camera_id
        ] = capture


    return captures


def inspect_video_geometry(
    captures,
):

    geometry = {}


    for camera_id, capture in (
        captures.items()
    ):

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        geometry[
            camera_id
        ] = {
            "width":
                width,

            "height":
                height,

            "fps":
                fps,

            "frame_count":
                frame_count,
        }


        print(
            f"c{camera_id}: "
            f"{width}x{height} | "
            f"{fps:.3f} FPS | "
            f"{frame_count} frames"
        )


    widths = {
        value[
            "width"
        ]
        for value in (
            geometry.values()
        )
    }

    heights = {
        value[
            "height"
        ]
        for value in (
            geometry.values()
        )
    }


    if (
        len(
            widths
        )
        != 1
        or len(
            heights
        )
        != 1
    ):

        raise RuntimeError(
            "Terrace camera videos do not "
            "share one resolution"
        )


    return geometry


def create_writer(
    path,
    width,
    height,
):

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )


    writer = cv2.VideoWriter(
        str(
            path
        ),
        fourcc,
        OUTPUT_FPS,
        (
            int(
                width
            ),
            int(
                height
            ),
        ),
    )


    if not writer.isOpened():

        raise RuntimeError(
            f"Unable to create video writer: "
            f"{path}"
        )


    return writer


# ============================================================
# SAVE FINAL IDENTITY TABLE
# ============================================================

def write_assignment_csv(
    output_path,
    rows,
):

    fieldnames = [
        "camera_id",
        "internal_track_id",
        "display_track",
        "start_frame",
        "end_frame",
        "gid",
        "trust",
        "pending",
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


        for row in sorted(
            rows,
            key=lambda item: (
                item[
                    "camera_id"
                ],
                item[
                    "start_frame"
                ],
                item[
                    "internal_track_id"
                ],
            ),
        ):

            writer.writerow(
                row
            )


# ============================================================
# RENDER
# ============================================================

def render_videos(
    overlays,
    output_dir,
    snapshot_frame,
    max_frames,
):

    captures = (
        open_captures()
    )


    try:

        print()
        print(
            "INPUT VIDEOS"
        )

        print(
            "============"
        )


        geometry = (
            inspect_video_geometry(
                captures
            )
        )


        width = (
            geometry[
                0
            ][
                "width"
            ]
        )

        height = (
            geometry[
                0
            ][
                "height"
            ]
        )


        total_frames = min(
            value[
                "frame_count"
            ]
            for value in (
                geometry.values()
            )
        )


        if (
            max_frames
            is not None
            and max_frames > 0
        ):

            total_frames = min(
                total_frames,
                int(
                    max_frames
                ),
            )


        individual_paths = {
            camera_id:
                (
                    output_dir
                    /
                    (
                        "terrace_global_ids_"
                        f"c{camera_id}.mp4"
                    )
                )

            for camera_id
            in range(
                4
            )
        }


        combined_path = (
            output_dir
            /
            "terrace_4cam_global_ids.mp4"
        )


        individual_writers = {
            camera_id:
                create_writer(
                    path=
                        individual_paths[
                            camera_id
                        ],

                    width=
                        width,

                    height=
                        height,
                )

            for camera_id
            in range(
                4
            )
        }


        combined_writer = (
            create_writer(
                path=
                    combined_path,

                width=
                    width * 2,

                height=
                    height * 2,
            )
        )


        snapshot_written = False


        print()
        print(
            "RENDERING"
        )

        print(
            "========="
        )

        print(
            "Frames:",
            total_frames,
        )

        print(
            "Output FPS:",
            OUTPUT_FPS,
        )


        try:

            for frame_id in range(
                total_frames
            ):

                rendered_frames = {}


                for camera_id in range(
                    4
                ):

                    ok, frame = (
                        captures[
                            camera_id
                        ].read()
                    )


                    if not ok:

                        raise RuntimeError(
                            f"Video read failed at "
                            f"camera {camera_id}, "
                            f"frame {frame_id}"
                        )


                    rendered = (
                        render_camera_frame(
                            frame=
                                frame,

                            camera_id=
                                camera_id,

                            frame_id=
                                frame_id,

                            frame_items=
                                overlays[
                                    camera_id
                                ].get(
                                    frame_id,
                                    [],
                                ),
                        )
                    )


                    rendered_frames[
                        camera_id
                    ] = rendered


                    individual_writers[
                        camera_id
                    ].write(
                        rendered
                    )


                top = np.hstack(
                    (
                        rendered_frames[
                            0
                        ],
                        rendered_frames[
                            1
                        ],
                    )
                )


                bottom = np.hstack(
                    (
                        rendered_frames[
                            2
                        ],
                        rendered_frames[
                            3
                        ],
                    )
                )


                combined = np.vstack(
                    (
                        top,
                        bottom,
                    )
                )


                combined_writer.write(
                    combined
                )


                if (
                    not snapshot_written
                    and frame_id
                    == snapshot_frame
                ):

                    snapshot_path = (
                        output_dir
                        /
                        (
                            "terrace_4cam_global_ids_"
                            f"frame_{frame_id:06d}.jpg"
                        )
                    )


                    cv2.imwrite(
                        str(
                            snapshot_path
                        ),
                        combined,
                    )


                    print(
                        "Snapshot:",
                        snapshot_path,
                    )


                    snapshot_written = True


                if (
                    frame_id % 250
                    == 0
                    or frame_id
                    == total_frames - 1
                ):

                    percent = (
                        100.0
                        * (
                            frame_id + 1
                        )
                        / total_frames
                    )


                    print(
                        f"Rendered "
                        f"{frame_id + 1}/"
                        f"{total_frames} "
                        f"({percent:.1f}%)"
                    )


        finally:

            for writer in (
                individual_writers.values()
            ):

                writer.release()


            combined_writer.release()


    finally:

        for capture in (
            captures.values()
        ):

            capture.release()


    print()
    print(
        "VIDEO RENDER COMPLETE"
    )

    print(
        "====================="
    )


    for camera_id in range(
        4
    ):

        path = (
            output_dir
            /
            (
                "terrace_global_ids_"
                f"c{camera_id}.mp4"
            )
        )


        print(
            path
        )


    print(
        output_dir
        /
        "terrace_4cam_global_ids.mp4"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Render validated Terrace "
            "four-camera global identities"
        )
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=
            DEFAULT_OUTPUT_DIR,
        help=(
            "Presentation output directory"
        ),
    )


    parser.add_argument(
        "--snapshot-frame",
        type=int,
        default=2000,
        help=(
            "Frame used for a synchronized "
            "2x2 JPEG snapshot"
        ),
    )


    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Optional rendering limit. "
            "0 renders the full sequence."
        ),
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


    (
        manager,
        tracklet_lookup,
        c3_records,
    ) = replay_four_camera_state()


    (
        overlays,
        assignment_rows,
    ) = build_frame_overlays(
        manager=
            manager,

        tracklet_lookup=
            tracklet_lookup,

        c3_records=
            c3_records,
    )


    assignment_path = (
        output_dir
        /
        "terrace_final_identity_assignments.csv"
    )


    write_assignment_csv(
        output_path=
            assignment_path,

        rows=
            assignment_rows,
    )


    print()
    print(
        "Identity table:",
        assignment_path,
    )


    render_videos(
        overlays=
            overlays,

        output_dir=
            output_dir,

        snapshot_frame=
            int(
                args.snapshot_frame
            ),

        max_frames=
            (
                None
                if args.max_frames <= 0
                else args.max_frames
            ),
    )


if __name__ == "__main__":
    main()
