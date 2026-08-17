from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "calibration"
    / "calibration-terrace.txt"
)


def normalize_camera_id(camera_id):

    if isinstance(camera_id, str):

        camera_id = camera_id.strip().lower()

        if camera_id.startswith("c"):
            camera_id = camera_id[1:]

    return int(camera_id)


def load_ground_homographies(
    calibration_path=DEFAULT_CALIBRATION_PATH,
):

    calibration_path = Path(
        calibration_path
    )

    if not calibration_path.exists():

        raise FileNotFoundError(
            f"Calibration file not found: "
            f"{calibration_path}"
        )

    lines = calibration_path.read_text().splitlines()

    homographies = {}

    current_camera = None

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # Example:
        #
        # # Camera 0
        # # Camera 1
        if line.lower().startswith("# camera"):

            parts = line.split()

            current_camera = int(
                parts[-1]
            )

        if (
            current_camera is not None
            and "ground plane homography"
            in line.lower()
        ):

            matrix_rows = []

            j = i + 1

            while (
                j < len(lines)
                and len(matrix_rows) < 3
            ):

                row_text = lines[j].strip()

                if (
                    row_text
                    and not row_text.startswith("#")
                ):

                    values = [
                        float(value)
                        for value
                        in row_text.split()
                    ]

                    if len(values) == 3:

                        matrix_rows.append(
                            values
                        )

                j += 1

            if len(matrix_rows) != 3:

                raise ValueError(
                    f"Could not read homography "
                    f"for camera {current_camera}"
                )

            homographies[
                current_camera
            ] = np.asarray(
                matrix_rows,
                dtype=np.float64,
            )

            i = j
            continue

        i += 1

    return homographies


def project_image_point(
    x,
    y,
    homography,
):

    point = np.asarray(
        [
            float(x),
            float(y),
            1.0,
        ],
        dtype=np.float64,
    )

    projected = (
        homography
        @ point
    )

    scale = projected[2]

    if abs(scale) < 1e-12:

        raise ValueError(
            "Homography projection produced "
            "a near-zero homogeneous scale."
        )

    ground_x = (
        projected[0]
        / scale
    )

    ground_y = (
        projected[1]
        / scale
    )

    return (
        float(ground_x),
        float(ground_y),
    )


def bbox_bottom_center(
    x1,
    y1,
    x2,
    y2,
):

    x = (
        float(x1)
        + float(x2)
    ) / 2.0

    y = float(y2)

    return x, y


def bbox_ground_point(
    x1,
    y1,
    x2,
    y2,
    camera_id,
    homographies,
):

    camera_id = normalize_camera_id(
        camera_id
    )

    if camera_id not in homographies:

        raise KeyError(
            f"No homography for camera "
            f"{camera_id}"
        )

    image_x, image_y = bbox_bottom_center(
        x1,
        y1,
        x2,
        y2,
    )

    return project_image_point(
        image_x,
        image_y,
        homographies[
            camera_id
        ],
    )


def detection_ground_point(
    detection,
    camera_id,
    homographies,
):

    return bbox_ground_point(
        x1=detection.x1,
        y1=detection.y1,
        x2=detection.x2,
        y2=detection.y2,
        camera_id=camera_id,
        homographies=homographies,
    )


def ground_distance(
    point_a,
    point_b,
):

    ax, ay = point_a
    bx, by = point_b

    return float(
        np.hypot(
            ax - bx,
            ay - by,
        )
    )
