from pathlib import Path
from collections import defaultdict

import cv2

from build_tracklets import load_tracklets


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_CSV = (
    PROJECT_ROOT
    / "output"
    / "terrace_bytetrack_tuned"
    / "tracks_c0.csv"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "videos"
    / "terrace1-c0.avi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "terrace_crops"
    / "c0"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_DETECTIONS = 25

CROPS_PER_TRACKLET = 8


# ============================================================
# SELECT REPRESENTATIVE DETECTIONS
# ============================================================

def select_detections(tracklet, count):

    detections = tracklet.detections

    if len(detections) <= count:
        return detections

    selected = []

    for i in range(count):

        index = round(
            i
            * (len(detections) - 1)
            / (count - 1)
        )

        selected.append(
            detections[index]
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD TRACKLETS
    # --------------------------------------------------------

    tracklets = load_tracklets(
        TRACK_CSV
    )

    useful_tracklets = [
        tracklet
        for tracklet in tracklets.values()
        if tracklet.num_detections >= MIN_DETECTIONS
    ]

    print()
    print("FAST CROP EXTRACTION")
    print("====================")
    print()

    print(
        f"Useful tracklets: {len(useful_tracklets)}"
    )

    print(
        f"Maximum crops per tracklet: {CROPS_PER_TRACKLET}"
    )

    print()

    # --------------------------------------------------------
    # BUILD FRAME LOOKUP
    # --------------------------------------------------------
    #
    # Instead of seeking the video for every crop:
    #
    # frame 121 -> [(tracklet, detection), ...]
    # frame 453 -> [(tracklet, detection), ...]
    #
    # Then we read the video only ONCE.
    #
    # --------------------------------------------------------

    frame_requests = defaultdict(list)

    for tracklet in useful_tracklets:

        selected = select_detections(
            tracklet,
            CROPS_PER_TRACKLET,
        )

        for detection in selected:

            frame_requests[
                detection.frame
            ].append(
                (
                    tracklet,
                    detection,
                )
            )

    requested_frames = set(
        frame_requests.keys()
    )

    print(
        f"Unique video frames needed: "
        f"{len(requested_frames)}"
    )

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # SINGLE SEQUENTIAL PASS
    # --------------------------------------------------------

    frame_index = 0

    total_saved = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # Only process frames that contain requested crops
        if frame_index in frame_requests:

            height, width = frame.shape[:2]

            requests = frame_requests[
                frame_index
            ]

            for tracklet, detection in requests:

                x1 = max(
                    0,
                    int(detection.x1),
                )

                y1 = max(
                    0,
                    int(detection.y1),
                )

                x2 = min(
                    width,
                    int(detection.x2),
                )

                y2 = min(
                    height,
                    int(detection.y2),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:
                    continue

                # --------------------------------------------
                # TRACK-SPECIFIC OUTPUT DIRECTORY
                # --------------------------------------------

                track_dir = (
                    OUTPUT_DIR
                    / f"track_{tracklet.local_track_id}"
                )

                track_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # --------------------------------------------
                # SAVE CROP
                # --------------------------------------------

                crop_path = (
                    track_dir
                    / f"frame_{frame_index:06d}.jpg"
                )

                success = cv2.imwrite(
                    str(crop_path),
                    crop,
                )

                if success:
                    total_saved += 1

        frame_index += 1

        if frame_index % 500 == 0:

            print(
                f"Read "
                f"{frame_index}/"
                f"{total_frames} frames | "
                f"saved {total_saved} crops"
            )

    cap.release()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("Crop extraction completed")
    print("=========================")
    print()

    print(
        f"Video frames read:  {frame_index}"
    )

    print(
        f"Tracklets:          {len(useful_tracklets)}"
    )

    print(
        f"Crops saved:        {total_saved}"
    )

    print(
        f"Output directory:   {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
