#!/usr/bin/env python3
"""Live-style multi-camera person tracking prototype.

Use the same command for the Terrace dataset today and office RTSP later.

Full terrace video (~200s):

    python src/run_live_multicam.py --preset terrace --no-display

Highlight GID 4 with a different thin box:

    python src/run_live_multicam.py --preset terrace --target-gid 4 --no-display

Two office cameras when your lead gives RTSP URLs:

    python src/run_live_multicam.py \\
        --sources rtsp://user:pass@cam1/stream rtsp://user:pass@cam2/stream \\
        --camera-ids 0 1 \\
        --enroll-image lead.jpg --target-name Lead

Keys while a window is open:
    q  quit
    p  pause
    t  set watchlist to the first visible GID
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.live_identity import (
        LiveIdentityGallery,
        clothing_color_name,
        histogram_embedding,
    )
except ImportError:
    from live_identity import (
        LiveIdentityGallery,
        clothing_color_name,
        histogram_embedding,
    )


DEFAULT_TERRACE_DIRS = [
    PROJECT_ROOT / "data" / "videos",
    Path.home() / "Downloads",
    Path("/home/delen02/Downloads"),
    Path("/home/jetson_agx_orin/Downloads"),
]

# Target person highlight (BGR). Everyone else uses a thin neutral GID color.
TARGET_COLOR = (0, 220, 255)

TRACKER_CONFIG = SRC_DIR / "bytetrack_terrace.yaml"
YOLO_CANDIDATES = [
    PROJECT_ROOT / "models" / "yolov8n.pt",
    Path("yolov8n.pt"),
]
REID_WEIGHTS = PROJECT_ROOT / "models" / "reid" / "osnet_x0_25_msmt17.pth"

GID_COLORS = [
    (40, 90, 230),
    (40, 180, 80),
    (230, 90, 40),
    (40, 200, 220),
    (200, 60, 200),
    (30, 160, 240),
    (80, 80, 220),
    (20, 210, 180),
]


def find_yolo_weights() -> str:
    for path in YOLO_CANDIDATES:
        if path.exists():
            return str(path)
    return "yolov8n.pt"


def find_terrace_video(camera_id: int) -> Path:
    name = f"terrace1-c{camera_id}.avi"
    for folder in DEFAULT_TERRACE_DIRS:
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {name}. Place it in data/videos or pass --sources."
    )


def select_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def gid_color(global_id: int) -> tuple:
    return GID_COLORS[(global_id - 1) % len(GID_COLORS)]


class ReIDBackend:
    def __init__(self, device: str):
        self.extractor = None
        try:
            try:
                from src.reid_extractor import ReIDExtractor
            except ImportError:
                from reid_extractor import ReIDExtractor

            if REID_WEIGHTS.exists():
                torch_device = "cuda" if device not in ("cpu",) else "cpu"
                self.extractor = ReIDExtractor(device=torch_device)
                print(f"ReID: OSNet weights {REID_WEIGHTS}")
            else:
                print(
                    "ReID: OSNet weights not found, using clothing histogram "
                    f"(looked for {REID_WEIGHTS})"
                )
        except Exception as exc:
            print(f"ReID: OSNet unavailable ({exc}). Using clothing histogram.")
            self.extractor = None

    def embed(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if self.extractor is not None:
            try:
                feature = self.extractor.extract(crop_bgr)
                if feature is not None:
                    return feature
            except Exception:
                pass
        return histogram_embedding(crop_bgr)


class CameraSource:
    def __init__(self, camera_id: int, uri: str, live: bool):
        self.camera_id = camera_id
        self.uri = uri
        self.live = live
        self.cap = None
        self._open()

    def _open(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}: {self.uri}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self.live and self.cap is not None:
            self.cap.grab()
        ok, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ok and self.live:
            time.sleep(0.4)
            try:
                self._open()
                ok, frame = self.cap.read()
            except RuntimeError:
                return False, None
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def crop_person(frame, box, min_width=24, min_height=48):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    if (x2 - x1) < min_width or (y2 - y1) < min_height:
        return None
    return frame[y1:y2, x1:x2]


def build_mosaic(frames: List[np.ndarray], labels: List[str], tile_w=640, tile_h=480):
    tiles = []
    for frame, label in zip(frames, labels):
        if frame is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(frame, (tile_w, tile_h))
        cv2.rectangle(tile, (0, 0), (tile_w, 28), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    while len(tiles) < 2:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    if len(tiles) == 2:
        return np.hstack(tiles)
    if len(tiles) == 3:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    top = np.hstack(tiles[0:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


def draw_person(frame, box, person, track_id, pending=False):
    x1, y1, x2, y2 = [int(v) for v in box]
    thickness = 1
    if pending or person is None:
        # Hold quietly until a GID is confirmed — avoids double boxes.
        return
    if person.is_watchlist:
        color = TARGET_COLOR
        label = f"{person.display_name()} TARGET"
    else:
        color = gid_color(person.global_id)
        label = person.display_name()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y_text = max(18, y1 - 4)
    cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0) if person.is_watchlist else (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def detections_for_draw(detections, assigned, gallery):
    """One drawn box per GID on this camera (largest box wins)."""

    by_gid = {}
    for detection in detections:
        gid = assigned.get(detection["track_id"])
        if gid is None:
            continue
        area = float(
            (detection["box"][2] - detection["box"][0])
            * (detection["box"][3] - detection["box"][1])
        )
        prev = by_gid.get(gid)
        if prev is None or area > prev[0]:
            by_gid[gid] = (area, detection)
    rows = []
    for gid, (_, detection) in by_gid.items():
        rows.append((detection, gallery.people.get(gid)))
    return rows


def draw_event_strip(mosaic, events, gallery):
    height, width = mosaic.shape[:2]
    strip_h = 150
    canvas = np.zeros((height + strip_h, width, 3), dtype=np.uint8)
    canvas[:height] = mosaic
    cv2.rectangle(canvas, (0, height), (width, height + strip_h), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        "Events  ENTER / CAMERA_ENTER / LEAVE / RE-ENTER",
        (10, height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    y = height + 44
    for event in events[-5:]:
        person = gallery.people.get(event.global_id)
        name = person.display_name() if person else f"GID {event.global_id}"
        line = (
            f"{event.timestamp:7.1f}s  {event.event:<13}  {name}  "
            f"cam {event.camera_id}  {event.detail}"
        )
        color = (80, 220, 80) if "ENTER" in event.event else (80, 180, 230)
        if event.event == "LEAVE":
            color = (80, 80, 230)
        cv2.putText(
            canvas,
            line[:140],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-camera live GID prototype (files or RTSP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preset",
        choices=["terrace"],
        help="Load terrace1-c0..c3 videos",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Video files or rtsp:// URLs, one per camera",
    )
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        type=int,
        help="Camera ids matching --sources (default 0 1 2 ...)",
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 = whole video (~200s / 5000 frames for terrace)",
    )
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "live_demo")
    parser.add_argument("--enroll-image", type=Path, help="Watchlist crop of the target person")
    parser.add_argument("--target-name", default="Target")
    parser.add_argument(
        "--target-gid",
        type=int,
        help="Highlight this GID with a different thin box color",
    )
    return parser.parse_args()


def resolve_sources(args) -> List[CameraSource]:
    if args.preset == "terrace":
        camera_ids = args.cameras
        uris = [str(find_terrace_video(camera_id)) for camera_id in camera_ids]
        live = False
    elif args.sources:
        uris = args.sources
        if args.camera_ids:
            camera_ids = args.camera_ids
        else:
            camera_ids = list(range(len(uris)))
        if len(camera_ids) != len(uris):
            raise ValueError("--camera-ids must match --sources")
        live = any(uri.lower().startswith("rtsp://") for uri in uris)
    else:
        raise ValueError("Pass --preset terrace or --sources ...")

    sources = []
    for camera_id, uri in zip(camera_ids, uris):
        print(f"Camera {camera_id}: {uri}")
        sources.append(CameraSource(camera_id, uri, live=live))
    return sources


def enroll_from_image(gallery: LiveIdentityGallery, reid: ReIDBackend, image_path: Path, name: str):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read enroll image: {image_path}")
    embedding = reid.embed(image)
    if embedding is None:
        raise RuntimeError("Could not embed enroll image")
    gid = gallery.enroll_embedding(
        embedding,
        name=name,
        watchlist=True,
        clothing_color=clothing_color_name(image),
    )
    print(f"Watchlist enrolled as GID {gid} ({name})")
    return gid


def save_outputs(output_dir: Path, gallery: LiveIdentityGallery):
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.csv"
    with events_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["timestamp", "frame", "event", "global_id", "camera_id", "detail"]
        )
        for event in gallery.events:
            writer.writerow(
                [
                    f"{event.timestamp:.3f}",
                    event.frame,
                    event.event,
                    event.global_id,
                    event.camera_id,
                    event.detail,
                ]
            )
    people_path = output_dir / "identities.json"
    people_path.write_text(json.dumps(gallery.snapshot_rows(), indent=2))
    print(f"Wrote {events_path}")
    print(f"Wrote {people_path}")


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    from ultralytics import YOLO

    yolo_weights = find_yolo_weights()
    print(f"YOLO: {yolo_weights}")
    if not TRACKER_CONFIG.exists():
        raise FileNotFoundError(TRACKER_CONFIG)

    sources = resolve_sources(args)
    models = [YOLO(yolo_weights) for _ in sources]
    reid = ReIDBackend(device)
    gallery = LiveIdentityGallery()

    if args.enroll_image:
        enroll_from_image(gallery, reid, args.enroll_image, args.target_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_index = 0
    paused = False
    start = time.time()

    try:
        while True:
            if args.max_frames and frame_index >= args.max_frames:
                break

            frames = []
            ended = False
            for source in sources:
                ok, frame = source.read()
                if not ok or frame is None:
                    ended = True
                    frames.append(None)
                else:
                    frames.append(frame)
            if ended and not any(source.live for source in sources):
                break
            if any(frame is None for frame in frames):
                if any(source.live for source in sources):
                    continue
                break

            if args.skip > 1 and frame_index % args.skip != 0:
                frame_index += 1
                continue

            timestamp = frame_index / 25.0
            annotated = []
            labels = []

            for source, model, frame in zip(sources, models, frames):
                results = model.track(
                    frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    classes=[0],
                    conf=args.conf,
                    device=device,
                    verbose=False,
                )
                result = results[0]
                vis = frame.copy()
                detections = []

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, track_ids):
                        crop = crop_person(frame, box)
                        embedding = reid.embed(crop) if crop is not None else None
                        if embedding is None:
                            continue
                        x1, y1, x2, y2 = box
                        detections.append(
                            {
                                "track_id": int(track_id),
                                "box": box,
                                "embedding": embedding,
                                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                                "clothing_color": clothing_color_name(crop),
                            }
                        )

                assigned = gallery.match_detections(
                    camera_id=source.camera_id,
                    frame=frame_index,
                    timestamp=timestamp,
                    detections=detections,
                )

                if args.target_gid:
                    gallery.set_target_gid(args.target_gid)

                for detection, person in detections_for_draw(
                    detections, assigned, gallery
                ):
                    draw_person(
                        vis,
                        detection["box"],
                        person,
                        detection["track_id"],
                        pending=person is None,
                    )

                annotated.append(vis)
                n_people = len({gid for gid in assigned.values()})
                labels.append(
                    f"Camera {source.camera_id}  frame {frame_index:04d}  people {n_people}"
                )

            gallery.close_missing(frame_index, timestamp)
            mosaic = build_mosaic(annotated, labels)
            mosaic = draw_event_strip(mosaic, gallery.events, gallery)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = args.output_dir / "live_mosaic.mp4"
                writer = cv2.VideoWriter(
                    str(out_path),
                    fourcc,
                    25.0,
                    (mosaic.shape[1], mosaic.shape[0]),
                )
                print(f"Writing {out_path}")

            writer.write(mosaic)

            if not args.no_display:
                cv2.imshow("live-multicam", mosaic)
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("t"):
                    visible = [
                        person
                        for person in gallery.people.values()
                        if person.visible_cameras
                    ]
                    if visible:
                        for person in gallery.people.values():
                            person.is_watchlist = False
                        visible[0].is_watchlist = True
                        print(f"Watchlist set to GID {visible[0].global_id}")

            if frame_index % 50 == 0:
                print(
                    f"frame {frame_index}  gids={len(gallery.people)}  "
                    f"events={len(gallery.events)}"
                )

            frame_index += 1

    finally:
        if writer is not None:
            writer.release()
        for source in sources:
            source.release()
        # Save first: Jetson OpenCV is often headless and destroyAllWindows
        # raises even after a successful --no-display run.
        save_outputs(args.output_dir, gallery)
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        elapsed = time.time() - start
        print(
            f"Done. frames={frame_index} people={len(gallery.people)} "
            f"events={len(gallery.events)} time={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
