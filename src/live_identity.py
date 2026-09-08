"""Persistent live identities for multi-camera person tracking.

One physical person -> one GID for the whole run.
Leave / re-enter reuses the same GID. New people get the next free ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Prefer reusing an existing person over minting a new GID.
GID_APPEARANCE_MIN = 0.58
GID_RECOVERY_MIN = 0.60
SAME_TRACK_MIN = 0.45
NEW_PERSON_MIN = 0.72
NEW_ID_CONFIRM_FRAMES = 5
NEW_ID_MAX_AGE = 30
EMA_ALPHA = 0.08
MAX_EMBEDDINGS = 60

# Keep person in gallery forever; only the "left camera" event uses this.
CAMERA_LEAVE_FRAMES = 50

# Same-camera box continuity when ByteTrack changes local ID.
IOU_CONTINUE_MIN = 0.35
CENTER_CONTINUE_MAX = 80.0
CONTINUE_MAX_GAP = 45

# Suppress duplicate YOLO boxes on one body.
NMS_IOU = 0.55


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def suppress_overlapping_detections(detections: List[dict], iou_thresh: float = NMS_IOU) -> List[dict]:
    """Keep one box per body. Prefer larger boxes (more complete crop)."""

    if len(detections) <= 1:
        return detections

    ranked = sorted(
        detections,
        key=lambda d: (
            float(d["box"][2] - d["box"][0]) * float(d["box"][3] - d["box"][1])
        ),
        reverse=True,
    )
    kept: List[dict] = []
    for det in ranked:
        if any(box_iou(det["box"], other["box"]) >= iou_thresh for other in kept):
            continue
        kept.append(det)
    return kept


def histogram_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    height, width = crop_bgr.shape[:2]
    if height < 16 or width < 8:
        return None

    y1 = int(height * 0.20)
    y2 = int(height * 0.80)
    torso = crop_bgr[y1:y2, :]
    if torso.size == 0:
        torso = crop_bgr

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [18, 16],
        [0, 180, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return l2_normalize(hist)


def clothing_color_name(crop_bgr: np.ndarray) -> str:
    """Optional attribute; not shown on boxes."""

    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    height, width = crop_bgr.shape[:2]
    y1 = int(height * 0.25)
    y2 = int(height * 0.70)
    torso = crop_bgr[y1:y2, :]
    if torso.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    if float(np.mean(val)) < 50:
        return "black"
    if float(np.mean(sat)) < 40:
        if float(np.mean(val)) > 170:
            return "white"
        return "gray"

    mean_hue = float(np.mean(hue[sat > 40])) if np.any(sat > 40) else float(np.mean(hue))
    bands = (
        (0, "red"),
        (15, "orange"),
        (30, "yellow"),
        (45, "green"),
        (75, "green"),
        (100, "cyan"),
        (125, "blue"),
        (150, "purple"),
        (170, "red"),
    )
    for boundary, label in bands:
        if mean_hue <= boundary:
            return label
    return "unknown"


@dataclass
class IdentityEvent:
    timestamp: float
    frame: int
    event: str
    global_id: int
    camera_id: int
    detail: str = ""


@dataclass
class PersonRecord:
    global_id: int
    name: str = ""
    embeddings: List[np.ndarray] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None
    clothing_color: str = "unknown"
    gender: str = "unknown"
    first_seen_frame: int = 0
    first_seen_time: float = 0.0
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    last_camera_id: int = -1
    last_center: Tuple[float, float] = (0.0, 0.0)
    last_track_by_camera: Dict[int, int] = field(default_factory=dict)
    last_box_by_camera: Dict[int, Tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    last_center_by_camera: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    visible_cameras: Dict[int, int] = field(default_factory=dict)
    enter_count: int = 0
    leave_count: int = 0
    is_watchlist: bool = False

    def appearance_score(self, embedding: np.ndarray) -> float:
        scores = []
        if self.prototype is not None:
            scores.append(cosine_similarity(embedding, self.prototype))
        for stored in self.embeddings[-20:]:
            scores.append(cosine_similarity(embedding, stored))
        if not scores:
            return -1.0
        return max(scores)

    def update_embedding(self, embedding: np.ndarray) -> None:
        embedding = l2_normalize(embedding)
        self.embeddings.append(embedding)
        if len(self.embeddings) > MAX_EMBEDDINGS:
            self.embeddings.pop(0)

        if self.prototype is None:
            self.prototype = embedding.copy()
            return

        updated = (1.0 - EMA_ALPHA) * self.prototype + EMA_ALPHA * embedding
        self.prototype = l2_normalize(updated)

    def display_name(self) -> str:
        if self.name:
            return self.name
        return f"GID {self.global_id}"

    def details_line(self) -> str:
        watch = " TARGET" if self.is_watchlist else ""
        return (
            f"{self.display_name()}{watch} | "
            f"enters:{self.enter_count} leaves:{self.leave_count}"
        )


class LiveIdentityGallery:
    def __init__(
        self,
        appearance_min: float = GID_APPEARANCE_MIN,
        recovery_min: float = GID_RECOVERY_MIN,
        camera_leave_frames: int = CAMERA_LEAVE_FRAMES,
    ):
        self.people: Dict[int, PersonRecord] = {}
        self.next_global_id = 1
        self.appearance_min = appearance_min
        self.recovery_min = recovery_min
        self.camera_leave_frames = camera_leave_frames
        self.events: List[IdentityEvent] = []
        self.local_bind: Dict[Tuple[int, int], int] = {}
        self._new_candidates: List[dict] = []

    def enroll_embedding(
        self,
        embedding: np.ndarray,
        name: str = "",
        watchlist: bool = True,
        clothing_color: str = "unknown",
    ) -> int:
        gid = self.next_global_id
        self.next_global_id += 1
        person = PersonRecord(
            global_id=gid,
            name=name,
            clothing_color=clothing_color,
            is_watchlist=watchlist,
        )
        person.update_embedding(embedding)
        self.people[gid] = person
        return gid

    def set_target_gid(self, target_gid: int) -> None:
        for person in self.people.values():
            person.is_watchlist = person.global_id == target_gid

    def _emit(
        self,
        timestamp: float,
        frame: int,
        event: str,
        person: PersonRecord,
        camera_id: int,
        detail: str = "",
    ) -> None:
        self.events.append(
            IdentityEvent(
                timestamp=timestamp,
                frame=frame,
                event=event,
                global_id=person.global_id,
                camera_id=camera_id,
                detail=detail or person.details_line(),
            )
        )

    def mark_seen(
        self,
        person: PersonRecord,
        camera_id: int,
        track_id: int,
        frame: int,
        timestamp: float,
        center: Tuple[float, float],
        box,
        embedding: Optional[np.ndarray],
        clothing_color: str,
    ) -> None:
        was_on_camera = camera_id in person.visible_cameras
        was_on_site = bool(person.visible_cameras)

        if not was_on_site:
            person.enter_count += 1
            event = "RE-ENTER" if person.leave_count else "ENTER"
            self._emit(
                timestamp,
                frame,
                event,
                person,
                camera_id,
                f"{event} site via camera {camera_id}",
            )
        elif not was_on_camera:
            self._emit(
                timestamp,
                frame,
                "CAMERA_ENTER",
                person,
                camera_id,
                f"appeared on camera {camera_id}",
            )

        person.visible_cameras[camera_id] = frame
        person.last_seen_frame = frame
        person.last_seen_time = timestamp
        person.last_camera_id = camera_id
        person.last_center = center
        person.last_track_by_camera[camera_id] = track_id
        person.last_box_by_camera[camera_id] = tuple(float(v) for v in box)
        person.last_center_by_camera[camera_id] = center
        self.local_bind[(camera_id, track_id)] = person.global_id

        if clothing_color != "unknown":
            person.clothing_color = clothing_color
        if embedding is not None:
            person.update_embedding(embedding)

    def close_missing(self, frame: int, timestamp: float) -> None:
        stale_binds = [
            key
            for key, gid in self.local_bind.items()
            if gid in self.people
            and (
                key[0] not in self.people[gid].visible_cameras
                or frame - self.people[gid].visible_cameras.get(key[0], -10**9)
                >= self.camera_leave_frames
            )
        ]
        for key in stale_binds:
            self.local_bind.pop(key, None)

        for person in self.people.values():
            missing_cameras = []
            for camera_id, last_frame in list(person.visible_cameras.items()):
                if frame - last_frame >= self.camera_leave_frames:
                    missing_cameras.append(camera_id)

            for camera_id in missing_cameras:
                person.visible_cameras.pop(camera_id, None)
                self._emit(
                    timestamp,
                    frame,
                    "CAMERA_LEAVE",
                    person,
                    camera_id,
                    f"left camera {camera_id}",
                )

            if missing_cameras and not person.visible_cameras:
                person.leave_count += 1
                self._emit(
                    timestamp,
                    frame,
                    "LEAVE",
                    person,
                    missing_cameras[0],
                    "left all cameras",
                )

        alive = []
        for candidate in self._new_candidates:
            if frame - candidate["last_frame"] <= NEW_ID_MAX_AGE:
                alive.append(candidate)
        self._new_candidates = alive

    def best_gallery_score(self, embedding: np.ndarray, used_gids: set) -> Tuple[float, Optional[int]]:
        best_score = -1.0
        best_gid = None
        for person in self.people.values():
            if person.global_id in used_gids:
                continue
            score = person.appearance_score(embedding)
            if score > best_score:
                best_score = score
                best_gid = person.global_id
        return best_score, best_gid

    def match_detections(
        self,
        camera_id: int,
        frame: int,
        timestamp: float,
        detections: List[dict],
    ) -> Dict[int, int]:
        """Map local track_id -> global_id. At most one box per GID on this camera."""

        detections = suppress_overlapping_detections(detections)
        assigned: Dict[int, int] = {}
        used_gids = set()

        # 1) Continue same ByteTrack local ID.
        for detection in detections:
            track_id = detection["track_id"]
            bind_key = (camera_id, track_id)
            if bind_key not in self.local_bind:
                continue
            gid = self.local_bind[bind_key]
            person = self.people.get(gid)
            if person is None or gid in used_gids:
                continue
            score = person.appearance_score(detection["embedding"])
            if score >= SAME_TRACK_MIN or score < 0:
                assigned[track_id] = gid
                used_gids.add(gid)

        # 2) Same-camera spatial continuity (tracker ID changed / short gap).
        spatial_scored = []
        for detection in detections:
            track_id = detection["track_id"]
            if track_id in assigned:
                continue
            for person in self.people.values():
                if person.global_id in used_gids:
                    continue
                last_box = person.last_box_by_camera.get(camera_id)
                last_center = person.last_center_by_camera.get(camera_id)
                last_seen = person.visible_cameras.get(camera_id, person.last_seen_frame)
                gap = frame - last_seen
                if last_box is None or gap < 0 or gap > CONTINUE_MAX_GAP:
                    continue
                iou = box_iou(detection["box"], last_box)
                dist = (
                    center_distance(detection["center"], last_center)
                    if last_center is not None
                    else 1e9
                )
                if iou < IOU_CONTINUE_MIN and dist > CENTER_CONTINUE_MAX:
                    continue
                appearance = person.appearance_score(detection["embedding"])
                if appearance < SAME_TRACK_MIN and iou < 0.55:
                    continue
                spatial_scored.append((iou + 0.15 * appearance, track_id, person.global_id))

        spatial_scored.sort(key=lambda item: item[0], reverse=True)
        for _, track_id, gid in spatial_scored:
            if track_id in assigned or gid in used_gids:
                continue
            assigned[track_id] = gid
            used_gids.add(gid)

        # 3) Appearance match to any existing GID (cross-camera / re-enter).
        scored = []
        for detection in detections:
            track_id = detection["track_id"]
            if track_id in assigned:
                continue
            embedding = detection["embedding"]
            for person in self.people.values():
                if person.global_id in used_gids:
                    continue
                missing = frame - person.last_seen_frame
                score = person.appearance_score(embedding)
                if missing > 0:
                    threshold = self.recovery_min
                else:
                    threshold = self.appearance_min
                if score < threshold:
                    continue
                scored.append((score, track_id, person.global_id))

        scored.sort(key=lambda item: item[0], reverse=True)
        for score, track_id, gid in scored:
            if track_id in assigned or gid in used_gids:
                continue
            assigned[track_id] = gid
            used_gids.add(gid)

        # 4) Only mint a new GID when gallery match is clearly weak.
        for detection in detections:
            track_id = detection["track_id"]
            if track_id in assigned:
                continue
            best_score, best_gid = self.best_gallery_score(
                detection["embedding"], used_gids
            )
            if best_gid is not None and best_score >= self.recovery_min:
                assigned[track_id] = best_gid
                used_gids.add(best_gid)
                continue

            gid = self._confirm_or_hold_new(
                detection,
                camera_id,
                frame,
                best_gallery_score=best_score,
            )
            if gid is None or gid in used_gids:
                continue
            assigned[track_id] = gid
            used_gids.add(gid)

        for detection in detections:
            track_id = detection["track_id"]
            gid = assigned.get(track_id)
            if gid is None:
                continue
            person = self.people[gid]
            if person.first_seen_frame == 0:
                person.first_seen_frame = frame
                person.first_seen_time = timestamp
            self.mark_seen(
                person=person,
                camera_id=camera_id,
                track_id=track_id,
                frame=frame,
                timestamp=timestamp,
                center=detection["center"],
                box=detection["box"],
                embedding=detection["embedding"],
                clothing_color=detection.get("clothing_color", "unknown"),
            )

        return assigned

    def _confirm_or_hold_new(
        self,
        detection: dict,
        camera_id: int,
        frame: int,
        best_gallery_score: float,
    ) -> Optional[int]:
        # Do not open a new person if an old GID is still a plausible match.
        if best_gallery_score >= NEW_PERSON_MIN - 0.08:
            return None

        embedding = detection["embedding"]
        best_index = -1
        best_score = -1.0

        for index, candidate in enumerate(self._new_candidates):
            score = cosine_similarity(embedding, candidate["prototype"])
            if score > best_score:
                best_score = score
                best_index = index

        if best_index >= 0 and best_score >= 0.68:
            candidate = self._new_candidates[best_index]
            candidate["count"] += 1
            candidate["last_frame"] = frame
            candidate["prototype"] = l2_normalize(
                (1.0 - EMA_ALPHA) * candidate["prototype"] + EMA_ALPHA * embedding
            )
            if candidate["count"] >= NEW_ID_CONFIRM_FRAMES:
                person = self._create_person(embedding, frame)
                del self._new_candidates[best_index]
                return person.global_id
            return None

        self._new_candidates.append(
            {
                "prototype": l2_normalize(embedding),
                "count": 1,
                "last_frame": frame,
                "camera_id": camera_id,
            }
        )
        return None

    def _create_person(self, embedding: np.ndarray, frame: int) -> PersonRecord:
        gid = self.next_global_id
        self.next_global_id += 1
        person = PersonRecord(global_id=gid, first_seen_frame=frame)
        person.update_embedding(embedding)
        self.people[gid] = person
        return person

    def snapshot_rows(self) -> List[dict]:
        rows = []
        for person in self.people.values():
            rows.append(
                {
                    "global_id": person.global_id,
                    "name": person.name,
                    "watchlist": person.is_watchlist,
                    "clothing_color": person.clothing_color,
                    "gender": person.gender,
                    "enter_count": person.enter_count,
                    "leave_count": person.leave_count,
                    "first_seen_time": round(person.first_seen_time, 2),
                    "last_seen_time": round(person.last_seen_time, 2),
                    "last_camera": person.last_camera_id,
                    "visible_cameras": sorted(person.visible_cameras.keys()),
                }
            )
        return rows
