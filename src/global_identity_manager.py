from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


try:
    from .global_identity import GlobalIdentity

    from .identity_compatibility import (
        identity_has_conflict,
        normalize_camera_id,
    )

    from .cross_camera_geometry_gate import (
        evaluate_cross_camera_geometry,
    )

except ImportError:
    from global_identity import GlobalIdentity

    from identity_compatibility import (
        identity_has_conflict,
        normalize_camera_id,
    )

    from cross_camera_geometry_gate import (
        evaluate_cross_camera_geometry,
    )


# ============================================================
# MATCH RESULT
# ============================================================

@dataclass
class MatchResult:

    status: str
    global_id: Optional[int]

    max_similarity: float
    mean_similarity: float
    topk_similarity: float

    second_global_id: Optional[int]
    second_topk_similarity: Optional[float]
    score_margin: Optional[float]


# ============================================================
# ASSIGNMENT RESULT
# ============================================================

@dataclass
class AssignmentResult:

    status: str
    global_id: Optional[int]

    created_new: bool
    merged: bool

    match: MatchResult

    trust_level: Optional[str] = None
    reason: Optional[str] = None


# ============================================================
# GLOBAL IDENTITY MANAGER
# ============================================================

class GlobalIdentityManager:

    def __init__(
        self,
        homographies=None,
    ):

        self.identities = {}

        self.next_global_id = 1

        self.pending_tracklets = {}

        self.homographies = homographies


        # ====================================================
        # APPEARANCE FALLBACK THRESHOLDS
        #
        # Appearance-only STRONG is only a hypothesis.
        # It is not direct merge permission.
        # ====================================================

        self.same_camera_strong = 0.92
        self.same_camera_pending = 0.88

        self.cross_camera_strong = 0.90
        self.cross_camera_pending = 0.84


        # ====================================================
        # STRICT / CORE CROSS-CAMERA POLICY
        # ====================================================

        self.strict_appearance_min = 0.88

        self.strict_geometry_median_max = 20.0


        # ====================================================
        # RELAXED CORE-SUPPORTED CROSS-CAMERA POLICY
        # ====================================================

        self.relaxed_gid_appearance_min = 0.84

        self.relaxed_core_direct_appearance_min = 0.86

        self.relaxed_geometry_median_max = 18.0

        self.relaxed_min_shared_frames = 15


        # ====================================================
        # SAME-CAMERA CORE CONTINUATION POLICY
        #
        # Validated conservative rule:
        #
        # same camera
        # CORE positive support
        # no overlap
        # gap <= 15
        # direct ReID >= .92
        # center distance <= 50 px
        # bottom distance <= 50 px
        #
        # Accepted member becomes RELAXED.
        # ====================================================

        self.same_camera_continuation_max_gap = 15

        self.same_camera_continuation_reid_min = 0.92

        self.same_camera_continuation_center_max = 50.0

        self.same_camera_continuation_bottom_max = 50.0


        # ====================================================
        # FINAL DUAL-EVIDENCE CONTINUATION POLICY
        #
        # This rule is deliberately weaker than the normal
        # same-camera ReID threshold ONLY because the same GID
        # must independently provide trusted same-camera CORE
        # support AND trusted cross-camera CORE geometry.
        #
        # Accepted members become RELAXED and never become
        # positive-support propagation sources.
        # ====================================================

        self.dual_same_camera_max_gap = 15

        self.dual_same_camera_reid_min = 0.90

        self.dual_same_camera_center_max = 50.0

        self.dual_same_camera_bottom_max = 50.0

        self.dual_cross_camera_reid_min = 0.85

        self.dual_cross_camera_min_shared_frames = 20

        self.dual_cross_camera_geometry_median_max = 18.0

        self.dual_gid_appearance_min = 0.84


        # ====================================================
        # FINAL CORE-ONLY HIGH-GALLERY CONTINUATION POLICY
        #
        # Positive evidence is CORE-only:
        #
        #   * at least 3 CORE gallery members
        #   * CORE gallery max >= .90
        #   * CORE gallery top-3 mean >= .90
        #   * cross-camera CORE direct ReID >= .83
        #   * cross-camera CORE shared frames >= 20
        #   * cross-camera CORE geometry median <= 18
        #
        # Accepted members become RELAXED.
        # ====================================================

        self.core_gallery_min_members = 3

        self.core_gallery_max_min = 0.90

        self.core_gallery_top3_min = 0.90

        self.core_gallery_cross_reid_min = 0.83

        self.core_gallery_cross_min_shared_frames = 20

        self.core_gallery_cross_geometry_median_max = 18.0


        # ====================================================
        # IDENTITY PROVENANCE
        # ====================================================

        self.anchored_gids = set()

        self.core_members = defaultdict(
            set
        )

        self.member_embeddings = {}


    # ========================================================
    # BASIC HELPERS
    # ========================================================

    def member_key(
        self,
        camera_id,
        local_track_id,
    ):

        return (
            normalize_camera_id(
                camera_id
            ),
            local_track_id,
        )


    def cosine_similarity(
        self,
        first,
        second,
    ):

        first = np.asarray(
            first,
            dtype=np.float32,
        )

        second = np.asarray(
            second,
            dtype=np.float32,
        )


        first_norm = np.linalg.norm(
            first
        )

        second_norm = np.linalg.norm(
            second
        )


        if (
            first_norm == 0
            or second_norm == 0
        ):

            return -1.0


        return float(
            np.dot(
                first / first_norm,
                second / second_norm,
            )
        )


    def register_member_embedding(
        self,
        camera_id,
        local_track_id,
        embedding,
    ):

        key = self.member_key(
            camera_id,
            local_track_id,
        )


        self.member_embeddings[
            key
        ] = np.asarray(
            embedding,
            dtype=np.float32,
        )


    # ========================================================
    # DETECTION HELPERS
    # ========================================================

    def first_detection(
        self,
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
        self,
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


    def bbox_center(
        self,
        detection,
    ):

        x = (
            detection.x1
            + detection.x2
        ) / 2.0


        y = (
            detection.y1
            + detection.y2
        ) / 2.0


        return np.asarray(
            [
                x,
                y,
            ],
            dtype=np.float32,
        )


    def bbox_bottom_center(
        self,
        detection,
    ):

        x = (
            detection.x1
            + detection.x2
        ) / 2.0


        y = detection.y2


        return np.asarray(
            [
                x,
                y,
            ],
            dtype=np.float32,
        )


    def chronological_pair(
        self,
        first,
        second,
    ):

        if (
            first.start_frame
            <= second.start_frame
        ):

            return (
                first,
                second,
            )


        return (
            second,
            first,
        )


    def same_camera_temporal_stats(
        self,
        first,
        second,
    ):

        first, second = (
            self.chronological_pair(
                first,
                second,
            )
        )


        gap = (
            second.start_frame
            - first.end_frame
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


        return (
            gap,
            overlap,
        )


    def same_camera_endpoint_distance(
        self,
        first,
        second,
    ):

        first, second = (
            self.chronological_pair(
                first,
                second,
            )
        )


        end_detection = (
            self.last_detection(
                first
            )
        )


        start_detection = (
            self.first_detection(
                second
            )
        )


        if (
            end_detection is None
            or start_detection is None
        ):

            return (
                None,
                None,
            )


        center_distance = float(
            np.linalg.norm(
                self.bbox_center(
                    end_detection
                )
                - self.bbox_center(
                    start_detection
                )
            )
        )


        bottom_distance = float(
            np.linalg.norm(
                self.bbox_bottom_center(
                    end_detection
                )
                - self.bbox_bottom_center(
                    start_detection
                )
            )
        )


        return (
            center_distance,
            bottom_distance,
        )


    # ========================================================
    # MEMBER TRUST
    # ========================================================

    def get_member_trust(
        self,
        global_id,
        camera_id,
        local_track_id,
    ):

        if global_id not in self.identities:

            return None


        key = self.member_key(
            camera_id,
            local_track_id,
        )


        if global_id not in self.anchored_gids:

            return "UNANCHORED"


        if key in self.core_members[
            global_id
        ]:

            return "CORE"


        return "RELAXED"


    def mark_core(
        self,
        global_id,
        camera_id,
        local_track_id,
    ):

        if global_id not in self.identities:

            raise KeyError(
                f"GID_{global_id:04d} does not exist"
            )


        self.anchored_gids.add(
            global_id
        )


        self.core_members[
            global_id
        ].add(
            self.member_key(
                camera_id,
                local_track_id,
            )
        )


    def bootstrap_identity_as_core(
        self,
        global_id,
    ):

        if global_id in self.anchored_gids:

            return


        identity = self.identities[
            global_id
        ]


        for member in identity.members:

            self.core_members[
                global_id
            ].add(
                self.member_key(
                    member.camera_id,
                    member.local_track_id,
                )
            )


        self.anchored_gids.add(
            global_id
        )


    # ========================================================
    # CREATE NEW IDENTITY
    # ========================================================

    def create_identity(
        self,
        camera_id,
        local_track_id,
        start_frame,
        end_frame,
        embedding,
        trust_level=None,
    ):

        camera_id = normalize_camera_id(
            camera_id
        )


        gid = self.next_global_id


        identity = GlobalIdentity(
            global_id=gid
        )


        identity.add_member(
            camera_id=camera_id,
            local_track_id=local_track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            embedding=embedding,
        )


        self.identities[
            gid
        ] = identity


        self.register_member_embedding(
            camera_id,
            local_track_id,
            embedding,
        )


        if trust_level == "CORE":

            self.mark_core(
                gid,
                camera_id,
                local_track_id,
            )


        self.next_global_id += 1


        return gid


    # ========================================================
    # ADD TO EXISTING IDENTITY
    # ========================================================

    def add_to_identity(
        self,
        global_id,
        camera_id,
        local_track_id,
        start_frame,
        end_frame,
        embedding,
        trust_level=None,
    ):

        if global_id not in self.identities:

            raise KeyError(
                f"GID_{global_id:04d} does not exist"
            )


        camera_id = normalize_camera_id(
            camera_id
        )


        self.identities[
            global_id
        ].add_member(
            camera_id=camera_id,
            local_track_id=local_track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            embedding=embedding,
        )


        self.register_member_embedding(
            camera_id,
            local_track_id,
            embedding,
        )


        if trust_level == "CORE":

            self.mark_core(
                global_id,
                camera_id,
                local_track_id,
            )


        # RELAXED intentionally does not enter core_members.


    # ========================================================
    # IDENTITY SCORE
    # ========================================================

    def score_identity(
        self,
        identity,
        embedding,
    ):

        return {
            "max":
                identity.max_similarity(
                    embedding
                ),

            "mean":
                identity.mean_similarity(
                    embedding
                ),

            "topk":
                identity.topk_similarity(
                    embedding,
                    k=3,
                ),
        }


    # ========================================================
    # TRACKLET LOOKUP
    # ========================================================

    def get_tracklet(
        self,
        camera_id,
        local_track_id,
        tracklet_lookup,
    ):

        if tracklet_lookup is None:

            return None


        camera = normalize_camera_id(
            camera_id
        )


        result = tracklet_lookup.get(
            (
                camera,
                local_track_id,
            )
        )


        if result is not None:

            return result


        return tracklet_lookup.get(
            (
                f"c{camera}",
                local_track_id,
            )
        )


    def get_member_tracklet(
        self,
        member,
        tracklet_lookup,
    ):

        return self.get_tracklet(
            member.camera_id,
            member.local_track_id,
            tracklet_lookup,
        )


    # ========================================================
    # GEOMETRY THRESHOLDS
    # ========================================================

    def is_strict_geometry(
        self,
        evidence,
    ):

        if evidence.status != "SUPPORTED":

            return False


        if evidence.median_distance is None:

            return False


        return (
            evidence.median_distance
            <= self.strict_geometry_median_max
        )


    def is_relaxed_core_geometry(
        self,
        evidence,
    ):

        if evidence.status != "SUPPORTED":

            return False


        if evidence.median_distance is None:

            return False


        if (
            evidence.median_distance
            > self.relaxed_geometry_median_max
        ):

            return False


        if (
            evidence.shared_frames
            < self.relaxed_min_shared_frames
        ):

            return False


        return True


    # ========================================================
    # CROSS-CAMERA CONTRADICTION
    #
    # Any member, CORE or RELAXED, may provide negative
    # evidence.
    # ========================================================

    def identity_geometry_contradictions(
        self,
        candidate_tracklet,
        identity,
        tracklet_lookup,
    ):

        contradictions = []


        if self.homographies is None:

            return contradictions


        if candidate_tracklet is None:

            return contradictions


        if tracklet_lookup is None:

            return contradictions


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


            if (
                member_camera
                == candidate_camera
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    existing_tracklet,
                    self.homographies,
                )
            )


            if (
                evidence.status
                == "CONTRADICTED"
            ):

                contradictions.append(
                    {
                        "camera_id":
                            member_camera,

                        "local_track_id":
                            member.local_track_id,

                        "shared_frames":
                            evidence.shared_frames,

                        "median_distance":
                            evidence.median_distance,
                    }
                )


        return contradictions


    def identity_geometry_contradicted(
        self,
        candidate_tracklet,
        identity,
        tracklet_lookup,
    ):

        return bool(
            self.identity_geometry_contradictions(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            )
        )


    # ========================================================
    # STRICT POSITIVE CROSS-CAMERA SUPPORT
    # ========================================================

    def strict_support_rows(
        self,
        global_id,
        identity,
        candidate_tracklet,
        embedding,
        tracklet_lookup,
    ):

        rows = []


        if self.homographies is None:

            return rows


        if candidate_tracklet is None:

            return rows


        if tracklet_lookup is None:

            return rows


        candidate_camera = (
            normalize_camera_id(
                candidate_tracklet.camera_id
            )
        )


        gid_is_anchored = (
            global_id
            in self.anchored_gids
        )


        for member in identity.members:

            member_camera = (
                normalize_camera_id(
                    member.camera_id
                )
            )


            if (
                member_camera
                == candidate_camera
            ):

                continue


            member_key = self.member_key(
                member_camera,
                member.local_track_id,
            )


            if (
                gid_is_anchored
                and
                member_key
                not in self.core_members[
                    global_id
                ]
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    existing_tracklet,
                    self.homographies,
                )
            )


            if not self.is_strict_geometry(
                evidence
            ):

                continue


            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            direct_similarity = None


            if member_embedding is not None:

                direct_similarity = (
                    self.cosine_similarity(
                        embedding,
                        member_embedding,
                    )
                )


            rows.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "shared_frames":
                        evidence.shared_frames,

                    "median_distance":
                        evidence.median_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


        return rows


    # ========================================================
    # RELAXED CORE-SUPPORTED CROSS-CAMERA SUPPORT
    # ========================================================

    def relaxed_core_support_rows(
        self,
        global_id,
        identity,
        candidate_tracklet,
        embedding,
        tracklet_lookup,
    ):

        rows = []


        if global_id not in self.anchored_gids:

            return rows


        if self.homographies is None:

            return rows


        if candidate_tracklet is None:

            return rows


        if tracklet_lookup is None:

            return rows


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


            if (
                member_camera
                == candidate_camera
            ):

                continue


            member_key = self.member_key(
                member_camera,
                member.local_track_id,
            )


            if (
                member_key
                not in self.core_members[
                    global_id
                ]
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    existing_tracklet,
                    self.homographies,
                )
            )


            if not self.is_relaxed_core_geometry(
                evidence
            ):

                continue


            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:

                continue


            direct_similarity = (
                self.cosine_similarity(
                    embedding,
                    member_embedding,
                )
            )


            if (
                direct_similarity
                < self.relaxed_core_direct_appearance_min
            ):

                continue


            rows.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "shared_frames":
                        evidence.shared_frames,

                    "median_distance":
                        evidence.median_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


        return rows


    # ========================================================
    # SAME-CAMERA CORE CONTINUATION SUPPORT
    # ========================================================

    def same_camera_core_support_rows(
        self,
        global_id,
        identity,
        candidate_tracklet,
        embedding,
        tracklet_lookup,
    ):

        rows = []


        if global_id not in self.anchored_gids:

            return rows


        if candidate_tracklet is None:

            return rows


        if tracklet_lookup is None:

            return rows


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


            if (
                member_camera
                != candidate_camera
            ):

                continue


            member_key = self.member_key(
                member_camera,
                member.local_track_id,
            )


            # Positive support comes from CORE only.
            if (
                member_key
                not in self.core_members[
                    global_id
                ]
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:

                continue


            (
                gap,
                overlap,
            ) = (
                self.same_camera_temporal_stats(
                    existing_tracklet,
                    candidate_tracklet,
                )
            )


            if overlap != 0:

                continue


            if gap < 0:

                continue


            if (
                gap
                > self.same_camera_continuation_max_gap
            ):

                continue


            (
                center_distance,
                bottom_distance,
            ) = (
                self.same_camera_endpoint_distance(
                    existing_tracklet,
                    candidate_tracklet,
                )
            )


            if (
                center_distance is None
                or bottom_distance is None
            ):

                continue


            if (
                center_distance
                > self.same_camera_continuation_center_max
            ):

                continue


            if (
                bottom_distance
                > self.same_camera_continuation_bottom_max
            ):

                continue


            direct_similarity = (
                self.cosine_similarity(
                    embedding,
                    member_embedding,
                )
            )


            if (
                direct_similarity
                < self.same_camera_continuation_reid_min
            ):

                continue


            rows.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "gap":
                        gap,

                    "overlap":
                        overlap,

                    "center_distance":
                        center_distance,

                    "bottom_distance":
                        bottom_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


        return rows


    # ========================================================
    # FINAL DUAL-EVIDENCE SAME-CAMERA CORE SUPPORT
    # ========================================================

    def dual_same_camera_core_support_rows(
        self,
        global_id,
        identity,
        candidate_tracklet,
        embedding,
        tracklet_lookup,
    ):

        rows = []


        if global_id not in self.anchored_gids:

            return rows


        if candidate_tracklet is None:

            return rows


        if tracklet_lookup is None:

            return rows


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


            if (
                member_camera
                != candidate_camera
            ):

                continue


            member_key = self.member_key(
                member_camera,
                member.local_track_id,
            )


            # Positive dual evidence comes from CORE only.
            if (
                member_key
                not in self.core_members[
                    global_id
                ]
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:

                continue


            (
                gap,
                overlap,
            ) = (
                self.same_camera_temporal_stats(
                    existing_tracklet,
                    candidate_tracklet,
                )
            )


            if overlap != 0:

                continue


            if gap < 0:

                continue


            if (
                gap
                > self.dual_same_camera_max_gap
            ):

                continue


            (
                center_distance,
                bottom_distance,
            ) = (
                self.same_camera_endpoint_distance(
                    existing_tracklet,
                    candidate_tracklet,
                )
            )


            if (
                center_distance is None
                or bottom_distance is None
            ):

                continue


            if (
                center_distance
                > self.dual_same_camera_center_max
            ):

                continue


            if (
                bottom_distance
                > self.dual_same_camera_bottom_max
            ):

                continue


            direct_similarity = (
                self.cosine_similarity(
                    embedding,
                    member_embedding,
                )
            )


            if (
                direct_similarity
                < self.dual_same_camera_reid_min
            ):

                continue


            rows.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "gap":
                        gap,

                    "overlap":
                        overlap,

                    "center_distance":
                        center_distance,

                    "bottom_distance":
                        bottom_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


        return rows


    # ========================================================
    # FINAL DUAL-EVIDENCE CROSS-CAMERA CORE SUPPORT
    # ========================================================

    def dual_cross_camera_core_support_rows(
        self,
        global_id,
        identity,
        candidate_tracklet,
        embedding,
        tracklet_lookup,
    ):

        rows = []


        if global_id not in self.anchored_gids:

            return rows


        if self.homographies is None:

            return rows


        if candidate_tracklet is None:

            return rows


        if tracklet_lookup is None:

            return rows


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


            if (
                member_camera
                == candidate_camera
            ):

                continue


            member_key = self.member_key(
                member_camera,
                member.local_track_id,
            )


            # Positive dual evidence comes from CORE only.
            if (
                member_key
                not in self.core_members[
                    global_id
                ]
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:

                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    existing_tracklet,
                    self.homographies,
                )
            )


            if (
                evidence.status
                != "SUPPORTED"
            ):

                continue


            if (
                evidence.shared_frames
                < self.dual_cross_camera_min_shared_frames
            ):

                continue


            if evidence.median_distance is None:

                continue


            if (
                evidence.median_distance
                > self.dual_cross_camera_geometry_median_max
            ):

                continue


            direct_similarity = (
                self.cosine_similarity(
                    embedding,
                    member_embedding,
                )
            )


            if (
                direct_similarity
                < self.dual_cross_camera_reid_min
            ):

                continue


            rows.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "shared_frames":
                        evidence.shared_frames,

                    "median_distance":
                        evidence.median_distance,

                    "mean_distance":
                        evidence.mean_distance,

                    "p90_distance":
                        evidence.p90_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


        return rows


    # ========================================================
    # FINAL CORE-ONLY GALLERY APPEARANCE SCORES
    # ========================================================

    def core_gallery_scores(
        self,
        global_id,
        embedding,
    ):

        rows = []


        for member_key in self.core_members.get(
            global_id,
            set(),
        ):

            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:

                continue


            similarity = (
                self.cosine_similarity(
                    embedding,
                    member_embedding,
                )
            )


            rows.append(
                (
                    similarity,
                    member_key,
                )
            )


        if not rows:

            return None


        rows.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )


        similarities = [
            row[0]

            for row in rows
        ]


        top_count = min(
            3,
            len(
                similarities
            ),
        )


        top3 = (
            sum(
                similarities[
                    :top_count
                ]
            )
            / top_count
        )


        return {
            "max":
                similarities[0],

            "mean":
                sum(
                    similarities
                )
                / len(
                    similarities
                ),

            "top3":
                top3,

            "count":
                len(
                    similarities
                ),

            "rows":
                rows,
        }


    # ========================================================
    # FINAL CORE-ONLY HIGH-GALLERY CROSS-CAMERA CORE SUPPORT
    # ========================================================

    def core_gallery_cross_camera_core_support_rows(
        self,
        global_id,
        identity,
        candidate_tracklet,
        embedding,
        tracklet_lookup,
    ):

        rows = []


        if global_id not in self.anchored_gids:

            return rows


        if self.homographies is None:

            return rows


        if candidate_tracklet is None:

            return rows


        if tracklet_lookup is None:

            return rows


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


            if (
                member_camera
                == candidate_camera
            ):

                continue


            member_key = self.member_key(
                member_camera,
                member.local_track_id,
            )


            # Positive support is CORE only.
            if (
                member_key
                not in self.core_members[
                    global_id
                ]
            ):

                continue


            existing_tracklet = (
                self.get_member_tracklet(
                    member,
                    tracklet_lookup,
                )
            )


            if existing_tracklet is None:

                continue


            member_embedding = (
                self.member_embeddings.get(
                    member_key
                )
            )


            if member_embedding is None:

                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    existing_tracklet,
                    self.homographies,
                )
            )


            if (
                evidence.status
                != "SUPPORTED"
            ):

                continue


            if (
                evidence.shared_frames
                < self.core_gallery_cross_min_shared_frames
            ):

                continue


            if evidence.median_distance is None:

                continue


            if (
                evidence.median_distance
                > self.core_gallery_cross_geometry_median_max
            ):

                continue


            direct_similarity = (
                self.cosine_similarity(
                    embedding,
                    member_embedding,
                )
            )


            if (
                direct_similarity
                < self.core_gallery_cross_reid_min
            ):

                continue


            rows.append(
                {
                    "camera_id":
                        member_camera,

                    "local_track_id":
                        member.local_track_id,

                    "shared_frames":
                        evidence.shared_frames,

                    "median_distance":
                        evidence.median_distance,

                    "mean_distance":
                        evidence.mean_distance,

                    "p90_distance":
                        evidence.p90_distance,

                    "direct_similarity":
                        direct_similarity,
                }
            )


        return rows


    # ========================================================
    # STRICT EXISTING GID CANDIDATES
    # ========================================================

    def find_strict_existing_candidates(
        self,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        candidates = []


        if (
            candidate_tracklet is None
            or tracklet_lookup is None
            or self.homographies is None
        ):

            return candidates


        for gid, identity in (
            self.identities.items()
        ):

            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            support = self.strict_support_rows(
                gid,
                identity,
                candidate_tracklet,
                embedding,
                tracklet_lookup,
            )


            if not support:

                continue


            scores = self.score_identity(
                identity,
                embedding,
            )


            if (
                scores["max"]
                < self.strict_appearance_min
            ):

                continue


            if (
                scores["topk"]
                < self.strict_appearance_min
            ):

                continue


            candidates.append(
                {
                    "global_id":
                        gid,

                    "max":
                        scores["max"],

                    "mean":
                        scores["mean"],

                    "topk":
                        scores["topk"],

                    "support":
                        support,
                }
            )


        candidates.sort(
            key=lambda item:
                item["topk"],
            reverse=True,
        )


        return candidates


    # ========================================================
    # RELAXED CROSS-CAMERA CORE CANDIDATES
    # ========================================================

    def find_relaxed_core_candidates(
        self,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        candidates = []


        if (
            candidate_tracklet is None
            or tracklet_lookup is None
            or self.homographies is None
        ):

            return candidates


        for gid in sorted(
            self.anchored_gids
        ):

            identity = self.identities.get(
                gid
            )


            if identity is None:

                continue


            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            support = (
                self.relaxed_core_support_rows(
                    gid,
                    identity,
                    candidate_tracklet,
                    embedding,
                    tracklet_lookup,
                )
            )


            if not support:

                continue


            scores = self.score_identity(
                identity,
                embedding,
            )


            if (
                scores["max"]
                < self.relaxed_gid_appearance_min
            ):

                continue


            if (
                scores["topk"]
                < self.relaxed_gid_appearance_min
            ):

                continue


            candidates.append(
                {
                    "global_id":
                        gid,

                    "max":
                        scores["max"],

                    "mean":
                        scores["mean"],

                    "topk":
                        scores["topk"],

                    "support":
                        support,
                }
            )


        candidates.sort(
            key=lambda item:
                item["topk"],
            reverse=True,
        )


        return candidates


    # ========================================================
    # SAME-CAMERA CONTINUATION CANDIDATES
    # ========================================================

    def find_same_camera_continuation_candidates(
        self,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        candidates = []


        if (
            candidate_tracklet is None
            or tracklet_lookup is None
        ):

            return candidates


        for gid in sorted(
            self.anchored_gids
        ):

            identity = self.identities.get(
                gid
            )


            if identity is None:

                continue


            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            # Same-camera continuation cannot override a
            # cross-camera geometry contradiction.
            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            support = (
                self.same_camera_core_support_rows(
                    gid,
                    identity,
                    candidate_tracklet,
                    embedding,
                    tracklet_lookup,
                )
            )


            if not support:

                continue


            support.sort(
                key=lambda row:
                    (
                        row[
                            "direct_similarity"
                        ],
                        -row[
                            "gap"
                        ],
                    ),
                reverse=True,
            )


            candidates.append(
                {
                    "global_id":
                        gid,

                    "support":
                        support,

                    "best":
                        support[0],
                }
            )


        candidates.sort(
            key=lambda item:
                (
                    item[
                        "best"
                    ][
                        "direct_similarity"
                    ],
                    -item[
                        "best"
                    ][
                        "gap"
                    ],
                ),
            reverse=True,
        )


        return candidates


    # ========================================================
    # FINAL DUAL-EVIDENCE CONTINUATION CANDIDATES
    #
    # Same GID must independently provide:
    #
    #   1. same-camera CORE continuity
    #   2. cross-camera CORE geometry
    #
    # Negative conflict/geometry vetoes remain active.
    # ========================================================

    def find_dual_evidence_candidates(
        self,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        candidates = []


        if (
            candidate_tracklet is None
            or tracklet_lookup is None
            or self.homographies is None
        ):

            return candidates


        for gid in sorted(
            self.anchored_gids
        ):

            identity = self.identities.get(
                gid
            )


            if identity is None:

                continue


            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            scores = self.score_identity(
                identity,
                embedding,
            )


            if (
                scores["max"]
                < self.dual_gid_appearance_min
            ):

                continue


            if (
                scores["topk"]
                < self.dual_gid_appearance_min
            ):

                continue


            same_support = (
                self.dual_same_camera_core_support_rows(
                    gid,
                    identity,
                    candidate_tracklet,
                    embedding,
                    tracklet_lookup,
                )
            )


            if not same_support:

                continue


            cross_support = (
                self.dual_cross_camera_core_support_rows(
                    gid,
                    identity,
                    candidate_tracklet,
                    embedding,
                    tracklet_lookup,
                )
            )


            if not cross_support:

                continue


            same_support.sort(
                key=lambda row:
                    (
                        row[
                            "direct_similarity"
                        ],
                        -row[
                            "gap"
                        ],
                    ),
                reverse=True,
            )


            cross_support.sort(
                key=lambda row:
                    (
                        row[
                            "direct_similarity"
                        ],
                        row[
                            "shared_frames"
                        ],
                        -row[
                            "median_distance"
                        ],
                    ),
                reverse=True,
            )


            candidates.append(
                {
                    "global_id":
                        gid,

                    "max":
                        scores["max"],

                    "mean":
                        scores["mean"],

                    "topk":
                        scores["topk"],

                    "same_support":
                        same_support,

                    "cross_support":
                        cross_support,

                    "best_same":
                        same_support[0],

                    "best_cross":
                        cross_support[0],
                }
            )


        candidates.sort(
            key=lambda item:
                (
                    item[
                        "best_same"
                    ][
                        "direct_similarity"
                    ],
                    item[
                        "best_cross"
                    ][
                        "direct_similarity"
                    ],
                    item[
                        "topk"
                    ],
                ),
            reverse=True,
        )


        return candidates


    # ========================================================
    # FINAL CORE-ONLY HIGH-GALLERY CONTINUATION CANDIDATES
    #
    # Every positive acceptance signal comes from CORE.
    # RELAXED members may still participate in negative vetoes
    # through identity_has_conflict() and geometry contradiction.
    # ========================================================

    def find_core_gallery_continuation_candidates(
        self,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        candidates = []


        if (
            candidate_tracklet is None
            or tracklet_lookup is None
            or self.homographies is None
        ):

            return candidates


        for gid in sorted(
            self.anchored_gids
        ):

            identity = self.identities.get(
                gid
            )


            if identity is None:

                continue


            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            scores = self.core_gallery_scores(
                gid,
                embedding,
            )


            if scores is None:

                continue


            if (
                scores[
                    "count"
                ]
                < self.core_gallery_min_members
            ):

                continue


            if (
                scores[
                    "max"
                ]
                < self.core_gallery_max_min
            ):

                continue


            if (
                scores[
                    "top3"
                ]
                < self.core_gallery_top3_min
            ):

                continue


            cross_support = (
                self.core_gallery_cross_camera_core_support_rows(
                    gid,
                    identity,
                    candidate_tracklet,
                    embedding,
                    tracklet_lookup,
                )
            )


            if not cross_support:

                continue


            cross_support.sort(
                key=lambda row:
                    (
                        row[
                            "direct_similarity"
                        ],
                        row[
                            "shared_frames"
                        ],
                        -row[
                            "median_distance"
                        ],
                    ),
                reverse=True,
            )


            candidates.append(
                {
                    "global_id":
                        gid,

                    "core_max":
                        scores[
                            "max"
                        ],

                    "core_mean":
                        scores[
                            "mean"
                        ],

                    "core_top3":
                        scores[
                            "top3"
                        ],

                    "core_count":
                        scores[
                            "count"
                        ],

                    "cross_support":
                        cross_support,

                    "best_cross":
                        cross_support[0],
                }
            )


        candidates.sort(
            key=lambda item:
                (
                    item[
                        "core_top3"
                    ],
                    item[
                        "core_max"
                    ],
                    item[
                        "best_cross"
                    ][
                        "direct_similarity"
                    ],
                ),
            reverse=True,
        )


        return candidates


    # ========================================================
    # STRICT CURRENT <-> PENDING CROSS-CAMERA PAIRS
    # ========================================================

    def find_strict_pending_candidates(
        self,
        camera_id,
        local_track_id,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        candidates = []


        if (
            candidate_tracklet is None
            or tracklet_lookup is None
            or self.homographies is None
        ):

            return candidates


        camera_id = normalize_camera_id(
            camera_id
        )


        current_key = (
            camera_id,
            local_track_id,
        )


        for pending_key, pending in (
            self.pending_tracklets.items()
        ):

            if pending_key == current_key:

                continue


            pending_camera = (
                normalize_camera_id(
                    pending[
                        "camera_id"
                    ]
                )
            )


            if pending_camera == camera_id:

                continue


            pending_tracklet = pending.get(
                "candidate_tracklet"
            )


            if pending_tracklet is None:

                pending_tracklet = (
                    self.get_tracklet(
                        pending_camera,
                        pending[
                            "local_track_id"
                        ],
                        tracklet_lookup,
                    )
                )


            if pending_tracklet is None:

                continue


            direct_similarity = (
                self.cosine_similarity(
                    embedding,
                    pending[
                        "embedding"
                    ],
                )
            )


            if (
                direct_similarity
                < self.strict_appearance_min
            ):

                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    pending_tracklet,
                    self.homographies,
                )
            )


            if not self.is_strict_geometry(
                evidence
            ):

                continue


            candidates.append(
                {
                    "pending_key":
                        pending_key,

                    "camera_id":
                        pending_camera,

                    "local_track_id":
                        pending[
                            "local_track_id"
                        ],

                    "similarity":
                        direct_similarity,

                    "shared_frames":
                        evidence.shared_frames,

                    "median_distance":
                        evidence.median_distance,

                    "pending":
                        pending,
                }
            )


        candidates.sort(
            key=lambda item:
                item["similarity"],
            reverse=True,
        )


        return candidates


    # ========================================================
    # APPEARANCE-ONLY BEST GID
    # ========================================================

    def find_best_identity(
        self,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        if not self.identities:

            return None


        candidates = []


        for gid, identity in (
            self.identities.items()
        ):

            if (
                candidate_tracklet is not None
                and tracklet_lookup is not None
            ):

                if identity_has_conflict(
                    candidate_tracklet,
                    identity,
                    tracklet_lookup,
                ):

                    continue


            if (
                candidate_tracklet is not None
                and tracklet_lookup is not None
                and self.homographies is not None
            ):

                if self.identity_geometry_contradicted(
                    candidate_tracklet,
                    identity,
                    tracklet_lookup,
                ):

                    continue


            scores = self.score_identity(
                identity,
                embedding,
            )


            candidates.append(
                {
                    "global_id":
                        gid,

                    "topk":
                        scores["topk"],

                    "max":
                        scores["max"],

                    "mean":
                        scores["mean"],
                }
            )


        if not candidates:

            return None


        candidates.sort(
            key=lambda item:
                item["topk"],
            reverse=True,
        )


        best = candidates[0]


        if len(candidates) >= 2:

            second = candidates[1]


            best[
                "second_global_id"
            ] = second[
                "global_id"
            ]


            best[
                "second_topk"
            ] = second[
                "topk"
            ]


            best[
                "margin"
            ] = (
                best["topk"]
                - second["topk"]
            )


        else:

            best[
                "second_global_id"
            ] = None


            best[
                "second_topk"
            ] = None


            best[
                "margin"
            ] = None


        return best


    # ========================================================
    # READ-ONLY APPEARANCE EVALUATION
    # ========================================================

    def evaluate(
        self,
        camera_id,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        best = self.find_best_identity(
            embedding,
            candidate_tracklet=
                candidate_tracklet,
            tracklet_lookup=
                tracklet_lookup,
        )


        if best is None:

            return MatchResult(
                status="NEW",
                global_id=None,

                max_similarity=-1.0,
                mean_similarity=-1.0,
                topk_similarity=-1.0,

                second_global_id=None,
                second_topk_similarity=None,
                score_margin=None,
            )


        identity = self.identities[
            best[
                "global_id"
            ]
        ]


        normalized_camera = (
            normalize_camera_id(
                camera_id
            )
        )


        identity_cameras = {
            normalize_camera_id(
                existing_camera
            )

            for existing_camera
            in identity.cameras_seen
        }


        same_camera = (
            normalized_camera
            in identity_cameras
        )


        if same_camera:

            strong_threshold = (
                self.same_camera_strong
            )

            pending_threshold = (
                self.same_camera_pending
            )


        else:

            strong_threshold = (
                self.cross_camera_strong
            )

            pending_threshold = (
                self.cross_camera_pending
            )


        if (
            best["max"]
            >= strong_threshold
            and
            best["topk"]
            >= pending_threshold
        ):

            status = "STRONG"


        elif (
            best["max"]
            >= pending_threshold
        ):

            status = "PENDING"


        else:

            status = "NEW"


        if status == "NEW":

            suggested_gid = None


        else:

            suggested_gid = (
                best[
                    "global_id"
                ]
            )


        return MatchResult(
            status=status,
            global_id=suggested_gid,

            max_similarity=
                best[
                    "max"
                ],

            mean_similarity=
                best[
                    "mean"
                ],

            topk_similarity=
                best[
                    "topk"
                ],

            second_global_id=
                best[
                    "second_global_id"
                ],

            second_topk_similarity=
                best[
                    "second_topk"
                ],

            score_margin=
                best[
                    "margin"
                ],
        )


    # ========================================================
    # BUILD MATCH RESULT FROM CANDIDATES
    # ========================================================

    def candidate_match_result(
        self,
        candidates,
        status="STRONG",
    ):

        best = candidates[0]


        if len(candidates) >= 2:

            second = candidates[1]

            second_gid = (
                second[
                    "global_id"
                ]
            )

            second_topk = (
                second[
                    "topk"
                ]
            )

            margin = (
                best[
                    "topk"
                ]
                - second[
                    "topk"
                ]
            )


        else:

            second_gid = None
            second_topk = None
            margin = None


        return MatchResult(
            status=status,

            global_id=
                best[
                    "global_id"
                ],

            max_similarity=
                best[
                    "max"
                ],

            mean_similarity=
                best[
                    "mean"
                ],

            topk_similarity=
                best[
                    "topk"
                ],

            second_global_id=
                second_gid,

            second_topk_similarity=
                second_topk,

            score_margin=
                margin,
        )


    # ========================================================
    # ADD PENDING
    # ========================================================

    def add_pending(
        self,
        camera_id,
        local_track_id,
        start_frame,
        end_frame,
        embedding,
        candidate_tracklet=None,
    ):

        camera_id = normalize_camera_id(
            camera_id
        )


        key = (
            camera_id,
            local_track_id,
        )


        self.pending_tracklets[
            key
        ] = {
            "camera_id":
                camera_id,

            "local_track_id":
                local_track_id,

            "start_frame":
                start_frame,

            "end_frame":
                end_frame,

            "embedding":
                np.asarray(
                    embedding,
                    dtype=np.float32,
                ),

            "candidate_tracklet":
                candidate_tracklet,
        }


    # ========================================================
    # CREATE CORE ANCHOR FROM PENDING PAIR
    # ========================================================

    def create_core_anchor_from_pending(
        self,
        camera_id,
        local_track_id,
        start_frame,
        end_frame,
        embedding,
        pending_candidate,
    ):

        camera_id = normalize_camera_id(
            camera_id
        )


        pending = pending_candidate[
            "pending"
        ]


        pending_camera = (
            normalize_camera_id(
                pending[
                    "camera_id"
                ]
            )
        )


        gid = self.create_identity(
            camera_id=
                pending_camera,

            local_track_id=
                pending[
                    "local_track_id"
                ],

            start_frame=
                pending[
                    "start_frame"
                ],

            end_frame=
                pending[
                    "end_frame"
                ],

            embedding=
                pending[
                    "embedding"
                ],

            trust_level=
                "CORE",
        )


        self.add_to_identity(
            global_id=
                gid,

            camera_id=
                camera_id,

            local_track_id=
                local_track_id,

            start_frame=
                start_frame,

            end_frame=
                end_frame,

            embedding=
                embedding,

            trust_level=
                "CORE",
        )


        self.pending_tracklets.pop(
            pending_candidate[
                "pending_key"
            ],
            None,
        )


        self.pending_tracklets.pop(
            (
                camera_id,
                local_track_id,
            ),
            None,
        )


        return gid


    # ========================================================
    # ASSIGN TRACKLET
    # ========================================================

    def assign_tracklet(
        self,
        camera_id,
        local_track_id,
        start_frame,
        end_frame,
        embedding,
        candidate_tracklet=None,
        tracklet_lookup=None,
    ):

        camera_id = normalize_camera_id(
            camera_id
        )


        current_key = (
            camera_id,
            local_track_id,
        )


        # ====================================================
        # FIRST IDENTITY
        # ====================================================

        if not self.identities:

            gid = self.create_identity(
                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                start_frame=
                    start_frame,

                end_frame=
                    end_frame,

                embedding=
                    embedding,
            )


            match = MatchResult(
                status="NEW",
                global_id=None,

                max_similarity=-1.0,
                mean_similarity=-1.0,
                topk_similarity=-1.0,

                second_global_id=None,
                second_topk_similarity=None,
                score_margin=None,
            )


            return AssignmentResult(
                status="NEW",
                global_id=gid,

                created_new=True,
                merged=False,

                match=match,

                trust_level="UNANCHORED",

                reason="FIRST_SINGLETON",
            )


        # ====================================================
        # PRIORITY 1:
        # STRICT EXISTING CROSS-CAMERA EVIDENCE
        # ====================================================

        strict_candidates = (
            self.find_strict_existing_candidates(
                embedding=
                    embedding,

                candidate_tracklet=
                    candidate_tracklet,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


        if len(
            strict_candidates
        ) == 1:

            candidate = (
                strict_candidates[0]
            )


            gid = candidate[
                "global_id"
            ]


            if gid not in self.anchored_gids:

                self.bootstrap_identity_as_core(
                    gid
                )


            self.add_to_identity(
                global_id=
                    gid,

                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                start_frame=
                    start_frame,

                end_frame=
                    end_frame,

                embedding=
                    embedding,

                trust_level=
                    "CORE",
            )


            self.pending_tracklets.pop(
                current_key,
                None,
            )


            match = (
                self.candidate_match_result(
                    strict_candidates,
                    status="STRONG",
                )
            )


            return AssignmentResult(
                status="STRONG",
                global_id=gid,

                created_new=False,
                merged=True,

                match=match,

                trust_level="CORE",

                reason=
                    "STRICT_EXISTING_GEOMETRY",
            )


        # ====================================================
        # PRIORITY 2:
        # RELAXED CROSS-CAMERA CORE SUPPORT
        # ====================================================

        relaxed_candidates = (
            self.find_relaxed_core_candidates(
                embedding=
                    embedding,

                candidate_tracklet=
                    candidate_tracklet,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


        if len(
            relaxed_candidates
        ) == 1:

            candidate = (
                relaxed_candidates[0]
            )


            gid = candidate[
                "global_id"
            ]


            self.add_to_identity(
                global_id=
                    gid,

                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                start_frame=
                    start_frame,

                end_frame=
                    end_frame,

                embedding=
                    embedding,

                trust_level=
                    "RELAXED",
            )


            self.pending_tracklets.pop(
                current_key,
                None,
            )


            match = (
                self.candidate_match_result(
                    relaxed_candidates,
                    status="STRONG",
                )
            )


            return AssignmentResult(
                status="STRONG",
                global_id=gid,

                created_new=False,
                merged=True,

                match=match,

                trust_level="RELAXED",

                reason=
                    "RELAXED_CORE_SUPPORTED",
            )


        # ====================================================
        # PRIORITY 3:
        # STRICT CURRENT <-> PENDING CROSS-CAMERA PAIR
        # ====================================================

        pending_candidates = (
            self.find_strict_pending_candidates(
                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                embedding=
                    embedding,

                candidate_tracklet=
                    candidate_tracklet,

                tracklet_lookup=
                    tracklet_lookup,
            )
        )


        if len(
            pending_candidates
        ) == 1:

            pending_candidate = (
                pending_candidates[0]
            )


            gid = (
                self.create_core_anchor_from_pending(
                    camera_id=
                        camera_id,

                    local_track_id=
                        local_track_id,

                    start_frame=
                        start_frame,

                    end_frame=
                        end_frame,

                    embedding=
                        embedding,

                    pending_candidate=
                        pending_candidate,
                )
            )


            direct_similarity = (
                pending_candidate[
                    "similarity"
                ]
            )


            match = MatchResult(
                status="STRONG",

                global_id=
                    gid,

                max_similarity=
                    direct_similarity,

                mean_similarity=
                    direct_similarity,

                topk_similarity=
                    direct_similarity,

                second_global_id=None,
                second_topk_similarity=None,
                score_margin=None,
            )


            return AssignmentResult(
                status="STRONG",
                global_id=gid,

                created_new=True,
                merged=True,

                match=match,

                trust_level="CORE",

                reason=
                    "STRICT_PENDING_GEOMETRY_ANCHOR",
            )


        # ====================================================
        # APPEARANCE FALLBACK
        # ====================================================

        match = self.evaluate(
            camera_id=
                camera_id,

            embedding=
                embedding,

            candidate_tracklet=
                candidate_tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )


        # Appearance-only STRONG is held.
        if match.status == "STRONG":

            self.add_pending(
                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                start_frame=
                    start_frame,

                end_frame=
                    end_frame,

                embedding=
                    embedding,

                candidate_tracklet=
                    candidate_tracklet,
            )


            return AssignmentResult(
                status="PENDING",
                global_id=
                    match.global_id,

                created_new=False,
                merged=False,

                match=match,

                trust_level=None,

                reason=
                    "APPEARANCE_ONLY_STRONG_HELD",
            )


        # Appearance pending is held.
        if match.status == "PENDING":

            self.add_pending(
                camera_id=
                    camera_id,

                local_track_id=
                    local_track_id,

                start_frame=
                    start_frame,

                end_frame=
                    end_frame,

                embedding=
                    embedding,

                candidate_tracklet=
                    candidate_tracklet,
            )


            return AssignmentResult(
                status="PENDING",
                global_id=
                    match.global_id,

                created_new=False,
                merged=False,

                match=match,

                trust_level=None,

                reason=
                    "APPEARANCE_PENDING_HELD",
            )


        # ====================================================
        # CLEAR NEW SINGLETON
        # ====================================================

        gid = self.create_identity(
            camera_id=
                camera_id,

            local_track_id=
                local_track_id,

            start_frame=
                start_frame,

            end_frame=
                end_frame,

            embedding=
                embedding,
        )


        self.pending_tracklets.pop(
            current_key,
            None,
        )


        return AssignmentResult(
            status="NEW",
            global_id=gid,

            created_new=True,
            merged=False,

            match=match,

            trust_level="UNANCHORED",

            reason="CLEAR_NEW_SINGLETON",
        )


    # ========================================================
    # SAFE PENDING RE-EVALUATION
    #
    # include_same_camera=False
    #
    #     Used DURING chronological processing.
    #
    #     Only performs the validated cross-camera
    #     CORE-supported pending sweep.
    #
    #
    # include_same_camera=True
    #
    #     Used ONLY after chronological processing has fully
    #     stabilized. Adds one conservative same-camera CORE
    #     continuation pass after the cross-camera sweep.
    #
    #
    # include_dual_evidence=True
    #
    #     Used ONLY in the final production pass. After the
    #     same-camera phase, adds one dual-evidence pass where
    #     the SAME GID must have independent same-camera CORE
    #     support and cross-camera CORE geometry support.
    #
    #
    # include_core_gallery=True
    #
    #     Used ONLY as the final final-only continuation stage.
    #     Positive evidence is entirely CORE-derived: strong
    #     CORE-only gallery agreement plus cross-camera CORE
    #     geometry/appearance support.
    #
    #
    # IMPORTANT:
    #
    # No new cross-camera sweep is performed after same-camera,
    # dual-evidence, or CORE-gallery additions. Final recovered
    # members are RELAXED and cannot propagate positive trust.
    # ========================================================

    def reevaluate_pending(
        self,
        tracklet_lookup=None,
        include_same_camera=False,
        include_dual_evidence=False,
        include_core_gallery=False,
    ):

        results = []


        # ====================================================
        # PHASE 1:
        # CROSS-CAMERA RELAXED CORE SWEEP TO FIXED POINT
        # ====================================================

        changed = True


        while changed:

            changed = False


            pending_items = list(
                self.pending_tracklets.items()
            )


            for key, pending in pending_items:

                if (
                    key
                    not in self.pending_tracklets
                ):

                    continue


                tracklet = pending.get(
                    "candidate_tracklet"
                )


                if (
                    tracklet is None
                    and tracklet_lookup is not None
                ):

                    tracklet = self.get_tracklet(
                        pending[
                            "camera_id"
                        ],
                        pending[
                            "local_track_id"
                        ],
                        tracklet_lookup,
                    )


                if (
                    tracklet is None
                    or tracklet_lookup is None
                ):

                    continue


                candidates = (
                    self.find_relaxed_core_candidates(
                        embedding=
                            pending[
                                "embedding"
                            ],

                        candidate_tracklet=
                            tracklet,

                        tracklet_lookup=
                            tracklet_lookup,
                    )
                )


                # Exactly one trusted GID required.
                if len(candidates) != 1:

                    continue


                candidate = candidates[0]


                gid = candidate[
                    "global_id"
                ]


                self.add_to_identity(
                    global_id=
                        gid,

                    camera_id=
                        pending[
                            "camera_id"
                        ],

                    local_track_id=
                        pending[
                            "local_track_id"
                        ],

                    start_frame=
                        pending[
                            "start_frame"
                        ],

                    end_frame=
                        pending[
                            "end_frame"
                        ],

                    embedding=
                        pending[
                            "embedding"
                        ],

                    trust_level=
                        "RELAXED",
                )


                del self.pending_tracklets[
                    key
                ]


                results.append(
                    {
                        "camera_id":
                            pending[
                                "camera_id"
                            ],

                        "local_track_id":
                            pending[
                                "local_track_id"
                            ],

                        "status":
                            "STRONG",

                        "global_id":
                            gid,

                        "trust_level":
                            "RELAXED",

                        "reason":
                            "RELAXED_CORE_PENDING_SWEEP",
                    }
                )


                changed = True


        # ====================================================
        # PHASE 2:
        # FINAL-ONLY SAME-CAMERA CORE CONTINUATION
        #
        # This phase is deliberately disabled by default.
        #
        # It must not execute during chronological identity
        # construction because its RELAXED embeddings would
        # modify the gallery seen by later tracklets.
        # ====================================================

        if include_same_camera:

            pending_items = list(
                self.pending_tracklets.items()
            )


            for key, pending in pending_items:

                if (
                    key
                    not in self.pending_tracklets
                ):

                    continue


                tracklet = pending.get(
                    "candidate_tracklet"
                )


                if (
                    tracklet is None
                    and tracklet_lookup is not None
                ):

                    tracklet = self.get_tracklet(
                        pending[
                            "camera_id"
                        ],
                        pending[
                            "local_track_id"
                        ],
                        tracklet_lookup,
                    )


                if (
                    tracklet is None
                    or tracklet_lookup is None
                ):

                    continue


                candidates = (
                    self.find_same_camera_continuation_candidates(
                        embedding=
                            pending[
                                "embedding"
                            ],

                        candidate_tracklet=
                            tracklet,

                        tracklet_lookup=
                            tracklet_lookup,
                    )
                )


                # Exactly one GID is required.
                if len(candidates) != 1:

                    continue


                candidate = candidates[0]


                gid = candidate[
                    "global_id"
                ]


                self.add_to_identity(
                    global_id=
                        gid,

                    camera_id=
                        pending[
                            "camera_id"
                        ],

                    local_track_id=
                        pending[
                            "local_track_id"
                        ],

                    start_frame=
                        pending[
                            "start_frame"
                        ],

                    end_frame=
                        pending[
                            "end_frame"
                        ],

                    embedding=
                        pending[
                            "embedding"
                        ],

                    trust_level=
                        "RELAXED",
                )


                del self.pending_tracklets[
                    key
                ]


                best = candidate[
                    "best"
                ]


                results.append(
                    {
                        "camera_id":
                            pending[
                                "camera_id"
                            ],

                        "local_track_id":
                            pending[
                                "local_track_id"
                            ],

                        "status":
                            "STRONG",

                        "global_id":
                            gid,

                        "trust_level":
                            "RELAXED",

                        "reason":
                            "SAME_CAMERA_CORE_CONTINUATION",

                        "support_camera_id":
                            best[
                                "camera_id"
                            ],

                        "support_local_track_id":
                            best[
                                "local_track_id"
                            ],

                        "gap":
                            best[
                                "gap"
                            ],

                        "direct_similarity":
                            best[
                                "direct_similarity"
                            ],

                        "center_distance":
                            best[
                                "center_distance"
                            ],

                        "bottom_distance":
                            best[
                                "bottom_distance"
                            ],
                    }
                )


        # ====================================================
        # PHASE 3:
        # FINAL-ONLY DUAL-EVIDENCE CONTINUATION
        #
        # This phase runs only after the optional same-camera
        # phase. It performs ONE pass and then stops.
        #
        # There is deliberately no cross-camera reevaluation
        # after these RELAXED additions.
        # ====================================================

        if include_dual_evidence:

            pending_items = list(
                self.pending_tracklets.items()
            )


            for key, pending in pending_items:

                if (
                    key
                    not in self.pending_tracklets
                ):

                    continue


                tracklet = pending.get(
                    "candidate_tracklet"
                )


                if (
                    tracklet is None
                    and tracklet_lookup is not None
                ):

                    tracklet = self.get_tracklet(
                        pending[
                            "camera_id"
                        ],
                        pending[
                            "local_track_id"
                        ],
                        tracklet_lookup,
                    )


                if (
                    tracklet is None
                    or tracklet_lookup is None
                ):

                    continue


                candidates = (
                    self.find_dual_evidence_candidates(
                        embedding=
                            pending[
                                "embedding"
                            ],

                        candidate_tracklet=
                            tracklet,

                        tracklet_lookup=
                            tracklet_lookup,
                    )
                )


                # Exactly one dual-evidence GID is required.
                if len(candidates) != 1:

                    continue


                candidate = candidates[0]


                gid = candidate[
                    "global_id"
                ]


                self.add_to_identity(
                    global_id=
                        gid,

                    camera_id=
                        pending[
                            "camera_id"
                        ],

                    local_track_id=
                        pending[
                            "local_track_id"
                        ],

                    start_frame=
                        pending[
                            "start_frame"
                        ],

                    end_frame=
                        pending[
                            "end_frame"
                        ],

                    embedding=
                        pending[
                            "embedding"
                        ],

                    trust_level=
                        "RELAXED",
                )


                del self.pending_tracklets[
                    key
                ]


                best_same = candidate[
                    "best_same"
                ]


                best_cross = candidate[
                    "best_cross"
                ]


                results.append(
                    {
                        "camera_id":
                            pending[
                                "camera_id"
                            ],

                        "local_track_id":
                            pending[
                                "local_track_id"
                            ],

                        "status":
                            "STRONG",

                        "global_id":
                            gid,

                        "trust_level":
                            "RELAXED",

                        "reason":
                            "DUAL_EVIDENCE_CONTINUATION",

                        "gid_max_similarity":
                            candidate[
                                "max"
                            ],

                        "gid_topk_similarity":
                            candidate[
                                "topk"
                            ],

                        "same_support_camera_id":
                            best_same[
                                "camera_id"
                            ],

                        "same_support_local_track_id":
                            best_same[
                                "local_track_id"
                            ],

                        "same_gap":
                            best_same[
                                "gap"
                            ],

                        "same_direct_similarity":
                            best_same[
                                "direct_similarity"
                            ],

                        "same_center_distance":
                            best_same[
                                "center_distance"
                            ],

                        "same_bottom_distance":
                            best_same[
                                "bottom_distance"
                            ],

                        "cross_support_camera_id":
                            best_cross[
                                "camera_id"
                            ],

                        "cross_support_local_track_id":
                            best_cross[
                                "local_track_id"
                            ],

                        "cross_direct_similarity":
                            best_cross[
                                "direct_similarity"
                            ],

                        "cross_shared_frames":
                            best_cross[
                                "shared_frames"
                            ],

                        "cross_median_distance":
                            best_cross[
                                "median_distance"
                            ],
                    }
                )


        # ====================================================
        # PHASE 4:
        # FINAL-ONLY CORE-ONLY HIGH-GALLERY CONTINUATION
        #
        # This is the final production continuation stage.
        # It performs ONE pass and then stops.
        #
        # Positive acceptance evidence is CORE-only.
        # Accepted members are RELAXED.
        # There is deliberately no reevaluation afterward.
        # ====================================================

        if include_core_gallery:

            pending_items = list(
                self.pending_tracklets.items()
            )


            for key, pending in pending_items:

                if (
                    key
                    not in self.pending_tracklets
                ):

                    continue


                tracklet = pending.get(
                    "candidate_tracklet"
                )


                if (
                    tracklet is None
                    and tracklet_lookup is not None
                ):

                    tracklet = self.get_tracklet(
                        pending[
                            "camera_id"
                        ],
                        pending[
                            "local_track_id"
                        ],
                        tracklet_lookup,
                    )


                if (
                    tracklet is None
                    or tracklet_lookup is None
                ):

                    continue


                candidates = (
                    self.find_core_gallery_continuation_candidates(
                        embedding=
                            pending[
                                "embedding"
                            ],

                        candidate_tracklet=
                            tracklet,

                        tracklet_lookup=
                            tracklet_lookup,
                    )
                )


                # Exactly one CORE-only high-gallery GID is required.
                if len(candidates) != 1:

                    continue


                candidate = candidates[0]


                gid = candidate[
                    "global_id"
                ]


                self.add_to_identity(
                    global_id=
                        gid,

                    camera_id=
                        pending[
                            "camera_id"
                        ],

                    local_track_id=
                        pending[
                            "local_track_id"
                        ],

                    start_frame=
                        pending[
                            "start_frame"
                        ],

                    end_frame=
                        pending[
                            "end_frame"
                        ],

                    embedding=
                        pending[
                            "embedding"
                        ],

                    trust_level=
                        "RELAXED",
                )


                del self.pending_tracklets[
                    key
                ]


                best_cross = candidate[
                    "best_cross"
                ]


                results.append(
                    {
                        "camera_id":
                            pending[
                                "camera_id"
                            ],

                        "local_track_id":
                            pending[
                                "local_track_id"
                            ],

                        "status":
                            "STRONG",

                        "global_id":
                            gid,

                        "trust_level":
                            "RELAXED",

                        "reason":
                            "CORE_GALLERY_CONTINUATION",

                        "core_gallery_max":
                            candidate[
                                "core_max"
                            ],

                        "core_gallery_top3":
                            candidate[
                                "core_top3"
                            ],

                        "core_gallery_mean":
                            candidate[
                                "core_mean"
                            ],

                        "core_gallery_count":
                            candidate[
                                "core_count"
                            ],

                        "cross_support_camera_id":
                            best_cross[
                                "camera_id"
                            ],

                        "cross_support_local_track_id":
                            best_cross[
                                "local_track_id"
                            ],

                        "cross_direct_similarity":
                            best_cross[
                                "direct_similarity"
                            ],

                        "cross_shared_frames":
                            best_cross[
                                "shared_frames"
                            ],

                        "cross_median_distance":
                            best_cross[
                                "median_distance"
                            ],
                    }
                )


        # ====================================================
        # REPORT REMAINING PENDING
        # ====================================================

        for key, pending in (
            self.pending_tracklets.items()
        ):

            match = self.evaluate(
                camera_id=
                    pending[
                        "camera_id"
                    ],

                embedding=
                    pending[
                        "embedding"
                    ],

                candidate_tracklet=
                    pending[
                        "candidate_tracklet"
                    ],

                tracklet_lookup=
                    tracklet_lookup,
            )


            results.append(
                {
                    "camera_id":
                        pending[
                            "camera_id"
                        ],

                    "local_track_id":
                        pending[
                            "local_track_id"
                        ],

                    "status":
                        "PENDING",

                    "global_id":
                        (
                            match.global_id
                            if match.status
                            != "NEW"
                            else None
                        ),

                    "trust_level":
                        None,

                    "reason":
                        "STILL_PENDING_NO_TRUSTED_EVIDENCE",
                }
            )


        return results


    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ):

        print()
        print(
            "GLOBAL IDENTITIES"
        )

        print(
            "================="
        )


        for gid in sorted(
            self.identities
        ):

            identity = self.identities[
                gid
            ]


            print(
                identity.summary()
            )


            if gid in self.anchored_gids:

                core_count = len(
                    self.core_members[
                        gid
                    ]
                )


                relaxed_count = (
                    len(
                        identity.members
                    )
                    - core_count
                )


                print(
                    "    trust: "
                    f"ANCHORED | "
                    f"CORE={core_count} | "
                    f"RELAXED={relaxed_count}"
                )


            else:

                print(
                    "    trust: UNANCHORED"
                )


        print()


        print(
            "Anchored GIDs:",
            len(
                self.anchored_gids
            ),
        )


        print(
            "CORE members:",
            sum(
                len(
                    members
                )

                for members
                in self.core_members.values()
            ),
        )


        relaxed_total = 0


        for gid in self.anchored_gids:

            identity = self.identities.get(
                gid
            )


            if identity is None:

                continue


            relaxed_total += (
                len(
                    identity.members
                )
                - len(
                    self.core_members[
                        gid
                    ]
                )
            )


        print(
            "RELAXED members:",
            relaxed_total,
        )


        print(
            "Pending tracklets:",
            len(
                self.pending_tracklets
            ),
        )
