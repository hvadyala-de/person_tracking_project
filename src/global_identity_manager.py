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

    # --------------------------------------------------------
    # New provenance information.
    #
    # None
    #     no identity membership was created
    #
    # CORE
    #     trusted member allowed to propagate identity
    #
    # RELAXED
    #     accepted member that is NOT allowed to propagate
    #     positive identity evidence
    # --------------------------------------------------------

    trust_level: Optional[str] = None

    # Human-readable decision reason for diagnostics.
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


        # ----------------------------------------------------
        # Unresolved / ambiguous tracklets.
        # ----------------------------------------------------

        self.pending_tracklets = {}


        # ----------------------------------------------------
        # Terrace ground-plane homographies.
        #
        # None means positive geometry evidence is unavailable.
        #
        # In that situation appearance-only matches will NOT
        # automatically modify an existing GID.
        # ----------------------------------------------------

        self.homographies = homographies


        # ====================================================
        # APPEARANCE-ONLY FALLBACK THRESHOLDS
        #
        # These preserve the original appearance evaluation
        # interface.
        #
        # IMPORTANT:
        #
        # STRONG no longer means "safe to merge".
        #
        # In assign_tracklet(), appearance-only STRONG and
        # PENDING are both held unresolved.
        # ====================================================

        self.same_camera_strong = 0.92
        self.same_camera_pending = 0.88

        self.cross_camera_strong = 0.90
        self.cross_camera_pending = 0.84


        # ====================================================
        # VALIDATED TERRACE EVIDENCE POLICY
        #
        # Experimental dataset-specific thresholds.
        #
        # They are NOT universal production thresholds.
        # ====================================================

        # ----------------------------------------------------
        # STRICT / CORE bootstrap
        #
        # Used to:
        #
        #   1. bootstrap an existing singleton into a trusted
        #      geometry-anchored identity
        #
        #   2. add further CORE members
        #
        #   3. create a CORE identity from a pending pair
        # ----------------------------------------------------

        self.strict_appearance_min = 0.88

        self.strict_geometry_median_max = 20.0


        # ----------------------------------------------------
        # RELAXED expansion
        #
        # Used ONLY for an already geometry-anchored GID.
        #
        # Positive geometry must come directly from a CORE
        # member.
        #
        # A RELAXED member can belong to the GID, but it can
        # never become a source of positive propagation.
        # ----------------------------------------------------

        self.relaxed_gid_appearance_min = 0.84

        self.relaxed_core_direct_appearance_min = 0.86

        self.relaxed_geometry_median_max = 18.0

        self.relaxed_min_shared_frames = 15


        # ====================================================
        # IDENTITY PROVENANCE
        # ====================================================

        # GIDs that have been established using trusted
        # cross-camera geometry.
        self.anchored_gids = set()


        # gid -> {(camera_id, local_track_id), ...}
        #
        # Only these members may propagate positive geometry
        # support.
        self.core_members = defaultdict(
            set
        )


        # ----------------------------------------------------
        # Tracklet embedding lookup.
        #
        # We keep the direct embedding for every assigned
        # member so relaxed evidence can be evaluated against
        # the exact CORE member that supplied geometry.
        # ----------------------------------------------------

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
    # CREATE NEW GID
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
    # ADD TRACKLET TO EXISTING GID
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


        # ----------------------------------------------------
        # RELAXED is deliberately NOT added to core_members.
        # ----------------------------------------------------


    # ========================================================
    # SCORE ONE IDENTITY
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
    # MEMBER TRACKLET LOOKUP
    # ========================================================

    def get_member_tracklet(
        self,
        member,
        tracklet_lookup,
    ):

        if tracklet_lookup is None:

            return None


        camera = normalize_camera_id(
            member.camera_id
        )


        result = tracklet_lookup.get(
            (
                camera,
                member.local_track_id,
            )
        )


        if result is not None:

            return result


        return tracklet_lookup.get(
            (
                f"c{camera}",
                member.local_track_id,
            )
        )


    # ========================================================
    # STRICT GEOMETRY
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


    # ========================================================
    # RELAXED CORE GEOMETRY
    # ========================================================

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
    # ALL CROSS-CAMERA GEOMETRY CONTRADICTIONS
    #
    # IMPORTANT:
    #
    # CORE and RELAXED members can both VETO a candidate.
    #
    # RELAXED members cannot provide positive propagation,
    # but synchronized contradictory geometry remains valid
    # negative evidence.
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


    # ========================================================
    # BACKWARD-COMPATIBLE BOOLEAN GEOMETRY VETO
    # ========================================================

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
    # STRICT POSITIVE SUPPORT ROWS
    #
    # Unanchored GID:
    #     any cross-camera existing member may establish the
    #     initial trusted geometry anchor.
    #
    # Anchored GID:
    #     ONLY CORE members may provide positive support.
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


            # ------------------------------------------------
            # Once anchored, positive geometry can ONLY come
            # from CORE.
            # ------------------------------------------------

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
    # RELAXED CORE SUPPORT ROWS
    #
    # A candidate may be accepted as RELAXED only when a CORE
    # member supplies BOTH:
    #
    #   1. direct appearance >= 0.86
    #   2. synchronized geometry support
    #
    # RELAXED members are never considered here.
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


            # ------------------------------------------------
            # POSITIVE SUPPORT MUST COME FROM CORE.
            # ------------------------------------------------

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
    # FIND STRICT EXISTING GID CANDIDATES
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

            # ------------------------------------------------
            # Same-camera incompatibility veto.
            # ------------------------------------------------

            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            # ------------------------------------------------
            # Geometry contradiction veto across ALL members.
            # ------------------------------------------------

            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            # ------------------------------------------------
            # Positive geometry support.
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Strict gallery appearance requirement.
            # ------------------------------------------------

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
    # FIND RELAXED CORE-SUPPORTED GID CANDIDATES
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


        for gid in self.anchored_gids:

            identity = self.identities.get(
                gid
            )


            if identity is None:

                continue


            # ------------------------------------------------
            # Same-camera incompatibility veto.
            # ------------------------------------------------

            if identity_has_conflict(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            # ------------------------------------------------
            # Negative geometry from ANY member vetoes.
            # ------------------------------------------------

            if self.identity_geometry_contradicted(
                candidate_tracklet,
                identity,
                tracklet_lookup,
            ):

                continue


            # ------------------------------------------------
            # Positive support MUST come from CORE.
            # ------------------------------------------------

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
    # FIND STRICT CURRENT <-> PENDING PAIRS
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

            # Do not match a pending entry to itself.
            if pending_key == current_key:

                continue


            pending_camera = (
                normalize_camera_id(
                    pending[
                        "camera_id"
                    ]
                )
            )


            # Strict geometry anchors are cross-camera.
            if pending_camera == camera_id:

                continue


            pending_tracklet = pending.get(
                "candidate_tracklet"
            )


            if pending_tracklet is None:

                pending_tracklet = (
                    tracklet_lookup.get(
                        (
                            pending_camera,
                            pending[
                                "local_track_id"
                            ],
                        )
                    )
                )


            if pending_tracklet is None:

                pending_tracklet = (
                    tracklet_lookup.get(
                        (
                            f"c{pending_camera}",
                            pending[
                                "local_track_id"
                            ],
                        )
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
    # APPEARANCE-ONLY BEST AND SECOND-BEST GID
    #
    # This method is retained for compatibility and for the
    # PENDING / NEW fallback.
    #
    # IMPORTANT:
    #
    # A high appearance result here is NOT permission to merge.
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

            # =================================================
            # GATE 1:
            # SAME-CAMERA SPATIOTEMPORAL CONFLICT
            # =================================================

            if (
                candidate_tracklet is not None
                and tracklet_lookup is not None
            ):

                conflict = (
                    identity_has_conflict(
                        candidate_tracklet,
                        identity,
                        tracklet_lookup,
                    )
                )


                if conflict:

                    continue


            # =================================================
            # GATE 2:
            # CROSS-CAMERA GEOMETRY CONTRADICTION
            #
            # Geometry contradiction remains a hard veto even
            # for appearance fallback candidate ranking.
            # =================================================

            if (
                candidate_tracklet is not None
                and tracklet_lookup is not None
                and self.homographies is not None
            ):

                contradicted = (
                    self.identity_geometry_contradicted(
                        candidate_tracklet,
                        identity,
                        tracklet_lookup,
                    )
                )


                if contradicted:

                    continue


            # =================================================
            # APPEARANCE SCORE
            # =================================================

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
    # EVALUATE APPEARANCE
    #
    # IMPORTANT:
    #
    # evaluate() remains READ-ONLY.
    #
    # STRONG here means:
    #
    #     strong appearance hypothesis
    #
    # It does NOT mean:
    #
    #     automatically safe to modify a GID
    #
    # assign_tracklet() applies the evidence policy.
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
    # BUILD MATCH RESULT FROM EXISTING-GID CANDIDATES
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
    # ADD PENDING TRACKLET
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
    # CREATE STRICT CORE ANCHOR FROM PENDING PAIR
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


        # ----------------------------------------------------
        # Pending member becomes first CORE member.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Current member becomes second CORE member.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Remove resolved pending entries.
        # ----------------------------------------------------

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
    # EVIDENCE-GATED ASSIGNMENT
    #
    # POLICY:
    #
    # 1. STRICT existing evidence
    #       -> merge as CORE
    #
    # 2. RELAXED CORE-supported evidence
    #       -> merge as RELAXED
    #
    # 3. STRICT current <-> pending pair
    #       -> create geometry-anchored GID
    #       -> both members CORE
    #
    # 4. Appearance-only STRONG
    #       -> PENDING
    #
    # 5. Appearance-only PENDING
    #       -> PENDING
    #
    # 6. Clearly NEW
    #       -> singleton GID
    #
    # Appearance alone NEVER modifies an existing identity.
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
        #
        # Starts as an UNANCHORED singleton.
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
        # STRICT GEOMETRY-SUPPORTED EXISTING GID
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


            # ------------------------------------------------
            # First strict geometry event establishes the
            # existing singleton as CORE.
            # ------------------------------------------------

            if gid not in self.anchored_gids:

                self.bootstrap_identity_as_core(
                    gid
                )


            # ------------------------------------------------
            # Strict member becomes CORE.
            # ------------------------------------------------

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
        # RELAXED EXPANSION OF AN ALREADY ANCHORED GID
        #
        # Positive evidence must come directly from CORE.
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
        # STRICT CURRENT <-> PENDING GEOMETRY PAIR
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
        # PRIORITY 4:
        # APPEARANCE-ONLY FALLBACK
        #
        # evaluate() is read-only.
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


        # ====================================================
        # APPEARANCE-ONLY STRONG
        #
        # OLD:
        #     merge directly
        #
        # NEW:
        #     HOLD PENDING
        # ====================================================

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


        # ====================================================
        # APPEARANCE PENDING
        # ====================================================

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
        # CLEAR NEW
        #
        # Appearance does not sufficiently support any
        # compatible existing GID.
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
    # IMPORTANT:
    #
    # This method deliberately does NOT use appearance-only
    # STRONG as permission to merge.
    #
    # It reproduces the validated CORE-only pending sweep:
    #
    #   pending track
    #       +
    #   existing ANCHORED GID
    #       +
    #   direct CORE appearance >= 0.86
    #       +
    #   shared >= 15
    #       +
    #   median geometry <= 18
    #       +
    #   no contradiction
    #
    #       -> RELAXED
    #
    # RELAXED members still do not become CORE.
    #
    # Pending tracks that remain unresolved are left pending.
    # They are NOT converted to NEW merely because appearance
    # changes later.
    # ========================================================

    def reevaluate_pending(
        self,
        tracklet_lookup=None,
    ):

        resolved_results = []


        # ----------------------------------------------------
        # Repeat because adding one relaxed gallery member can
        # slightly change GID aggregate appearance scoring.
        #
        # Positive geometry must still come directly from CORE,
        # so RELAXED -> RELAXED geometry propagation remains
        # impossible.
        # ----------------------------------------------------

        changed = True


        while changed:

            changed = False


            pending_items = list(
                self.pending_tracklets.items()
            )


            for key, pending in pending_items:

                # Entry may already have been resolved during
                # this sweep.
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


                # ------------------------------------------------
                # Conservative uniqueness requirement.
                # ------------------------------------------------

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


                resolved_results.append(
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


        # ----------------------------------------------------
        # Report remaining entries without changing them.
        # ----------------------------------------------------

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


            resolved_results.append(
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


        return resolved_results


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
