from dataclasses import dataclass
from typing import Optional

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
        # None means geometry checking is disabled.
        # ----------------------------------------------------

        self.homographies = homographies

        # ----------------------------------------------------
        # TEMPORARY TERRACE THRESHOLDS
        #
        # These are experimental and NOT final production
        # thresholds.
        # ----------------------------------------------------

        self.same_camera_strong = 0.92
        self.same_camera_pending = 0.88

        self.cross_camera_strong = 0.90
        self.cross_camera_pending = 0.84


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
    ):

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
    ):

        if global_id not in self.identities:

            raise KeyError(
                f"GID_{global_id:04d} does not exist"
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
    # CROSS-CAMERA GEOMETRY CONTRADICTION
    #
    # IMPORTANT:
    #
    # Geometry is currently used ONLY AS A VETO.
    #
    # SUPPORTED:
    #     does not automatically promote a match
    #
    # UNKNOWN:
    #     appearance remains in control
    #
    # CONTRADICTED:
    #     remove this GID from consideration
    # ========================================================

    def identity_geometry_contradicted(
        self,
        candidate_tracklet,
        identity,
        tracklet_lookup,
    ):

        # No homography data means geometry checking
        # is disabled.
        if self.homographies is None:
            return False


        candidate_camera = (
            normalize_camera_id(
                candidate_tracklet.camera_id
            )
        )


        # ----------------------------------------------------
        # Compare candidate against cross-camera members
        # already belonging to this GID.
        # ----------------------------------------------------

        for member in identity.members:

            member_camera = (
                normalize_camera_id(
                    member.camera_id
                )
            )


            # Geometry gate here is specifically
            # cross-camera.
            if (
                member_camera
                == candidate_camera
            ):
                continue


            # ------------------------------------------------
            # Try integer camera lookup first:
            #
            # (0, track_id)
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
            # Also support:
            #
            # ("c0", track_id)
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


            if existing_tracklet is None:
                continue


            evidence = (
                evaluate_cross_camera_geometry(
                    candidate_tracklet,
                    existing_tracklet,
                    self.homographies,
                )
            )


            # ------------------------------------------------
            # HARD GEOMETRY VETO
            #
            # Example discovered in Terrace:
            #
            # c0:399 <-> c1:503
            #
            # ReID:
            #     ~0.916
            #
            # Geometry median:
            #     ~347
            #
            # Therefore this visually attractive match must
            # not be allowed into the same GID.
            # ------------------------------------------------

            if (
                evidence.status
                == "CONTRADICTED"
            ):

                return True


        return False


    # ========================================================
    # FIND BEST AND SECOND-BEST GID
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
            #
            # Two people visible simultaneously in the same
            # camera and repeatedly far apart cannot belong
            # to the same GID.
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
            # CROSS-CAMERA GEOMETRY VETO
            #
            # If synchronized ground-plane geometry says
            # the candidate is physically inconsistent with
            # this identity, remove the GID completely.
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
            # APPEARANCE / GALLERY SCORE
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


        # ----------------------------------------------------
        # Every existing GID may have been eliminated
        # by compatibility or geometry.
        # ----------------------------------------------------

        if not candidates:
            return None


        # ----------------------------------------------------
        # Rank GIDs using gallery top-k similarity.
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item:
                item["topk"],
            reverse=True,
        )


        best = candidates[0]


        # ----------------------------------------------------
        # SECOND-BEST COMPETITOR + SCORE MARGIN
        # ----------------------------------------------------

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
    # EVALUATE TRACKLET
    #
    # IMPORTANT:
    #
    # evaluate() DOES NOT change any GID.
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
            candidate_tracklet=candidate_tracklet,
            tracklet_lookup=tracklet_lookup,
        )


        # ----------------------------------------------------
        # No compatible existing GID.
        #
        # Possible reasons:
        #
        # 1. no identity exists
        #
        # 2. same-camera spatial conflict rejected all
        #
        # 3. cross-camera geometry rejected all
        # ----------------------------------------------------

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
            best["global_id"]
        ]


        # ----------------------------------------------------
        # SAME-CAMERA OR CROSS-CAMERA APPEARANCE THRESHOLDS?
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CURRENT APPEARANCE DECISION
        #
        # max:
        #     strongest individual gallery member
        #
        # topk:
        #     gallery support
        #
        # score_margin:
        #     measured, but not yet used for automatic
        #     acceptance/rejection.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # NEW must NEVER suggest an existing GID.
        # ----------------------------------------------------

        if status == "NEW":

            suggested_gid = None

        else:

            suggested_gid = (
                best["global_id"]
            )


        return MatchResult(
            status=status,
            global_id=suggested_gid,

            max_similarity=
                best["max"],

            mean_similarity=
                best["mean"],

            topk_similarity=
                best["topk"],

            second_global_id=
                best["second_global_id"],

            second_topk_similarity=
                best["second_topk"],

            score_margin=
                best["margin"],
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
                embedding,

            "candidate_tracklet":
                candidate_tracklet,
        }


    # ========================================================
    # SAFE OPEN-SET ASSIGNMENT
    #
    # STRONG
    #     -> merge into existing GID
    #
    # PENDING
    #     -> keep unresolved
    #
    # NEW
    #     -> create new GID
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


        # ----------------------------------------------------
        # FIRST IDENTITY
        # ----------------------------------------------------

        if not self.identities:

            gid = self.create_identity(
                camera_id=camera_id,
                local_track_id=local_track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                embedding=embedding,
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
            )


        # ----------------------------------------------------
        # EVALUATE AGAINST EXISTING GIDS
        # ----------------------------------------------------

        match = self.evaluate(
            camera_id=camera_id,
            embedding=embedding,
            candidate_tracklet=candidate_tracklet,
            tracklet_lookup=tracklet_lookup,
        )


        # ====================================================
        # STRONG
        #
        # Safe enough according to our current gates and
        # appearance evidence.
        # ====================================================

        if match.status == "STRONG":

            self.add_to_identity(
                global_id=match.global_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                embedding=embedding,
            )


            # If this track was previously pending,
            # remove the stale pending copy.
            pending_key = (
                camera_id,
                local_track_id,
            )

            self.pending_tracklets.pop(
                pending_key,
                None,
            )


            return AssignmentResult(
                status="STRONG",
                global_id=match.global_id,

                created_new=False,
                merged=True,

                match=match,
            )


        # ====================================================
        # PENDING
        #
        # Keep the tracklet unresolved.
        #
        # DO NOT contaminate a GID.
        # DO NOT create a new identity yet.
        # ====================================================

        if match.status == "PENDING":

            self.add_pending(
                camera_id=camera_id,
                local_track_id=local_track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                embedding=embedding,
                candidate_tracklet=candidate_tracklet,
            )


            return AssignmentResult(
                status="PENDING",
                global_id=match.global_id,

                created_new=False,
                merged=False,

                match=match,
            )


        # ====================================================
        # NEW
        #
        # No sufficiently compatible existing GID.
        # ====================================================

        gid = self.create_identity(
            camera_id=camera_id,
            local_track_id=local_track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            embedding=embedding,
        )


        pending_key = (
            camera_id,
            local_track_id,
        )

        self.pending_tracklets.pop(
            pending_key,
            None,
        )


        return AssignmentResult(
            status="NEW",
            global_id=gid,

            created_new=True,
            merged=False,

            match=match,
        )


    # ========================================================
    # RE-EVALUATE PENDING TRACKLETS
    # ========================================================

    def reevaluate_pending(
        self,
        tracklet_lookup=None,
    ):

        results = []


        # Work on a copy because resolved entries
        # can be removed from pending_tracklets.
        pending_items = list(
            self.pending_tracklets.items()
        )


        for key, pending in pending_items:

            match = self.evaluate(
                camera_id=pending[
                    "camera_id"
                ],

                embedding=pending[
                    "embedding"
                ],

                candidate_tracklet=pending[
                    "candidate_tracklet"
                ],

                tracklet_lookup=tracklet_lookup,
            )


            # =================================================
            # NOW STRONG
            # =================================================

            if match.status == "STRONG":

                self.add_to_identity(
                    global_id=match.global_id,

                    camera_id=pending[
                        "camera_id"
                    ],

                    local_track_id=pending[
                        "local_track_id"
                    ],

                    start_frame=pending[
                        "start_frame"
                    ],

                    end_frame=pending[
                        "end_frame"
                    ],

                    embedding=pending[
                        "embedding"
                    ],
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
                            match.global_id,
                    }
                )


            # =================================================
            # NOW CLEARLY NEW
            # =================================================

            elif match.status == "NEW":

                gid = self.create_identity(
                    camera_id=pending[
                        "camera_id"
                    ],

                    local_track_id=pending[
                        "local_track_id"
                    ],

                    start_frame=pending[
                        "start_frame"
                    ],

                    end_frame=pending[
                        "end_frame"
                    ],

                    embedding=pending[
                        "embedding"
                    ],
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
                            "NEW",

                        "global_id":
                            gid,
                    }
                )


            # =================================================
            # STILL AMBIGUOUS
            # =================================================

            else:

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
                            match.global_id,
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
        print("GLOBAL IDENTITIES")
        print("=================")


        for gid in sorted(
            self.identities
        ):

            print(
                self.identities[
                    gid
                ].summary()
            )


        print()

        print(
            "Pending tracklets:",
            len(
                self.pending_tracklets
            ),
        )
