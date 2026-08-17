from dataclasses import dataclass
from typing import Optional

try:
    from .global_identity import GlobalIdentity
    from .identity_compatibility import identity_has_conflict
except ImportError:
    from global_identity import GlobalIdentity
    from identity_compatibility import identity_has_conflict


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

    def __init__(self):

        self.identities = {}

        self.next_global_id = 1

        # ----------------------------------------------------
        # Unresolved tracklets are stored here.
        # ----------------------------------------------------

        self.pending_tracklets = {}

        # ----------------------------------------------------
        # TEMPORARY TERRACE THRESHOLDS
        #
        # These are experimental values.
        # They are not final production thresholds.
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

        self.identities[gid] = identity

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
            "max": identity.max_similarity(
                embedding
            ),

            "mean": identity.mean_similarity(
                embedding
            ),

            "topk": identity.topk_similarity(
                embedding,
                k=3,
            ),
        }


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


        for gid, identity in self.identities.items():

            # ------------------------------------------------
            # SAME-CAMERA SPATIAL CONFLICT GATE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # APPEARANCE / GALLERY SCORE
            # ------------------------------------------------

            scores = self.score_identity(
                identity,
                embedding,
            )

            candidates.append(
                {
                    "global_id": gid,
                    "topk": scores["topk"],
                    "max": scores["max"],
                    "mean": scores["mean"],
                }
            )


        # Every existing identity may have been rejected
        # by the compatibility gate.
        if not candidates:
            return None


        # ----------------------------------------------------
        # Rank by gallery top-k similarity.
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: item["topk"],
            reverse=True,
        )

        best = candidates[0]


        # ----------------------------------------------------
        # SECOND-BEST COMPETITOR
        # ----------------------------------------------------

        if len(candidates) >= 2:

            second = candidates[1]

            best["second_global_id"] = (
                second["global_id"]
            )

            best["second_topk"] = (
                second["topk"]
            )

            best["margin"] = (
                best["topk"]
                - second["topk"]
            )

        else:

            best["second_global_id"] = None
            best["second_topk"] = None
            best["margin"] = None


        return best


    # ========================================================
    # EVALUATE TRACKLET
    #
    # IMPORTANT:
    #
    # This method does NOT modify any GID.
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
        # No compatible existing identity.
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
        # SAME CAMERA OR CROSS CAMERA?
        # ----------------------------------------------------

        same_camera = (
            camera_id
            in identity.cameras_seen
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
        # Margin is measured but not yet used as an
        # automatic accept/reject rule.
        # ----------------------------------------------------

        if (
            best["max"] >= strong_threshold
            and best["topk"] >= pending_threshold
        ):

            status = "STRONG"

        elif (
            best["max"] >= pending_threshold
        ):

            status = "PENDING"

        else:

            status = "NEW"


        # ----------------------------------------------------
        # NEW must not point to an existing GID.
        # ----------------------------------------------------

        if status == "NEW":

            suggested_gid = None

        else:

            suggested_gid = best[
                "global_id"
            ]


        return MatchResult(
            status=status,
            global_id=suggested_gid,

            max_similarity=best["max"],
            mean_similarity=best["mean"],
            topk_similarity=best["topk"],

            second_global_id=best[
                "second_global_id"
            ],

            second_topk_similarity=best[
                "second_topk"
            ],

            score_margin=best[
                "margin"
            ],
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

        key = (
            camera_id,
            local_track_id,
        )

        self.pending_tracklets[key] = {
            "camera_id": camera_id,

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
    #     -> store for later
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

        # ----------------------------------------------------
        # FIRST IDENTITY IN SYSTEM
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
        # EVALUATE EXISTING IDENTITIES
        # ----------------------------------------------------

        match = self.evaluate(
            camera_id=camera_id,
            embedding=embedding,
            candidate_tracklet=candidate_tracklet,
            tracklet_lookup=tracklet_lookup,
        )


        # ----------------------------------------------------
        # STRONG
        #
        # Merge into existing identity.
        # ----------------------------------------------------

        if match.status == "STRONG":

            self.add_to_identity(
                global_id=match.global_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                embedding=embedding,
            )

            return AssignmentResult(
                status="STRONG",
                global_id=match.global_id,
                created_new=False,
                merged=True,
                match=match,
            )


        # ----------------------------------------------------
        # PENDING
        #
        # Do not contaminate the gallery.
        # Store for future re-evaluation.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # NEW
        #
        # Existing identities are not sufficiently
        # compatible.
        # ----------------------------------------------------

        gid = self.create_identity(
            camera_id=camera_id,
            local_track_id=local_track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            embedding=embedding,
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

        # Copy items because resolved entries will
        # be deleted from the pending dictionary.
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


            # ------------------------------------------------
            # NOW STRONG
            #
            # Additional gallery evidence made the
            # identity sufficiently convincing.
            # ------------------------------------------------

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


            # ------------------------------------------------
            # NOW CLEARLY NEW
            # ------------------------------------------------

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


            # ------------------------------------------------
            # STILL PENDING
            # ------------------------------------------------

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

    def summary(self):

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
