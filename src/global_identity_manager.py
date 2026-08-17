from dataclasses import dataclass
from typing import Optional

try:
    from .global_identity import GlobalIdentity
    from .identity_compatibility import identity_has_conflict
except ImportError:
    from global_identity import GlobalIdentity
    from identity_compatibility import identity_has_conflict


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


@dataclass
class AssignmentResult:
    status: str
    global_id: Optional[int]

    created_new: bool
    merged: bool

    match: MatchResult


class GlobalIdentityManager:

    def __init__(self):

        self.identities = {}

        self.next_global_id = 1

        # ------------------------------------------------
        # TEMPORARY TERRACE THRESHOLDS
        #
        # These are not final production thresholds.
        # ------------------------------------------------

        self.same_camera_strong = 0.92
        self.same_camera_pending = 0.88

        self.cross_camera_strong = 0.90
        self.cross_camera_pending = 0.84


    # ====================================================
    # CREATE NEW GID
    # ====================================================

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


    # ====================================================
    # ADD TRACKLET TO EXISTING GID
    # ====================================================

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


    # ====================================================
    # SCORE ONE GID
    # ====================================================

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


    # ====================================================
    # FIND BEST AND SECOND-BEST GID
    # ====================================================

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
            #
            # If a candidate is simultaneously visible with
            # an existing same-camera member and they are
            # repeatedly spatially separated, this GID is
            # impossible and is removed from consideration.
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


        # Every GID may have been rejected by
        # compatibility constraints.
        if not candidates:
            return None


        # ------------------------------------------------
        # Rank identities by gallery top-k score.
        # ------------------------------------------------

        candidates.sort(
            key=lambda x: x["topk"],
            reverse=True,
        )


        best = candidates[0]


        # ------------------------------------------------
        # SECOND-BEST COMPETITOR
        # ------------------------------------------------

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


    # ====================================================
    # EVALUATE A TRACKLET
    #
    # This method DOES NOT modify the identity database.
    # ====================================================

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


        # ------------------------------------------------
        # No compatible GID.
        # ------------------------------------------------

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


        # ------------------------------------------------
        # SAME-CAMERA OR CROSS-CAMERA?
        # ------------------------------------------------

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


        # ------------------------------------------------
        # CURRENT DECISION RULE
        #
        # max:
        #     strongest gallery member
        #
        # topk:
        #     support from the gallery
        #
        # Margin is currently measured but NOT yet used
        # to automatically accept/reject.
        # ------------------------------------------------

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


        # ------------------------------------------------
        # NEW must never point to an existing GID.
        # ------------------------------------------------

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


    # ====================================================
    # SAFE OPEN-SET ASSIGNMENT
    #
    # STRONG:
    #     merge into existing GID
    #
    # PENDING:
    #     do not modify anything
    #
    # NEW:
    #     create a new GID
    # ====================================================

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

        # ------------------------------------------------
        # FIRST PERSON IN THE SYSTEM
        # ------------------------------------------------

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


        # ------------------------------------------------
        # FIND BEST EXISTING IDENTITY
        # ------------------------------------------------

        match = self.evaluate(
            camera_id=camera_id,
            embedding=embedding,
            candidate_tracklet=candidate_tracklet,
            tracklet_lookup=tracklet_lookup,
        )


        # ------------------------------------------------
        # STRONG MATCH
        #
        # Attach this local tracklet to the existing GID.
        # ------------------------------------------------

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


        # ------------------------------------------------
        # PENDING
        #
        # Do NOT:
        #
        #   - merge into existing GID
        #   - create a new GID
        #
        # Keep it unresolved for additional evidence.
        # ------------------------------------------------

        if match.status == "PENDING":

            return AssignmentResult(
                status="PENDING",
                global_id=match.global_id,
                created_new=False,
                merged=False,
                match=match,
            )


        # ------------------------------------------------
        # NEW PERSON
        #
        # Existing GIDs are not sufficiently compatible.
        # ------------------------------------------------

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


    # ====================================================
    # SUMMARY
    # ====================================================

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
