from dataclasses import dataclass
from typing import Optional

try:
    from .global_identity import GlobalIdentity
except ImportError:
    from global_identity import GlobalIdentity


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


class GlobalIdentityManager:

    def __init__(self):

        self.identities = {}

        self.next_global_id = 1

        # ------------------------------------------------
        # Temporary thresholds learned from Terrace C0/C1
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
    # ADD TRACK TO EXISTING GID
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
    ):

        if not self.identities:
            return None

        candidates = []

        for gid, identity in self.identities.items():

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

        # Rank identities using gallery top-k similarity.
        candidates.sort(
            key=lambda x: x["topk"],
            reverse=True,
        )

        best = candidates[0]

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
    # EVALUATE NEW TRACKLET
    # ====================================================

    def evaluate(
        self,
        camera_id,
        embedding,
    ):

        best = self.find_best_identity(
            embedding
        )

        # No existing identities.
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
        # Determine whether matching against this GID is
        # same-camera or cross-camera.
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
        # Current decision logic.
        #
        # NOTE:
        # We are measuring score_margin now, but we are
        # NOT using it for the decision yet.
        #
        # We first want to observe real Terrace margins.
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


        # NEW means no existing GID should be suggested.
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
