from dataclasses import dataclass, field
from typing import List

import numpy as np


# ============================================================
# LOCAL TRACK REFERENCE
# ============================================================

@dataclass
class IdentityMember:

    camera_id: int
    local_track_id: int

    start_frame: int
    end_frame: int


# ============================================================
# GLOBAL IDENTITY
# ============================================================

@dataclass
class GlobalIdentity:

    global_id: int

    members: List[IdentityMember] = field(
        default_factory=list
    )

    embedding_gallery: List[np.ndarray] = field(
        default_factory=list
    )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:

            raise ValueError(
                "Embedding has zero norm"
            )

        return (
            embedding
            / norm
        )


    # ========================================================
    # ADD MEMBER
    # ========================================================

    def add_member(
        self,
        camera_id: int,
        local_track_id: int,
        start_frame: int,
        end_frame: int,
        embedding: np.ndarray,
    ) -> None:

        # Avoid accidentally adding the exact same
        # local track twice.

        if self.contains(
            camera_id,
            local_track_id,
        ):

            return

        member = IdentityMember(
            camera_id=camera_id,
            local_track_id=local_track_id,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        normalized_embedding = (
            self.normalize_embedding(
                embedding
            )
        )

        self.members.append(
            member
        )

        self.embedding_gallery.append(
            normalized_embedding
        )


    # ========================================================
    # MEMBER CHECK
    # ========================================================

    def contains(
        self,
        camera_id: int,
        local_track_id: int,
    ) -> bool:

        return any(
            member.camera_id == camera_id
            and member.local_track_id == local_track_id

            for member in self.members
        )


    # ========================================================
    # SIMILARITY TO ENTIRE GALLERY
    # ========================================================

    def similarities(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:

        if not self.embedding_gallery:

            return np.array(
                [],
                dtype=np.float32,
            )

        query = self.normalize_embedding(
            embedding
        )

        gallery = np.stack(
            self.embedding_gallery,
            axis=0,
        )

        # All vectors are normalized.
        # Dot product = cosine similarity.

        return (
            gallery
            @ query
        )


    # ========================================================
    # BEST GALLERY SIMILARITY
    # ========================================================

    def max_similarity(
        self,
        embedding: np.ndarray,
    ) -> float:

        scores = self.similarities(
            embedding
        )

        if len(scores) == 0:
            return -1.0

        return float(
            np.max(scores)
        )


    # ========================================================
    # AVERAGE GALLERY SIMILARITY
    # ========================================================

    def mean_similarity(
        self,
        embedding: np.ndarray,
    ) -> float:

        scores = self.similarities(
            embedding
        )

        if len(scores) == 0:
            return -1.0

        return float(
            np.mean(scores)
        )


    # ========================================================
    # TOP-K GALLERY SIMILARITY
    # ========================================================

    def topk_similarity(
        self,
        embedding: np.ndarray,
        k: int = 3,
    ) -> float:

        scores = self.similarities(
            embedding
        )

        if len(scores) == 0:
            return -1.0

        k = min(
            k,
            len(scores),
        )

        top_scores = np.sort(
            scores
        )[-k:]

        return float(
            np.mean(top_scores)
        )


    # ========================================================
    # CENTROID EMBEDDING
    # ========================================================

    @property
    def centroid(self):

        if not self.embedding_gallery:
            return None

        centroid = np.mean(
            np.stack(
                self.embedding_gallery,
                axis=0,
            ),
            axis=0,
        )

        return self.normalize_embedding(
            centroid
        )


    # ========================================================
    # CAMERAS SEEN
    # ========================================================

    @property
    def cameras_seen(self):

        return sorted(
            {
                member.camera_id
                for member in self.members
            }
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> str:

        member_text = ", ".join(
            (
                f"c{member.camera_id}:"
                f"{member.local_track_id}"
            )

            for member in self.members
        )

        return (
            f"GID_{self.global_id:04d} | "
            f"members={len(self.members)} | "
            f"cameras={self.cameras_seen} | "
            f"[{member_text}]"
        )
