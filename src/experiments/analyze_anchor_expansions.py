import csv
from pathlib import Path

from src.global_identity_manager import (
    GlobalIdentityManager,
)

from src.identity_compatibility import (
    normalize_camera_id,
)

from src.cross_camera_geometry_gate import (
    evaluate_cross_camera_geometry,
)

from src.terrace_geometry import (
    load_ground_homographies,
)

from src.experiments.run_anchor_expansion_experiment import (
    STRICT_APPEARANCE_MIN,
    ANCHORED_EXPANSION_APPEARANCE_MIN,
    TRUSTED_GEOMETRY_MEDIAN_MAX,
    load_data,
    cosine_similarity,
    is_trusted_geometry,
    find_existing_geometry_candidates,
    find_strict_pending_candidates,
    add_to_existing_gid,
    create_anchor_from_pending,
    hold_pending,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "anchor_expansion_audit.csv"
)


# ============================================================
# AUDIT-ONLY BORDERLINE BANDS
#
# IMPORTANT:
#
# These do NOT affect assignment.
#
# They are only used to label a relaxed expansion as
# BORDERLINE for inspection.
#
# Current relaxed assignment threshold:
#     appearance >= 0.84
#
# Audit warnings:
#     GID top-k < 0.86
#     direct support appearance < 0.86
#     shared frames < 20
#     median geometry > 18
# ============================================================

AUDIT_APPEARANCE_WARNING = 0.86

AUDIT_MIN_SHARED_WARNING = 20

AUDIT_MEDIAN_WARNING = 18.0


# ============================================================
# BUILD EMBEDDING LOOKUP
# ============================================================

def build_embedding_lookup(
    items,
):

    lookup = {}


    for tracklet, embedding in items:

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        lookup[
            (
                camera,
                tracklet.local_track_id,
            )
        ] = embedding


    return lookup


# ============================================================
# MEMBER STRING
# ============================================================

def identity_member_string(
    identity,
):

    parts = []


    for member in identity.members:

        camera = normalize_camera_id(
            member.camera_id
        )


        parts.append(
            f"c{camera}:"
            f"{member.local_track_id}"
        )


    return ",".join(
        parts
    )


# ============================================================
# GET MEMBER TRACKLET
# ============================================================

def get_member_tracklet(
    member,
    tracklet_lookup,
):

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


# ============================================================
# DETAILED MEMBER EVIDENCE
#
# Unlike the actual assignment rule, this captures:
#
#   - geometry state
#   - direct pairwise appearance
#   - shared frames
#   - median geometry
#
# for every cross-camera GID member.
# ============================================================

def collect_member_evidence(
    current_tracklet,
    current_embedding,
    identity,
    tracklet_lookup,
    embedding_lookup,
    homographies,
):

    current_camera = normalize_camera_id(
        current_tracklet.camera_id
    )


    rows = []


    for member in identity.members:

        member_camera = normalize_camera_id(
            member.camera_id
        )


        # Geometry comparison is cross-camera only.
        if member_camera == current_camera:
            continue


        member_tracklet = get_member_tracklet(
            member,
            tracklet_lookup,
        )


        if member_tracklet is None:
            continue


        evidence = evaluate_cross_camera_geometry(
            current_tracklet,
            member_tracklet,
            homographies,
        )


        member_embedding = embedding_lookup.get(
            (
                member_camera,
                member.local_track_id,
            )
        )


        if member_embedding is None:

            direct_appearance = None

        else:

            direct_appearance = cosine_similarity(
                current_embedding,
                member_embedding,
            )


        rows.append(
            {
                "camera_id":
                    member_camera,

                "local_track_id":
                    member.local_track_id,

                "status":
                    evidence.status,

                "trusted":
                    is_trusted_geometry(
                        evidence
                    ),

                "shared":
                    evidence.shared_frames,

                "median":
                    evidence.median_distance,

                "direct_appearance":
                    direct_appearance,
            }
        )


    return rows


# ============================================================
# CLASSIFY ONE RELAXED EXPANSION
# ============================================================

def classify_expansion(
    gid_topk,
    evidence_rows,
):

    contradictions = [
        row
        for row in evidence_rows
        if row["status"] == "CONTRADICTED"
    ]


    trusted_supports = [
        row
        for row in evidence_rows
        if row["trusted"]
    ]


    # --------------------------------------------------------
    # This should NEVER occur because the assignment logic
    # rejects GIDs containing geometry contradictions.
    #
    # If it appears here, it is a serious regression.
    # --------------------------------------------------------

    if contradictions:

        return (
            "CONTRADICTED",
            [
                "geometry_contradiction",
            ],
        )


    # --------------------------------------------------------
    # Multiple independent cross-camera members geometrically
    # support this new tracklet.
    # --------------------------------------------------------

    if len(
        trusted_supports
    ) >= 2:

        return (
            "MULTI-SUPPORTED",
            [],
        )


    # --------------------------------------------------------
    # No trusted support should also be impossible for a
    # relaxed anchored expansion.
    # --------------------------------------------------------

    if not trusted_supports:

        return (
            "NO-TRUSTED-SUPPORT",
            [
                "no_trusted_geometry",
            ],
        )


    # --------------------------------------------------------
    # Exactly one trusted member.
    #
    # Determine whether it sits close to any of our current
    # experimental boundaries.
    # --------------------------------------------------------

    support = trusted_supports[0]


    reasons = []


    if (
        gid_topk
        < AUDIT_APPEARANCE_WARNING
    ):

        reasons.append(
            "gid_topk<0.86"
        )


    direct_appearance = support[
        "direct_appearance"
    ]


    if (
        direct_appearance is not None
        and
        direct_appearance
        < AUDIT_APPEARANCE_WARNING
    ):

        reasons.append(
            "direct_app<0.86"
        )


    if (
        support["shared"]
        < AUDIT_MIN_SHARED_WARNING
    ):

        reasons.append(
            "shared<20"
        )


    median = support[
        "median"
    ]


    if (
        median is not None
        and
        median > AUDIT_MEDIAN_WARNING
    ):

        reasons.append(
            "median>18"
        )


    if reasons:

        return (
            "BORDERLINE",
            reasons,
        )


    return (
        "SINGLE-SUPPORTED",
        [],
    )


# ============================================================
# CREATE AUDIT RECORD
# ============================================================

def make_audit_record(
    source,
    current_tracklet,
    current_embedding,
    gid,
    candidate,
    manager,
    tracklet_lookup,
    embedding_lookup,
    homographies,
):

    camera = normalize_camera_id(
        current_tracklet.camera_id
    )


    identity = manager.identities[
        gid
    ]


    members_before = (
        identity_member_string(
            identity
        )
    )


    evidence_rows = collect_member_evidence(
        current_tracklet,
        current_embedding,
        identity,
        tracklet_lookup,
        embedding_lookup,
        homographies,
    )


    classification, reasons = (
        classify_expansion(
            candidate["topk"],
            evidence_rows,
        )
    )


    trusted_supports = [
        row
        for row in evidence_rows
        if row["trusted"]
    ]


    weak_supports = [
        row
        for row in evidence_rows
        if (
            row["status"] == "SUPPORTED"
            and not row["trusted"]
        )
    ]


    contradictions = [
        row
        for row in evidence_rows
        if row["status"] == "CONTRADICTED"
    ]


    unknown = [
        row
        for row in evidence_rows
        if row["status"] == "UNKNOWN"
    ]


    return {
        "source":
            source,

        "camera_id":
            camera,

        "local_track_id":
            current_tracklet.local_track_id,

        "gid":
            gid,

        "classification":
            classification,

        "reasons":
            reasons,

        "gid_max":
            candidate["max"],

        "gid_mean":
            candidate["mean"],

        "gid_topk":
            candidate["topk"],

        "members_before":
            members_before,

        "trusted_supports":
            trusted_supports,

        "weak_supports":
            weak_supports,

        "contradictions":
            contradictions,

        "unknown":
            unknown,
    }


# ============================================================
# PRINT ONE GEOMETRY MEMBER ROW
# ============================================================

def format_member_evidence(
    row,
):

    median = row[
        "median"
    ]


    if median is None:

        median_text = "None"

    else:

        median_text = (
            f"{median:.2f}"
        )


    direct = row[
        "direct_appearance"
    ]


    if direct is None:

        appearance_text = "None"

    else:

        appearance_text = (
            f"{direct:.4f}"
        )


    return (
        f"c{row['camera_id']}:"
        f"{row['local_track_id']} | "
        f"status={row['status']} | "
        f"direct_app={appearance_text} | "
        f"shared={row['shared']} | "
        f"median={median_text}"
    )


# ============================================================
# AUDITED RELAXED PENDING SWEEP
#
# This reproduces the previous experiment's sweep, but records
# the state of the GID BEFORE each relaxed member is added.
# ============================================================

def audited_pending_sweep(
    manager,
    tracklet_lookup,
    embedding_lookup,
    homographies,
    anchored_gids,
    audit_records,
):

    resolved_count = 0

    changed = True


    while changed:

        changed = False


        pending_items = list(
            manager.pending_tracklets.items()
        )


        for key, pending in pending_items:

            tracklet = pending.get(
                "candidate_tracklet"
            )


            if tracklet is None:

                camera = normalize_camera_id(
                    pending["camera_id"]
                )


                tracklet = tracklet_lookup.get(
                    (
                        camera,
                        pending[
                            "local_track_id"
                        ],
                    )
                )


            if tracklet is None:
                continue


            candidates = (
                find_existing_geometry_candidates(
                    manager,
                    tracklet,
                    pending["embedding"],
                    tracklet_lookup,
                    homographies,
                    anchored_gids,
                    relaxed_anchored=True,
                )
            )


            # Same conservative rule as previous experiment:
            # exactly one anchored GID must qualify.
            if len(candidates) != 1:
                continue


            candidate = candidates[0]

            gid = candidate[
                "gid"
            ]


            record = make_audit_record(
                source=
                    "SWEEP",

                current_tracklet=
                    tracklet,

                current_embedding=
                    pending["embedding"],

                gid=
                    gid,

                candidate=
                    candidate,

                manager=
                    manager,

                tracklet_lookup=
                    tracklet_lookup,

                embedding_lookup=
                    embedding_lookup,

                homographies=
                    homographies,
            )


            audit_records.append(
                record
            )


            add_to_existing_gid(
                manager,
                tracklet,
                pending["embedding"],
                gid,
            )


            resolved_count += 1

            changed = True


    return resolved_count


# ============================================================
# CSV HELPERS
# ============================================================

def member_rows_to_string(
    rows,
):

    parts = []


    for row in rows:

        median = row[
            "median"
        ]


        direct = row[
            "direct_appearance"
        ]


        median_text = (
            ""
            if median is None
            else f"{median:.4f}"
        )


        direct_text = (
            ""
            if direct is None
            else f"{direct:.4f}"
        )


        parts.append(
            (
                f"c{row['camera_id']}:"
                f"{row['local_track_id']}"
                f"|{row['status']}"
                f"|app={direct_text}"
                f"|shared={row['shared']}"
                f"|median={median_text}"
            )
        )


    return ";".join(
        parts
    )


def write_csv(
    records,
):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fieldnames = [
        "source",
        "camera_id",
        "local_track_id",
        "gid",
        "classification",
        "reasons",
        "gid_max",
        "gid_mean",
        "gid_topk",
        "members_before",
        "trusted_supports",
        "weak_supports",
        "contradictions",
        "unknown_count",
    ]


    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        for record in records:

            writer.writerow(
                {
                    "source":
                        record[
                            "source"
                        ],

                    "camera_id":
                        record[
                            "camera_id"
                        ],

                    "local_track_id":
                        record[
                            "local_track_id"
                        ],

                    "gid":
                        record[
                            "gid"
                        ],

                    "classification":
                        record[
                            "classification"
                        ],

                    "reasons":
                        ",".join(
                            record[
                                "reasons"
                            ]
                        ),

                    "gid_max":
                        f"{record['gid_max']:.6f}",

                    "gid_mean":
                        f"{record['gid_mean']:.6f}",

                    "gid_topk":
                        f"{record['gid_topk']:.6f}",

                    "members_before":
                        record[
                            "members_before"
                        ],

                    "trusted_supports":
                        member_rows_to_string(
                            record[
                                "trusted_supports"
                            ]
                        ),

                    "weak_supports":
                        member_rows_to_string(
                            record[
                                "weak_supports"
                            ]
                        ),

                    "contradictions":
                        member_rows_to_string(
                            record[
                                "contradictions"
                            ]
                        ),

                    "unknown_count":
                        len(
                            record[
                                "unknown"
                            ]
                        ),
                }
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "ANCHOR EXPANSION AUDIT"
    )

    print(
        "======================"
    )


    print(
        "Bootstrap appearance:",
        STRICT_APPEARANCE_MIN,
    )


    print(
        "Relaxed anchored appearance:",
        ANCHORED_EXPANSION_APPEARANCE_MIN,
    )


    print(
        "Geometry median maximum:",
        TRUSTED_GEOMETRY_MEDIAN_MAX,
    )


    print()
    print(
        "Audit warning bands only:"
    )

    print(
        "  GID top-k <",
        AUDIT_APPEARANCE_WARNING,
    )

    print(
        "  direct support appearance <",
        AUDIT_APPEARANCE_WARNING,
    )

    print(
        "  shared frames <",
        AUDIT_MIN_SHARED_WARNING,
    )

    print(
        "  geometry median >",
        AUDIT_MEDIAN_WARNING,
    )


    # ========================================================
    # DATA
    # ========================================================

    items, tracklet_lookup = (
        load_data()
    )


    embedding_lookup = (
        build_embedding_lookup(
            items
        )
    )


    homographies = (
        load_ground_homographies()
    )


    manager = GlobalIdentityManager(
        homographies=
            homographies
    )


    anchored_gids = set()

    audit_records = []


    strict_existing_count = 0

    strict_pending_count = 0

    relaxed_live_count = 0

    relaxed_sweep_count = 0

    strong_held = 0

    pending_held = 0

    new_gid_count = 0


    # ========================================================
    # REPLAY
    # ========================================================

    for tracklet, embedding in items:

        camera = normalize_camera_id(
            tracklet.camera_id
        )


        track_id = (
            tracklet.local_track_id
        )


        key = (
            camera,
            track_id,
        )


        # ----------------------------------------------------
        # FIRST TRACK
        # ----------------------------------------------------

        if not manager.identities:

            manager.create_identity(
                camera_id=
                    camera,

                local_track_id=
                    track_id,

                start_frame=
                    tracklet.start_frame,

                end_frame=
                    tracklet.end_frame,

                embedding=
                    embedding,
            )


            new_gid_count += 1

            continue


        # ====================================================
        # PRIORITY 1
        # STRICT EXISTING GEOMETRY
        # ====================================================

        strict_existing = (
            find_existing_geometry_candidates(
                manager,
                tracklet,
                embedding,
                tracklet_lookup,
                homographies,
                anchored_gids,
                relaxed_anchored=False,
            )
        )


        if len(
            strict_existing
        ) == 1:

            candidate = (
                strict_existing[0]
            )


            gid = candidate[
                "gid"
            ]


            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )


            anchored_gids.add(
                gid
            )


            strict_existing_count += 1


            relaxed_sweep_count += (
                audited_pending_sweep(
                    manager,
                    tracklet_lookup,
                    embedding_lookup,
                    homographies,
                    anchored_gids,
                    audit_records,
                )
            )


            continue


        # ====================================================
        # PRIORITY 2
        # RELAXED EXISTING -> ANCHORED GID
        # ====================================================

        relaxed_existing = (
            find_existing_geometry_candidates(
                manager,
                tracklet,
                embedding,
                tracklet_lookup,
                homographies,
                anchored_gids,
                relaxed_anchored=True,
            )
        )


        if len(
            relaxed_existing
        ) == 1:

            candidate = (
                relaxed_existing[0]
            )


            gid = candidate[
                "gid"
            ]


            record = make_audit_record(
                source=
                    "LIVE",

                current_tracklet=
                    tracklet,

                current_embedding=
                    embedding,

                gid=
                    gid,

                candidate=
                    candidate,

                manager=
                    manager,

                tracklet_lookup=
                    tracklet_lookup,

                embedding_lookup=
                    embedding_lookup,

                homographies=
                    homographies,
            )


            audit_records.append(
                record
            )


            add_to_existing_gid(
                manager,
                tracklet,
                embedding,
                gid,
            )


            relaxed_live_count += 1


            relaxed_sweep_count += (
                audited_pending_sweep(
                    manager,
                    tracklet_lookup,
                    embedding_lookup,
                    homographies,
                    anchored_gids,
                    audit_records,
                )
            )


            continue


        # ====================================================
        # PRIORITY 3
        # STRICT PENDING PAIR -> NEW ANCHORED GID
        # ====================================================

        pending_candidates = (
            find_strict_pending_candidates(
                manager,
                tracklet,
                embedding,
                tracklet_lookup,
                homographies,
            )
        )


        if len(
            pending_candidates
        ) == 1:

            pending_candidate = (
                pending_candidates[0]
            )


            gid = create_anchor_from_pending(
                manager,
                tracklet,
                embedding,
                pending_candidate,
            )


            anchored_gids.add(
                gid
            )


            strict_pending_count += 1


            relaxed_sweep_count += (
                audited_pending_sweep(
                    manager,
                    tracklet_lookup,
                    embedding_lookup,
                    homographies,
                    anchored_gids,
                    audit_records,
                )
            )


            continue


        # ====================================================
        # APPEARANCE-ONLY
        #
        # Same evidence-gated policy as previous experiment:
        # appearance NEVER modifies a GID.
        # ====================================================

        match = manager.evaluate(
            camera_id=
                camera,

            embedding=
                embedding,

            candidate_tracklet=
                tracklet,

            tracklet_lookup=
                tracklet_lookup,
        )


        if match.status == "STRONG":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )


            strong_held += 1

            continue


        if match.status == "PENDING":

            hold_pending(
                manager,
                tracklet,
                embedding,
            )


            pending_held += 1

            continue


        # ====================================================
        # NEW SINGLETON
        # ====================================================

        manager.create_identity(
            camera_id=
                camera,

            local_track_id=
                track_id,

            start_frame=
                tracklet.start_frame,

            end_frame=
                tracklet.end_frame,

            embedding=
                embedding,
        )


        manager.pending_tracklets.pop(
            key,
            None,
        )


        new_gid_count += 1


    # ========================================================
    # FINAL RELAXED SWEEP
    # ========================================================

    relaxed_sweep_count += (
        audited_pending_sweep(
            manager,
            tracklet_lookup,
            embedding_lookup,
            homographies,
            anchored_gids,
            audit_records,
        )
    )


    # ========================================================
    # WRITE CSV
    # ========================================================

    write_csv(
        audit_records
    )


    # ========================================================
    # CLASSIFICATION COUNTS
    # ========================================================

    classification_counts = {}


    for record in audit_records:

        label = record[
            "classification"
        ]


        classification_counts[
            label
        ] = (
            classification_counts.get(
                label,
                0,
            )
            + 1
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "AUDIT SUMMARY"
    )

    print(
        "============="
    )


    print(
        "Tracklets:",
        len(items),
    )


    print(
        "Anchored GIDs:",
        len(
            anchored_gids
        ),
    )


    print(
        "Strict existing:",
        strict_existing_count,
    )


    print(
        "Strict pending anchors:",
        strict_pending_count,
    )


    print(
        "Relaxed LIVE additions:",
        relaxed_live_count,
    )


    print(
        "Relaxed SWEEP additions:",
        relaxed_sweep_count,
    )


    print(
        "Total relaxed additions audited:",
        len(
            audit_records
        ),
    )


    print(
        "Appearance STRONG held:",
        strong_held,
    )


    print(
        "Appearance PENDING held:",
        pending_held,
    )


    print(
        "Final GIDs:",
        len(
            manager.identities
        ),
    )


    print(
        "Final pending:",
        len(
            manager.pending_tracklets
        ),
    )


    print()
    print(
        "CLASSIFICATION COUNTS"
    )

    print(
        "====================="
    )


    labels = [
        "MULTI-SUPPORTED",
        "SINGLE-SUPPORTED",
        "BORDERLINE",
        "CONTRADICTED",
        "NO-TRUSTED-SUPPORT",
    ]


    for label in labels:

        print(
            f"{label:>20}: "
            f"{classification_counts.get(label, 0)}"
        )


    # ========================================================
    # COMPACT TABLE
    # ========================================================

    print()
    print(
        "RELAXED EXPANSION TABLE"
    )

    print(
        "======================="
    )


    print(
        f"{'Src':<6} "
        f"{'Track':<10} "
        f"{'GID':>5} "
        f"{'Class':<20} "
        f"{'TopK':>7} "
        f"{'Max':>7} "
        f"{'Sup':>4} "
        f"{'Con':>4} "
        f"Reasons"
    )


    print(
        "-" * 95
    )


    for record in audit_records:

        track_text = (
            f"c{record['camera_id']}:"
            f"{record['local_track_id']}"
        )


        reasons = (
            ",".join(
                record[
                    "reasons"
                ]
            )
            if record["reasons"]
            else "-"
        )


        print(
            f"{record['source']:<6} "
            f"{track_text:<10} "
            f"{record['gid']:>5} "
            f"{record['classification']:<20} "
            f"{record['gid_topk']:>7.4f} "
            f"{record['gid_max']:>7.4f} "
            f"{len(record['trusted_supports']):>4} "
            f"{len(record['contradictions']):>4} "
            f"{reasons}"
        )


    # ========================================================
    # DETAILED AUDIT
    # ========================================================

    print()
    print(
        "DETAILED RELAXED EXPANSIONS"
    )

    print(
        "==========================="
    )


    for index, record in enumerate(
        audit_records,
        start=1,
    ):

        print()
        print(
            "=" * 110
        )


        print(
            f"{index}. "
            f"{record['source']} | "
            f"c{record['camera_id']}:"
            f"{record['local_track_id']}"
            f" -> GID {record['gid']} | "
            f"{record['classification']}"
        )


        print(
            f"GID appearance: "
            f"mean={record['gid_mean']:.4f} "
            f"topk={record['gid_topk']:.4f} "
            f"max={record['gid_max']:.4f}"
        )


        print(
            "Members before:",
            record[
                "members_before"
            ],
        )


        if record["reasons"]:

            print(
                "Borderline reasons:",
                ", ".join(
                    record[
                        "reasons"
                    ]
                ),
            )


        print()
        print(
            "TRUSTED SUPPORTS:"
        )


        if record[
            "trusted_supports"
        ]:

            for row in record[
                "trusted_supports"
            ]:

                print(
                    "  ",
                    format_member_evidence(
                        row
                    ),
                )


        else:

            print(
                "  None"
            )


        print(
            "WEAK SUPPORTED:"
        )


        if record[
            "weak_supports"
        ]:

            for row in record[
                "weak_supports"
            ]:

                print(
                    "  ",
                    format_member_evidence(
                        row
                    ),
                )


        else:

            print(
                "  None"
            )


        print(
            "CONTRADICTIONS:"
        )


        if record[
            "contradictions"
        ]:

            for row in record[
                "contradictions"
            ]:

                print(
                    "  ",
                    format_member_evidence(
                        row
                    ),
                )


        else:

            print(
                "  None"
            )


        print(
            "UNKNOWN cross-camera members:",
            len(
                record[
                    "unknown"
                ]
            ),
        )


    print()
    print(
        "CSV written to:"
    )

    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()
