import re

from src.analyze_physical_identity_graph import (
    UnionFind,
    build_current_state,
    build_graph,
    collect_fragments,
    component_compatible,
)


LOG_PATH = "/tmp/reciprocal_stitch_scan.log"


SHORT_MAX_GAP = 75
SHORT_MIN_REID = 0.88
SHORT_MAX_ENDPOINT = 40.0

MEDIUM_MAX_GAP = 150
MEDIUM_MIN_REID = 0.90
MEDIUM_MAX_ENDPOINT = 40.0


# ============================================================
# RECIPROCAL LOG PARSING
# ============================================================

def load_reciprocal_rows():

    profile = None
    rows = []

    profile_re = re.compile(
        r"PROFILE\s+(\S+)"
    )

    row_re = re.compile(
        r"^c([0-3]) \| "
        r"(.+?)\s+->\s+(.+?)\s+"
        r"\| owners=(.+?)->(.+?) "
        r"\| ReID=([0-9.]+) "
        r"\| gap=\s*([0-9]+) "
        r"\| endpoint=([0-9.]+|NA)"
    )

    with open(
        LOG_PATH,
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            profile_match = (
                profile_re.search(
                    line
                )
            )

            if profile_match:

                profile = (
                    profile_match.group(
                        1
                    )
                )

                continue


            row_match = (
                row_re.search(
                    line.strip()
                )
            )


            if (
                row_match is None
                or profile is None
            ):

                continue


            endpoint_text = (
                row_match.group(
                    8
                )
            )


            if endpoint_text == "NA":

                continue


            rows.append(
                {
                    "profile":
                        profile,

                    "camera":
                        int(
                            row_match.group(
                                1
                            )
                        ),

                    "fragment_a":
                        row_match.group(
                            2
                        ).strip(),

                    "fragment_b":
                        row_match.group(
                            3
                        ).strip(),

                    "owner_a":
                        row_match.group(
                            4
                        ).strip(),

                    "owner_b":
                        row_match.group(
                            5
                        ).strip(),

                    "reid":
                        float(
                            row_match.group(
                                6
                            )
                        ),

                    "gap":
                        int(
                            row_match.group(
                                7
                            )
                        ),

                    "endpoint":
                        float(
                            endpoint_text
                        ),
                }
            )


    return rows


# ============================================================
# TIERED LOCAL STITCHES
# ============================================================

def build_local_edges(
    rows,
):

    candidates = []


    for row in rows:

        # ----------------------------------------------------
        # TIER A:
        # short occlusion
        # ----------------------------------------------------

        if (
            row["profile"]
            == "SHORT_STRICT"

            and

            row["gap"]
            <= SHORT_MAX_GAP

            and

            row["reid"]
            >= SHORT_MIN_REID

            and

            row["endpoint"]
            <= SHORT_MAX_ENDPOINT
        ):

            candidates.append(
                {
                    **row,
                    "tier":
                        "TIER_A_LOCAL",
                }
            )

            continue


        # ----------------------------------------------------
        # TIER B:
        # medium occlusion
        # ----------------------------------------------------

        if (
            row["profile"]
            == "MEDIUM"

            and

            row["gap"]
            > SHORT_MAX_GAP

            and

            row["gap"]
            <= MEDIUM_MAX_GAP

            and

            row["reid"]
            >= MEDIUM_MIN_REID

            and

            row["endpoint"]
            <= MEDIUM_MAX_ENDPOINT
        ):

            candidates.append(
                {
                    **row,
                    "tier":
                        "TIER_B_LOCAL",
                }
            )


    # ========================================================
    # Keep only strongest candidate for each owner pair.
    # ========================================================

    best_by_pair = {}


    for row in candidates:

        pair = tuple(
            sorted(
                (
                    row["owner_a"],
                    row["owner_b"],
                )
            )
        )


        tier_priority = (
            2
            if row["tier"]
            == "TIER_A_LOCAL"
            else 1
        )


        rank = (
            tier_priority,
            row["reid"],
            -row["gap"],
            -row["endpoint"],
        )


        current = (
            best_by_pair.get(
                pair
            )
        )


        if (
            current is None
            or rank > current[0]
        ):

            best_by_pair[
                pair
            ] = (
                rank,
                row,
            )


    edges = []


    for pair, (_, row) in (
        best_by_pair.items()
    ):

        edges.append(
            {
                "owner_a":
                    pair[0],

                "owner_b":
                    pair[1],

                "result":
                    {
                        "accepted":
                            True,

                        "hard_negative":
                            False,

                        "reason":
                            "TIERED_LOCAL_STITCH",

                        "modes":
                            [
                                row[
                                    "tier"
                                ]
                            ],

                        "details":
                            {
                                "camera":
                                    row[
                                        "camera"
                                    ],

                                "reid":
                                    row[
                                        "reid"
                                    ],

                                "gap":
                                    row[
                                        "gap"
                                    ],

                                "endpoint":
                                    row[
                                        "endpoint"
                                    ],
                            },
                    },

                "source":
                    "LOCAL",
            }
        )


    return edges


# ============================================================
# EDGE PRIORITY
# ============================================================

def edge_priority(
    edge,
):

    modes = (
        edge[
            "result"
        ][
            "modes"
        ]
    )


    if (
        "LONG_GEOMETRY_CONSENSUS"
        in modes
    ):

        return 60


    if (
        "MULTICAM_XCAM"
        in modes
    ):

        return 55


    if (
        "MULTICAM_SEQUENTIAL"
        in modes
    ):

        return 50


    if (
        "TIER_A_LOCAL"
        in modes
    ):

        return 40


    if (
        "TIER_B_LOCAL"
        in modes
    ):

        return 35


    if (
        "STRONG_XCAM"
        in modes
    ):

        return 30


    if (
        "STRONG_SAMECAM"
        in modes
    ):

        return 20


    return 10


# ============================================================
# COMBINE BASE + LOCAL EDGES
# ============================================================

def combine_edges(
    base_edges,
    local_edges,
):

    combined = {}


    for original_edge in base_edges:

        edge = dict(
            original_edge
        )

        edge[
            "source"
        ] = "BASE"


        pair = tuple(
            sorted(
                (
                    edge[
                        "owner_a"
                    ],
                    edge[
                        "owner_b"
                    ],
                )
            )
        )


        combined[
            pair
        ] = edge


    for edge in local_edges:

        pair = tuple(
            sorted(
                (
                    edge[
                        "owner_a"
                    ],
                    edge[
                        "owner_b"
                    ],
                )
            )
        )


        current = (
            combined.get(
                pair
            )
        )


        if (
            current is None
            or
            edge_priority(
                edge
            )
            >
            edge_priority(
                current
            )
        ):

            combined[
                pair
            ] = edge


    edges = list(
        combined.values()
    )


    edges.sort(
        key=lambda edge: (
            -edge_priority(
                edge
            ),
            edge[
                "owner_a"
            ],
            edge[
                "owner_b"
            ],
        )
    )


    return edges


# ============================================================
# COMPONENT-SAFE UNION
# ============================================================

def cluster_safely(
    owners,
    pair_results,
    combined_edges,
):

    union_find = (
        UnionFind(
            owners
        )
    )


    applied = []
    blocked = []


    for edge in combined_edges:

        owner_a = (
            edge[
                "owner_a"
            ]
        )

        owner_b = (
            edge[
                "owner_b"
            ]
        )


        root_a = (
            union_find.find(
                owner_a
            )
        )

        root_b = (
            union_find.find(
                owner_b
            )
        )


        if root_a == root_b:

            continue


        groups = (
            union_find.groups()
        )


        component_a = (
            groups[
                root_a
            ]
        )

        component_b = (
            groups[
                root_b
            ]
        )


        (
            compatible,
            reason,
        ) = component_compatible(
            component_a,
            component_b,
            pair_results,
        )


        if not compatible:

            blocked.append(
                {
                    "edge":
                        edge,

                    "reason":
                        reason,
                }
            )

            continue


        union_find.union(
            owner_a,
            owner_b,
        )


        applied.append(
            edge
        )


    return (
        union_find.groups(),
        applied,
        blocked,
    )


# ============================================================
# FINAL HARD-NEGATIVE AUDIT
# ============================================================

def audit_components(
    groups,
    pair_results,
):

    failures = []


    for owners in (
        groups.values()
    ):

        for index, owner_a in enumerate(
            owners
        ):

            for owner_b in owners[
                index + 1:
            ]:

                pair = tuple(
                    sorted(
                        (
                            owner_a,
                            owner_b,
                        )
                    )
                )


                result = (
                    pair_results.get(
                        pair
                    )
                )


                if result is None:

                    continue


                if result.get(
                    "hard_negative",
                    False,
                ):

                    failures.append(
                        {
                            "owner_a":
                                owner_a,

                            "owner_b":
                                owner_b,

                            "reason":
                                result[
                                    "reason"
                                ],
                        }
                    )


    return failures


# ============================================================
# REPORT HELPERS
# ============================================================

def build_component_map(
    groups,
):

    component_map = {}


    for root, owners in (
        groups.items()
    ):

        for owner in owners:

            component_map[
                owner
            ] = root


    return component_map


def same_component(
    component_map,
    *owners,
):

    roots = {
        component_map.get(
            owner
        )

        for owner in owners
    }


    return (
        None not in roots

        and

        len(
            roots
        ) == 1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TIERED COMPONENT SAFETY AUDIT"
    )

    print(
        "=============================="
    )


    reciprocal_rows = (
        load_reciprocal_rows()
    )


    local_edges = (
        build_local_edges(
            reciprocal_rows
        )
    )


    print(
        "Reciprocal rows parsed:",
        len(
            reciprocal_rows
        ),
    )

    print(
        "Tiered local candidate edges:",
        len(
            local_edges
        ),
    )


    tier_a_count = sum(
        "TIER_A_LOCAL"
        in edge[
            "result"
        ][
            "modes"
        ]

        for edge in local_edges
    )


    tier_b_count = sum(
        "TIER_B_LOCAL"
        in edge[
            "result"
        ][
            "modes"
        ]

        for edge in local_edges
    )


    print(
        "Tier A candidates:",
        tier_a_count,
    )

    print(
        "Tier B candidates:",
        tier_b_count,
    )


    # ========================================================
    # Full validated 4-camera state
    # ========================================================

    (
        manager,
        tracklet_lookup,
    ) = build_current_state()


    population = (
        collect_fragments(
            manager,
            tracklet_lookup,
        )
    )


    owner_to_fragments = (
        population[
            "owner_to_fragments"
        ]
    )


    (
        owners,
        pair_results,
        base_edges,
    ) = build_graph(
        manager,
        owner_to_fragments,
    )


    print()
    print(
        "BASE GRAPH"
    )

    print(
        "=========="
    )

    print(
        "Owners:",
        len(
            owners
        ),
    )

    print(
        "Base conservative edges:",
        len(
            base_edges
        ),
    )


    combined_edges = (
        combine_edges(
            base_edges,
            local_edges,
        )
    )


    print(
        "Unique combined candidate edges:",
        len(
            combined_edges
        ),
    )


    (
        groups,
        applied,
        blocked,
    ) = cluster_safely(
        owners,
        pair_results,
        combined_edges,
    )


    hard_failures = (
        audit_components(
            groups,
            pair_results,
        )
    )


    local_applied = sum(
        edge[
            "source"
        ] == "LOCAL"

        for edge in applied
    )


    base_applied = sum(
        edge[
            "source"
        ] == "BASE"

        for edge in applied
    )


    local_blocked = sum(
        row[
            "edge"
        ][
            "source"
        ] == "LOCAL"

        for row in blocked
    )


    print()
    print(
        "COMPONENT-SAFE RESULT"
    )

    print(
        "====================="
    )

    print(
        "Initial owners:",
        len(
            owners
        ),
    )

    print(
        "Final components:",
        len(
            groups
        ),
    )

    print(
        "Total applied edges:",
        len(
            applied
        ),
    )

    print(
        "Base edges applied:",
        base_applied,
    )

    print(
        "Tiered local edges applied:",
        local_applied,
    )

    print(
        "Total blocked edges:",
        len(
            blocked
        ),
    )

    print(
        "Tiered local edges blocked:",
        local_blocked,
    )

    print(
        "Hard-negative failures inside final components:",
        len(
            hard_failures
        ),
    )


    print()
    print(
        "BLOCKED EDGES"
    )

    print(
        "============="
    )


    if not blocked:

        print(
            "None"
        )


    for row in blocked:

        edge = (
            row[
                "edge"
            ]
        )


        print(
            f"{edge['owner_a']:14s}"
            f" <-> "
            f"{edge['owner_b']:14s}"
            f" | source="
            f"{edge['source']}"
            f" | modes="
            f"{'+'.join(edge['result']['modes'])}"
            f" | block="
            f"{row['reason']}"
        )


    if hard_failures:

        print()
        print(
            "HARD FAILURES"
        )

        print(
            "============="
        )


        for failure in hard_failures:

            print(
                f"{failure['owner_a']}"
                f" <-> "
                f"{failure['owner_b']}"
                f" | "
                f"{failure['reason']}"
            )


    merged_groups = [
        values

        for values in groups.values()

        if len(
            values
        ) > 1
    ]


    print()
    print(
        "LARGEST SAFE COMPONENTS"
    )

    print(
        "======================="
    )


    for values in sorted(
        merged_groups,
        key=lambda values: (
            -len(
                values
            ),
            values,
        ),
    )[
        :35
    ]:

        print(
            f"size={len(values):2d} | "
            +
            ", ".join(
                values
            )
        )


    component_map = (
        build_component_map(
            groups
        )
    )


    print()
    print(
        "KNOWN REGRESSION CHECKS"
    )

    print(
        "======================="
    )


    checks = [
        (
            same_component(
                component_map,
                "G1",
                "G110",
            ),

            (
                "GID 1 + GID 110 "
                "are one person"
            ),
        ),

        (
            same_component(
                component_map,
                "G78",
                "G111",
                "G113",
            ),

            (
                "GID 78 + GID 111 + GID 113 "
                "are one person"
            ),
        ),

        (
            not same_component(
                component_map,
                "G111",
                "P:c3:15",
            ),

            (
                "c3:15 remains different "
                "from GID 111"
            ),
        ),

        (
            len(
                hard_failures
            ) == 0,

            (
                "No hard negative exists "
                "inside any final component"
            ),
        ),
    ]


    passed = 0


    for success, label in checks:

        if success:

            passed += 1


        print(
            (
                "PASS"
                if success
                else "FAIL"
            ),
            "|",
            label,
        )


    print()
    print(
        f"{passed}/{len(checks)} "
        "checks passed"
    )


    if passed != len(
        checks
    ):

        raise RuntimeError(
            "Tiered component safety "
            "regression failed"
        )


    print()
    print(
        "TIERED COMPONENT SAFETY AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
