from pathlib import Path

from src.build_tracklets import load_tracklets
from src.identity_compatibility import same_camera_conflict


# ------------------------------------------------------------
# SAME PERSON / TRACK FRAGMENTATION
#
# Expected:
#     conflict = False
# ------------------------------------------------------------

NO_CONFLICT_PAIRS = [
    (15, 26),
    (701, 740),
    (56, 149),
    (707, 746),
    (188, 221),
]


# ------------------------------------------------------------
# DIFFERENT PEOPLE SEEN AT THE SAME TIME AND FAR APART
#
# Expected:
#     conflict = True
# ------------------------------------------------------------

CONFLICT_PAIRS = [
    (719, 845),
    (716, 775),
    (379, 593),
    (570, 593),
    (609, 716),
    (37, 78),
]


def main():

    tracklets = load_tracklets(
        Path(
            "output/terrace_bytetrack_tuned/"
            "tracks_c0.csv"
        )
    )

    if isinstance(tracklets, dict):
        values = tracklets.values()
    else:
        values = tracklets

    by_id = {
        tracklet.local_track_id: tracklet
        for tracklet in values
    }


    passed = 0
    total = 0


    print()
    print("NO-CONFLICT TESTS")
    print("=================")

    for a, b in NO_CONFLICT_PAIRS:

        result = same_camera_conflict(
            by_id[a],
            by_id[b],
        )

        ok = result is False

        total += 1

        if ok:
            passed += 1

        print(
            f"c0:{a:<4} vs c0:{b:<4} | "
            f"conflict={str(result):<5} | "
            f"{'OK' if ok else 'WRONG'}"
        )


    print()
    print("CONFLICT TESTS")
    print("==============")

    for a, b in CONFLICT_PAIRS:

        result = same_camera_conflict(
            by_id[a],
            by_id[b],
        )

        ok = result is True

        total += 1

        if ok:
            passed += 1

        print(
            f"c0:{a:<4} vs c0:{b:<4} | "
            f"conflict={str(result):<5} | "
            f"{'OK' if ok else 'WRONG'}"
        )


    print()
    print("RESULT")
    print("======")
    print(
        f"Passed: {passed}/{total}"
    )


if __name__ == "__main__":
    main()
