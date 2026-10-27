"""Distribution of the kernel differences in tests/test_equivalence.py's sweep.

Runs the fast sweep (and, with ``--slow``, the large scenarios) and prints, per
compared quantity, the number of comparisons and the median and maximum
relative difference; the distribution of the tolerances; and the stages not
compared with the oracle, by reason.

    python scripts/equivalence_summary.py [--slow]
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
)
import test_equivalence as eq  # noqa: E402

scenarios = [eq.scenario(i) for i in range(eq.N_FAST)]
if "--slow" in sys.argv:
    scenarios += [eq._large_scenario(i) for i in range(eq.N_SLOW)]
for sc in scenarios:
    eq.compare(sc)
diffs: dict[str, list[float]] = defaultdict(list)
for _, name, rel, _, _ in eq.RECORDS:
    diffs[name].append(rel)
tols = np.array([tol for *_, tol, _ in eq.RECORDS])
print(f"{len(eq.RECORDS)} comparisons over {len(scenarios)} scenarios")
print("| quantity | n | median | max |")
print("|-|-|-|-|")
for name in sorted(diffs):
    v = np.array(diffs[name])
    print(f"| {name} | {v.size} | {np.median(v):.1e} | {v.max():.1e} |")
print(
    f"tolerances: median {np.median(tols):.1e}, max {tols.max():.1e}, "
    f"{np.sum(tols > 1e-8)} above 1e-8, {np.sum(tols >= 1e-6)} at or above 1e-6"
)


def reason_of(r: str) -> str:
    """The skip reason without its numbers."""
    if "amplification" in r:
        return r.split(":")[0] + ": amplification too large"
    return r.split(" (")[0]


groups: dict[str, list[int]] = defaultdict(list)
for i, r in eq.SKIPS:
    groups[reason_of(r)].append(i)
print(f"{len(eq.SKIPS)} stages not compared with the oracle:")
for reason, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    scen = sorted(set(idx))
    more = " ..." if len(scen) > 12 else ""
    print(f"  {len(idx):4d}  {reason}  ({len(scen)} scenarios: {scen[:12]}{more})")
for i, r in eq.SKIPS:
    if "amplification" in r:
        print("   ", r, "|", next(sc for sc in scenarios if sc.index == i))
