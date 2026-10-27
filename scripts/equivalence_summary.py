"""Distribution of the kernel differences in tests/test_equivalence.py's sweep.

Runs the fast sweep (and, with ``--slow``, the large scenarios) and prints, per
compared quantity, how many comparisons used the plain tolerance (RTOL) and
the median and maximum relative difference among them, and the same for the
comparisons whose tolerance the conditioning raised.

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

indices = list(range(eq.N_FAST))
for i in indices:
    eq.compare(eq.scenario(i))
if "--slow" in sys.argv:
    for i in range(eq.N_SLOW):
        eq.compare(eq._large_scenario(i))
        indices.append(10_000 + i)
plain: dict[str, list[float]] = defaultdict(list)
raised: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
for index, name, rel, tol in eq.RECORDS:
    if tol <= eq.RTOL:
        plain[name].append(rel)
    else:
        raised[name].append((rel, tol, index))
print(f"{len(eq.RECORDS)} comparisons over {len(indices)} scenarios")
print("| quantity | n (tol 1e-10) | median | max | n (raised tol) | max diff / tol |")
print("|-|-|-|-|-|-|")
for name in sorted(set(plain) | set(raised)):
    p = np.array(plain.get(name, [np.nan]))
    r = raised.get(name, [])
    ratio = max((d / t for d, t, _ in r), default=float("nan"))
    print(
        f"| {name} | {len(plain.get(name, []))} | {np.nanmedian(p):.1e} | "
        f"{np.nanmax(p):.1e} | {len(r)} | {ratio:.1e} |"
    )
loose = sorted({i for v in raised.values() for _, t, i in v if t > 1e-6})
print("scenarios with a tolerance above 1e-6 (degenerate):", loose)
for i in loose:
    print("  ", eq.scenario(i) if i < 10_000 else eq._large_scenario(i - 10_000))
