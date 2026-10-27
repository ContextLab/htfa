"""Write tests/data/fits_3298278.npz: the fits of tests/test_fit_equivalence.py's
FIT_CASES by the code that is importable as ``htfa`` (run it with the code of
commit 3298278 first on the path).

    git archive 3298278 htfa | tar -x -C /tmp/htfa_3298278
    PYTHONPATH=/tmp/htfa_3298278 python scripts/make_fit_references.py 3298278
"""

import datetime
import os
import platform
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "tests"))
import test_fit_equivalence as fe  # noqa: E402

commit = sys.argv[1]
arrays = {}
for i, case in enumerate(fe.FIT_CASES):
    for k, v in fe.fit_case(case).items():
        if k in ("centers", "widths", "weights", "cost"):
            arrays[f"case{i}_{k}"] = v
arrays["provenance"] = np.array(
    f"commit {commit}; TFA defaults, FIT_CASES of tests/test_fit_equivalence.py; numpy "
    f"{np.__version__}; {platform.platform()}; "
    f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"
)
os.makedirs(os.path.join(here, "..", "tests", "data"), exist_ok=True)
np.savez_compressed(
    os.path.join(here, "..", "tests", "data", f"fits_{commit}.npz"), **arrays
)
print("wrote", len(fe.FIT_CASES), "fits;", arrays["provenance"])
