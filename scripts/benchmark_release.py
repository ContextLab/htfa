"""Matched current-API benchmarks on an otherwise idle host.

Run: python scripts/benchmark_release.py OUTPUT.json
Requires psutil for external-process CPU monitoring. Downloads, preprocessing,
voxel standardization, and warmup are excluded from timing. Results record source
and input hashes, fit settings, convergence, and competing CPU activity. Reject
rows marked contaminated or without monitoring samples.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from urllib.error import URLError

import numpy as np
import psutil
import requests
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

import htfa
from htfa.tfa import standardize_data

BASELINE_PERCENT = 10.0  # percent of the entire machine
PROCESS_PERCENT = 25.0  # percent of one CPU core, as reported by psutil.Process
SUSTAINED_SAMPLES = 3
SAMPLE_INTERVAL = 1.0


class ExternalCPU:
    """Track process CPU deltas; the initial call primes every process counter."""

    def __init__(self):
        self.processes = {}
        self.sample()  # cpu_percent's first call measures nothing

    def sample(self):
        excluded = {os.getpid()} | {
            p.pid for p in psutil.Process().children(recursive=True)
        }
        active = set()
        readings = []
        for p in psutil.process_iter():
            if p.pid in excluded:
                continue
            try:
                key = (p.pid, p.create_time())  # distinguish a reused PID
                active.add(key)
                if key not in self.processes:
                    self.processes[key] = p
                    p.cpu_percent()
                    continue
                process = self.processes[key]
                cpu = process.cpu_percent()
                readings.append(
                    dict(pid=p.pid, created=key[1], name=p.name(), cpu_percent=cpu)
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.processes = {
            key: value for key, value in self.processes.items() if key in active
        }
        readings.sort(key=lambda row: row["cpu_percent"], reverse=True)
        return dict(
            external_cpu_percent=sum(row["cpu_percent"] for row in readings),
            processes=[row for row in readings if row["cpu_percent"] > 0],
        )


def idle():
    """Require three quiet host samples; record them without stopping other work."""
    reader = ExternalCPU()
    history = []
    quiet = 0
    for _ in range(60):
        usage = psutil.cpu_percent(interval=2)
        row = reader.sample()
        row["host_cpu_percent"] = usage
        history.append(row)
        # A machine-wide average alone can hide another single-core job.
        largest = max((p["cpu_percent"] for p in row["processes"]), default=0)
        quiet = (
            quiet + 1 if usage < BASELINE_PERCENT and largest <= PROCESS_PERCENT else 0
        )
        if quiet == 3:
            return history
    raise RuntimeError("Host is busy; stop competing jobs and rerun the benchmark")


class Monitor:
    """Record CPU activity outside the benchmark's process family during a fit."""

    def __enter__(self):
        self.reader = ExternalCPU()  # prime outside the timed region
        self.stop = threading.Event()
        self.samples = []
        self.started = self.last_sample = time.perf_counter()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        row = self.reader.sample()
        now = time.perf_counter()
        row.update(
            elapsed_seconds=now - self.started, interval_seconds=now - self.last_sample
        )
        self.last_sample = now
        self.samples.append(row)

    def run(self):
        while not self.stop.wait(SAMPLE_INTERVAL):
            self.sample()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        # Include the remaining partial interval rather than missing its activity.
        if time.perf_counter() - self.last_sample >= 0.05:
            self.sample()

    def assessment(self):
        counts = {}
        offenders = {}
        for sample in self.samples:
            high = {
                (p["pid"], p["created"]): p
                for p in sample["processes"]
                if p["cpu_percent"] > PROCESS_PERCENT
            }
            counts = {key: counts.get(key, 0) + 1 for key in high}
            for key, count in counts.items():
                if count >= SUSTAINED_SAMPLES:
                    offenders[key] = high[key]
        return dict(
            uncontended=bool(self.samples) and not offenders,
            external_cpu_max=max(
                (s["external_cpu_percent"] for s in self.samples), default=None
            ),
            sustained_competing_processes=list(offenders.values()),
            sample_count=len(self.samples),
            samples=self.samples,
        )


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    imported = Path(htfa.__file__).resolve().parent
    if imported != (root / "htfa").resolve():
        raise RuntimeError(
            "Benchmark source and imported htfa differ; install this checkout "
            "with pip install -e . or set PYTHONPATH to its root"
        )
    hashes = {}
    for path in sorted((root / "htfa").rglob("*.py")):
        hashes[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_status=subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).splitlines(),
        source_file_sha256=hashes,
        imported_package=str(imported),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def fetch_data():
    """Retry transport failures before any timed work; preserve cached downloads."""
    for attempt in range(4):
        try:
            return htfa.datasets.fetch_development_fmri(n_subjects=3)
        except (requests.exceptions.RequestException, URLError, TimeoutError) as error:
            if attempt == 3:
                raise
            delay = 10 * 2**attempt
            print(
                f"Dataset transport failure ({error}); retrying in {delay} s",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def main():
    target = Path(sys.argv[1])
    if target.exists():
        raise FileExistsError(target)
    ds = fetch_data()
    # The loader detrends/regresses confounds; it does not z-score voxels.
    # Standardize once outside timing, then fit these identical arrays in both modes.
    data = [standardize_data(x, "auto") for x in ds.data]
    meta = dict(
        **source_provenance(),
        python=platform.python_version(),
        platform=platform.platform(),
        processor=subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if sys.platform == "darwin"
        else platform.processor(),
        cpu_count=os.cpu_count(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        htfa=htfa.__version__,
        threadpools=threadpool_info(),
        subject_ids=ds.subject_ids,
        shapes=[list(x.shape) for x in data],
        data_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in data],
        coordinates_sha256=[hashlib.sha256(r.tobytes()).hexdigest() for r in ds.coords],
        preprocessing=(
            "Loader confound regression and detrending, then voxel-wise temporal "
            "z-scoring with standardize_data(auto), outside timing."
        ),
        protocol=(
            "Two repetitions; mode order alternates; one fit at a time; "
            "2 BLAS threads per TFA; no downloads, preprocessing, or "
            "standardization in timed region."
        ),
        monitoring=dict(
            baseline_host_percent_below=BASELINE_PERCENT,
            baseline_external_process_percent_at_most=PROCESS_PERCENT,
            baseline_consecutive_samples=3,
            external_process_percent_above=PROCESS_PERCENT,
            contamination_consecutive_samples=SUSTAINED_SAMPLES,
            sample_interval_seconds=SAMPLE_INTERVAL,
            percent_units="host: percent of all cores; process: percent of one core",
        ),
        rows=[],
    )
    X, R, _ = htfa.make_tfa_data(
        grid_shape=(8, 8, 8), n_samples=20, K=3, random_state=0
    )
    with threadpool_limits(limits=2):
        htfa.TFA(K=3, init="ncorr", max_iter=2, random_state=0).fit(X, R)
    for repeat in range(2):
        for k in (50, 100):
            for s, (X, R) in enumerate(zip(data, ds.coords, strict=True)):
                for mode in ("exact", "fast") if repeat == 0 else ("fast", "exact"):
                    baseline = idle()
                    settings = dict(
                        K=k,
                        mode=mode,
                        init="ncorr",
                        backend="numpy",
                        standardize=False,
                        max_iter=100,
                        threshold=1e-3,
                        random_state=0,
                    )
                    with (
                        threadpool_limits(limits=2),
                        warnings.catch_warnings(record=True) as caught,
                        Monitor() as monitor,
                    ):
                        start = time.perf_counter()
                        model = htfa.TFA(**settings).fit(X, R)
                        seconds = time.perf_counter() - start
                    activity = monitor.assessment()
                    row = dict(
                        kind="TFA",
                        repeat=repeat,
                        subject=ds.subject_ids[s],
                        settings=settings,
                        seconds=seconds,
                        cost=float(model.cost_),
                        iterations=int(model.n_iter_),
                        stop=model.converged_reason_,
                        warnings=[str(w.message) for w in caught],
                        baseline=baseline,
                        **activity,
                    )
                    meta["rows"].append(row)
                    target.write_text(json.dumps(meta, indent=2) + "\n")
                    print(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("baseline", "samples")
                        },
                        flush=True,
                    )
    print("Complete", target, flush=True)


if __name__ == "__main__":
    main()
