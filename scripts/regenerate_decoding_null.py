"""Checkpoint tutorial 09 nulls against a specified regenerated fit (no upload).

Usage: python scripts/regenerate_decoding_null.py FIT_FILE OUTPUT [P=100]
Each permutation is saved separately and validated against input hashes on resume.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import htfa


def main():
    fit_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    output.mkdir(parents=True, exist_ok=True)
    fit_sha = hashlib.sha256(fit_path.read_bytes()).hexdigest()
    saved = dict(np.load(fit_path, allow_pickle=False))
    embeddings = htfa.datasets.fetch_tutorial_fit("mitchell2008_embeddings")
    e_sha = hashlib.sha256(embeddings["E"].tobytes()).hexdigest()
    notebook = json.loads(
        Path("docs/tutorials/09_word_decoding_mitchell2008.ipynb").read_text()
    )
    ns = {"__name__": "__main__"}

    def cell(index):
        exec(
            compile(
                "".join(notebook["cells"][index]["source"]),
                f"tutorial09-{index}",
                "exec",
            ),
            ns,
        )

    cell(3)
    cell(5)
    model = htfa.HTFA(**json.loads(str(saved["settings"])))
    for name in ("centers", "widths", "weights"):
        setattr(model, name + "_", list(saved[name]))
    model.global_centers_, model.global_widths_ = (
        saved["global_centers"],
        saved["global_widths"],
    )
    subjects = ns["subjects"]
    ns.update(
        model=model,
        E=embeddings["E"],
        res=htfa.HTFAResults(
            model,
            htfa.load(
                [s["X"] for s in subjects], coords=[s["coords"] for s in subjects]
            ),
        ),
    )
    cell(11)
    cell(15)
    # Define the permutation statistic without executing the cache branch.
    source = "".join(notebook["cells"][20]["source"]).split("if RECOMPUTE_NULL:")[0]
    exec(compile(source, "tutorial09-permutation", "exec"), ns)
    observed_start = time.perf_counter()
    with threadpool_limits(limits=1):
        cell(17)
    observed = np.r_[ns["recon_r"].mean(), ns["brain"].mean(0), ns["word"].mean(0)]
    print(
        "Observed decoding seconds:", time.perf_counter() - observed_start, flush=True
    )

    def one(index):
        path = output / f"decoding-permutation-{index:03d}.npz"
        if path.exists():
            cached = dict(np.load(path, allow_pickle=False))
            if (
                str(cached["fit_sha256"]) != fit_sha
                or str(cached["embeddings_sha256"]) != e_sha
            ):
                raise RuntimeError(f"Checkpoint inputs differ: {path}")
            return cached["statistic"]
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            statistic = ns["permuted_means"](1000 + index)
        np.savez(
            path,
            statistic=statistic,
            fit_sha256=fit_sha,
            embeddings_sha256=e_sha,
            seed=1000 + index,
            seconds=time.perf_counter() - started,
        )
        print("Saved permutation", index + 1, flush=True)
        return statistic

    started = time.perf_counter()
    null = np.array(Parallel(n_jobs=4)(delayed(one)(i) for i in range(count)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for n in sorted({min(20, count), count}):
        target = output / f"htfa_mitchell2008_null_{n}.npz"
        np.savez_compressed(
            target,
            null=null[:n],
            statistics=ns["names"],
            n_permutations=n,
            seeds=1000 + np.arange(n),
            seconds=time.perf_counter() - started,
            generation_permutations=count,
            timing_kind="full generation run wall time; subset uses a prefix",
            observed=observed,
            fit=fit_path.name,
            fit_sha256=fit_sha,
            embeddings="mitchell2008_embeddings",
            embeddings_sha256_of_E=e_sha,
            embedding_model=embeddings["embedding_model"],
            embedding_revision=embeddings["embedding_revision"],
            htfa_commit=commit,
            htfa_version=htfa.__version__,
            machine=f"{platform.system()} {platform.machine()} {os.cpu_count()} CPUs",
        )
        print(target, hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
