# 2026-11-21: issue #188 (network plots and animations), overnight run

## State
- Request came from the timecorr session (peer `timecorr-35`); user confirmed: all 7 items, option 3 (kwargs plus `compose=`), branch + PR + 0.2.0 prep, docs and tutorials updated. No PyPI publish without explicit go.
- User went to sleep (2026-11-21 evening): proceed without input, use codex for review, surface consequential decisions in the morning.
- Branch: `epic/network-plots-188` (local, from main after `git pull`).
- Spec: `docs/superpowers/specs/2026-11-21-network-plots-188-design.md` (committed a8bd78b; builder check folded in).
- Plan: `docs/superpowers/plans/2026-11-21-network-plots-188.md` (10 tasks, sequential: all touch `htfa/brainnet.py`).

## Decisions made without the user (surface in the morning)
1. `(T, K)` sizes are scaled jointly over all frames, not per frame.
2. Blank-render check and restarts run only for `n_jobs != 1`; sequential path untouched (keeps 0.1.0 output identical).
3. Restart only on blank render or dead worker (max 5); other exceptions raise at once. Private worker pool, not joblib's shared one.
4. `connectivity=None` allowed in `brainnet.animate_network`; `HTFAResults.animate_network` unchanged (None = compute ISFC).
5. Layout reuses `annotate_pair`'s look (caption under titles, divider); single view gets no title; layout path triggered by `views`, `legend` or `frame_size`.
6. K colors / per-frame sizes or colors rejected in component-lobe mode. Alpha in colors dropped.
7. Flat 3- or 4-number `node_color` is always one color.
8. `compose` excludes `legend`, `captions`, `frame_size`. `plot_network` has no `compose`.
9. Legend row wider than the frame raises `ValueError`. Odd `frame_size` raises.
10. New style key `supersample` (default 2). No x264 `preset` option.
11. 0.2.0 section goes in existing `docs/changelog.rst`; no new CHANGELOG.md.
12. Dark background: bands take the background color, text turns white.

## Peer info
- timecorr session: black frames likely whole views, macOS, GPU memory pressure with parallel workers; unverified. Keep the tutorial's 5 percent rule. Send it a note when the PR is open (which items landed, version).

## Task list
- [x] Tasks 1-10 of the plan
- [x] codex review per task and whole branch (two-part final review, one fix wave, scoped re-review, two residual test fixes)
- [ ] push branch, open PR, wait for CI on every workflow, note to timecorr session
- [ ] morning summary for the user
- [ ] NOT done on purpose: PyPI publish (needs the user's explicit go)

## Status as of 2026-11-22 morning
- Branch head 667fa9f, 37 commits on top of main (de4232f). Version 0.2.0 in pyproject and changelog.
- Local checks at e409e0e: make lint clean; make test 1216 passed, 19 skipped, 0 warnings; make docs 0 warnings. 667fa9f changes two tests only (brainnet+headless: 171 passed, 5 skipped).
- Final codex review: 6 major library findings and 6 major test findings, all fixed in one wave (64ea99d..e409e0e) and confirmed by a scoped re-review; two residual test items fixed in 667fa9f.
- Ledger with every commit range, review result and ruling: `.superpowers/sdd/2026-11-21-network-plots-188/progress.md` (git-ignored; delete after merge).

## Found along the way
- macOS render-window leak (about 9 MB per render, "Context leak detected" line), present on main too: fixed with VTK's Cocoa autorelease pool, scoped to the call that builds the scenes; skipped on VTK < 9.5.
- First frame of an animation varies run to run by a few dozen pixels (main too). Cause is in VTK (depth peeling of translucent edges over the translucent mesh). Not fixed.
- n_jobs: 300 frames took 61/50/35/31 s with 1/2/3/4 workers; about 4 s start-up, so short movies gain nothing. Docstring says so.
- 7 ConvergenceWarnings from tests/test_stats.py (main too): the test now asserts the expected one and allows the optional TFA one.

## Open for the user
- Each view renders three times per frame in set_edges (pyvista remove_actor/add_mesh). Removing the extra renders would speed up every animation but changes translucent pixels of existing output.
- Encoding could overlap with drawing in the n_jobs path (needs a drawing thread).
- No suite test for the import-order fix (would need patching imports); the VTK < 9.5 branch is not exercised on this machine.
- Crop-reference fix changes render order in layout movies: tutorial 10's legend GIF differs by 1-27 px per frame from the first version on this branch.
- fade_by_mass shows 1-(1-a)^2 opacity on translucent spheres (0.1.0 behavior, unchanged, now asserted).
- brainiak cross-checks do not run locally (brainiak not installed); CI workflow covers them.

## More decisions made without the user (add to the morning list)
13. Static (K, K) connectivity accepted with time-varying sizes/colors (fixed edges).
14. With frame_size, views scale to fit both up and down.
15. Legend font floor 12 px; at natural width the frame widens to fit the legend; with frame_size a too-wide legend raises.
16. Captions/titles never widen the frame.
17. Blank-render rule: >5% near-black pixels AND a near-black corner pixel.
18. Workers inherit the parent's GL probe result.
19. Cleanup failure of the temp frame folder: OSError on success path; joins the exception chain when a render error is in flight.
20. Subagent commits carry the trailer of the model that wrote them (Opus/Sonnet on some); not rewritten.
21. Three verification findings outside #188's scope folded into this branch as separate commits (first-frame variation, ConvergenceWarnings, leak).
22. Odd frame_size: allowed for stills and GIF; ValueError only for mp4 (supersedes item 9's "odd frame_size raises").
23. Pool restarts only on a dead worker or a blank render; other broken-pool errors raise at once.
24. Crop of a movie comes from opaque nodes at their largest radius in one contrasting color.
25. content_box / annotate_pair / montage on white need neither pyvista nor matplotlib.
26. Memory-growth test is macOS-only and slow-marked.

## CI round 1 on PR #189 (f42aa68, 2026-11-22): docs and brainiak green; tests red on all three OSes
- Linux: not a branch regression. ubuntu-latest moved from 24.04 (Mesa 25.2.8) to 26.04 (Mesa 26.0.8), where an actor colored by cell data draws nothing (main fails the same way in an Ubuntu 26.04 container). Fix 93e465f: edge and node colors are per-point. macOS images pixel-identical before/after.
- Windows: test helper used os.kill(pid, 0) (invalid on Windows) and workers appended to one shared log file. Fix 1dfaff0 (tests only). Unverified until CI.
- macOS runners: one layout test differed (same cell-color lookup on the virtual GPU, expected fixed by 93e465f); on 3.10, numpy 2.2.6 emits a spurious matmul RuntimeWarning (inputs and result finite), now allowed in test_stats (2a208d8).
- Reproduce Linux CI locally: Docker, ubuntu:26.04 amd64, apt libegl1 libegl-mesa0, pip install -e ".[dev]". Do this before the first push of any rendering change.
- CI timings: tests about 10 min per Ubuntu job, 22-29 min per macOS/Windows job; docs 5 min; brainiak 13 min; candidate 17 min.

## CI round 2 (59a1e15): docs, brainiak green; test matrix 14 of 15 green (all Linux, all macOS, Windows 3.10-3.13)
- Windows 3.14: test_n_jobs_temp_folder_removed read an empty record file: a worker the pool terminated had created its record but not yet written it. Test records are now written under a .tmp name and renamed (tests only).
- Local at 59a1e15: make test 1216 passed, 19 skipped; make docs succeeded; lint clean.
