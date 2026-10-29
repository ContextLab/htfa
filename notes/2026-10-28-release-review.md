# Release candidate review — publication blocked

User instruction: continue release preparation, **do NOT publish to PyPI until
Jeremy has carefully reviewed and explicitly approved it**. TestPyPI and GitHub
release publication also remain approval-gated under the release design.
The candidate workflow has read-only permissions and no publishing job.

Base candidate: 08b3332. PCA exploration was removed and remains local-only.

Review findings fixed in this pass:
- Tutorial 09 used os.getloadavg(), which does not exist on Windows. A portable
  helper now prints NaN for unavailable load averages.
- htfa.stats was added to __all__ after earlier test collection, but the module
  itself was missing from the API index. Added it; API/statistics tests rerun.

Added scripts/regenerate_tutorial_fits.py for provenance-recorded local fits,
scripts/check_installed.py to test a wheel outside the source checkout, and a
manual/PR candidate workflow across 3 OSes x Python 3.10--3.14. No publication.
The installed-wheel check excludes network/BrainIAK markers per project config;
those require separate checks, as do executed tutorials and sdist installation.

Active validation:
- Tutorial 09--11 regeneration: /private/tmp/htfa-regenerate-review.log,
  output /private/tmp/htfa-release-review-assets, source library 08b3332.
- Fresh wheel/sdist: /private/tmp/htfa-review-dist; twine check passed.
- Installed-wheel suite: /private/tmp/htfa-review-installed-tests.log,
  environment /private/tmp/htfa-review-venv.
- BrainIAK: /private/tmp/htfa-review-brainiak.log.

Outstanding gates: full independent review; finalize regenerated fit hashes and
matching tutorial 09 nulls (never reuse against changed factors/weights), verify
13 assets via fresh-cache downloads, execute all tutorials against installed
package, final cross-platform CI, sdist install, final release review packet.

Further review findings:
- Tutorial 09 now checks saved-null fit and embedding hashes before using it.
- Statistics reject complex arrays instead of silently dropping imaginary parts;
  unrepresentable squared-error scales fail with rescaling guidance. Computing
  log(RSS)-log(n) avoids underflow in the variance division. Added regression
  cases; 17 API/statistics tests pass (/private/tmp/htfa-review-stats-tests.log).
- Draft PR https://github.com/ContextLab/htfa/pull/184 starts existing CI and the
  installed-wheel matrix. No publishing step exists in the new workflow.
- scripts/check_tutorials.py verifies a site-packages import in each kernel and
  retains executed notebooks outside the checkout. Initial installed-wheel
  tutorial run: /private/tmp/htfa-installed-tutorials.log.

Latest checks:
- Three refits completed from 08b3332 (tutorials 09/10/11: 531/122/43 seconds).
- Old vs regenerated tutorial 09 fit arrays differ; recomputing all 100 null
  permutations, with checkpoints in /private/tmp/htfa-release-review-nulls.
  Helper control: old fit + seed 1000 reproduces the saved old null exactly.
- Five immutable data assets uploaded to existing tutorial-data-v1; no prior
  asset overwritten. Manifest: /private/tmp/htfa-release-upload/manifest.json.
  All five downloaded into an empty cache via fetch_tutorial_fit and verified.
- Installed-wheel tutorials 01–06, 12–15 passed. Initial installed full suite:
  999 passed, 17 skipped, 28 deselected. Hardened-wheel full rerun active.
- Fresh sdist install package/stats smoke: 15 passed.
- macOS/Python 3.14 CI exposed a brittle bit-exact renderer assertion: 12 of
  129600 channels differ, maximum 16/255, across separate OpenGL contexts.
  Added exact geometry/topology/weight equality checks and bounded pixel
  differences (less than 0.1% channels; at most 32/255); rerunning real renders.

Completion evidence for this stage:
- Hardened installed-wheel suite: 1003 passed, 17 skipped, 28 deselected,
  /private/tmp/htfa-review-final-tests.log (477.70 s).
- All 9 registered tutorial download/cache checks pass after filename/hash updates:
  /private/tmp/htfa-review-asset-tests.log.
- Refreshed installed-wheel tutorials 10–11 pass; executed outputs copied back,
  all 31 embedded PNGs inspected plus BrainNet stills and GIF metadata. Logs:
  /private/tmp/htfa-installed-tutorials-10-11.log.
- Rendering regression tests: 14 passed (/private/tmp/htfa-component-review-tests.log).
- Existing macOS 3.13/3.14 CI failures both isolate to pixel-exact screenshot
  comparison; geometry plus bounded rendering comparison is awaiting final CI.
- Read-only checks: no remote v0.1.0 tag; PyPI/TestPyPI project endpoints return
  404. Inactive publication example in release-preparation/; no index configuration/upload.

Final decoding assets:
- 100 permutations complete, seeds 1000–1099, all finite. 20-permutation
  asset is the exact prefix; both record full-run generation timing explicitly.
- New assets added to tutorial-data-v1 without replacing any prior files; fit
  and both null registry entries updated together. Manifest below.
- Fresh-cache registered asset tests: 9 passed, 10 deselected,
  /private/tmp/htfa-final-assets-tests.log.
- Strict Sphinx build passed: /private/tmp/htfa-release-docs.log.
- Final candidate wheel/sdist build and twine check pass:
  /private/tmp/htfa-final-candidate-dist.
- Refreshed notebook 09 passed against installed final candidate wheel; six
  embedded PNGs visually inspected, executed output copied back. Log:
  /private/tmp/htfa-installed-tutorial09.log. This completes local execution
  of all 13 notebooks against an installed wheel.
- Final lint, format and mypy checks pass. Final-head platform matrix pending.

Standalone-documentation audit (PR #185, in progress)
--------------------------------------------------
User requests present-tense public prose, controlled benchmarks, current
HyperTools API, PyPI installation preview, fresh critical reviewers, all tests
and tutorials. No publication; Zenodo DOI/badge is the final post-review step.

- Main contains PR184 at 3f40f31. Audit branch codex/standalone-docs-audit
  uses /private/tmp/htfa-integration. Latest numerical-source commit 03bef8c.
- Fresh reviewers identified/fixed nonfinite permutation p-values, mismatched
  image/mask affines and stacked coordinates, complex input truncation, and
  label-dependent per-factor permutation matching. Independent follow-up
  found/fixed normalization of custom mixture-start reference weights.
- Public docs/tutorials rewritten; HyperTools1.1.0 Figure API verified; all13
  notebooks executed. 01/09/12 rerun for figure-label/crop fixes and inspected.
- Initial audited installed suite1037passed/17skipped/28deselected. Subsequent
  suite1048passed/1failed: only obsolete rawNilearn-colorbar image expectation.
  Strict geometry comparison now hides colorbars; results+tick tests36passed.
- Isolated benchmark workflow40800345659 succeeded: all24 matched fits pass
  CPU-interference checks; rawartifact /private/tmp/htfa-isolated-benchmark,
  retained under docs/_static/benchmarks. No local competing jobs influence it.
- Recomputed13null running checkpointed4processes x3subjectworkers, exact public
  label sequence, /private/tmp/htfa-audit-group-null-parallel.log; output
  /private/tmp/htfa-audit-group-null-100.npz. Serial exploratory run interrupted
  in favor of exact-equivalent parallel helper. Existing fit hash01a094f61846
  remains unchanged; newnull must replace registry with immutable name/hash.
- Local network and BrainIAK marker suites active in htfa-audit-network.log and
  htfa-audit-brainiak.log under /private/tmp. Network first test has failed;
  inspect final traceback before diagnosing. All tests must be accounted for.
- Strict Sphinx build passes;10790localHTML filelinks/113image references
  resolve. Browser UI unavailable; PNGs/GIF sampledframes inspected directly.
- Latest remote audit head0fb3092 has CI running (candidate40801887178; data
  warmup succeeded). It predates the testexpectation/mixture-anchor fix and
  newgroupnull; final-head CI still required.

Audit follow-up evidence:
- Installed wheel at03bef8c:1053passed17skipped28deselected (692.79s),
  /private/tmp/htfa-audit-validated-tests.log. BrainIAK27passed (913.89s).
- Network suite27passed1failed2skipped: failedNITRC mesh archive timeout;
  exacttest test_fetch_mesh_repairs_corrupt_cache rerun1passed22.28s.
  /private/tmp/htfa-audit-network.log and htfa-audit-network-retry.log.
- Fresh current sdistinstall+15smoke/statistics tests pass;
  /private/tmp/htfa-audit-sdist-tests.log. Lint/format/mypy/twine pass.
- Finalreader inspected all9galleryfigures; corrected reconstructiongalleryR2
  to publicstats temporal-meanbaseline. Tutorial12 explicitly explains its
  different grand-meanbaseline. Strictgallerybuild passes;10680links/113images
  resolve. /private/tmp/htfa-audit-gallery-final.log.
- Candidate40804508416 Ubuntu3.11 fails Tutorial14 two-component HTFA:
  near-vanished component yields excessiveJacobianinverse scale, overflow
  in bounded_lm scale**2/H_s. Log htfa-audit-candidate-ubuntu311-clean.log.
  APIreviewer diagnosing isolated /private/tmp/htfa-damping-audit; do not
  mutate active nullrun numericalsource until regeneration ends.
- Source-test matrix40804508450 at03bef8c completes successfully across all15
  OS/Python combinations plus minimumversions; BrainIAK40804508432 and
  docs40804508410 also succeed. Candidate/tutorial matrix is still active;
  Ubuntu3.10/3.13/3.14 pass, Ubuntu3.11 reveals the mixtureoverflow above.

Final audit candidate evidence:
- Numerical source 6e534cc2dfd6b43aa128f6ce3ad747d2b67d7689 includes reviewed
  mixture scaling and stable norm overflow handling. Fourteen new regressions,
  46 existing solver checks, and 341 identical RBF fingerprint arrays support
  the numerical changes. Public source docstrings are rewritten without code
  changes; executable ASTs match this commit except the new null registry entry.
- Installed candidate: 1067 passed, 17 skipped, 28 deselected (400.37 s),
  /private/tmp/htfa-audit-review-tests.log. All 13 notebooks pass from the same
  installed candidate, /private/tmp/htfa-audit-review-tutorials.log. Outputs
  copied after executable-cell equality checks; reviewed markdown preserved.
- Source-distribution install and smoke/statistics checks: 15 passed,
  /private/tmp/htfa-audit-review-sdist.log. Strict Sphinx build, Ruff, mypy,
  formatting, and package metadata checks pass.
- BrainIAK run40811513557: 27 passed, 1110 deselected (702.19 s).
- New group null: 100 exact-sequence permutations, finite statistics, common
  fixed reference and matching fit/checkpoint hashes. SHA256
  5bf2ae0b61f729f10be2820ba84cdd5f00448f5b51910024b66d7cca821b30eb.
  Immutable asset added to tutorial-data-v1; fresh registered download passes.
  Uncorrected counts at .05: center1,width2,connectivity2; FWE counts all0.
- Final isolated benchmark40811532627 succeeds on6e534cc; independent review
  verifies all24rows,1070monitoring samples, quiet baselines, no competing
  process streaks, source/data hashes, paired protocol and public summaries.
  /private/tmp/htfa-audit-final-benchmark/benchmark.json is the original artifact.
- Fresh contextual reviewers audit all13notebooks, README, allpublicguides,
  gallery examples and API docstrings.28markdown passages introduce examples
  and scope assumptions. Independent follow-up resolves age-ordered inference
  example and unsupported noise claim. Tutorial13 group heatmaps share a color
  scale; ISFC is explicitly distinct from the tested static-connectivity metric.
- Final current outputs13–15 inspected; earlier complete PNG/GIF/gallery review
  remains applicable to unchanged figures. PyPI and Colab badges are present in
  all13notebooks. No package publication or DOI operation is performed.
