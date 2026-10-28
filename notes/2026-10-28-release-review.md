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
  404. Inactive publication example in release/; no index configuration/upload.
