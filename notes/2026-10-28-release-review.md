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
