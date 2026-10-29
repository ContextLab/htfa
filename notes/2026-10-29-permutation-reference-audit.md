# Permutation factor-reference audit

## Result

Factorwise permutation inference needs one label-independent definition of each
factor's statistic across the permutation orbit. Aligning every refit to the
observed grouped template does not supply that definition in general. The global
maximum over factors is invariant to reordering, so the problem concerns the
uncorrected factorwise nulls, not the maximum-statistic null.

The implementation uses the first subject's unpenalized stage-0 factor centers as
a fixed reference. For mixtures it uses their mixing-weighted centers. The
unpenalized fits are computed before group-dependent fitting. Caller-supplied
starts must also be independent of the tested labels; this assumption cannot be
verified from the arrays themselves.

## Counterexample

Take two-factor templates consisting of opposite unit vectors in 2D, oriented
at 0, 120 and 240 degrees in three possible label states. Hungarian matching
between every pair swaps the two factors. Thus matching A to B and B to C gives
an identity permutation, whereas matching A directly to C gives a swap.
Pairwise matching is not transitive.

Give each state's factors the statistics [1, 0] in their original order. If each
state serves as the observed reference, its first statistic is 1 and both other
states' first matched statistics are 0. The exact-orbit p-value is consequently
1/3 in every state. A fixed reference instead yields the same statistic matrix
for every choice of observed state, and its exact ranks are superuniform. This
is a counterexample to the matching rule's general validity, not a measured
false-positive rate for fitted HTFA datasets.

The regression `test_observed_reference_matching_is_not_transitive` reproduces
this counterexample with the actual Hungarian matching implementation. A small
end-to-end HTFA test checks that a null sample under one observed labeling equals
the observed statistic under that labeling when it becomes the observed model,
after restoring canonical reference order.

## Display order and interpretation

After aligning to the fixed reference, one common permutation maps every
observed/null statistic array to the supplied model's global-template order.
This changes only how the family of tests is displayed. `reference_centers` and
`reference_indices` retain the identity of the fixed reference hypotheses.
Uncorrected inference concerns prespecified reference factors. Selecting an
interesting factor from observed effects is a selection problem and requires
multiplicity control; reordering the output does not grant selective validity.

Saved `GroupTest` arrays without reference metadata remain readable, with empty
reference arrays. This compatibility does not certify how those nulls were
computed. Tutorial 13's cached null must be recomputed with the fixed reference;
its saved fit can be retained without refitting its stage-0 estimates.

## Regeneration

After committing the source and script:

```bash
PYTHONPATH=. .venv/bin/python scripts/regenerate_group_null.py \
  /private/tmp/htfa-tutorial13-assets/htfa_development_fmri_10children_10adults_K30.npz \
  /private/tmp/htfa-group-reference-null/htfa_development_fmri_children_adults_null_100.npz \
  --permutations 100 --n-jobs 6 --n-threads 2
```

The script preserves the fit and records its exact SHA-256, participant/mask
checks, full refit parameters, numerical-source commit, reference metadata and
elapsed time. It does not upload anything. The previous 100-permutation archive
records 6,758 seconds on a 12-CPU Apple Silicon host, so allow roughly two hours;
this historical duration is a scheduling estimate, not a performance benchmark.

## Statistical background

Permutation validity assumes exchangeability under the null and a consistent
statistic across the rearrangements. See [Winkler et al., *Permutation inference
for the general linear model*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/).
The nontransitive-template counterexample above is a code-specific derivation.
