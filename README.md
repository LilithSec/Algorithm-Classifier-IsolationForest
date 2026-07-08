# Algorithm::Classifier::IsolationForest

Isolation Forest (Liu, Fei Tony & Ting, Kai & Zhou, Zhi-Hua, 2008) detects anomalies by
random partitioning rather than by modelling normal points. Each tree repeatedly splits
the data. Points that get isolated after only a few splits are likely anomalies. The score
is the average isolation depth across many trees, normalised so values approach 1 for
anomalies and stay below 0.5 for normal points.

In extended mode the module implements the Extended Isolation Forest variant. Each split
is a random hyperplane instead of an axis-aligned cut, which removes the rectangular,
axis-aligned bias in the score field and tends to help on elongated or multi-modal data.

With `voting => 'majority'` the module implements the Majority Voting Isolation
Forest (MVIForest, Chabchoub, Togbe, Boly & Chiky 2022): each tree votes a
sample anomalous or normal against the decision threshold on its own score,
`predict` labels the sample by the majority of the votes and stops walking
trees as soon as the outcome is decided, and `score_samples` returns the
anomaly vote fraction. Trees are built identically either way, so majority
voting composes with both axis and extended mode, and an existing model can be
flipped between the two modes without refitting — with the `set_voting` method
in Perl or the `iforest set_voting` command on a saved model. A
contamination-learned threshold does not carry across modes (it is a quantile
of a different per-point quantity in each), so switching relearns it for the
target mode and therefore needs the original training data.

```perl
use Algorithm::Classifier::IsolationForest;

my @data = ([0.1, -0.2], [0.0, 0.1], [5.0, 6.0], ...);

# Classic, axis-parallel Isolation Forest
my $iforest = Algorithm::Classifier::IsolationForest->new(
    n_trees     => 100,
    sample_size => 256,
    seed        => 42,
);
$iforest->fit(\@data);

my $scores = $iforest->score_samples(\@data);  # arrayref, each in (0,1]
my $flags  = $iforest->predict(\@data, 0.6);    # arrayref of 0/1

# Save and reload
$iforest->save('model.json');
my $reloaded = Algorithm::Classifier::IsolationForest->load('model.json');

# Extended Isolation Forest (oblique hyperplane splits)
my $eif = IsolationForest->new(mode => 'extended', seed => 42);
$eif->fit(\@data);

# Majority Voting Isolation Forest (per-tree votes, majority label)
my $mv = IsolationForest->new(voting => 'majority', seed => 42);
$mv->fit(\@data);
my $labels = $mv->predict(\@data, 0.6);   # threshold is the per-tree cutoff here

# Switch an existing model's aggregation without refitting. No data needed
# unless it was fit with contamination, in which case pass the training set
# so the decision threshold is recalibrated for the target mode.
$iforest->set_voting('majority', \@data);  # ->set_voting('mean') if no contamination
```

# Online (streaming) Isolation Forest

For data that arrives as a stream and may drift over time, the companion
class `Algorithm::Classifier::IsolationForest::Online` implements Online
Isolation Forest (Leveni, Weigert Cassales, Pfahringer, Bifet & Boracchi
2024). There is no `fit`: the model `learn`s points as they arrive and,
once more than `window_size` points have been seen, forgets the oldest
point for every new one, so the model always reflects the most recent
part of the stream. Trees never store data points — nodes keep only
counts and bounding boxes; leaves split by simulating points inside
their box, and forgetting collapses under-populated subtrees back into
leaves. Pure Perl (the Inline::C accelerator assumes immutable trees and
does not apply).

```perl
use Algorithm::Classifier::IsolationForest::Online;

my $oif = Algorithm::Classifier::IsolationForest::Online->new(
    n_trees          => 100,
    window_size      => 2048,   # points the model reflects; 0 = never forget
    max_leaf_samples => 32,     # points a leaf accumulates before splitting
    contamination    => 0.05,   # optional: learn the label cutoff from the window
    seed             => 42,
);

$oif->learn(\@warmup_rows);                 # warm-up / plain learning
my $scores = $oif->score_learn(\@rows);     # prequential: score, then learn, per row
my $flags  = $oif->predict(\@query_rows);   # score without learning

# After the stream drifts, refresh the contamination cutoff:
$oif->relearn_threshold;

# Persistence keeps the sliding window, so a reloaded model resumes the
# stream where it left off. load() on the parent class dispatches on the
# stored format tag, so either model type loads through either class.
$oif->save('oiforest_model.json');
my $resumed = Algorithm::Classifier::IsolationForest->load('oiforest_model.json');
```

On the command line the `iforest stream` subcommand runs the same loop
over a CSV: it creates or resumes the model at `-m`, scores + learns each
row (prequentially), prints `score,label` lines, and saves the updated
state back — so repeated invocations continue the stream.

```shell
iforest stream -i batch1.csv -m om.json -n 100 --window 2048 --eta 32 -c 0.05
iforest stream -i batch2.csv -m om.json               # resumes om.json
iforest stream -i suspect.csv -m om.json --score-only # score without learning
iforest info -m om.json                               # online-aware model info
```

# Performance options

A handful of constructor / method-level knobs unlock measurable speedups
for specific workloads.  All of them are no-ops when the optional
Inline::C backend is absent.

## `parallel_fit => N` — fork-based parallel training

Builds the `n_trees` across `N` forked workers (Unix-like platforms; no-op
elsewhere).  Each worker gets a derived RNG seed, so parallel fits are
reproducible across runs at fixed worker count — though the trees
*differ* from a serial fit with the same seed, because the RNG draws
happen in a different order.  Inference results are unaffected.

```perl
my $f = Algorithm::Classifier::IsolationForest->new(
    n_trees      => 200,
    sample_size  => 256,
    seed         => 42,
    parallel_fit => 4,       # 4 forked workers
)->fit(\@training_data);
```

## `pack_data` — score the same dataset many times faster

`pack_data` returns an opaque wrapper that the scoring methods accept
directly, skipping the per-call walk over the arrayref-of-arrayrefs.
Use it when the same dataset is scored repeatedly (interactive threshold
tuning, dashboards, plotting that updates as parameters change).

```perl
my $packed = $f->pack_data(\@data);
my $scores = $f->score_samples($packed);
my $flags  = $f->predict($packed, 0.6);
my ($s, $l) = $f->score_predict_split($packed);  # two flat arrayrefs
```

## `score_predict_split` — get scores + labels without the AV-of-AVs

When you want both anomaly scores and 0/1 labels but don't need them
paired together row-by-row, `score_predict_split` returns the two as
flat arrayrefs and skips the ~`2 * n_pts` SV allocations that the
classic `score_predict_samples` shape requires.

```perl
my ($scores, $labels) = $f->score_predict_split(\@data, 0.6);
```

# Native acceleration (Inline::C, OpenMP, SIMD)

The scoring hot path (`score_samples`, `predict`, `path_lengths`,
`score_predict_samples`, `score_predict_split`) is automatically
accelerated through [`Inline::C`](https://metacpan.org/pod/Inline::C)
when it is installed and a working C compiler is present.  On top of
that:

* if the toolchain accepts `-fopenmp` and can link against `libgomp`,
  the per-point tree walk runs in parallel across all available CPU
  cores using OpenMP;
* on OpenMP 4.0+ compilers the extended-mode oblique dot product is
  vectorised via `#pragma omp simd` — substantially faster for
  high-feature-count extended models.

When `Inline::C` is available while the distribution is being built,
the C backend is compiled once during `make` and installed with the
module — at run time it loads like any XS module, with no compiler,
Inline, or `_Inline/` cache directory needed.  Otherwise detection
happens once at module load and the build is cached under `_Inline/`.
None of these dependencies are required: without them the module falls
back to a pure-Perl implementation that produces identical results,
just slower.

Check which backend is active on your machine:

```shell
iforest accel
```

Sample output on a host with everything wired up:

```
Algorithm::Classifier::IsolationForest acceleration status
  Inline::C : available
  OpenMP    : available
  SIMD      : available
  C object  : prebuilt at install time
  Build flags: -O3 -march=x86-64-v3

Active backend: Inline::C with OpenMP + SIMD -- prebuilt at install time
```

User code that wants to introspect the active backend can read these
package variables:

```perl
$Algorithm::Classifier::IsolationForest::HAS_C       # 0/1
$Algorithm::Classifier::IsolationForest::HAS_OPENMP  # 0/1
$Algorithm::Classifier::IsolationForest::HAS_SIMD    # 0/1
$Algorithm::Classifier::IsolationForest::C_SOURCE    # 'prebuilt' / 'runtime' / ''
```

# Install

## Source

```shell
perl Makefile.PL
make
make test
make install
```

On x86-64 machines from roughly the last decade, configuring with

```shell
IF_ARCH=x86-64-v3 perl Makefile.PL
```

bakes `-march=x86-64-v3` (AVX2 + FMA, no AVX-512) into the installed C
backend, which can speed up extended-mode scoring — how much is
hardware-dependent, so benchmark before assuming.  Results stay
bit-identical to the pure-Perl backend either way.  See "Tuning the C
build" in the module documentation for the other `IF_*` knobs and why
`-march=native` is not always the better choice.

## FreeBSD

```shell
pkg install p5-App-Cmd p5-File-Slurp p5-App-cpanminus \
            p5-Inline p5-Inline-C gcc
cpanm Algorithm::Classifier::IsolationForest
```

`gcc` ships with `libgomp` and provides the OpenMP runtime; the system
clang does not by default.  `p5-Inline-C` is what makes the C backend
build (at install time, or at first module load from a plain checkout).

## Debian

```shell
apt-get install libapp-cmd-perl libfile-slurp-perl cpanminus \
                libinline-c-perl gcc
cpanm Algorithm::Classifier::IsolationForest
```

`libinline-c-perl` brings in `libinline-perl`.  `gcc` pulls in `libgomp1`
(the OpenMP runtime), which is what enables the parallel tree-walk.  Both
dependencies are optional — leave them out and the module installs and
runs in pure-Perl mode.
