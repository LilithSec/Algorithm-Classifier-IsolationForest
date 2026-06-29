#!/usr/bin/perl
# benchmarking/bench-sklearn-scoring.pl
#
# Compares scoring throughput between this module and scikit-learn's
# IsolationForest across a range of query set sizes.
#
# The same training CSV and query CSVs are used by both sides so the
# comparison is on identical data.  Models are pre-trained before any
# timing starts.
#
# Method correspondence:
#   Perl score_samples         <-->  clf.score_samples(X)      (same formula, opposite sign)
#   Perl predict               <-->  clf.predict(X)            (same semantics, 0/1 vs -1/+1)
#   Perl score_predict_samples <-->  (no sklearn equivalent)
#   Perl path_lengths          <-->  (no sklearn equivalent)
#   (no Perl equivalent)       <-->  clf.decision_function(X)  (threshold-shifted score)
#
# The table shows "ratio = sklearn ops/s / Perl ops/s" for methods that
# have direct equivalents.  >1 means sklearn is faster; <1 means Perl.
#
# scikit-learn is optional: if not installed, only Perl results are shown.
#
# Run with:
#   perl -Ilib benchmarking/bench-sklearn-scoring.pl

use strict;
use warnings;
use lib '../lib';
use Time::HiRes qw(time);
use File::Temp  qw(tempfile);
use JSON::PP    ();
use Algorithm::Classifier::IsolationForest;

use constant PI => 3.14159265358979;

# -----------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------
sub gaussian {
    my ( $mu, $sigma ) = @_;
    return $mu + $sigma
        * sqrt( -2 * log( rand() || 1e-12 ) )
        * cos( 2 * PI * rand() );
}

sub make_data {
    my ( $n, $nf ) = @_;
    my @rows = map { [ map { gaussian( 0, 1 ) } 1 .. $nf ] } 1 .. $n;
    for ( 1 .. int( $n * 0.05 ) ) {
        my $r = 5 + rand() * 3;
        push @rows, [ map { $r * ( rand() > 0.5 ? 1 : -1 ) } 1 .. $nf ];
    }
    return \@rows;
}

# -----------------------------------------------------------------------
# Timing helper: warm up for 0.3 s then measure for $secs wall-clock seconds.
# Returns ops/second.
# -----------------------------------------------------------------------
sub bench {
    my ( $code, $secs ) = @_;
    my $t0 = time();
    $code->() while time() - $t0 < 0.3;
    $t0 = time();
    my $n = 0;
    $code->(), $n++ while time() - $t0 < $secs;
    return $n / ( time() - $t0 );
}

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
my $N_TRAIN    = 1000;
my $N_FEATURES = 2;
my $N_TREES    = 100;
my $PSI        = 256;
my $BENCH_SECS = 2;

my @query_sizes = ( 100, 500, 1_000, 5_000, 10_000 );

# -----------------------------------------------------------------------
# Generate data (once, outside all timing)
# -----------------------------------------------------------------------
srand(42);
my $train_data = make_data( $N_TRAIN, $N_FEATURES );

my %query_data;
$query_data{$_} = make_data( $_, $N_FEATURES ) for @query_sizes;

# -----------------------------------------------------------------------
# Train Perl model
# -----------------------------------------------------------------------
my $model = Algorithm::Classifier::IsolationForest->new(
    n_trees     => $N_TREES,
    sample_size => $PSI,
    seed        => 1,
)->fit($train_data);

# -----------------------------------------------------------------------
# Write CSVs for Python (training + one per query size)
# -----------------------------------------------------------------------
my ( $train_fh, $train_csv ) = tempfile( SUFFIX => '.csv', UNLINK => 1 );
print $train_fh join( ',', @$_ ) . "\n" for @$train_data;
close $train_fh;

my %query_csvs;
for my $sz (@query_sizes) {
    my ( $fh, $path ) = tempfile( SUFFIX => '.csv', UNLINK => 1 );
    print $fh join( ',', @$_ ) . "\n" for @{ $query_data{$sz} };
    close $fh;
    $query_csvs{$sz} = $path;
}

# -----------------------------------------------------------------------
# Locate Python + scikit-learn
# -----------------------------------------------------------------------
my $python_bin;
for my $cmd (qw(python3 python)) {
    my $probe = `$cmd -c "import sklearn; print('ok')" 2>/dev/null`;
    if ( defined $probe && $probe =~ /\bok\b/ ) {
        $python_bin = $cmd;
        last;
    }
}

# -----------------------------------------------------------------------
# Python benchmarking script (embedded, written to a temp file).
#
# Receives:  train.csv  bench_secs  path1:size1  path2:size2 ...
# Outputs:   JSON { "100": { "score_samples": N, "predict": N, ... }, ... }
# -----------------------------------------------------------------------
my $py_script = <<'END_PY';
import sys, json
import time as pytime
import numpy as np
from sklearn.ensemble import IsolationForest

def bench(fn, seconds):
    t0 = pytime.perf_counter()
    while pytime.perf_counter() - t0 < 0.3:
        fn()
    t0 = pytime.perf_counter()
    n = 0
    while pytime.perf_counter() - t0 < seconds:
        fn()
        n += 1
    return n / (pytime.perf_counter() - t0)

train_csv  = sys.argv[1]
bench_secs = float(sys.argv[2])
specs      = sys.argv[3:]          # "path:size" pairs

with open(train_csv) as f:
    X_train = np.array([[float(v) for v in ln.strip().split(',')]
                        for ln in f if ln.strip()])

psi = min(256, len(X_train))
clf = IsolationForest(n_estimators=100, max_samples=psi,
                      contamination='auto', random_state=1)
clf.fit(X_train)

results = {}
for spec in specs:
    path, size = spec.rsplit(':', 1)
    size = int(size)
    with open(path) as f:
        X_q = np.array([[float(v) for v in ln.strip().split(',')]
                        for ln in f if ln.strip()])
    results[size] = {
        'score_samples':     bench(lambda X=X_q: clf.score_samples(X),     bench_secs),
        'predict':           bench(lambda X=X_q: clf.predict(X),           bench_secs),
        'decision_function': bench(lambda X=X_q: clf.decision_function(X), bench_secs),
    }

print(json.dumps(results))
END_PY

# -----------------------------------------------------------------------
# Run Python (one subprocess, all sizes, to avoid repeated import cost)
# -----------------------------------------------------------------------
my $sk;
if ( defined $python_bin ) {
    my ( $py_fh, $py_path ) = tempfile( SUFFIX => '.py', UNLINK => 1 );
    print $py_fh $py_script;
    close $py_fh;

    my $specs = join( ' ', map { qq("$query_csvs{$_}:$_") } @query_sizes );
    my $raw   = `$python_bin "$py_path" "$train_csv" $BENCH_SECS $specs 2>/dev/null`;
    $sk       = eval { JSON::PP->new->decode($raw) };
}

# -----------------------------------------------------------------------
# Run Perl benchmarks (all methods, all sizes)
# -----------------------------------------------------------------------
my %pl;
for my $sz (@query_sizes) {
    my $q = $query_data{$sz};
    $pl{$sz} = {
        score_samples         => bench( sub { $model->score_samples($q)         }, $BENCH_SECS ),
        predict               => bench( sub { $model->predict($q)               }, $BENCH_SECS ),
        score_predict_samples => bench( sub { $model->score_predict_samples($q) }, $BENCH_SECS ),
        path_lengths          => bench( sub { $model->path_lengths($q)          }, $BENCH_SECS ),
    };
}

# -----------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------
print "=" x 67, "\n";
print " Perl vs scikit-learn -- scoring speed (ops/second, higher = faster)\n";
print "=" x 67, "\n";
printf " Training: %d samples, %d features, n_trees=%d, sample_size=%d\n",
    $N_TRAIN, $N_FEATURES, $N_TREES, $PSI;
printf " Each measurement: %.0fs wall-clock with 0.3s warmup\n", $BENCH_SECS;
print " ratio = sklearn ops/s / Perl ops/s  (>1 = sklearn faster)\n";
print " --  = no equivalent method on that side\n\n";

unless ( defined $sk ) {
    print " (scikit-learn not available; showing Perl results only)\n\n";
}

# Row definitions: [ label, perl_key, sklearn_key ]
my @rows = (
    [ 'score_samples',         'score_samples',         'score_samples'     ],
    [ 'predict',               'predict',               'predict'           ],
    [ 'score_predict_samples', 'score_predict_samples', undef               ],
    [ 'path_lengths',          'path_lengths',          undef               ],
    [ 'decision_function',     undef,                   'decision_function' ],
);

for my $sz (@query_sizes) {
    printf "--- %d query points ---\n", $sz;

    if ( defined $sk ) {
        printf "  %-28s  %12s  %14s  %8s\n",
            'method', 'Perl (ops/s)', 'sklearn (ops/s)', 'ratio';
        printf "  %-28s  %12s  %14s  %8s\n",
            '-' x 28, '-' x 12, '-' x 14, '-' x 8;
    }
    else {
        printf "  %-28s  %12s\n", 'method', 'Perl (ops/s)';
        printf "  %-28s  %12s\n", '-' x 28, '-' x 12;
    }

    for my $row (@rows) {
        my ( $label, $pl_key, $sk_key ) = @$row;
        my $pl_rate = $pl_key ? $pl{$sz}{$pl_key} : undef;
        my $sk_rate = ( $sk_key && $sk ) ? $sk->{$sz}{$sk_key} : undef;

        if ( defined $sk ) {
            my $ratio = ( $pl_rate && $sk_rate )
                ? sprintf( '%.2f', $sk_rate / $pl_rate )
                : '--';
            printf "  %-28s  %12s  %14s  %8s\n",
                $label,
                $pl_rate ? sprintf( '%.1f', $pl_rate ) : '--',
                $sk_rate ? sprintf( '%.1f', $sk_rate ) : '--',
                $ratio;
        }
        else {
            printf "  %-28s  %12s\n",
                $label,
                $pl_rate ? sprintf( '%.1f', $pl_rate ) : '--';
        }
    }
    print "\n";
}
