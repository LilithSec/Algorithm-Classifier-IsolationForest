#!/usr/bin/perl
# benchmarking/bench-score.pl
#
# Benchmarks the four public scoring/prediction methods:
#   score_samples, predict, score_predict_samples, path_lengths
#
# Sections:
#   1. Scoring method comparison  -- which method has the lowest overhead
#   2. Query set size scaling     -- throughput vs number of points scored
#   3. n_trees scaling on scoring -- effect of model size on score time
#
# Models are pre-trained before any timing begins.
#
# Run with:
#   perl -Ilib benchmarking/bench-score.pl

use strict;
use warnings;
use lib '../lib';
use Benchmark qw(cmpthese);
use Algorithm::Classifier::IsolationForest;

use constant PI => 3.14159265358979;

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

print "=" x 62, "\n";
print " scoring benchmarks -- Algorithm::Classifier::IsolationForest\n";
print "=" x 62, "\n";
print "(rates shown as calls/second; higher is faster)\n";

# -----------------------------------------------------------------------
# Pre-train models with different n_trees on the same 1000-sample dataset
# -----------------------------------------------------------------------
srand(42);
my $train = make_data( 1000, 2 );
my %model;
for my $nt ( 10, 50, 100, 200, 500 ) {
    $model{$nt} = Algorithm::Classifier::IsolationForest->new(
        n_trees     => $nt,
        sample_size => 256,
        seed        => 1,
    )->fit($train);
}

# Pre-generate query sets of varying sizes
srand(99);
my %q;
$q{$_} = make_data( $_, 2 ) for ( 100, 500, 1_000, 5_000, 10_000 );
my $q1k = $q{1_000};

# -----------------------------------------------------------------------
# 1. Scoring methods compared  (n_trees=100, 1000 query points)
# -----------------------------------------------------------------------
print "\n--- scoring methods  (n_trees=100, 1000 query points) ---\n";
my $m = $model{100};
cmpthese(
    -2,
    {
        'score_samples'         => sub { $m->score_samples($q1k)         },
        'predict'               => sub { $m->predict($q1k)               },
        'score_predict_samples' => sub { $m->score_predict_samples($q1k) },
        'path_lengths'          => sub { $m->path_lengths($q1k)          },
    }
);

# -----------------------------------------------------------------------
# 2. Query set size  (n_trees=100, score_samples)
# -----------------------------------------------------------------------
print "\n--- query set size  (n_trees=100, score_samples) ---\n";
cmpthese(
    -2,
    {
        '100 pts'   => sub { $m->score_samples( $q{100}    ) },
        '500 pts'   => sub { $m->score_samples( $q{500}    ) },
        '1k pts'    => sub { $m->score_samples( $q{1_000}  ) },
        '5k pts'    => sub { $m->score_samples( $q{5_000}  ) },
        '10k pts'   => sub { $m->score_samples( $q{10_000} ) },
    }
);

# -----------------------------------------------------------------------
# 3. n_trees effect on scoring  (1000 query points, score_samples)
# -----------------------------------------------------------------------
print "\n--- n_trees effect on score_samples  (1000 query points) ---\n";
cmpthese(
    -2,
    {
        'n_trees=10'  => sub { $model{10} ->score_samples($q1k) },
        'n_trees=50'  => sub { $model{50} ->score_samples($q1k) },
        'n_trees=100' => sub { $model{100}->score_samples($q1k) },
        'n_trees=200' => sub { $model{200}->score_samples($q1k) },
        'n_trees=500' => sub { $model{500}->score_samples($q1k) },
    }
);
