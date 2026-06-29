#!/usr/bin/perl
# benchmarking/bench-fit-parallel.pl
#
# Measures fit() throughput across a sweep of parallel_fit worker
# counts.  The training data is the same for every run; only the
# `parallel_fit` constructor option changes.
#
# Tree-building uses Perl rand() + recursion -- parallel_fit forks
# workers and divvies up n_trees across them, with each worker fitting
# its share in a child process and returning the trees to the parent
# via Storable on a pipe.  Speedup is bounded by:
#   * core count
#   * Storable freeze/thaw + pipe IPC overhead per worker
#   * fork() cost (small but non-zero)
#
# Run with:
#   perl -Ilib benchmarking/bench-fit-parallel.pl

use strict;
use warnings;
use lib '../lib';
use Time::HiRes qw(time);
use Config;
use Algorithm::Classifier::IsolationForest;

use constant PI => 3.14159265358979;

sub gaussian {
    my ( $mu, $sigma ) = @_;
    return $mu + $sigma
        * sqrt( -2 * log( rand() || 1e-12 ) )
        * cos( 2 * PI * rand() );
}

# -----------------------------------------------------------------------
# Bench helper: warm up briefly, then time exactly $reps fits and report
# the median.  fit() is much slower than scoring so we run it a handful
# of times rather than fitting a 2-second budget.
# -----------------------------------------------------------------------
sub time_fit_median {
    my ( $build_iforest, $train, $reps ) = @_;
    # One warm-up fit (covers any first-time Inline::C compile path etc.)
    $build_iforest->()->fit($train);

    my @times;
    for ( 1 .. $reps ) {
        my $f  = $build_iforest->();
        my $t0 = time();
        $f->fit($train);
        push @times, time() - $t0;
    }
    my @s = sort { $a <=> $b } @times;
    return $s[ int( @s / 2 ) ];
}

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
my $N_TRAIN     = 5_000;
my $N_FEATURES  = 8;
my $N_TREES     = 200;
my $SAMPLE_SIZE = 256;
my $REPS        = 5;
my @workers     = ( 1, 2, 4, 8 );

# -----------------------------------------------------------------------
# Generate training data once (outside any timing).
# -----------------------------------------------------------------------
srand(42);
my @train = map {
    [ map { gaussian( 0, 1 ) } 1 .. $N_FEATURES ]
} 1 .. $N_TRAIN;
push @train, [ map { 6 + rand() } 1 .. $N_FEATURES ] for 1 .. 50;

# -----------------------------------------------------------------------
# Detect environment
# -----------------------------------------------------------------------
my $can_fork = ( $Config{d_fork} || '' ) eq 'define';
my $has_c
    = $Algorithm::Classifier::IsolationForest::HAS_C    ? 'yes' : 'no';
my $has_openmp
    = $Algorithm::Classifier::IsolationForest::HAS_OPENMP ? 'yes' : 'no';

print "=" x 67, "\n";
print " fit() parallelism sweep (parallel_fit constructor option)\n";
print "=" x 67, "\n";
printf " Training: %d samples, %d features, %d trees, sample_size=%d\n",
    $N_TRAIN, $N_FEATURES, $N_TREES, $SAMPLE_SIZE;
printf " Median of %d fits per worker count\n", $REPS;
printf " Inline::C: %s   OpenMP: %s   fork(): %s\n\n",
    $has_c, $has_openmp, $can_fork ? 'yes' : 'no';

unless ($can_fork) {
    print " (fork() not available on this platform; parallel_fit falls back to serial)\n";
}

# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------
printf "  %-12s  %14s  %14s\n", 'workers', 'fit (s, median)', 'speedup vs 1';
printf "  %-12s  %14s  %14s\n", '-' x 12, '-' x 14, '-' x 14;

my $serial_time;
for my $w (@workers) {
    my $label = $w == 1 ? 'serial' : "fork=$w";
    my $fit_time = time_fit_median(
        sub {
            Algorithm::Classifier::IsolationForest->new(
                n_trees      => $N_TREES,
                sample_size  => $SAMPLE_SIZE,
                seed         => 1,
                parallel_fit => $w == 1 ? undef : $w,
            );
        },
        \@train,
        $REPS,
    );
    $serial_time = $fit_time if $w == 1;
    my $speedup
        = ( defined $serial_time && $serial_time > 0 )
        ? sprintf( '%.2fx', $serial_time / $fit_time )
        : '--';
    printf "  %-12s  %14.3f  %14s\n", $label, $fit_time, $speedup;
}

print "\n";
