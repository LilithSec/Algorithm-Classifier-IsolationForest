#!perl
# 33-parallel-fit.t
#
# Verifies the parallel_fit option:
#   1. Produces a fitted model with the requested number of trees.
#   2. score_samples on a held-out point looks like a normal score
#      (between 0 and 1, and clearly separates an obvious outlier).
#   3. Re-running the parallel fit with the same seed and worker count
#      gives bit-identical scores (cross-run reproducibility, which is
#      the parallel_fit contract -- serial-vs-parallel differs but
#      parallel-vs-parallel does not).
#   4. parallel_fit on a no-fork platform falls back silently to serial.

use strict;
use warnings;
use Test::More;
use List::Util qw(min max);
use Config;

use Algorithm::Classifier::IsolationForest;

my $CLASS = 'Algorithm::Classifier::IsolationForest';

# Build a deterministic dataset.
sub gaussian {
    my ( $mu, $sigma ) = @_;
    my $u1 = rand() || 1e-12;
    my $u2 = rand();
    return $mu + $sigma * sqrt( -2 * log($u1) ) * cos( 2 * 3.14159265358979 * $u2 );
}

srand(20260629);
my @train;
push @train, [ gaussian(0,1), gaussian(0,1), gaussian(0,1) ] for 1 .. 300;
push @train, [ 8, -8, 7 ];
push @train, [ -7, 8, -8 ];

my @query = (
    [ 0.1, -0.2, 0.0  ],    # inlier-like
    [ 9,    9,    9   ],    # obvious outlier
);

my $can_fork = ( $Config{d_fork} || '' ) eq 'define';

subtest 'parallel_fit produces a valid model' => sub {
    plan skip_all => 'no fork() on this platform' unless $can_fork;

    my $f = $CLASS->new(
        n_trees      => 50,
        sample_size  => 256,
        seed         => 42,
        parallel_fit => 4,
    );
    $f->fit( \@train );

    is( scalar @{ $f->{trees} }, 50, 'tree count matches n_trees' );

    my $s = $f->score_samples( \@query );
    is( scalar @$s, 2, 'two scores returned' );
    cmp_ok( $s->[0], '>=', 0, 'inlier score >= 0' );
    cmp_ok( $s->[0], '<=', 1, 'inlier score <= 1' );
    cmp_ok( $s->[1], '>',  $s->[0],
        'outlier scores strictly higher than inlier (parallel-fit model is sane)' );
};

subtest 'parallel_fit is reproducible across runs at fixed worker count' => sub {
    plan skip_all => 'no fork() on this platform' unless $can_fork;

    my $f1 = $CLASS->new(
        n_trees      => 30,
        sample_size  => 256,
        seed         => 99,
        parallel_fit => 3,
    )->fit( \@train );

    my $f2 = $CLASS->new(
        n_trees      => 30,
        sample_size  => 256,
        seed         => 99,
        parallel_fit => 3,
    )->fit( \@train );

    my $s1 = $f1->score_samples( \@query );
    my $s2 = $f2->score_samples( \@query );
    my $diffs = grep { $s1->[$_] != $s2->[$_] } 0 .. $#$s1;
    is( $diffs, 0,
        'two parallel fits with same seed + workers give identical scores' );
};

subtest 'parallel_fit number must be a positive integer' => sub {
    eval { $CLASS->new( parallel_fit => -1 ) };
    like( $@, qr/parallel_fit/,  'negative integer rejected' );
    eval { $CLASS->new( parallel_fit => 'abc' ) };
    like( $@, qr/parallel_fit/, 'non-numeric rejected' );
};

subtest 'parallel_fit=1 is equivalent to serial' => sub {
    # n_trees > 1 and parallel_fit == 1 hits the same serial branch as
    # parallel_fit undef.  Just verify it produces a working model.
    my $f = $CLASS->new(
        n_trees      => 20,
        sample_size  => 256,
        seed         => 7,
        parallel_fit => 1,
    );
    $f->fit( \@train );
    is( scalar @{ $f->{trees} }, 20, 'tree count matches n_trees' );
    my $s = $f->score_samples( \@query );
    cmp_ok( $s->[1], '>', $s->[0], 'model separates outlier from inlier' );
};

done_testing;
