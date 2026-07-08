#!/usr/bin/perl
# benchmarking/bench-online-score-accel.pl
#
# Benchmarks Online Isolation Forest batch scoring under each acceleration
# backend:
#   pure_perl   -- use_c => 0                   (pure Perl tree walk)
#   c_serial    -- use_c => 1, use_openmp => 0  (C tree walk, single thread)
#   c_openmp    -- use_c => 1, use_openmp => 1  (C tree walk, OpenMP parallel)
#
# The online class scores through the parent's C backend by lazily packing
# its mutable trees into the parent's node layout; learning invalidates the
# packed snapshot and the next scoring call repacks once.  Sections 1 and 2
# measure steady-state scoring (snapshot reused across calls); section 3
# interleaves a learned point before every scoring call, so each call pays
# the repack -- the worst case for the C path.
#
# Reference numbers (2026-07-08, 8-core dev box, 100 trees, window 2048,
# 5 features, 20k query points): pure Perl ~3.6 s/call, C serial ~58 ms,
# C+OpenMP ~9 ms.
#
# Run with:
#   perl -Ilib benchmarking/bench-online-score-accel.pl

use strict;
use warnings;
use lib '../lib';
use FindBin;
use lib "$FindBin::Bin";
use BenchAccel                                     qw(wall_cmpthese);
use Algorithm::Classifier::IsolationForest         ();
use Algorithm::Classifier::IsolationForest::Online ();

use constant PI => 3.14159265358979;

sub gaussian {
	my ( $mu, $sigma ) = @_;
	return $mu + $sigma * sqrt( -2 * log( rand() || 1e-12 ) ) * cos( 2 * PI * rand() );
}

sub make_data {
	my ( $n, $nf ) = @_;
	return [
		map {
			[ map { gaussian( 0, 1 ) } 1 .. $nf ]
		} 1 .. $n
	];
}

my $HAS_C      = $Algorithm::Classifier::IsolationForest::HAS_C;
my $HAS_OPENMP = $Algorithm::Classifier::IsolationForest::HAS_OPENMP;

# One model per accel config, sharing seed and stream so the trees are
# identical -- learning is pure Perl on all three, only scoring differs.
sub build_models {
	my ( $stream, %opts ) = @_;
	my %m;
	$m{pure_perl} = Algorithm::Classifier::IsolationForest::Online->new( %opts, use_c => 0 );
	$m{c_serial}  = Algorithm::Classifier::IsolationForest::Online->new( %opts, use_c => 1, use_openmp => 0 )
		if $HAS_C;
	$m{c_openmp} = Algorithm::Classifier::IsolationForest::Online->new( %opts, use_c => 1, use_openmp => 1 )
		if $HAS_C && $HAS_OPENMP;
	$_->learn($stream) for values %m;
	return \%m;
} ## end sub build_models

print "=" x 70, "\n";
print " online (streaming) scoring accel benchmarks\n";
print " Algorithm::Classifier::IsolationForest::Online\n";
print "=" x 70, "\n";
printf "Backend availability: HAS_C=%d  HAS_OPENMP=%d\n", $HAS_C, $HAS_OPENMP;
print "(rates shown as calls/second wall-clock; higher is faster)\n";

srand(42);
my $stream = make_data( 3000, 5 );
my $models = build_models(
	$stream,
	n_trees          => 100,
	window_size      => 2048,
	max_leaf_samples => 32,
	seed             => 1,
);

# -----------------------------------------------------------------------
# 1. Scoring method comparison  (1000 query points, snapshot reused)
# -----------------------------------------------------------------------
print "\n--- scoring methods  (100 trees, 1000 query points, 5 features) ---\n";
srand(43);
my $q1k = make_data( 1000, 5 );

for my $method (
	qw(score_samples predict score_predict_samples
	score_predict_split path_lengths)
	)
{
	printf "\n  %s\n", $method;
	my %v;
	for my $name ( keys %$models ) {
		my $m = $models->{$name};
		$v{$name} = sub { my @r = $m->$method($q1k); 1 };
	}
	wall_cmpthese( 1, \%v );
} ## end for my $method ( qw(score_samples predict score_predict_samples...))

# -----------------------------------------------------------------------
# 2. Query set size scaling  (where OpenMP parallelism shines)
# -----------------------------------------------------------------------
for my $n_q ( 1000, 10000, 50000 ) {
	print "\n--- score_samples, $n_q query points ---\n";
	srand(44);
	my $q = make_data( $n_q, 5 );
	my %v;
	for my $name ( keys %$models ) {
		my $m = $models->{$name};
		$v{$name} = sub { my $s = $m->score_samples($q); 1 };
	}
	wall_cmpthese( 1, \%v );
} ## end for my $n_q ( 1000, 10000, 50000 )

# -----------------------------------------------------------------------
# 3. Interleaved learn + score  (every call repacks the snapshot)
# -----------------------------------------------------------------------
print "\n--- learn(1 row) + score_samples(1000)  (repack per call) ---\n";
srand(45);
my $q_mut = make_data( 1000, 5 );
my @drip  = @{ make_data( 100000, 5 ) };
my %v;
for my $name ( keys %$models ) {
	my $m = $models->{$name};
	$v{$name} = sub {
		$m->learn( [ shift(@drip) // [ (0) x 5 ] ] );
		my $s = $m->score_samples($q_mut);
		1;
	};
}
wall_cmpthese( 1, \%v );

print "\ndone\n";
