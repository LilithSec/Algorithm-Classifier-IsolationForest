package Algorithm::Classifier::IsolationForest::App::Command::accel;

use strict;
use warnings;
use Algorithm::Classifier::IsolationForest ();
use Algorithm::Classifier::IsolationForest::App -command;

sub opt_spec { () }

sub abstract { 'Report which (if any) native acceleration backend is active' }

sub description { 'Initialises Algorithm::Classifier::IsolationForest, fits
a tiny synthetic dataset to exercise the optional native code path, then
reports which acceleration (if any) is wired up:

  * Inline::C  -- C scoring backend compiled at module load
  * OpenMP     -- parallel tree-walk across CPU cores (requires libgomp)

The Inline::C / OpenMP detection happens automatically the first time the
module is loaded (the build is cached under _Inline/).  If neither
backend is active the module falls back to a pure-Perl implementation.
' }

sub validate { 1 }

sub execute {
	my ( $self, $opt, $args ) = @_;

	# Tiny deterministic dataset.  Fitting + scoring confirms the chosen
	# backend is callable end-to-end, not merely that it compiled.
	srand(1);
	my @data = map { [ rand(), rand(), rand() ] } 1 .. 30;
	push @data, [ 10, 10, 10 ], [ -10, -10, -10 ];

	my $iforest = Algorithm::Classifier::IsolationForest->new(
		n_trees     => 10,
		sample_size => 32,
		seed        => 1,
	);
	$iforest->fit( \@data );
	$iforest->score_samples( [ [ 0.5, 0.5, 0.5 ] ] );

	my $has_c
		= $Algorithm::Classifier::IsolationForest::HAS_C ? 1 : 0;
	my $has_openmp
		= $Algorithm::Classifier::IsolationForest::HAS_OPENMP ? 1 : 0;

	print "Algorithm::Classifier::IsolationForest acceleration status\n";
	print "  Inline::C : ", ( $has_c      ? "available\n" : "not available\n" );
	print "  OpenMP    : ", ( $has_openmp ? "available\n" : "not available\n" );
	print "\n";

	if ( $has_c && $has_openmp ) {
		print "Active backend: Inline::C with OpenMP (multi-core parallel scoring)\n";
	}
	elsif ($has_c) {
		print "Active backend: Inline::C (single-threaded)\n";
	}
	else {
		print "Active backend: pure Perl (no native acceleration)\n";
	}

	return 1;
}

return 1;
