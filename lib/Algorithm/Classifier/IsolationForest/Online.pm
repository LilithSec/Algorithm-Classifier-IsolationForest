package Algorithm::Classifier::IsolationForest::Online;

use strict;
use warnings;
use Carp        qw(croak);
use JSON::PP    ();
use File::Slurp qw(read_file write_file);

# Runtime-only dependency: tagged_row_to_array is delegated to the parent
# class (identical semantics, no point duplicating it) and the
# contamination threshold selection reuses _threshold_from_ranked.  The
# parent never loads this module at compile time (its from_json requires
# it on demand), so there is no cycle.
use Algorithm::Classifier::IsolationForest ();

our $VERSION = '0.6.0';

# Node layout.  Unlike the batch forest's nodes, online nodes are mutable
# and carry a running point count plus the bounding box (per-feature
# lo/hi) of every point that has passed through them -- that box is what
# split simulation samples from, since points themselves are never stored
# in the tree.  Both node types share the first four slots so the
# learn/unlearn bookkeeping never has to branch on type:
#
#   leaf:     [0, count, \@lo, \@hi]
#   internal: [1, count, \@lo, \@hi, attr, split, left, right]
#
# The type tag mirrors the parent's convention (0 is falsy, so
# while ($node->[0]) walks to a leaf).  A leaf built from an empty
# synthetic partition has count 0 and an undef box (slots 2/3); the box
# is initialised from the first real point that reaches it.
use constant _N_TYPE  => 0;
use constant _N_COUNT => 1;
use constant _N_LO    => 2;
use constant _N_HI    => 3;
use constant _N_ATTR  => 4;
use constant _N_SPLIT => 5;
use constant _N_LEFT  => 6;
use constant _N_RIGHT => 7;

use constant _NT_LEAF => 0;
use constant _NT_AXIS => 1;

# Trees are binary (the reference implementation's branching_factor == 2),
# which fixes the depth-budget log base at log(2 * 2).
use constant _LOG4 => log(4);
use constant _LOG2 => log(2);

# DBL_EPSILON, added to the normalisation factor before dividing so a
# just-started model (normaliser 0) yields well-defined scores instead of
# a division by zero -- the same guard the reference implementation uses.
use constant _EPS => 2.220446049250313e-16;

=head1 NAME

Algorithm::Classifier::IsolationForest::Online - Online (streaming) Isolation Forest anomaly detection

=head1 SYNOPSIS

    use Algorithm::Classifier::IsolationForest::Online;

    my $oif = Algorithm::Classifier::IsolationForest::Online->new(
        n_trees          => 100,
        window_size      => 2048,
        max_leaf_samples => 32,
        seed             => 42,
    );

    # stream data through the model; each point is learned and old
    # points beyond the window are forgotten automatically
    $oif->learn(\@warmup_rows);

    # prequential operation: score each point against the model as it
    # stood BEFORE that point was learned, then learn it
    my $scores = $oif->score_learn(\@new_rows);

    # or score without learning
    my $scores2 = $oif->score_samples(\@query_rows);
    my $labels  = $oif->predict(\@query_rows);

    # persistence keeps the window, so a reloaded model keeps forgetting
    # correctly as the stream continues
    $oif->save('oiforest_model.json');
    my $resumed = Algorithm::Classifier::IsolationForest::Online->load('oiforest_model.json');

=head1 DESCRIPTION

Implements Online Isolation Forest (Online-iForest; Leveni, Weigert
Cassales, Pfahringer, Bifet & Boracchi 2024 -- see REFERENCES), a
streaming variant of Isolation Forest for data that arrives continuously
and whose distribution may drift.  There is no C<fit()>: the model
C<learn>s points as they arrive and, once more than C<window_size> points
have been seen, forgets the oldest point for every new one so the model
always reflects the most recent C<window_size> points of the stream.

Trees never store data points.  Each node keeps only a running count of
the points that passed through it and the bounding box of their feature
values.  A leaf splits once enough points have accumulated (see
C<max_leaf_samples> and C<growth>); because the actual points are gone,
the split simulates them by sampling uniformly inside the leaf's bounding
box.  Forgetting reverses the process: counts are decremented along the
forgotten point's path and a subtree whose count falls below its split
requirement is collapsed back into a leaf.

Scoring follows the classic Isolation Forest intuition -- anomalies
isolate at shallow depth -- but normalises by the depth budget
C<log(n/max_leaf_samples) / log(4)> of the current window rather than the
batch model's C<c(psi)>.  Scores are in (0, 1] with high values
anomalous, directly comparable in spirit (though not numerically) to the
parent class's scores.

This class is pure Perl; the parent's Inline::C accelerator does not
apply, as its packed-buffer scoring assumes immutable trees.  The
per-point cost is one root-to-leaf walk per tree, which is cheap at
typical stream rates.

A model needs to have seen at least C<max_leaf_samples> points before
tree structure exists at all; until then every point scores 1.0.  Give
the model a warm-up C<learn()> pass before trusting scores or labels.

Models saved by this class carry their own C<format> tag.
C<< Algorithm::Classifier::IsolationForest->load >> recognises it and
dispatches here, so callers can load either model type through the
parent class.

=head1 GENERAL METHODS

=head2 new(%args)

Inits the object.

  - n_trees :: number of isolation trees in the ensemble
      default :: 100

  - window_size :: how many of the most recent points the model reflects.
          Once the stream exceeds this, learning a point forgets the
          oldest retained point.  0 or undef disables forgetting: the
          model then learns from the whole stream and retains no window
          (so nothing is ever unlearned and threshold relearning needs
          caller-supplied data).
      default :: 2048

  - max_leaf_samples :: how many points a leaf must accumulate before it
          splits (eta in the paper).  Also the unit of the depth budget:
          trees stop splitting past log(n/eta)/log(4).
      default :: 32

  - growth :: how the split requirement scales with depth (the reference
          implementation's `type` parameter).
            adaptive :: a leaf at depth k needs max_leaf_samples * 2**k
                        points to split -- deeper splits need
                        exponentially more evidence
            fixed    :: max_leaf_samples points regardless of depth
      default :: adaptive

  - subsample :: probability in (0, 1] that a given tree learns (or
          forgets) a given point, drawn independently per tree per point.
          Values below 1 increase diversity among trees on very dense
          streams.  Note that, as in the reference implementation, learn
          and forget draws are independent, so per-tree counts are only
          approximate under subsampling.
      default :: 1.0

  - seed :: optional integer to seed srand with, for reproducible trees
          given the same stream in the same order.  Processed via
          abs(int()).  Seeding happens here in new(), since there is no
          fit() to do it in.
      default :: undef

  - contamination :: expected fraction of anomalies, in (0, 0.5]. When
          set, the first predict()-family call learns a score threshold
          that flags this fraction of the current window, and uses it as
          the default cutoff.  The threshold does NOT track the stream
          automatically afterwards; call relearn_threshold() to refresh
          it.  undef => no learned threshold (predict() falls back to
          0.5).
      default :: undef

  - missing :: how learn() treats undef (missing) feature cells.  Scoring
          always tolerates undef (mapped to 0), matching the parent
          class's long-standing behaviour.
            die  :: croak if a learned point contains an undef cell
            zero :: treat a missing cell as the value 0
      default :: die

  - feature_names :: optional arrayref of per-feature labels enabling the
          *_tagged methods.
      default :: undef

=cut

sub new {
	my ( $class, %args ) = @_;

	my $growth = $args{growth} // 'adaptive';
	croak "growth must be 'adaptive' or 'fixed'"
		unless $growth =~ /\A(?:adaptive|fixed)\z/;

	my $missing = $args{missing} // 'die';
	croak "missing must be one of: die, zero"
		unless $missing =~ /\A(?:die|zero)\z/;

	if ( defined( $args{seed} ) ) {
		$args{seed} = abs( int( $args{seed} ) );
	}

	# window_size => 0 and window_size => undef both mean "no forgetting";
	# normalise to 0 so the rest of the code has one falsy spelling.
	my $window_size = exists $args{window_size} ? ( $args{window_size} // 0 ) : 2048;

	my $self = {
		n_trees          => $args{n_trees} // 100,
		window_size      => $window_size,
		max_leaf_samples => $args{max_leaf_samples} // 32,
		growth           => $growth,
		subsample        => $args{subsample} // 1.0,
		seed             => $args{seed},
		contamination    => $args{contamination},
		missing          => $missing,
		feature_names    => $args{feature_names},
		threshold        => undef,                           # learned lazily if contamination set
		n_features       => undef,                           # learned from the first row
		seen             => 0,                               # total points learned over the model's lifetime
		window           => [],                              # the retained rows, oldest first
		trees            => [],
	};

	croak "n_trees must be >= 1"          unless $self->{n_trees} >= 1;
	croak "max_leaf_samples must be >= 1" unless $self->{max_leaf_samples} >= 1;
	croak "window_size must be 0 (unbounded) or >= max_leaf_samples"
		if $self->{window_size} && $self->{window_size} < $self->{max_leaf_samples};
	croak "subsample must be in (0, 1]"
		unless $self->{subsample} > 0 && $self->{subsample} <= 1;
	croak "contamination must be a number in (0, 0.5]"
		if defined $self->{contamination}
		&& !( $self->{contamination} > 0 && $self->{contamination} <= 0.5 );

	$self->{trees} = [ map { { root => undef, count => 0, depth_limit => 0 } } 1 .. $self->{n_trees} ];

	srand( $self->{seed} ) if defined $self->{seed};

	return bless $self, $class;
} ## end sub new

=head2 learn(\@data)

Learns the passed samples, in order, as the next points of the stream.
Once the model has seen more than C<window_size> points, each learned
point also forgets the oldest retained point, so the model tracks the
most recent C<window_size> points.

The data format matches the parent class's C<fit>: an arrayref of
arrayrefs, each inner arrayref one sample of numeric features.  All
samples must have the same feature count; the count is locked in by the
first sample ever learned.

Returns C<$self>, so it chains.

    $oif->learn(\@rows);

=cut

sub learn {
	my ( $self, $data ) = @_;
	croak "learn() expects a non-empty arrayref of samples"
		unless ref $data eq 'ARRAY' && @$data;
	for my $row (@$data) {
		$self->_learn_row( $self->_prep_row( $row, 'learn' ) );
	}
	return $self;
}

=head2 learn_tagged(\%row)

Learns a single sample supplied as a hashref of named feature values.
The model must have C<feature_names> set.  Returns C<$self>.

    $oif->learn_tagged({ cpu => 0.9, mem => 0.4, disk => 0.1 });

Croaks under the same conditions as L</tagged_row_to_array>.

=cut

sub learn_tagged {
	my ( $self, $row ) = @_;
	my $vec = $self->tagged_row_to_array( $row, 'learn_tagged' );
	return $self->learn( [$vec] );
}

=head2 score_learn(\@data)

Prequential (test-then-train) operation, the usual way to run a streaming
detector: each sample is scored against the model as it stood I<before>
that sample was learned, then learned.  Returns an arrayref of anomaly
scores, one per sample, in input order.

Unlike the pure scoring methods this works on a brand-new model too (the
first points of a stream simply score 1.0, as nothing is known yet).

    my $scores = $oif->score_learn(\@rows);

=cut

sub score_learn {
	my ( $self, $data ) = @_;
	croak "score_learn() expects a non-empty arrayref of samples"
		unless ref $data eq 'ARRAY' && @$data;
	my @scores;
	for my $row (@$data) {
		my $r = $self->_prep_row( $row, 'score_learn' );
		push @scores, $self->_score_row($r);
		$self->_learn_row($r);
	}
	return \@scores;
} ## end sub score_learn

=head2 score_learn_tagged(\%row)

Prequential score-then-learn for a single sample supplied as a hashref of
named feature values.  Returns the scalar anomaly score the sample had
before it was learned.

    my $score = $oif->score_learn_tagged({ cpu => 0.9, mem => 0.4 });

Croaks under the same conditions as L</tagged_row_to_array>.

=cut

sub score_learn_tagged {
	my ( $self, $row ) = @_;
	my $vec    = $self->tagged_row_to_array( $row, 'score_learn_tagged' );
	my $result = $self->score_learn( [$vec] );
	return $result->[0];
}

=head2 score_samples(\@data)

Returns an arrayref of anomaly scores in (0, 1] without learning
anything.  Scores near 1 are strong anomalies (isolated at shallow
depth); scores well below 0.5 are normal.

    my $scores = $oif->score_samples(\@data);

=cut

sub score_samples {
	my ( $self, $data ) = @_;
	$self->_check_learned;
	croak "score_samples() expects an arrayref of samples"
		unless ref $data eq 'ARRAY';
	my $sums = $self->_depth_sums($data);
	my $inv  = $self->_score_inv;
	return [ map { exp( -$_ * $inv ) } @$sums ];
}

=head2 score_sample_tagged(\%row)

Scores a single sample supplied as a hashref of named feature values,
without learning it.  Returns a scalar anomaly score in (0, 1].

    my $score = $oif->score_sample_tagged({ cpu => 0.9, mem => 0.4 });

Croaks under the same conditions as L</tagged_row_to_array>.

=cut

sub score_sample_tagged {
	my ( $self, $row ) = @_;
	my $vec    = $self->tagged_row_to_array( $row, 'score_sample_tagged' );
	my $result = $self->score_samples( [$vec] );
	return $result->[0];
}

=head2 path_lengths(\@data)

Returns an arrayref of the mean isolation depth per sample across the
trees, for inspection -- the streaming counterpart of the parent class's
method of the same name.  Depths include the per-leaf count adjustment.

    my $depths = $oif->path_lengths(\@data);

=cut

sub path_lengths {
	my ( $self, $data ) = @_;
	$self->_check_learned;
	croak "path_lengths() expects an arrayref of samples"
		unless ref $data eq 'ARRAY';
	my $sums = $self->_depth_sums($data);
	my $t    = $self->{n_trees};
	return [ map { $_ / $t } @$sums ];
}

=head2 predict(\@data, $threshold)

Returns an arrayref of 0/1 labels for the specified data, without
learning it.

If C<$threshold> is not given, the contamination-learned cutoff is used
when available (learned from the current window on first use -- see
C<contamination> in L</new>), otherwise 0.5.

Note that absolute score levels depend on C<window_size> and
C<max_leaf_samples> (shallower depth budgets compress scores downward),
so the 0.5 fallback is a blunt default here -- anomalies reliably rank
above normal points, but may sit below 0.5.  Setting C<contamination>,
or passing a threshold calibrated from observed scores, is recommended.

    my $labels = $oif->predict(\@data);

=cut

sub predict {
	my ( $self, $data, $threshold ) = @_;
	$self->_check_learned;
	$self->_ensure_threshold;
	$threshold
		= defined $threshold         ? $threshold
		: defined $self->{threshold} ? $self->{threshold}
		:                              0.5;
	my $scores = $self->score_samples($data);
	return [ map { $_ >= $threshold ? 1 : 0 } @$scores ];
} ## end sub predict

=head2 predict_tagged(\%row, $threshold)

Predicts whether a single sample, supplied as a hashref of named feature
values, is an anomaly.  Returns a scalar 1 (anomaly) or 0 (normal).
C<$threshold> defaults the same way as in L</predict>.

    my $label = $oif->predict_tagged({ cpu => 0.9, mem => 0.4 });

Croaks under the same conditions as L</tagged_row_to_array>.

=cut

sub predict_tagged {
	my ( $self, $row, $threshold ) = @_;
	my $vec    = $self->tagged_row_to_array( $row, 'predict_tagged' );
	my $result = $self->predict( [$vec], $threshold );
	return $result->[0];
}

=head2 score_predict_samples(\@data, $threshold)

Returns an arrayref of C<[$score, $label]> pairs, one per sample, without
learning.  C<$threshold> defaults the same way as in L</predict>.

    my $results = $oif->score_predict_samples(\@data);

=cut

sub score_predict_samples {
	my ( $self, $data, $threshold ) = @_;
	$self->_check_learned;
	$self->_ensure_threshold;
	$threshold
		= defined $threshold         ? $threshold
		: defined $self->{threshold} ? $self->{threshold}
		:                              0.5;
	my $scores = $self->score_samples($data);
	return [ map { [ $_, ( $_ >= $threshold ? 1 : 0 ) ] } @$scores ];
} ## end sub score_predict_samples

=head2 score_predict_sample_tagged(\%row, $threshold)

Scores and classifies a single sample supplied as a hashref of named
feature values.  Returns a two-element arrayref C<[$score, $label]>.
C<$threshold> defaults the same way as in L</predict>.

    my $pair = $oif->score_predict_sample_tagged({ cpu => 0.9, mem => 0.4 });

Croaks under the same conditions as L</tagged_row_to_array>.

=cut

sub score_predict_sample_tagged {
	my ( $self, $row, $threshold ) = @_;
	my $vec    = $self->tagged_row_to_array( $row, 'score_predict_sample_tagged' );
	my $result = $self->score_predict_samples( [$vec], $threshold );
	return $result->[0];
}

=head2 score_predict_split(\@data, $threshold)

Same values as L</score_predict_samples> but returned as two flat
arrayrefs.  In list context returns C<($scores_aref, $labels_aref)>.

    my ($scores, $labels) = $oif->score_predict_split(\@data);

=cut

sub score_predict_split {
	my ( $self, $data, $threshold ) = @_;
	$self->_check_learned;
	$self->_ensure_threshold;
	$threshold
		= defined $threshold         ? $threshold
		: defined $self->{threshold} ? $self->{threshold}
		:                              0.5;
	my $scores = $self->score_samples($data);
	my @labels = map { $_ >= $threshold ? 1 : 0 } @$scores;
	return ( $scores, \@labels );
} ## end sub score_predict_split

=head2 relearn_threshold(\@data)

Re-derives the contamination decision threshold so it flags the requested
fraction of the current window (or of C<\@data>, when passed).  Call this
after the stream has drifted, or on whatever cadence threshold freshness
matters; learning alone never moves the threshold.

Requires C<contamination> to have been set.  With C<< window_size => 0 >>
no window is retained, so C<\@data> must be supplied.

Returns C<$self>, so it chains.

    $oif->relearn_threshold;

=cut

sub relearn_threshold {
	my ( $self, $data ) = @_;
	croak "relearn_threshold requires contamination to have been set in new()"
		unless defined $self->{contamination};
	my $rows = defined $data ? $data : $self->{window};
	croak "relearn_threshold: no retained window to learn a threshold from "
		. "(window_size is 0); pass an arrayref of recent data"
		unless ref $rows eq 'ARRAY' && @$rows;

	my $scores = $self->score_samples($rows);
	my @desc   = sort { $b <=> $a } @$scores;
	my $n_pts  = scalar @desc;
	my $k      = int( $self->{contamination} * $n_pts + 0.5 );
	$k                 = 1      if $k < 1;
	$k                 = $n_pts if $k > $n_pts;
	$self->{threshold} = Algorithm::Classifier::IsolationForest::_threshold_from_ranked( \@desc, $k );
	return $self;
} ## end sub relearn_threshold

=head2 decision_threshold

The score cutoff the predict methods use by default; undef unless
C<contamination> was set and a predict-family method or
L</relearn_threshold> has run.

=cut

sub decision_threshold { return $_[0]->{threshold} }

=head2 feature_names

Returns the arrayref of feature name strings stored with the model, or
undef if none were provided.

=cut

sub feature_names { return $_[0]->{feature_names} }

=head2 window_count

Returns how many points the model currently retains in its sliding
window (0 when C<< window_size => 0 >>).

=cut

sub window_count { return scalar @{ $_[0]->{window} } }

=head2 seen

Returns the total number of points learned over the model's lifetime,
including points that have since been forgotten.

=cut

sub seen { return $_[0]->{seen} }

=head2 tagged_row_to_array(\%row, $caller)

Validates a hashref of named feature values against the model's stored
C<feature_names> and returns a positional arrayref.  Identical semantics
to the parent class's method of the same name (to which it delegates);
see there for the croak conditions.

=cut

sub tagged_row_to_array {
	my $self = shift;
	return Algorithm::Classifier::IsolationForest::tagged_row_to_array( $self, @_ );
}

=head1 MODEL SAVE/LOAD METHODS

Persistence keeps the sliding window alongside the trees, so a reloaded
model continues forgetting correctly as the stream resumes.  This makes
saved online models larger than batch models by O(window_size *
n_features).  Perl's RNG state is not persisted: a save/reload point
breaks bit-for-bit reproducibility of subsequent learning versus an
uninterrupted run, though scoring of the reloaded model is exact.

=head2 to_json

Returns a JSON representation of the model.

    my $json = $oif->to_json;

=cut

sub to_json {
	my ($self) = @_;
	my $payload = {
		format  => 'Algorithm::Classifier::IsolationForest::Online',
		version => 1,
		params  => {
			n_trees          => $self->{n_trees},
			window_size      => $self->{window_size},
			max_leaf_samples => $self->{max_leaf_samples},
			growth           => $self->{growth},
			subsample        => $self->{subsample},
			contamination    => $self->{contamination},
			threshold        => $self->{threshold},
			n_features       => $self->{n_features},
			missing          => $self->{missing},
			feature_names    => $self->{feature_names},
			seen             => $self->{seen},
		},
		trees  => [ map { { count => $_->{count}, root => $_->{root} } } @{ $self->{trees} } ],
		window => $self->{window},
	};
	return JSON::PP->new->canonical(1)->encode($payload);
} ## end sub to_json

=head2 from_json($json)

Init the object from the model in the specified JSON string.

    my $oif = Algorithm::Classifier::IsolationForest::Online->from_json($json);

=cut

sub from_json {
	my ( $class, $text ) = @_;
	my $payload = JSON::PP->new->decode($text);
	croak "not an online IsolationForest model"
		unless ref $payload eq 'HASH'
		&& defined $payload->{format}
		&& $payload->{format} eq 'Algorithm::Classifier::IsolationForest::Online';

	my $p = $payload->{params} || {};

	my $self = {
		n_trees          => $p->{n_trees},
		window_size      => $p->{window_size} // 0,
		max_leaf_samples => $p->{max_leaf_samples},
		growth           => $p->{growth}    // 'adaptive',
		subsample        => $p->{subsample} // 1.0,
		seed             => undef,
		contamination    => $p->{contamination},
		threshold        => $p->{threshold},
		n_features       => $p->{n_features},
		missing          => $p->{missing} // 'die',
		feature_names    => $p->{feature_names},
		seen             => $p->{seen}         // 0,
		window           => $payload->{window} // [],
		trees            => [],
	};

	my $trees = $payload->{trees};
	croak "model contains no trees" unless ref $trees eq 'ARRAY' && @$trees;

	my $model = bless $self, $class;

	# depth_limit is a pure function of the tree's count, so recompute it
	# rather than trusting a stored float.
	$self->{trees}
		= [ map { { count => $_->{count}, root => $_->{root}, depth_limit => $model->_rpl( $_->{count} ) } } @$trees ];

	return $model;
} ## end sub from_json

=head2 save($path)

Saves the model to the specified path.

    $oif->save($path);

=cut

sub save {
	my ( $self, $path ) = @_;
	write_file( $path, { 'atomic' => 1 }, $self->to_json );
}

=head2 load($path)

Init the object from the model in the specified file.

    my $oif = Algorithm::Classifier::IsolationForest::Online->load($path);

=cut

sub load {
	my ( $class, $path ) = @_;
	my $raw_model = read_file($path);
	return $class->from_json($raw_model);
}

=head1 REFERENCES

Filippo Leveni, Guilherme Weigert Cassales, Bernhard Pfahringer, Albert
Bifet, Giacomo Boracchi (2024). Online Isolation Forest. Proceedings of
the 41st International Conference on Machine Learning (ICML), PMLR 235.

L<https://proceedings.mlr.press/v235/leveni24a.html>

L<https://github.com/ineveLoppiliF/Online-Isolation-Forest>

=cut

###
###
### internal stuff below
###
###

sub _check_learned {
	my ($self) = @_;
	croak "model has not learned any data yet; call learn() first"
		unless $self->{seen} > 0;
}

# Validate one incoming sample, apply the missing-value strategy, and
# return a fresh dense copy (the window owns its rows; the caller may
# reuse or mutate the original).  Locks in n_features on first contact.
sub _prep_row {
	my ( $self, $row, $caller ) = @_;
	croak "$caller: each sample must be an arrayref of features"
		unless ref $row eq 'ARRAY' && @$row;

	if ( !defined $self->{n_features} ) {
		$self->{n_features} = scalar @$row;
	} elsif ( scalar @$row != $self->{n_features} ) {
		croak "$caller: sample has " . scalar(@$row) . " features but model expects " . $self->{n_features};
	}

	if ( $self->{missing} eq 'die' ) {
		for my $f ( 0 .. $#$row ) {
			next if defined $row->[$f];
			croak "$caller: undef feature value at column $f; "
				. "construct with missing => 'zero' to learn from data with missing values";
		}
		return [@$row];
	}

	# zero: a missing cell counts as the value 0.
	return [ map { $_ // 0 } @$row ];
} ## end sub _prep_row

# The depth budget for n points: how deep a tree fed n points is allowed
# (learn) or expected (scoring normalisation, per-leaf adjustment) to
# go.  log base 4 = log(2 * branching_factor) with binary trees.  Under
# max_leaf_samples points there is nothing to isolate: 0.
sub _rpl {
	my ( $self, $n ) = @_;
	my $eta = $self->{max_leaf_samples};
	return 0 if $n < $eta;
	return log( $n / $eta ) / _LOG4;
}

# How many points a node at $depth needs before it may split (or below
# which, on forgetting, it collapses back into a leaf).
sub _split_threshold {
	my ( $self, $depth ) = @_;
	return $self->{max_leaf_samples} * ( $self->{growth} eq 'adaptive' ? 2**$depth : 1 );
}

# Number of points the model currently reflects: the window fill, or the
# whole stream when forgetting is disabled.
sub _data_size {
	my ($self) = @_;
	return $self->{window_size} ? scalar @{ $self->{window} } : $self->{seen};
}

# exp() multiplier turning a per-sample depth SUM into the normalised
# anomaly score: 2**(-(sum/t)/norm) == exp(-sum * log(2)/(t*norm)).
# _EPS keeps a zero normaliser (fewer than max_leaf_samples points seen)
# well-defined; every depth is 0 then, so everything scores 1.0.
sub _score_inv {
	my ($self) = @_;
	my $norm = $self->_rpl( $self->_data_size * $self->{subsample} );
	return _LOG2 / ( $self->{n_trees} * ( $norm + _EPS ) );
}

#-------------------------------------------------------------------------------
# Learning.
#-------------------------------------------------------------------------------

# Advance the stream by one (already prepped) row: every tree learns it
# (subject to subsampling), it enters the window, and the oldest point
# beyond the window is forgotten.
sub _learn_row {
	my ( $self, $r ) = @_;
	my $sub = $self->{subsample};

	for my $tree ( @{ $self->{trees} } ) {
		next if $sub < 1 && rand() >= $sub;
		$self->_tree_learn( $tree, $r );
	}
	$self->{seen}++;

	if ( $self->{window_size} ) {
		push @{ $self->{window} }, $r;
		if ( @{ $self->{window} } > $self->{window_size} ) {
			my $old = shift @{ $self->{window} };
			for my $tree ( @{ $self->{trees} } ) {
				next if $sub < 1 && rand() >= $sub;
				$self->_tree_unlearn( $tree, $old );
			}
		}
	} ## end if ( $self->{window_size} )
	return;
} ## end sub _learn_row

sub _tree_learn {
	my ( $self, $tree, $x ) = @_;
	$tree->{count}++;
	$tree->{depth_limit} = $self->_rpl( $tree->{count} );
	if ( !defined $tree->{root} ) {
		$tree->{root} = [ _NT_LEAF, 1, [@$x], [@$x] ];
	} else {
		$tree->{root} = $self->_node_learn( $tree->{root}, $x, 0, $tree->{depth_limit} );
	}
	return;
} ## end sub _tree_learn

# Route $x down to its leaf, growing counts and bounding boxes along the
# path.  A leaf that has accumulated its split requirement (and still has
# depth budget) is replaced by a subtree built from synthetic points
# sampled inside its box -- the return value replaces the node in the
# parent, which is how leaves turn into subtrees in place.
sub _node_learn {
	my ( $self, $node, $x, $depth, $limit ) = @_;

	$node->[_N_COUNT]++;
	if ( !defined $node->[_N_LO] ) {

		# Leaf born from an empty synthetic partition: first real point
		# initialises the box.
		$node->[_N_LO] = [@$x];
		$node->[_N_HI] = [@$x];
	} else {
		my ( $lo, $hi ) = ( $node->[_N_LO], $node->[_N_HI] );
		for my $f ( 0 .. $#$x ) {
			my $v = $x->[$f];
			$lo->[$f] = $v if $v < $lo->[$f];
			$hi->[$f] = $v if $v > $hi->[$f];
		}
	}

	if ( $node->[_N_TYPE] == _NT_LEAF ) {
		if ( $node->[_N_COUNT] >= $self->_split_threshold($depth) && $depth < $limit ) {
			my $pts = $self->_sample_box( $node->[_N_LO], $node->[_N_HI], $node->[_N_COUNT] );
			return $self->_build_from_points( $pts, $depth, $limit );
		}
		return $node;
	}

	my $ci = $x->[ $node->[_N_ATTR] ] < $node->[_N_SPLIT] ? _N_LEFT : _N_RIGHT;
	$node->[$ci] = $self->_node_learn( $node->[$ci], $x, $depth + 1, $limit );
	return $node;
} ## end sub _node_learn

# $n synthetic points drawn uniformly inside the box -- the stand-in for
# the real points the tree never stored.
sub _sample_box {
	my ( $self, $lo, $hi, $n ) = @_;
	my @pts;
	for ( 1 .. $n ) {
		push @pts, [ map { my $w = $hi->[$_] - $lo->[$_]; $w > 0 ? $lo->[$_] + rand() * $w : $lo->[$_] } 0 .. $#$lo ];
	}
	return \@pts;
}

# Recursively build a subtree over (synthetic) points: random feature,
# uniform split value within the points' range on it, recurse on the
# partitions.  Leaves keep the partition's count and box.
sub _build_from_points {
	my ( $self, $pts, $depth, $limit ) = @_;
	my $n = scalar @$pts;
	my ( $lo, $hi ) = _box_of($pts);

	if ( $n < $self->_split_threshold($depth) || $depth >= $limit ) {
		return [ _NT_LEAF, $n, $lo, $hi ];
	}

	my $attr = int( rand( $self->{n_features} ) );
	my ( $pmin, $pmax ) = ( $pts->[0][$attr], $pts->[0][$attr] );
	for my $p (@$pts) {
		$pmin = $p->[$attr] if $p->[$attr] < $pmin;
		$pmax = $p->[$attr] if $p->[$attr] > $pmax;
	}
	my $split = $pmin + rand() * ( $pmax - $pmin );

	my ( @l, @r );
	for my $p (@$pts) {
		if   ( $p->[$attr] < $split ) { push @l, $p }
		else                          { push @r, $p }
	}

	my $left  = $self->_build_from_points( \@l, $depth + 1, $limit );
	my $right = $self->_build_from_points( \@r, $depth + 1, $limit );
	return [ _NT_AXIS, $n, $lo, $hi, $attr, $split, $left, $right ];
} ## end sub _build_from_points

#-------------------------------------------------------------------------------
# Forgetting.
#-------------------------------------------------------------------------------

sub _tree_unlearn {
	my ( $self, $tree, $x ) = @_;
	$tree->{count}--;
	$tree->{depth_limit} = $self->_rpl( $tree->{count} );
	return unless defined $tree->{root};
	$tree->{root} = $self->_node_unlearn( $tree->{root}, $x, 0 );
	return;
}

# Route the forgotten point down its (current) path, decrementing counts.
# An internal node whose count no longer justifies its split collapses
# back into a leaf; otherwise its box is refreshed to the union of its
# children's, which is how boxes shrink as old extremes age out.
sub _node_unlearn {
	my ( $self, $node, $x, $depth ) = @_;

	$node->[_N_COUNT]--;
	return $node            if $node->[_N_TYPE] == _NT_LEAF;
	return _collapse($node) if $node->[_N_COUNT] < $self->_split_threshold($depth);

	my $ci = $x->[ $node->[_N_ATTR] ] < $node->[_N_SPLIT] ? _N_LEFT : _N_RIGHT;
	$node->[$ci] = $self->_node_unlearn( $node->[$ci], $x, $depth + 1 );

	my ( $lo, $hi ) = _box_union( $node->[_N_LEFT], $node->[_N_RIGHT] );
	if ( defined $lo ) {
		$node->[_N_LO] = $lo;
		$node->[_N_HI] = $hi;
	}
	return $node;
} ## end sub _node_unlearn

# Aggregate a subtree back into a single leaf holding the subtree's
# (already decremented) count and the union of its descendants' boxes.
sub _collapse {
	my ($node) = @_;
	return $node if $node->[_N_TYPE] == _NT_LEAF;
	my $l = _collapse( $node->[_N_LEFT] );
	my $r = _collapse( $node->[_N_RIGHT] );
	my ( $lo, $hi ) = _box_union( $l, $r );
	if ( !defined $lo ) {

		# Both children empty: keep the node's own box.
		( $lo, $hi ) = ( $node->[_N_LO], $node->[_N_HI] );
	}
	return [ _NT_LEAF, $node->[_N_COUNT], $lo, $hi ];
} ## end sub _collapse

# (lo, hi) of the union of two nodes' boxes, as fresh arrays (parent
# boxes grow in place, so they must never alias a child's).  Nodes with
# no box yet (empty leaves) are skipped; (undef, undef) if neither has
# one.
sub _box_union {
	my ( $a, $b ) = @_;
	my @boxed = grep { defined $_->[_N_LO] } ( $a, $b );
	return ( undef, undef ) unless @boxed;
	my $lo = [ @{ $boxed[0][_N_LO] } ];
	my $hi = [ @{ $boxed[0][_N_HI] } ];
	if ( @boxed == 2 ) {
		my ( $blo, $bhi ) = ( $boxed[1][_N_LO], $boxed[1][_N_HI] );
		for my $f ( 0 .. $#$lo ) {
			$lo->[$f] = $blo->[$f] if $blo->[$f] < $lo->[$f];
			$hi->[$f] = $bhi->[$f] if $bhi->[$f] > $hi->[$f];
		}
	}
	return ( $lo, $hi );
} ## end sub _box_union

# (lo, hi) bounding box of a point set; (undef, undef) when empty.
sub _box_of {
	my ($pts) = @_;
	return ( undef, undef ) unless @$pts;
	my $lo = [ @{ $pts->[0] } ];
	my $hi = [ @{ $pts->[0] } ];
	for my $p (@$pts) {
		for my $f ( 0 .. $#$p ) {
			$lo->[$f] = $p->[$f] if $p->[$f] < $lo->[$f];
			$hi->[$f] = $p->[$f] if $p->[$f] > $hi->[$f];
		}
	}
	return ( $lo, $hi );
} ## end sub _box_of

#-------------------------------------------------------------------------------
# Scoring.
#-------------------------------------------------------------------------------

# Depth of the leaf $x lands in, plus the leaf's own depth budget -- the
# streaming analogue of the batch scorer's c(leaf size) adjustment.
# Scoring tolerates undef cells (mapped to 0), matching the parent class.
sub _depth_of {
	my ( $self, $x, $node ) = @_;
	my $depth = 0;
	while ( $node->[_N_TYPE] ) {
		$node = ( $x->[ $node->[_N_ATTR] ] // 0 ) < $node->[_N_SPLIT] ? $node->[_N_LEFT] : $node->[_N_RIGHT];
		$depth++;
	}
	return $depth + $self->_rpl( $node->[_N_COUNT] );
}

# Per-sample depth sums across all trees (tree-outer, sample-inner for
# cache locality, mirroring the parent's pure-Perl loops).
sub _depth_sums {
	my ( $self, $data ) = @_;
	my @sums = (0) x @$data;
	for my $tree ( @{ $self->{trees} } ) {
		my $root = $tree->{root};
		next unless defined $root;
		for my $i ( 0 .. $#$data ) {
			$sums[$i] += $self->_depth_of( $data->[$i], $root );
		}
	}
	return \@sums;
} ## end sub _depth_sums

# Single-row score against the current model state; used by the
# prequential score_learn loop, where the normaliser moves as points are
# learned and so must be recomputed per row.
sub _score_row {
	my ( $self, $r ) = @_;
	my $sum = 0;
	for my $tree ( @{ $self->{trees} } ) {
		$sum += $self->_depth_of( $r, $tree->{root} ) if defined $tree->{root};
	}
	return exp( -$sum * $self->_score_inv );
}

# Lazily learn the contamination threshold from the current window the
# first time a predict-family method needs it.  A model with no retained
# window (window_size 0) stays on the 0.5 fallback until the caller runs
# relearn_threshold with data.
sub _ensure_threshold {
	my ($self) = @_;
	return
		   if !defined $self->{contamination}
		|| defined $self->{threshold}
		|| !@{ $self->{window} };
	$self->relearn_threshold;
	return;
}

1;
