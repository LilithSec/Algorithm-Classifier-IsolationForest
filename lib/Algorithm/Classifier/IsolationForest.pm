package Algorithm::Classifier::IsolationForest;

use strict;
use warnings;
use Carp        qw(croak);
use List::Util  qw(min);
use POSIX       qw(ceil);
use JSON::PP    ();
use File::Slurp qw(read_file write_file);

our $VERSION = '0.1.0';

use constant EULER  => 0.5772156649015329;
use constant TWO_PI => 6.283185307179586;

# Node-type tags stored in index 0 of every tree node arrayref.
# 0 is falsy, so  while ($node->[0])  acts as  while (!leaf).
use constant _NODE_LEAF    => 0;
use constant _NODE_AXIS    => 1;
use constant _NODE_OBLIQUE => 2;

# ---------------------------------------------------------------------------
# Optional Inline::C accelerator for the scoring hot path.
#
# score_tree_xs(nodes_sv, coefs_sv, x_sv, sums_sv, n_pts, n_feats)
# accumulates path lengths for all n_pts query points through one pre-packed
# tree directly in C, avoiding Perl call-overhead on the innermost loop.
#
# Node layout (6 doubles per node, "IF_NZ = 6"):
#   leaf:    [0, size, 0,   0,  0, 0]
#   axis:    [1, attr, split, li, ri, 0]
#   oblique: [2, coff, nf,  li, ri, b]
#
# coefs: flat array of (feat_idx, coef_val) pairs, indexed by coff*2.
# x:     row-major doubles, n_pts rows of n_feats each.
# sums:  in/out double array of length n_pts; C adds to existing values.
# ---------------------------------------------------------------------------
our $HAS_C = 0;
{
    my $C_CODE = <<'__INLINE_C__';
#include <math.h>
#define IF_NZ 6
static double _ifc(double n){
    if(n<=1.0)return 0.0;
    if(n<2.5) return 1.0;
    double h=log(n-1.0)+0.5772156649015329;
    return 2.0*h-2.0*(n-1.0)/n;
}
/* pack_input_xs(data_sv, out_sv, n_pts, n_feats)
 *
 * Walks a Perl arrayref-of-arrayrefs (n_pts rows of n_feats doubles each)
 * directly in C and writes the packed double buffer into out_sv (which the
 * caller pre-allocates with "\0" x (n_pts*n_feats*8)).  Replaces
 *
 *   pack('d*', map { my $r=$_; map { $r->[$_] // 0 } 0..$nf-1 } @$data)
 *
 * which was the dominant per-call overhead for high feature counts.
 * Undef cells (and missing rows) are coerced to 0.0 with no warning.
 * Matches the "fill an SV in place" convention used by score_tree_xs. */
void pack_input_xs(SV* data_sv, SV* out_sv, int n_pts, int n_feats){
    STRLEN tl;
    double* out;
    AV* outer;
    int i, k;

    if (!SvROK(data_sv) || SvTYPE(SvRV(data_sv)) != SVt_PVAV) {
        croak("pack_input_xs: data must be an arrayref");
    }
    outer = (AV*)SvRV(data_sv);
    out   = (double*)SvPVbyte_force(out_sv, tl);

    for (i = 0; i < n_pts; i++) {
        SV** row_pp = av_fetch(outer, i, 0);
        double* dst = out + (size_t)i * (size_t)n_feats;
        if (!row_pp || !*row_pp || !SvROK(*row_pp) ||
            SvTYPE(SvRV(*row_pp)) != SVt_PVAV) {
            for (k = 0; k < n_feats; k++) dst[k] = 0.0;
            continue;
        }
        {
            AV* row = (AV*)SvRV(*row_pp);
            for (k = 0; k < n_feats; k++) {
                SV** v = av_fetch(row, k, 0);
                if (v && *v && SvOK(*v)) {
                    dst[k] = SvNV(*v);
                } else {
                    dst[k] = 0.0;
                }
            }
        }
    }
}

void score_tree_xs(SV*nd_sv,SV*co_sv,SV*x_sv,SV*sm_sv,int n_pts,int n_feats){
    STRLEN tl;
    const double*nd=(const double*)SvPVbyte(nd_sv,tl);
    const double*co=(const double*)SvPVbyte(co_sv,tl);
    const double*xd=(const double*)SvPVbyte(x_sv,tl);
    double*sm=(double*)SvPVbyte_force(sm_sv,tl);
    int i,k;
    for(i=0;i<n_pts;i++){
        const double*xi=xd+(size_t)i*(size_t)n_feats;
        int ni=0,depth=0;
        for(;;){
            const double*node=nd+(size_t)ni*IF_NZ;
            int type=(int)node[0];
            if(type==0){sm[i]+=depth+_ifc(node[1]);break;}
            if(type==1){
                int attr=(int)node[1];
                double fv=(attr<n_feats)?xi[attr]:0.0;
                ni=(fv<node[2])?(int)node[3]:(int)node[4];
            }else{
                int coff=(int)node[1],nf=(int)node[2];
                double b=node[5],dot=0.0;
                const double*cp=co+(size_t)coff*2;
                for(k=0;k<nf;k++){
                    int fi=(int)cp[k*2];
                    dot+=cp[k*2+1]*(fi<n_feats?xi[fi]:0.0);
                }
                ni=(dot<=b)?(int)node[3]:(int)node[4];
            }
            depth++;
        }
    }
}
__INLINE_C__
    local $@;
    eval {
        require Inline;
        Inline->import( C => $C_CODE, LIBS => '-lm' );
        $HAS_C = 1;
    };
}

=head1 NAME

Algorithm::Classifier::IsolationForest - unsupervised anomaly detection via Isolation Forest or Extended Isolation Forest

=head1 SYNOPSIS

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

=head1 DESCRIPTION

Isolation Forest (Liu, Fei Tony & Ting, Kai & Zhou, Zhi-Hua, 2008) detects anomalies by random
partitioning rather than by modelling normal points. Each tree repeatedly
splits the data. Points that get isolated after only a few splits are likely
anomalies. The score is the average isolation depth across many trees,
normalised so values approach 1 for anomalies and stay below 0.5 for normal
points.

In extended mode the module implements the Extended Isolation Forest
variant. Each split is a random hyperplane instead of an axis-aligned cut,
which removes the rectangular, axis-aligned bias in the score field and
tends to help on elongated or multi-modal data.

psi refernced below is ψ or the pitchfork math symbol refrenced in paper,
Liu, Fei Tony & Ting, Kai & Zhou, Zhi-Hua. (2008). Isolation Forest. 413 - 422. 10.1109/ICDM.2008.17.

... or max samples.

L<https://www.researchgate.net/publication/224384174_Isolation_Forest>

=head1 GENERAL METHODS

=head2 new(%args)

Inits the object.

  - n_trees :: number of isolation trees in the ensemble
      default :: 100

  - sample_size :: sub-sample size used to build each tree... max samples
      default :: 256

   - max_depth :: per-tree height limit... if not defined is set to ceil(log2(psi))
       default :: undef

   - seed :: optional integer to seed srand with for reproducible trees...
           see perldoc -f srand for more info. This number is processed via abs(int()).
       default :: undef

   - mode :: if it should be IF or EIF
        axis :: classic axis-parallel splits (IF)
        extended :: oblique hyperplane splits (EIF)
      default :: axis

   - extension_level :: extended mode only... how many features take partin each
           split. 0 behaves like a single-feature (axis) cut; the
           maximum (n_features - 1) uses every varying feature. undef
           => maximum. Clamped to [0, n_features - 1] at fit time.

    - contamination :: expected fraction of anomalies, in (0, 0.5]. When given,
          fit() learns a score threshold that flags this fraction of
          the training set, and predict() uses it by default. undef
          => no learned threshold (predict() falls back to 0.5).
        default :: undef

Note: log2 under Perl is as below...

    log($psi) / log(2)

=cut

sub new {
	my ( $class, %args ) = @_;

	my $mode = $args{mode} // 'axis';
	croak "mode must be 'axis' or 'extended'"
		unless $mode eq 'axis' || $mode eq 'extended';

	if ( defined( $args{seed} ) ) {
		$args{seed} = abs( int( $args{seed} ) );
	}

	my $self = {
		n_trees         => $args{n_trees}     // 100,
		sample_size     => $args{sample_size} // 256,
		max_depth       => $args{max_depth},          # undef => auto
		seed            => $args{seed},               # undef => non-deterministic
		mode            => $mode,
		extension_level => $args{extension_level},    # undef => max, resolved in fit()
		contamination   => $args{contamination},      # undef => no learned threshold
		threshold       => undef,                     # learned in fit() if contamination set
		trees           => [],
		c_psi           => undef,                     # c(psi), set during fit()
		n_features      => undef,
	};

	croak "n_trees must be >= 1"     unless $self->{n_trees} >= 1;
	croak "sample_size must be >= 1" unless $self->{sample_size} >= 1;
	croak "extension_level must be >= 0"
		if defined $self->{extension_level} && $self->{extension_level} < 0;
	croak "contamination must be a number in (0, 0.5]"
		if defined $self->{contamination}
		&& !( $self->{contamination} > 0 && $self->{contamination} <= 0.5 );

	return bless $self, $class;
} ## end sub new

=head2 decision_threshold

The score cutoff C<predict> uses by default; undef unless C<contamination> was
set.

=cut

sub decision_threshold { return $_[0]->{threshold} }

=head2 fit

Trains the model on the specified data.

The data taken is an array of arrays. Each sub-array is one sample and must
contain one or more numeric features. All samples must have the same number
of features. There is no upper limit on dimensionality.

    @training_data = (
        [ 3, 5 ],
        [ 2.3, 1 ],
        [ 5, 9 ],
        ...
    );

    # Three-feature example
    @training_data = (
        [ 1.0, 2.0, 3.0 ],
        [ 1.1, 1.9, 3.1 ],
        ...
    );

Below shows a example of building a gausing cluster and using that for training.

    # so it is reproducible
    srand(7);

    # build a gaussian cluster and add a handful out outliers...

    use constant PI => 3.14159265358979;
    sub gaussian {
        my ($mu, $sigma) = @_;
        my $u1 = rand() || 1e-12;
        my $u2 = rand();
        my $z  = sqrt(-2 * log($u1)) * cos(2 * PI * $u2);
        return $mu + $sigma * $z;
    }

    # add some normal items
    for (1 .. 500) {
        push @data,  [ gaussian(0, 1), gaussian(0, 1) ];
        push @truth, 0;
    }
    # add some outliers
    for (1 .. 20) {
        my $angle  = rand() * 2 * PI;
        my $radius = 5 + rand() * 3;             # distance 5..8 from the origin
        push @data,  [ $radius * cos($angle), $radius * sin($angle) ];
        push @truth, 1;
    }

    $iforest->fit(\@training_data);

=cut

sub fit {
	my ( $self, $data ) = @_;

	croak "fit() expects a non-empty arrayref of samples"
		unless ref $data eq 'ARRAY' && @$data;
	croak "each sample must be an arrayref of features"
		unless ref $data->[0] eq 'ARRAY' && @{ $data->[0] };

	my $n          = scalar @$data;
	my $n_features = scalar @{ $data->[0] };
	$self->{n_features} = $n_features;

	# The sub-sample cannot be larger than the data set itself.
	my $psi = min( $self->{sample_size}, $n );
	$self->{c_psi}    = _c($psi);
	$self->{psi_used} = $psi;

	# Resolve the extension level against the data's dimensionality.
	if ( $self->{mode} eq 'extended' ) {
		my $max_ext = $n_features - 1;
		my $ext
			= defined $self->{extension_level}
			? $self->{extension_level}
			: $max_ext;
		$ext                          = 0        if $ext < 0;
		$ext                          = $max_ext if $ext > $max_ext;
		$self->{extension_level_used} = $ext;
	} else {
		$self->{extension_level_used} = undef;
	}

	# Height limit: the average tree height ceil(log2(psi)). Past this depth the
	# remaining points are scored using the c(size) adjustment instead.
	my $limit
		= defined $self->{max_depth}
		? $self->{max_depth}
		: ceil( log($psi) / log(2) );
	$limit = 1 if $limit < 1;
	$self->{max_depth_used} = $limit;

	srand( $self->{seed} ) if defined $self->{seed};

	my @trees;
	for ( 1 .. $self->{n_trees} ) {
		my $sample = _subsample( $data, $psi );
		push @trees, $self->_build_tree( $sample, 0, $limit );
	}
	$self->{trees} = \@trees;

	# If a contamination rate was requested, learn the score cutoff that flags
	# that fraction of the training set. We place the threshold midway between
	# the k-th and (k+1)-th highest training scores, so it sits in the gap
	# between flagged and unflagged points -- unambiguous and robust to the
	# tiny float rounding introduced by JSON serialisation.
	if ( defined $self->{contamination} ) {
		my $scores = $self->score_samples($data);
		my @desc   = sort { $b <=> $a } @$scores;
		my $n_pts  = scalar @desc;
		my $k      = int( $self->{contamination} * $n_pts + 0.5 );
		$k                 = 1      if $k < 1;
		$k                 = $n_pts if $k > $n_pts;
		$self->{threshold} = $k < $n_pts
			? ( $desc[ $k - 1 ] + $desc[$k] ) / 2.0    # midpoint of the boundary
			: $desc[ $n_pts - 1 ] - 1e-9;              # k == n: flag everything
	} ## end if ( defined $self->{contamination} )

	$self->_rebuild_c_trees() if $HAS_C;
	return $self;
} ## end sub fit

=head2 path_lengths(\@data)

Returns the mean isolation depth per sample, for inspection.

    my @lenghts = $forest->path_lengths(\@data);

    print "x, y, length\n";

    my $int=0;
    while (defined($data[$int])) {
        print $data[$int][0].', '.$data[$int][1].', '.$lenghts[$int]."\n";

        $int++;
    }

=cut

sub path_lengths {
	my ( $self, $data ) = @_;
	$self->_check_fitted;
	my $trees = $self->{trees};
	my $t     = scalar @$trees;

	if ( $HAS_C && $self->{_c_nodes} ) {
		my $n_pts   = scalar @$data;
		my $nf      = $self->{n_features};
		my $x_packed    = "\0" x ( $n_pts * $nf * 8 );
		pack_input_xs( $data, $x_packed, $n_pts, $nf );
		my $sums_packed = pack( 'd*', (0.0) x $n_pts );
		my $c_nodes = $self->{_c_nodes};
		my $c_coefs = $self->{_c_coefs};
		for my $ti ( 0 .. $t - 1 ) {
			score_tree_xs( $c_nodes->[$ti], $c_coefs->[$ti],
				$x_packed, $sums_packed, $n_pts, $nf );
		}
		my @sums = unpack( 'd*', $sums_packed );
		return [ map { $_ / $t } @sums ];
	}

	# Pure-Perl fallback (tree-outer, sample-inner for cache locality).
	my @sums = (0) x @$data;
	for my $tree (@$trees) {
		for my $i ( 0 .. $#$data ) {
			$sums[$i] += _path_length( $data->[$i], $tree, 0 );
		}
	}
	return [ map { $_ / $t } @sums ];
} ## end sub path_lengths

=head predict(\@data, $threshold)

Returns an arrayref of 0/1 labels for the specified data.

If theshold is not specified it uses whatever the set default.

    my $results = $forest->predict(\@data, $threshold);

    print "x, y, result\n";

    my $int=0;
    while (defined($data[$int])) {
        print $data[$int][0].', '.$data[$int][1].', '.$results->[$int]."\n";

        $int++;
    }

=cut

sub predict {
	my ( $self, $data, $threshold ) = @_;
	$threshold
		= defined $threshold         ? $threshold
		: defined $self->{threshold} ? $self->{threshold}
		:                              0.5;
	my $scores = $self->score_samples($data);
	return [ map { $_ >= $threshold ? 1 : 0 } @$scores ];
}

=head2 score_samples(\@data)

Returns an arrayref of anomaly scores, between 0 and 1.

Scores near 1 are strong anomalies (isolated quickly).

Scores well below 0.5 are normal.

Scores ~0.5 means the points are hard to tell apart.

    my $scores = $forest->path_lengths(\@data);

    print "x, y, length\n";

    my $int=0;
    while (defined($data[$int])) {
        print $data[$int][0].', '.$data[$int][1].', '.$scores->[$int]."\n";

        $int++;
    }

=cut

sub score_samples {
	my ( $self, $data ) = @_;
	$self->_check_fitted;
	my $c     = $self->{c_psi};
	my $trees = $self->{trees};
	my $t     = scalar @$trees;

	if ( $HAS_C && $self->{_c_nodes} ) {
		my $n_pts   = scalar @$data;
		my $nf      = $self->{n_features};
		my $x_packed    = "\0" x ( $n_pts * $nf * 8 );
		pack_input_xs( $data, $x_packed, $n_pts, $nf );
		my $sums_packed = pack( 'd*', (0.0) x $n_pts );
		my $c_nodes = $self->{_c_nodes};
		my $c_coefs = $self->{_c_coefs};
		for my $ti ( 0 .. $t - 1 ) {
			score_tree_xs( $c_nodes->[$ti], $c_coefs->[$ti],
				$x_packed, $sums_packed, $n_pts, $nf );
		}
		my @sums = unpack( 'd*', $sums_packed );
		if ( $c > 0 ) {
			my $inv = log(2) / ( $c * $t );
			return [ map { exp( -$_ * $inv ) } @sums ];
		}
		return [ (0.5) x @sums ];
	}

	# Pure-Perl fallback (tree-outer, sample-inner for cache locality).
	my @sums = (0) x @$data;
	for my $tree (@$trees) {
		for my $i ( 0 .. $#$data ) {
			$sums[$i] += _path_length( $data->[$i], $tree, 0 );
		}
	}

	# Precompute the single normalising factor; exp() is a direct FPU
	# instruction and faster than Perl's general-purpose 2**x (pow).
	# Derivation: 2**(-avg/c) = 2**(-(sum/t)/c) = exp(-sum * log(2)/(c*t))
	if ( $c > 0 ) {
		my $inv = log(2) / ( $c * $t );
		return [ map { exp( -$_ * $inv ) } @sums ];
	}
	return [ (0.5) x @sums ];
} ## end sub score_samples

=head2 score_predict_samples

Returns a array ref of arrays. First value of each sub array is the score with the second being
0/1 for if it is a anomaly or not.

    my $results = $forest->predict(\@data, $threshold);

    print "x, y, score, result\n";

    my $int=0;
    while (defined($data[$int])) {
        print $data[$int][0].', '.$data[$int][1].', '.$results->[$int][0].', '.$results->[$int][1]."\n";

        $int++;
    }

=cut

sub score_predict_samples {
	my ( $self, $data, $threshold ) = @_;
	$threshold
		= defined $threshold         ? $threshold
		: defined $self->{threshold} ? $self->{threshold}
		:                              0.5;
	my $scores = $self->score_samples($data);

	my @to_return;
	foreach my $score ( @{$scores} ) {
		if ( $score >= $threshold ) {
			push @to_return, [ $score, 1 ];
		} else {
			push @to_return, [ $score, 0 ];
		}
	}

	return \@to_return;
} ## end sub score_predict_samples

=head1 MODEL SAVE/LOAD METHODS

=head2 to_json

Returns a JSON representation of the module.

Required being fit having to be called.

    my $json = $iforest->to_json;

=cut

sub to_json {
	my ($self) = @_;
	$self->_check_fitted;
	my $payload = {
		format  => 'Algorithm::Classifier::IsolationForest',
		version => 1,
		params  => {
			n_trees         => $self->{n_trees},
			sample_size     => $self->{sample_size},
			mode            => $self->{mode},
			extension_level => $self->{extension_level_used},
			contamination   => $self->{contamination},
			threshold       => $self->{threshold},
			n_features      => $self->{n_features},
			psi_used        => $self->{psi_used},
			c_psi           => $self->{c_psi},
			max_depth_used  => $self->{max_depth_used},
		},
		trees => $self->{trees},
	};
	return JSON::PP->new->canonical(1)->encode($payload);
} ## end sub to_json

=head2 from_json($json)

Init the object from the model in the specified JSON string.

    my $iforest = Algorithm::Classifier::IsolationForest->from_json($json);

=cut

sub from_json {
	my ( $class, $text ) = @_;
	my $payload = JSON::PP->new->decode($text);
	croak "not an IsolationForest model"
		unless ref $payload eq 'HASH'
		&& defined $payload->{format}
		&& $payload->{format} eq 'Algorithm::Classifier::IsolationForest';

	my $p = $payload->{params} || {};

	# version 0 used hash-based nodes; version 1+ uses array-based nodes.
	# Convert old models on load so the rest of the code only sees arrays.
	my $trees = $payload->{trees} || [];
	if ( ( $payload->{version} // 0 ) < 1 ) {
		$trees = [ map { _hash_node_to_array($_) } @$trees ];
	}

	my $self = {
		n_trees              => $p->{n_trees},
		sample_size          => $p->{sample_size},
		max_depth            => undef,
		seed                 => undef,
		mode                 => $p->{mode} // 'axis',
		extension_level      => $p->{extension_level},
		extension_level_used => $p->{extension_level},
		contamination        => $p->{contamination},
		threshold            => $p->{threshold},
		n_features           => $p->{n_features},
		psi_used             => $p->{psi_used},
		c_psi                => $p->{c_psi},
		max_depth_used       => $p->{max_depth_used},
		trees                => $trees,
	};
	croak "model contains no trees" unless @{ $self->{trees} };

	# Recompute the normalising constant from the (integer, exact) sub-sample
	# size rather than trusting the stored float, so a reloaded model's scores
	# are bit-for-bit identical to the original's.
	$self->{c_psi} = _c( $self->{psi_used} ) if defined $self->{psi_used};

	my $model = bless $self, $class;
	$model->_rebuild_c_trees() if $HAS_C;
	return $model;
} ## end sub from_json

=head2 save($path)

Saves the model to the specified path.

    $iforest->save($path);

=cut

sub save {
	my ( $self, $path ) = @_;
	write_file( $path, { 'atomic' => 1 }, $self->to_json );
}

=head2 load($path);

Init the object from the model in the specified file.

    my $iforest = Algorithm::Classifier::IsolationForest->load($path);

=cut

sub load {
	my ( $class, $path ) = @_;
	my $raw_model = read_file($path);
	return $class->from_json($raw_model);
}

=head1 REFERENCES

Liu, Fei Tony & Ting, Kai & Zhou, Zhi-Hua. (2008). Isolation Forest. 413 - 422. 10.1109/ICDM.2008.17.

L<https://www.researchgate.net/publication/224384174_Isolation_Forest>

L<https://ieeexplore.ieee.org/abstract/document/4781136>

Sahand Hariri, Matias Carrasco Kind, Robert J. Brunner (2020). Extended Isolation Forest. 1479 - 1489. 10.1109/TKDE.2019.2947676

L<https://ieeexplore.ieee.org/document/8888179>

=cut

###
###
### internal stuff below
###
###

#-------------------------------------------------------------------------------
# c(n): the expected path length of an unsuccessful search in a binary search
# tree of n nodes. Isolation Forest uses it (a) to adjust the path length when a
# leaf still holds more than one point (depth limit reached), and (b) to
# normalise the average path length into a 0..1 anomaly score.
#-------------------------------------------------------------------------------
sub _c {
	my ($n) = @_;
	return 0.0 if $n <= 1;
	return 1.0 if $n == 2;
	my $harmonic = log( $n - 1 ) + EULER;    # H(n-1) ~= ln(n-1) + gamma
	return 2.0 * $harmonic - ( 2.0 * ( $n - 1 ) / $n );
}

# One draw from the standard normal N(0,1) via Box-Muller. Used to pick the
# random hyperplane orientations in Extended Isolation Forest mode.
sub _randn {
	my $u1 = rand() || 1e-12;
	my $u2 = rand();
	return sqrt( -2.0 * log($u1) ) * cos( TWO_PI * $u2 );
}

#-------------------------------------------------------------------------------
# Draw $k samples without replacement via a partial Fisher-Yates shuffle of the
# index array. Returns an arrayref of (shared, read-only) sample refs.
#-------------------------------------------------------------------------------
sub _subsample {
	my ( $data, $k ) = @_;
	my $n   = scalar @$data;
	my @idx = ( 0 .. $n - 1 );
	for my $i ( 0 .. $k - 1 ) {
		my $j = $i + int( rand( $n - $i ) );
		@idx[ $i, $j ] = @idx[ $j, $i ];
	}
	my @chosen = @idx[ 0 .. $k - 1 ];
	return [ @{$data}[@chosen] ];
} ## end sub _subsample

#-------------------------------------------------------------------------------
# Recursively build one isolation tree.
#
# A node is one of:
#   leaf     { leaf => 1, size => N }
#   axis     { attr => A, split => S,            left => ..., right => ... }
#   oblique  { idx => [..], coef => [..], b => B, left => ..., right => ... }
#
# In both split styles the choice is restricted to features that actually vary
# across the points reaching the node: this avoids wasted levels on constant
# columns and lets a node leaf out exactly when its points are indistinguishable.
#-------------------------------------------------------------------------------
sub _build_tree {
	my ( $self, $X, $depth, $limit ) = @_;

	my $size = scalar @$X;
	return [ _NODE_LEAF, $size ]
		if $depth >= $limit || $size <= 1;

	my $nf = $self->{n_features};

	# Per-feature min and max within this node, in a single pass.
	my ( @lo, @hi );
	for my $row (@$X) {
		for my $f ( 0 .. $nf - 1 ) {
			my $v = $row->[$f];
			$lo[$f] = $v if !defined $lo[$f] || $v < $lo[$f];
			$hi[$f] = $v if !defined $hi[$f] || $v > $hi[$f];
		}
	}

	# Features with spread are the only ones that can split the data.
	my @varying = grep { $lo[$_] < $hi[$_] } 0 .. $nf - 1;

	# No spread on any feature => all points identical => cannot isolate.
	return [ _NODE_LEAF, $size ] unless @varying;

	my $node
		= $self->{mode} eq 'extended'
		? $self->_oblique_split( $X, \@varying, \@lo, \@hi )
		: _axis_split( $X, \@varying, \@lo, \@hi );

	# Split functions leave the raw point arrays at the child slots so that
	# _build_tree can recurse into them; the subtree refs replace them in-place.
	# Axis nodes:   left at [3], right at [4]
	# Oblique nodes: left at [4], right at [5]
	my ( $li, $ri ) = $node->[0] == _NODE_AXIS ? ( 3, 4 ) : ( 4, 5 );
	$node->[$li] = $self->_build_tree( $node->[$li], $depth + 1, $limit );
	$node->[$ri] = $self->_build_tree( $node->[$ri], $depth + 1, $limit );

	return $node;
} ## end sub _build_tree

# Axis-parallel cut: random varying feature, random threshold in its range.
# Returns [_NODE_AXIS, attr, split, \@left_pts, \@right_pts].
# _build_tree overwrites slots 3 and 4 with the recursed subtrees.
sub _axis_split {
	my ( $X, $varying, $lo, $hi ) = @_;

	my $attr  = $varying->[ int( rand( scalar @$varying ) ) ];
	my $split = $lo->[$attr] + rand() * ( $hi->[$attr] - $lo->[$attr] );

	my ( @left, @right );
	for my $row (@$X) {
		if   ( $row->[$attr] < $split ) { push @left,  $row }
		else                            { push @right, $row }
	}
	return [ _NODE_AXIS, $attr, $split, \@left, \@right ];
} ## end sub _axis_split

# Oblique cut (Extended Isolation Forest): a random hyperplane. We activate
# (extension_level + 1) of the varying features, give each a Gaussian
# coefficient, and place the plane through a random point in the bounding box.
# A point goes left when coef . x <= b, where b = coef . p.
# Returns [_NODE_OBLIQUE, \@idx, \@coef, $b, \@left_pts, \@right_pts].
# _build_tree overwrites slots 4 and 5 with the recursed subtrees.
sub _oblique_split {
	my ( $self, $X, $varying, $lo, $hi ) = @_;

	my $active = $self->{extension_level_used} + 1;
	$active = scalar @$varying if $active > scalar @$varying;

	# Pick which varying features take part (partial shuffle of their indices).
	my @pool = @$varying;
	for my $i ( 0 .. $active - 1 ) {
		my $j = $i + int( rand( scalar(@pool) - $i ) );
		@pool[ $i, $j ] = @pool[ $j, $i ];
	}
	my @idx = @pool[ 0 .. $active - 1 ];

	my ( @coef, $b );
	$b = 0.0;
	for my $f (@idx) {
		my $c = _randn();
		my $p = $lo->[$f] + rand() * ( $hi->[$f] - $lo->[$f] );    # point in the box
		push @coef, $c;
		$b += $c * $p;
	}

	my ( @left, @right );
	for my $row (@$X) {
		my $dot = 0.0;
		$dot += $coef[$_] * $row->[ $idx[$_] ] for 0 .. $#idx;
		if   ( $dot <= $b ) { push @left,  $row }
		else                { push @right, $row }
	}
	return [ _NODE_OBLIQUE, \@idx, \@coef, $b, \@left, \@right ];
} ## end sub _oblique_split

#-------------------------------------------------------------------------------
# Path length of a single point in a single tree: edges traversed until a leaf,
# plus c(leaf size) when the leaf still holds several points.
#
# Node layout (arrayref, slot 0 = type):
#   _NODE_LEAF    [0, size]
#   _NODE_AXIS    [1, attr, split, left, right]
#   _NODE_OBLIQUE [2, \@idx, \@coef, b, left, right]
#
# The type tag is also used as a loop sentinel: 0 (_NODE_LEAF) is falsy.
# No $self argument -- the node type encodes everything needed.
#-------------------------------------------------------------------------------
sub _path_length {
	my ( $x, $node, $depth ) = @_;
	while ( $node->[0] ) {                       # false only for leaf (type 0)
		if ( $node->[0] == _NODE_AXIS ) {        # [1, attr, split, left, right]
			$node = ( $x->[ $node->[1] ] // 0 ) < $node->[2]
				? $node->[3] : $node->[4];
		} else {                                 # [2, \@idx, \@coef, b, left, right]
			my ( $idx, $coef, $b ) = ( $node->[1], $node->[2], $node->[3] );
			my $dot = 0.0;
			$dot += $coef->[$_] * ( $x->[ $idx->[$_] ] // 0 ) for 0 .. $#$idx;
			$node = $dot <= $b ? $node->[4] : $node->[5];
		}
		$depth++;
	}
	return $depth + _c( $node->[1] );            # leaf size at slot 1
} ## end sub _path_length

# Recursively convert a version-0 hash-based tree node to the version-1
# array format.  Called by from_json when loading an old saved model.
sub _hash_node_to_array {
	my ($node) = @_;
	if ( $node->{leaf} ) {
		return [ _NODE_LEAF, $node->{size} ];
	} elsif ( exists $node->{attr} ) {
		return [
			_NODE_AXIS,
			$node->{attr},
			$node->{split},
			_hash_node_to_array( $node->{left} ),
			_hash_node_to_array( $node->{right} ),
		];
	} else {
		return [
			_NODE_OBLIQUE,
			$node->{idx},
			$node->{coef},
			$node->{b},
			_hash_node_to_array( $node->{left} ),
			_hash_node_to_array( $node->{right} ),
		];
	}
} ## end sub _hash_node_to_array

# ---------------------------------------------------------------------------
# _pack_tree($root) -- flatten one tree into two packed double strings.
#
# Returns ($nodes_packed, $coefs_packed) where:
#   nodes_packed: 6 doubles per node (see score_tree_xs comment above)
#   coefs_packed: pairs (feat_idx, coef_val) for oblique nodes
#
# Nodes are numbered in DFS pre-order: the root is always index 0 and
# children always get indices larger than their parent's.
# ---------------------------------------------------------------------------
sub _pack_tree {
	my ($root) = @_;
	my ( @node_data, @coefs );

	my $assign;
	$assign = sub {
		my ($node) = @_;
		my $my_idx = scalar @node_data;
		push @node_data, undef;    # reserve slot; filled in after children

		if ( $node->[0] == _NODE_LEAF ) {
			$node_data[$my_idx] = [ 0.0, $node->[1] + 0.0, 0.0, 0.0, 0.0, 0.0 ];
		}
		elsif ( $node->[0] == _NODE_AXIS ) {
			my $li = $assign->( $node->[3] );
			my $ri = $assign->( $node->[4] );
			$node_data[$my_idx] = [
				1.0,
				$node->[1] + 0.0,    # attr
				$node->[2] + 0.0,    # split
				$li + 0.0,
				$ri + 0.0,
				0.0,
			];
		}
		else {                       # _NODE_OBLIQUE
			my ( $idx_arr, $coef_arr, $b ) = ( $node->[1], $node->[2], $node->[3] );
			my $coef_off = scalar(@coefs) / 2;
			my $num      = scalar @$idx_arr;
			for my $i ( 0 .. $num - 1 ) {
				push @coefs, $idx_arr->[$i] + 0.0, $coef_arr->[$i] + 0.0;
			}
			my $li = $assign->( $node->[4] );
			my $ri = $assign->( $node->[5] );
			$node_data[$my_idx] = [
				2.0,
				$coef_off + 0.0,
				$num + 0.0,
				$li + 0.0,
				$ri + 0.0,
				$b + 0.0,
			];
		}
		return $my_idx;
	};
	$assign->($root);

	my $nodes_packed = pack( 'd*', map { @$_ } @node_data );
	my $coefs_packed = @coefs ? pack( 'd*', @coefs ) : pack( 'd*' );
	return ( $nodes_packed, $coefs_packed );
} ## end sub _pack_tree

# Build packed C-ready representations for all trees and store them in
# $self->{_c_nodes} and $self->{_c_coefs}.  Called after fit() and from_json()
# when $HAS_C is true.
sub _rebuild_c_trees {
	my ($self) = @_;
	my ( @c_nodes, @c_coefs );
	for my $tree ( @{ $self->{trees} } ) {
		my ( $np, $cp ) = _pack_tree($tree);
		push @c_nodes, $np;
		push @c_coefs, $cp;
	}
	$self->{_c_nodes} = \@c_nodes;
	$self->{_c_coefs} = \@c_coefs;
} ## end sub _rebuild_c_trees

sub _check_fitted {
	my ($self) = @_;
	croak "model is not fitted yet; call fit() first"
		unless ref $self->{trees} eq 'ARRAY' && @{ $self->{trees} };
}

1;
