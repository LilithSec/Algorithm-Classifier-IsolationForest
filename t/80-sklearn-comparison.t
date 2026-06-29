#!perl
# 80-sklearn-comparison.t
#
# Cross-language validation: trains both this module and Python scikit-learn's
# IsolationForest on the same dataset and verifies that the two implementations
# agree on anomaly ordering.  The whole file is skipped when Python or
# scikit-learn is not installed.
#
# Agreement is verified by three complementary checks:
#   1. Both models clearly separate the obvious outliers from the inliers
#      (score direction test -- Perl: higher = anomalous; sklearn: lower).
#   2. Both models rank the same 8 obvious outliers as the top-8 anomalies.
#   3. The Spearman rank correlation between the two score vectors is >= 0.85.
#
# Because the models use different RNG implementations they cannot produce
# identical floating-point scores, but any faithful Isolation Forest
# implementation produces highly correlated anomaly rankings on
# well-separated data.

use strict;
use warnings;
use Test::More;
use List::Util qw(sum min max);
use File::Temp qw(tempfile);
use JSON::PP   ();

use Algorithm::Classifier::IsolationForest;

my $CLASS = 'Algorithm::Classifier::IsolationForest';

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
sub mean { @_ ? sum(@_) / @_ : 0 }

# Assign 1-based ranks; lower value gets lower rank.
sub _assign_ranks {
    my @v   = @_;
    my @idx = sort { $v[$a] <=> $v[$b] } 0 .. $#v;
    my @r;
    $r[ $idx[$_] ] = $_ + 1 for 0 .. $#idx;
    return @r;
}

# Pearson correlation of two rank vectors (= Spearman rho of the originals).
sub spearman_rho {
    my ( $xs, $ys ) = @_;
    my @rx = _assign_ranks(@$xs);
    my @ry = _assign_ranks(@$ys);
    my $n  = scalar @rx;
    my ( $sa, $sb, $saa, $sbb, $sab ) = (0) x 5;
    for my $i ( 0 .. $n - 1 ) {
        $sa  += $rx[$i];
        $sb  += $ry[$i];
        $saa += $rx[$i]**2;
        $sbb += $ry[$i]**2;
        $sab += $rx[$i] * $ry[$i];
    }
    my ( $ma, $mb ) = ( $sa / $n, $sb / $n );
    my $cov = $sab / $n - $ma * $mb;
    my $da  = sqrt( $saa / $n - $ma**2 );
    my $db  = sqrt( $sbb / $n - $mb**2 );
    return ( $da > 0 && $db > 0 ) ? $cov / ( $da * $db ) : 0;
}

# -----------------------------------------------------------------------
# Dataset: 225 inliers (regular grid in [-1,1]^2) + 8 obvious outliers
# -----------------------------------------------------------------------
my ( @inliers, @outliers );
for my $i ( -7 .. 7 ) {
    for my $j ( -7 .. 7 ) {
        push @inliers, [ $i / 7.0, $j / 7.0 ];
    }
}
@outliers = (
    [ 6, 6 ], [ -6, 6 ], [ 6, -6 ], [ -6, -6 ],
    [ 0, 8 ], [ 8, 0 ],  [ -8, 0 ], [ 0,  -8 ]
);
my @all_data = ( @inliers, @outliers );
my $n_in  = scalar @inliers;    # 225
my $n_out = scalar @outliers;   # 8

# -----------------------------------------------------------------------
# Perl IsolationForest
# -----------------------------------------------------------------------
my $f = $CLASS->new( n_trees => 100, sample_size => 256, seed => 42 );
$f->fit( \@all_data );

my $perl_in_scores  = $f->score_samples( \@inliers );
my $perl_out_scores = $f->score_samples( \@outliers );
my $perl_all_scores = $f->score_samples( \@all_data );

# -----------------------------------------------------------------------
# Locate Python + scikit-learn; skip the whole file if unavailable
# -----------------------------------------------------------------------
my $python_bin;
for my $cmd (qw(python3 python)) {
    my $probe = `$cmd -c "import sklearn; print('ok')" 2>/dev/null`;
    if ( defined $probe && $probe =~ /\bok\b/ ) {
        $python_bin = $cmd;
        last;
    }
}

unless ( defined $python_bin ) {
    plan skip_all =>
        'Python with scikit-learn is not installed; skipping cross-language comparison';
}

# -----------------------------------------------------------------------
# Python helper: train sklearn IsolationForest on stdin CSV, emit JSON
#
# sklearn score_samples convention: lower score = more anomalous.
# That is the opposite direction from this module (higher = more anomalous),
# so we negate sklearn scores before computing rank correlation.
# -----------------------------------------------------------------------
my $py_script = <<'END_PY';
import sys, json
import numpy as np
from sklearn.ensemble import IsolationForest

rows = []
for line in sys.stdin:
    line = line.strip()
    if line:
        rows.append([float(x) for x in line.split(',')])

X = np.array(rows)
psi = min(256, len(X))

clf = IsolationForest(
    n_estimators=100,
    max_samples=psi,
    contamination='auto',
    random_state=42,
)
clf.fit(X)

# score_samples: lower value = more anomalous (opposite sign from Perl)
scores = clf.score_samples(X).tolist()

print(json.dumps({"scores": scores}))
END_PY

my ( $py_fh, $py_path ) = tempfile( SUFFIX => '.py', UNLINK => 1 );
print $py_fh $py_script;
close $py_fh;

my ( $csv_fh, $csv_path ) = tempfile( SUFFIX => '.csv', UNLINK => 1 );
for my $row (@all_data) {
    print $csv_fh join( ',', @$row ) . "\n";
}
close $csv_fh;

my $raw = `$python_bin "$py_path" < "$csv_path" 2>/dev/null`;
my $py  = eval { JSON::PP->new->decode($raw) };

unless ( defined $py && ref $py eq 'HASH' && ref $py->{scores} eq 'ARRAY' ) {
    plan skip_all => 'Python/sklearn script returned unusable output; skipping';
}

my $sk_all  = $py->{scores};
my @sk_in   = @{$sk_all}[ 0 .. $n_in - 1 ];
my @sk_out  = @{$sk_all}[ $n_in .. $n_in + $n_out - 1 ];

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

subtest 'Perl: outliers score clearly higher than inliers' => sub {
    cmp_ok( mean( @$perl_out_scores ), '>', mean( @$perl_in_scores ) + 0.2,
        'mean outlier Perl score exceeds mean inlier score by at least 0.2' );
    cmp_ok( min( @$perl_out_scores ), '>', max( @$perl_in_scores ),
        'every outlier has a strictly higher Perl score than every inlier' );
};

subtest 'sklearn: outliers score clearly lower (more anomalous) than inliers' => sub {
    cmp_ok( mean(@sk_out), '<', mean(@sk_in),
        'mean outlier sklearn score is lower (more anomalous) than mean inlier score' );
    cmp_ok( max(@sk_out), '<', min(@sk_in),
        'every outlier has a strictly lower sklearn score than every inlier' );
};

subtest "both models rank all $n_out outliers in the top-$n_out anomalies" => sub {
    # Perl: sort by descending score; highest scores are the most anomalous.
    my @perl_rank = sort { $perl_all_scores->[$b] <=> $perl_all_scores->[$a] }
                    0 .. $#$perl_all_scores;
    my %perl_top = map { $_ => 1 } @perl_rank[ 0 .. $n_out - 1 ];

    # sklearn: sort by ascending score; lowest scores are the most anomalous.
    my @sk_rank = sort { $sk_all->[$a] <=> $sk_all->[$b] }
                  0 .. $#$sk_all;
    my %sk_top  = map { $_ => 1 } @sk_rank[ 0 .. $n_out - 1 ];

    my $perl_caught = grep { $perl_top{$_} } $n_in .. $n_in + $n_out - 1;
    my $sk_caught   = grep { $sk_top{$_}   } $n_in .. $n_in + $n_out - 1;

    is( $perl_caught, $n_out, "Perl top-$n_out contains all $n_out outlier points" );
    is( $sk_caught,   $n_out, "sklearn top-$n_out contains all $n_out outlier points" );
};

subtest 'Perl predict at 0.5 threshold flags all outliers and almost no inliers' => sub {
    my $in_labels  = $f->predict( \@inliers );
    my $out_labels = $f->predict( \@outliers );

    is( sum( @$out_labels ), $n_out,
        "Perl predict() flags all $n_out outliers at the 0.5 threshold" );
    cmp_ok( sum( @$in_labels ), '<', 0.05 * $n_in,
        'fewer than 5% of inliers are flagged by Perl predict()' );
};

subtest 'Spearman rank correlation between Perl and sklearn scores >= 0.85' => sub {
    # Negate sklearn scores so both vectors point in the same direction
    # (higher value = more anomalous) before ranking.
    my @neg_sk = map { -$_ } @$sk_all;
    my $rho = spearman_rho( $perl_all_scores, \@neg_sk );
    cmp_ok( $rho, '>=', 0.85,
        sprintf( 'Spearman rho(Perl, -sklearn) = %.4f (must be >= 0.85)', $rho ) );
};

done_testing;
