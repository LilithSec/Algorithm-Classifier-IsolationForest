#!perl
# 80-sklearn-comparison-undef.t
#
# Verifies consistent handling of undef (Perl) / NaN (Python) in one feature
# column during scoring and prediction.
#
# Perl coerces undef to 0 in numeric comparisons, so score_samples and
# predict on data with an undef column are bit-for-bit identical to the same
# calls with an explicit 0 in that column.  The Python side uses
# numpy.where(isnan, 0, x) to apply the same substitution before scoring.
#
# Subtests 1 and 2 are pure Perl and always run:
#   1. score_samples([x, undef]) == score_samples([x, 0])  -- exact equality
#   2. predict([x, undef])       == predict([x, 0])        -- exact equality
#
# Subtests 3 and 4 cross-check against scikit-learn (skipped if unavailable):
#   3. Spearman rho between Perl(undef→0) and sklearn(NaN→0) scores >= 0.90
#   4. Both implementations still rank the x-axis outliers above the inliers
#      after the y-column is erased.

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

sub _assign_ranks {
    my @v   = @_;
    my @idx = sort { $v[$a] <=> $v[$b] } 0 .. $#v;
    my @r;
    $r[ $idx[$_] ] = $_ + 1 for 0 .. $#idx;
    return @r;
}

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
# Training dataset: same 2-D grid + outliers as the main sklearn test
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
my @train = ( @inliers, @outliers );

# -----------------------------------------------------------------------
# Test points: column 1 (y) is undef / NaN; only column 0 (x) carries
# signal.  Two groups:
#   inlier-like  -- x well inside [-1, 1]; with y=0 they land in the
#                   training cluster.
#   outlier-like -- |x| >= 6; with y=0 they are still far outside the
#                   cluster along the x-axis.
# -----------------------------------------------------------------------
my @undef_test = (
    ( map { [ $_ * 0.1, undef ] } -9 .. 9 ),              # 19 inlier-like
    ( map { [ $_, undef ] } ( 6, 7, 8, -6, -7, -8 ) ),    # 6  outlier-like
);
my @zero_test = map { [ $_->[0], 0.0 ] } @undef_test;

my $n_in_test  = 19;
my $n_out_test = 6;

# -----------------------------------------------------------------------
# Train Perl model
# -----------------------------------------------------------------------
my $f = $CLASS->new( n_trees => 100, sample_size => 256, seed => 42 );
$f->fit( \@train );

# -----------------------------------------------------------------------
# Subtest 1: score_samples -- undef column is bit-for-bit identical to 0
# -----------------------------------------------------------------------
subtest 'Perl score_samples: undef column gives identical scores to explicit 0' => sub {
    my ( $s_undef, $s_zero );
    {
        local $SIG{__WARN__} = sub { };    # suppress "uninitialized value" warnings
        $s_undef = $f->score_samples( \@undef_test );
    }
    $s_zero = $f->score_samples( \@zero_test );

    is( scalar @$s_undef, scalar @$s_zero, 'same number of scores returned' );

    my $diffs = grep { $s_undef->[$_] != $s_zero->[$_] } 0 .. $#$s_undef;
    is( $diffs, 0,
        'every score with undef column is bit-for-bit identical to score with explicit 0'
    );
};

# -----------------------------------------------------------------------
# Subtest 2: predict -- undef column gives identical labels to 0
# -----------------------------------------------------------------------
subtest 'Perl predict: undef column gives identical labels to explicit 0' => sub {
    my ( $l_undef, $l_zero );
    {
        local $SIG{__WARN__} = sub { };
        $l_undef = $f->predict( \@undef_test );
    }
    $l_zero = $f->predict( \@zero_test );

    is( scalar @$l_undef, scalar @$l_zero, 'same number of labels returned' );

    my $diffs = grep { $l_undef->[$_] != $l_zero->[$_] } 0 .. $#$l_undef;
    is( $diffs, 0,
        'every predict label with undef column is identical to label with explicit 0'
    );
};

# -----------------------------------------------------------------------
# Locate Python + scikit-learn; finish gracefully if absent
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
    note 'Python with scikit-learn not found; skipping cross-language subtests';
    done_testing;
    exit 0;
}

# -----------------------------------------------------------------------
# Python helper: train on the first $n_train CSV rows (clean data), then
# score the remaining rows with NaN → 0 imputation in every column.
# Emits JSON {"scores": [...]} on stdout.
# -----------------------------------------------------------------------
my $py_script = <<'END_PY';
import sys, json
import numpy as np
from sklearn.ensemble import IsolationForest

rows = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    row = []
    for tok in line.split(','):
        tok = tok.strip()
        row.append(float('nan') if tok.lower() in ('nan', 'undef', '') else float(tok))
    rows.append(row)

n_train = int(sys.argv[1])
X_train = np.array(rows[:n_train], dtype=float)
X_test  = np.array(rows[n_train:], dtype=float)

# Impute NaN → 0 (mirrors Perl's undef-to-0 numeric coercion)
X_test_clean = np.where(np.isnan(X_test), 0.0, X_test)

psi = min(256, len(X_train))
clf = IsolationForest(
    n_estimators=100,
    max_samples=psi,
    contamination='auto',
    random_state=42,
)
clf.fit(X_train)

# score_samples: lower value = more anomalous (opposite sign from Perl)
scores = clf.score_samples(X_test_clean).tolist()
print(json.dumps({"scores": scores}))
END_PY

my ( $py_fh, $py_path ) = tempfile( SUFFIX => '.py', UNLINK => 1 );
print $py_fh $py_script;
close $py_fh;

# Build the combined CSV: training rows first, then test rows.
# undef is encoded as "nan" so Python can parse it.
my ( $csv_fh, $csv_path ) = tempfile( SUFFIX => '.csv', UNLINK => 1 );
for my $row (@train) {
    print $csv_fh join( ',', @$row ) . "\n";
}
for my $row (@undef_test) {
    print $csv_fh join( ',', map { defined $_ ? $_ : 'nan' } @$row ) . "\n";
}
close $csv_fh;

my $n_train  = scalar @train;
my $raw      = `$python_bin "$py_path" $n_train < "$csv_path" 2>/dev/null`;
my $py       = eval { JSON::PP->new->decode($raw) };

unless ( defined $py && ref $py eq 'HASH' && ref $py->{scores} eq 'ARRAY' ) {
    note 'Python/sklearn script did not return usable output; skipping cross-language subtests';
    done_testing;
    exit 0;
}

my $sk_scores = $py->{scores};    # lower = more anomalous in sklearn

# Perl scores for the same test points (undef → 0 coercion)
my $perl_scores;
{
    local $SIG{__WARN__} = sub { };
    $perl_scores = $f->score_samples( \@undef_test );
}

# -----------------------------------------------------------------------
# Subtest 3: Spearman rank correlation between Perl(undef→0) and
#            sklearn(NaN→0) on the same 25 test points
# -----------------------------------------------------------------------
subtest 'Spearman rank correlation Perl(undef→0) vs sklearn(NaN→0) >= 0.90' => sub {
    # Negate sklearn scores so both axes point in the same direction
    # (higher value = more anomalous) before computing rank correlation.
    my @neg_sk = map { -$_ } @$sk_scores;
    my $rho    = spearman_rho( $perl_scores, \@neg_sk );
    cmp_ok( $rho, '>=', 0.90,
        sprintf( 'Spearman rho(Perl, -sklearn) = %.4f (must be >= 0.90)', $rho ) );
};

# -----------------------------------------------------------------------
# Subtest 4: both implementations still clearly separate the x-axis
#            outliers from the inliers after the y-column is erased
# -----------------------------------------------------------------------
subtest 'both agree: x-axis outliers still flagged after y-column erasure' => sub {
    my @perl_in  = @{$perl_scores}[ 0 .. $n_in_test - 1 ];
    my @perl_out = @{$perl_scores}[ $n_in_test .. $n_in_test + $n_out_test - 1 ];

    cmp_ok( mean(@perl_out), '>', mean(@perl_in) + 0.2,
        'Perl: mean outlier score (undef y) exceeds mean inlier score by at least 0.2' );
    cmp_ok( min(@perl_out), '>', max(@perl_in),
        'Perl: every x-axis outlier scores strictly higher than every inlier (undef y)' );

    my @sk_in  = @{$sk_scores}[ 0 .. $n_in_test - 1 ];
    my @sk_out = @{$sk_scores}[ $n_in_test .. $n_in_test + $n_out_test - 1 ];

    cmp_ok( mean(@sk_out), '<', mean(@sk_in),
        'sklearn: mean outlier score (NaN y) is lower (more anomalous) than mean inlier score' );
    cmp_ok( max(@sk_out), '<', min(@sk_in),
        'sklearn: every x-axis outlier scores strictly lower than every inlier (NaN y)' );
};

done_testing;
