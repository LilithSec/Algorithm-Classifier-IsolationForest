package BenchAccel;

# Shared helper for the bench-*-accel.pl scripts.
#
# Benchmark::cmpthese is unsafe for comparing OpenMP-parallel code
# against serial code: its rate column is computed from CPU time
# (user + sys), and an OpenMP `parallel for` running on N cores
# consumes ~N x the CPU time of its serial counterpart even when
# wall-clock time drops.  That makes the c_openmp variant look
# *slower* than c_serial in cmpthese output -- the opposite of what
# a user actually experiences.
#
# wall_cmpthese measures Time::HiRes wall-clock time instead, so the
# rate column reflects what the user waits for.  Output layout matches
# Benchmark::cmpthese: rows sorted slowest -> fastest, with a pairwise
# speedup matrix showing percent difference from each column variant.

use strict;
use warnings;
use Time::HiRes qw(time);
use Exporter qw(import);

our @EXPORT_OK = qw(wall_cmpthese);

sub wall_cmpthese {
    my ( $target_secs, $vars ) = @_;
    my $target = abs( $target_secs || 0 ) || 1;

    my %res;
    for my $name ( sort keys %$vars ) {
        my $code = $vars->{$name};

        # Warm-up: one call absorbs first-touch and cache-miss spikes
        # so the calibration window measures steady-state cost.
        $code->();

        # Calibrate over 50 ms so the real run lands close to $target s
        # regardless of how fast or slow the variant is.
        my $cal_iters = 0;
        my $cal_t0    = time;
        while ( time - $cal_t0 < 0.05 ) { $code->(); $cal_iters++ }
        my $cal_elapsed = ( time - $cal_t0 ) || 1e-9;
        my $iters
            = int( $cal_iters / $cal_elapsed * $target ) || 1;

        # Real timed run.
        my $t0 = time;
        $code->() for 1 .. $iters;
        my $elapsed = ( time - $t0 ) || 1e-9;
        $res{$name} = { iters => $iters, rate => $iters / $elapsed };
    }

    my @names = sort { $res{$a}{rate} <=> $res{$b}{rate} } keys %res;
    my $name_w = 1;
    for my $n (@names) { $name_w = length $n if length $n > $name_w }
    my $col_w = $name_w < 8 ? 8 : $name_w;

    printf "  %-*s  %10s", $name_w, '', 'Rate';
    printf "  %*s", $col_w, $_ for @names;
    print "\n";

    for my $a (@names) {
        printf "  %-*s  %10s", $name_w, $a, _fmt_rate( $res{$a}{rate} );
        for my $b (@names) {
            if ( $a eq $b ) {
                printf "  %*s", $col_w, '--';
                next;
            }
            my $pct
                = ( $res{$a}{rate} - $res{$b}{rate} )
                / $res{$b}{rate}
                * 100;
            printf "  %*s", $col_w, sprintf( '%+d%%', int($pct) );
        }
        print "\n";
    }
}

sub _fmt_rate {
    my $r = shift;
    return sprintf '%.2g/s', $r if $r < 1;
    return sprintf '%.2f/s', $r if $r < 100;
    return sprintf '%.0f/s',  $r;
}

1;
