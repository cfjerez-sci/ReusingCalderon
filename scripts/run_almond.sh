#!/usr/bin/env bash
#
# The whole NASA almond campaign, in order, from the project root.
#
#   ./scripts/run_almond.sh --check        preflight only, seconds
#   ./scripts/run_almond.sh                stages 1-7
#   ./scripts/run_almond.sh --only 4       one stage
#   NFRESH=3 ./scripts/run_almond.sh       fresh on 3 of 10 seeds: 8.3 h not 11.9
#   ELL=d4 ./scripts/run_almond.sh         after re-exporting at ell = d/4
#
# Every driver appends to its CSV and skips samples already recorded, so this
# script is resumable: rerun it after an interruption and it continues. It
# stops at the first failing stage rather than burning hours on a broken
# premise, and each stage's output is teed to data/sweep/logs/.
#
# Leaving it running unattended:
#   nohup ./scripts/run_almond.sh > run_almond.out 2>&1 &
#   tail -f run_almond.out
#
set -u -o pipefail

JULIA=${JULIA:-julia}
JLFLAGS=${JLFLAGS:-+1.11 --project}
ELL=${ELL:-d5}                 # primary correlation length; see export_for_julia.py
ELL2=${ELL2:-d10}              # sensitivity correlation length
NFRESH=${NFRESH:-10}           # fresh solves per 10-seed cell in the sweep
NCAMP=${NCAMP:-100}            # Monte Carlo sample count
NCFRESH=${NCFRESH:-20}         # of those, also solved fresh

cd "$(dirname "$0")/.."
ROOT=$(pwd)
LOGS="$ROOT/data/sweep/logs"
mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d-%H%M%S)

ONLY=""
CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --only)  ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
secs_hm() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }

# ---------------------------------------------------------------- preflight
say "preflight  (ell = $ELL)"
if ! $JULIA $JLFLAGS scripts/p35_almond_preflight.jl --ell "$ELL" \
        --n "$NCAMP" 2>&1 | tee "$LOGS/preflight-$STAMP.log"; then
  echo "preflight failed -- nothing was run. See $LOGS/preflight-$STAMP.log" >&2
  exit 1
fi
if [ "$CHECK_ONLY" = 1 ]; then
  echo "preflight only; stopping here."
  exit 0
fi

# Disk: the lambda/20 stage caches two 7269^2 complex matrices per sample.
FREE_GB=$(df -Pg "$ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 40 ]; then
  echo "WARNING: only ${FREE_GB} GB free; stage 5 (lambda/20) caches several" \
       "GB per sample. Consider skipping it with --only, or freeing space." >&2
fi

# ------------------------------------------------------------------- stages
# name | estimated wall clock | command
run_stage () {
  local n="$1" name="$2" est="$3"; shift 3
  if [ -n "$ONLY" ] && [ "$ONLY" != "$n" ]; then return 0; fi
  local log="$LOGS/stage$n-$STAMP.log"
  say "stage $n: $name   (estimated $est)"
  echo "   log: $log"
  local t0=$(date +%s)
  if "$@" 2>&1 | tee "$log"; then
    local dt=$(( $(date +%s) - t0 ))
    printf '   stage %s done in %s\n' "$n" "$(secs_hm $dt)"
    return 0
  else
    local dt=$(( $(date +%s) - t0 ))
    echo "   stage $n FAILED after $(secs_hm $dt); see $log" >&2
    echo "   later stages not started. Fix, then rerun -- finished samples" \
         "are skipped." >&2
    return 1
  fi
}

T0=$(date +%s)

run_stage 2 "amplitude sweep, 5 levels x 2 cases x 10 seeds" \
  "$( [ "$NFRESH" -lt 10 ] && echo 2.1h || echo 6.1h )" \
  $JULIA $JLFLAGS scripts/p32_almond_sweep.jl --ell "$ELL" --nfresh "$NFRESH" \
  || exit 1

run_stage 3 "correlation-length sensitivity, levels 1-2" \
  "$( [ "$NFRESH" -lt 10 ] && echo 0.8h || echo 2.5h )" \
  $JULIA $JLFLAGS scripts/p32_almond_sweep.jl --ell "$ELL2" --levels 1,2 \
  --nfresh "$NFRESH" || exit 1

run_stage 4 "Monte Carlo campaign, $NCAMP frozen + $NCFRESH fresh" "1.5h" \
  $JULIA $JLFLAGS scripts/p33_almond_campaign.jl --ell "$ELL" --n "$NCAMP" \
  --nfresh "$NCFRESH" || exit 1

run_stage 5 "mesh check at lambda/20, case U, 3 seeds" "1.4h" \
  $JULIA $JLFLAGS scripts/p32_almond_sweep.jl --mesh almond_lam20 \
  --ell "$ELL" --levels 5 --cases U --seeds 3 --nfresh 3 || exit 1

run_stage 6 "monostatic azimuth sweep" "0.4h" \
  $JULIA $JLFLAGS scripts/p34_almond_monostatic.jl --ell "$ELL" || exit 1

# ------------------------------------------------- tables and figures (cheap)
if [ -z "$ONLY" ] || [ "$ONLY" = 7 ]; then
  say "stage 7: tables and figures"
  PY=${PY:-python3}
  ( cd almond \
    && $PY make_almond_tables.py "$ROOT/data/sweep" > "$ROOT/data/sweep/sec59_results.tex" \
    && $PY make_fig_almond_geometry.py "$ROOT/data/almond" \
         "$ROOT/data/sweep/fig_almond_geometry.pdf" \
    && $PY make_fig_almond_sweep.py \
         "$ROOT/data/sweep/p32_almond_sweep_summary.csv" \
         "$ROOT/data/sweep/fig_almond_sweep.pdf" \
    && $PY make_fig_almond_campaign.py \
         "$ROOT/data/sweep/p33_almond_campaign_summary.csv" \
         "$ROOT/data/sweep/p33_almond_campaign_rcs.csv" \
         "$ROOT/data/sweep/fig_almond_campaign.pdf" ) \
    2>&1 | tee "$LOGS/stage7-$STAMP.log"
  echo "   LaTeX in data/sweep/sec59_results.tex, figures beside it."
  echo "   Copy the figures into the manuscript's figures/ when you are happy"
  echo "   with them; this script does not write into the manuscript."
fi

say "all requested stages complete in $(secs_hm $(( $(date +%s) - T0 )))"
echo "Summaries:"
ls -la "$ROOT"/data/sweep/p3[234]_almond_*.csv 2>/dev/null || true
