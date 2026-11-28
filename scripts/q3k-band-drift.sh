#!/usr/bin/env bash
# Regression check for the Q3_K band's cost per prompt chunk.
#
# The band is the largest single phase of a chunk (45% at ~4k, 69% at the start — see
# `docs/benchmarks.md`, "Q3_K at real depths"), and its absolute cost is flat with depth. So one
# 128-token chunk is enough to see a release regress it: the number that has to stay put is
# `other_dot` (Q3_K + Q5_K) milliseconds for the first chunk, not a throughput ratio, because a ratio
# against a decoder or a second arm would not notice both halves getting slower.
#
# It is a same-machine check: the baseline records the CPU it was taken on, and on anything else the
# script prints why it is skipping instead of failing. Run it before a release, or after a change to
# Q3_K, and update the baseline deliberately when the new number is the intended one.
#
#   scripts/q3k-band-drift.sh                 # compare against scripts/q3k-band-baseline.txt
#   VIPERA_DRIFT_TOLERANCE=4 ...              # tighter than the default 8%
#
# The default tolerance is set by the noise floor rather than by taste: two back-to-back runs agreed
# to 0.04% (5657 and 5655 ms), but a run taken while the host was busy read 5972 ms, +5.6%. An 8%
# default therefore catches a real regression without reporting somebody else's compile as one.
#   VIPERA_TEST_MODEL=/path/model.gguf ...    # a different model (the baseline says which)
set -euo pipefail

readonly root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly baseline="$root/scripts/q3k-band-baseline.txt"
tolerance="${VIPERA_DRIFT_TOLERANCE:-8}"
model="${VIPERA_TEST_MODEL:-$HOME/.vipera/models/q8src-imbf-q3km.gguf}"

if [[ ! -f "$model" ]]; then
    echo "skip: no model at $model (set VIPERA_TEST_MODEL)" >&2
    exit 0
fi

read_field() { awk -v key="$1" '$1 == key { print $2 }' "$baseline"; }
base_other="$(read_field other_dot_ms)"
base_chunk="$(read_field chunk_ms)"
base_host="$(sed -n 's/^# host: //p' "$baseline" | head -1)"
base_model="$(sed -n 's/^# model: //p' "$baseline" | head -1)"

host="$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
if [[ "$host" != "$base_host" ]]; then
    echo "skip: baseline was taken on '$base_host', this machine is '$host'" >&2
    exit 0
fi
if [[ "$(basename "$model")" != "$base_model" ]]; then
    echo "skip: baseline was taken on '$base_model', this run uses '$(basename "$model")'" >&2
    exit 0
fi

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

echo "running one 128-token chunk on $base_model ..."
VIPERA_PROFILE_STEP=1 VIPERA_TEST_MODEL="$model" VIPERA_THREADS=8 \
    taskset -c 2-7 cargo test -p vipera-backend-cpu --release --locked \
    --test win_vs_llama prefill_one_chunk -- --ignored --nocapture >"$log" 2>&1 || {
    echo "run failed:" >&2
    tail -20 "$log" >&2
    exit 1
}

line="$(grep -m1 '^PROFILEPASS chunk' "$log" || true)"
if [[ -z "$line" ]]; then
    echo "no PROFILEPASS line: the profiler switch did not take effect" >&2
    exit 1
fi

other="$(sed -E 's/.*other_dot ([0-9.]+)ms.*/\1/' <<<"$line")"
chunk="$(sed -E 's/^PROFILEPASS chunk [0-9]+ ([0-9.]+) ms:.*/\1/' <<<"$line")"

awk -v new="$other" -v base="$base_other" -v tol="$tolerance" \
    -v chunk="$chunk" -v base_chunk="$base_chunk" 'BEGIN {
    drift = (new - base) / base * 100
    drift_abs = drift < 0 ? -drift : drift
    chunk_drift = (chunk - base_chunk) / base_chunk * 100
    printf "other_dot %.0f ms vs baseline %.0f ms (%+.1f%%), chunk %.0f ms vs %.0f ms (%+.1f%%), tolerance %.0f%%\n",
        new, base, drift, chunk, base_chunk, chunk_drift, tol
    if (drift_abs > tol) {
        printf "DRIFT FAIL: the Q3_K band moved %.1f%% against %s\n", drift, "scripts/q3k-band-baseline.txt"
        exit 1
    }
    print "ok: within tolerance"
}'
