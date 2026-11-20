#!/usr/bin/env bash
# The gate, in one place.
#
# CI runs this script and nothing else, stage by stage, so "green locally" and "green on CI" cannot
# drift apart: there is no command here that CI spells differently, and no command in ci.yml that is
# not here. A human runs the same stages by hand.
#
#   scripts/gate.sh            # every stage, in order (what a release needs)
#   scripts/gate.sh lint       # one stage
#   scripts/gate.sh --list     # the stages
#
# `CONSTRAINTS.md` § How to run the checks is the written bar; this is its executable form. If the
# two disagree, the file wins and this script is wrong.
#
# Exit status is the first failing stage's, so a caller can tell a lint red from a test red.
set -euo pipefail

readonly root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# Every stage: name, then the command. Kept as functions so `--list` and the dispatch below read
# from the same definitions.
fmt() { cargo fmt --all --check; }

# `acme` is off by default, so it is named here: code the default build never compiles is code the
# gate never checks.
lint() { cargo clippy --workspace --all-targets --locked --features acme -- -D warnings; }

build() { cargo build --workspace --release --locked; }

test() { cargo test --workspace --locked; }

features() { cargo test -p vipera-server --features acme --locked; }

# The two measurement tests that assert (EXC-004). Release because a debug build cannot time them;
# without a model they print the reason and pass.
measure() {
    cargo test -p vipera-backend-cpu --release --locked \
        --test prefix_cache --test alignment_probe -- --ignored
}

# The CUDA seam: the paged KV layout, the `ForwardPass` implementation and the parity harness. Without
# a CUDA device the device part prints its skip reason and passes; `VIPERA_REQUIRE_GPU=1` turns a
# skip into a failure, which is what a GPU runner sets.
gpu() { cargo test -p vipera-backend-gpu --locked -- --nocapture; }

STAGES=(fmt lint build test features measure gpu)

usage() {
    echo "usage: ${0##*/} [all] | one of: ${STAGES[*]}" >&2
    exit 2
}

case "${1:-all}" in
--list)
    printf '%s\n' "${STAGES[@]}"
    exit 0
    ;;
all)
    for stage in "${STAGES[@]}"; do
        printf '\n=== %s\n' "$stage"
        "$stage"
    done
    ;;
"")
    usage
    ;;
*)
    if [[ " ${STAGES[*]} " != *" $1 "* ]]; then
        usage
    fi
    printf '=== %s\n' "$1"
    "$1"
    ;;
esac
