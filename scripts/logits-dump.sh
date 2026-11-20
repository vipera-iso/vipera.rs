#!/usr/bin/env bash
# Build scripts/logits-dump.c against a llama.cpp tree and run it.
#
#   scripts/logits-dump.sh ~/.vipera/models/qwen3.gguf 16 12095 13 3555
#
# The llama.cpp tree default is `./llama-latest` (or a sibling checkout of that name), with a
# `build/` that has libllama.so; point `LLAMA_CPP` at another one. The binary is cached in `target/logits-dump`, and a
# run that takes longer than a minute is a 14B decode, not a hang.
set -euo pipefail

model=${1:?usage: $0 <model.gguf> <token> [token ...]}
shift

root=$(cd "$(dirname "$0")/.." && pwd)
llama_cpp=${LLAMA_CPP:-}
if [ -z "$llama_cpp" ]; then
    for candidate in "$root/llama-latest" "$root/../llama-latest"; do
        if [ -f "$candidate/include/llama.h" ]; then
            llama_cpp=$candidate
            break
        fi
    done
fi
if [ -z "$llama_cpp" ] || [ ! -f "$llama_cpp/include/llama.h" ]; then
    echo "no llama.cpp tree found; set LLAMA_CPP to one with include/llama.h" >&2
    exit 1
fi
binary=$root/target/logits-dump

if [ ! -x "$binary" ] || [ "$root/scripts/logits-dump.c" -nt "$binary" ]; then
    cc=${CC:-gcc}
    "$cc" -O2 "$root/scripts/logits-dump.c" \
        -I"$llama_cpp/include" -I"$llama_cpp/ggml/include" \
        -L"$llama_cpp/build/bin" -lllama \
        -Wl,-rpath,"$llama_cpp/build/bin" \
        -o "$binary"
fi

exec "$binary" "$model" "$@"
