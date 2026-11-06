#!/usr/bin/env bash
# Compare `vipera tokenize` with llama.cpp's /tokenize on the same model.
#
# Both implementations must agree on every id, because a tokenizer that differs by one merge sends a
# prompt the model was not trained on. The corpus covers the cases where they historically diverged:
# multi-byte scripts, emoji, runs of whitespace, code indentation, and control markers.
#
# Usage:
#   scripts/parity-tokenize.sh MODEL.gguf [--port 18099]
#
# Requires llama-server on PATH, or LLAMA_SERVER=/path/to/llama-server, and a release build of
# `vipera`. Exits non-zero on the first mismatch, on a missing prerequisite, or if the server never
# becomes healthy. Nothing is skipped silently.

set -euo pipefail

MODEL="${1:?usage: parity-tokenize.sh MODEL.gguf [--port N]}"
shift || true
PORT=18099
while [ "$#" -gt 0 ]; do
    case "$1" in
        --port)
            PORT="${2:?--port needs a value}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SERVER="${LLAMA_SERVER:-llama-server}"
if ! command -v "$SERVER" >/dev/null 2>&1; then
    echo "parity: cannot find $SERVER; set LLAMA_SERVER to the llama.cpp server binary" >&2
    exit 1
fi

VIPERA="$(cd "$(dirname "$0")/.." && pwd)/target/release/vipera"
if [ ! -x "$VIPERA" ]; then
    echo "parity: $VIPERA is missing; run: cargo build --release -p vipera" >&2
    exit 1
fi

LOG="$(mktemp)"
"$SERVER" -m "$MODEL" --port "$PORT" -c 512 -ngl 0 >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    rm -f "$LOG"
}
trap cleanup EXIT

HEALTHY=0
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "parity: llama-server exited before becoming healthy:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sleep 1
done
if [ "$HEALTHY" -ne 1 ]; then
    echo "parity: llama-server did not answer /health within 120s" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

python3 - "$MODEL" "$PORT" "$VIPERA" <<'PY'
import json
import subprocess
import sys
import urllib.request

model, port, vipera = sys.argv[1], sys.argv[2], sys.argv[3]

corpus = [
    "hello world",
    "The capital of France is",
    "def f(x):\n    return x * 2\n",
    '{"a": [1, 2, 3], "b": null}',
    "  leading and trailing  ",
    "\t\ttabs\tand\ttabs\t",
    "line one\nline two\r\nline three\n",
    "Tiếng Việt có dấu: Hà Nội, Đà Nẵng, Cần Thơ.",
    "日本語のテキストと中文文本",
    "Русский текст и ελληνικά",
    "emoji 🦀🔥🚀 and symbols ✓ § ±",
    "1234567890 3.14159 -42 1e-9",
    "a" * 200,
    "<|im_start|>user\nhello<|im_end|>\n",
    "</s><s>[INST] hi [/INST]",
    "mixed: don't can't it's — dash “quotes”",
    "\\n is not a newline",
    "   ",
    "=" * 40 + "\n" + "-" * 40,
    "function name(arg) { return { a: 1 }; }",
]


def reference_ids(text):
    payload = json.dumps({"content": text}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=payload,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["tokens"]


failures = 0
for index, text in enumerate(corpus):
    ours = subprocess.run(
        [vipera, "tokenize", model, "--text", text, "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    mine = json.loads(ours.stdout)["ids"]
    theirs = reference_ids(text)
    if mine != theirs:
        failures += 1
        print(f"MISMATCH [{index}] {text!r}")
        print(f"  vipera    ({len(mine)}): {mine[:24]}")
        print(f"  llama.cpp ({len(theirs)}): {theirs[:24]}")
        first = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), None)
        if first is not None:
            print(f"  first difference at index {first}")

print(f"parity: {len(corpus) - failures}/{len(corpus)} inputs tokenize identically")
sys.exit(0 if failures == 0 else 1)
PY
