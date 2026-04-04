#!/bin/bash
# Decompile all Gemma 4 family models via ollama
# Requires: ollama with gemma3:4b gemma4 gemma4:26b gemma4:31b pulled
# Build: cc -O2 -o decompile decompile.c
set -e
cd "$(dirname "$0")"
[ -f decompile ] || cc -O2 -o decompile decompile.c

blob() { ollama show "$1" --modelfile 2>/dev/null | grep "^FROM" | awk '{print $2}'; }

mkdir -p results
for model in "gemma3:4b" "gemma4" "gemma4:26b" "gemma4:31b"; do
    b=$(blob "$model")
    out="results/$(echo "$model" | tr ':' '_').txt"
    [ -f "$b" ] || { echo "SKIP $model (not pulled)"; continue; }
    echo "Decompiling $model ($(du -h "$b" | cut -f1))..."
    # Unload model to free RAM for mmap
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$model\",\"keep_alive\":0}" > /dev/null 2>&1
    sleep 1
    ./decompile "$b" > "$out" 2>&1
    echo "  -> $out ($(wc -l < "$out") lines)"
done
echo "Done. Results in results/"
