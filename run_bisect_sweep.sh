#!/usr/bin/env bash
# Sweep: run ./build/bstar_bisect x 64 --verbose > rat_x.log
# for x = 16, 32, ..., 256 (step 16).
set -u
cd "$(dirname "$0")"

from=${1:-16}
to=${2:-128}
step=${3:-16}

echo "building ..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j "$(nproc)"

echo "sweeping bits from $from to $to in steps of $step (threads=64)"
for x in $(seq "$from" "$step" "$to"); do
    log="rat_${x}.log"
    echo "=== bits=$x -> $log ==="
    start=$(date +%s)
    if ./build/bstar_bisect "$x" 64 --verbose > "$log" 2>&1; then
        dur=$(( $(date +%s) - start ))
        evals=$(grep -m1 "evaluations" "$log" || true)
        approx=$(grep -m1 "B\* lower (decimal)" "$log" || true)
        printf "  [%ds] %s | %s\n" "$dur" "$evals" "$approx"
    else
        echo "  FAILED (bits=$x), see $log"
    fi
done
echo "done."
