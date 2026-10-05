#!/bin/bash
# b5 runner: parallel over 2 GPUs (6 workers), resumable (skips existing CSVs).
set -u
cd ~/projects/chainfl_v3
mkdir -p results/raw_b5 logs
cat jobs_b5.txt | xargs -P 6 -I{} bash -c '
  line="{}"
  out=$(echo "$line" | sed "s/.*--out \([^ ]*\).*/\1/")
  [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
  n=$(( $(date +%s) % 2 ))
  f=$(basename "$out" .csv)
  CUDA_VISIBLE_DEVICES=$n bash -c "$line" >> logs/b5_${f}.log 2>&1 \
    || echo "FAILED: $line" >> logs/b5.failed
  echo "DONE $f" >> logs/b5.status
'
echo "b5 finished: $(ls results/raw_b5/*.csv 2>/dev/null | grep -v chain | wc -l) csv, failed=$(cat logs/b5.failed 2>/dev/null | wc -l)"
