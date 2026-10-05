#!/bin/bash
set -u
cd ~/projects/chainfl_v3
mkdir -p results/raw_abl logs
cat jobs_abl.txt | xargs -P 6 -I{} bash -c '
  line="{}"
  out=$(echo "$line" | sed "s/.*--out \([^ ]*\).*/\1/")
  [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
  n=$(( $(date +%s) % 2 ))
  f=$(basename "$out" .csv)
  CUDA_VISIBLE_DEVICES=$n bash -c "$line" >> logs/abl_${f}.log 2>&1 \
    || echo "FAILED: $line" >> logs/abl.failed
  echo "DONE $f" >> logs/abl.status
'
echo "abl finished: $(ls results/raw_abl/*.csv 2>/dev/null | grep -v "_chain.csv" | grep -v "_clients.csv" | wc -l)"
