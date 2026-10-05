#!/bin/bash
set -u
cd ~/projects/chainfl_v3
mkdir -p results/raw_cifar logs
cat jobs_cifar.txt | xargs -P 6 -I{} bash -c '
  line="{}"
  out=$(echo "$line" | sed "s/.*--out \([^ ]*\).*/\1/")
  [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
  n=$(( $(date +%s) % 2 ))
  f=$(basename "$out" .csv)
  CUDA_VISIBLE_DEVICES=$n bash -c "$line" >> logs/cifar_${f}.log 2>&1 \
    || echo "FAILED: $line" >> logs/cifar.failed
  echo "DONE $f" >> logs/cifar.status
'
echo "cifar finished: $(ls results/raw_cifar/*.csv 2>/dev/null | grep -v chain | wc -l)/115"
