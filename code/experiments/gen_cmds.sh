#!/bin/bash
PY="C:/Users/ts/.workbuddy/binaries/python/envs/flbc310/Scripts/python.exe"
RAW="../results/raw"
COMMON="--clients 10 --rounds 25 --local-epochs 2 --lr 0.05 --max-train 12000 --max-test 2000 --data-root ../data"
emit() { local tag=$1; shift; echo "$PY -m experiments.run_fl $COMMON $@ --out $RAW/${tag}.csv --chain-out $RAW/${tag}_chain.csv > $RAW/logs/${tag}.log 2>&1"; }
for ds in mnist fmnist; do for part in iid dirichlet; do for algo in fedavg fedprox scaffold krum chainfl; do for s in 0 1 2; do
  emit "e1_${ds}_${part}_${algo}_s${s}" --algo $algo --dataset $ds --partition $part --alpha 0.5 --seed $s
done; done; done; done
for atk in labelflip scale; do for mr in 0.1 0.2 0.3; do for algo in fedavg krum chainfl; do for s in 0 1; do
  emit "e2_mnist_dir_${algo}_${atk}_m${mr}_s${s}" --algo $algo --dataset mnist --partition dirichlet --alpha 0.5 --attack $atk --malicious-ratio $mr --seed $s
done; done; done; done
for ab in full no_rep no_qual avg; do for s in 0 1; do
  emit "e3_mnist_dir_chainfl_${ab}_s${s}" --algo chainfl --ablate $ab --dataset mnist --partition dirichlet --alpha 0.5 --attack labelflip --malicious-ratio 0.2 --seed $s
done; done
for nc in 10 20 50; do
  emit "e4_overhead_c${nc}" --algo chainfl --dataset mnist --partition iid --clients $nc --rounds 5 --seed 0
done
