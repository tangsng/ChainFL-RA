# -*- coding: utf-8 -*-
"""Detection-level metrics for the new suite (raw_v2):
- Table IV: slash precision / recall / honest false-slash rate (post warm-up)
- Table V: cumulative three-zone token balances
- Norm-gate checks:
  (a) scaling adversaries suppressed during warm-up (rounds 1-2 weight share ~ 0)
  (b) honest clients essentially never zero-weighted during warm-up in
      attack-free runs (norm-gate false-flag proxy)
Usage: python scripts/analyze_detection.py [raw_dir]
"""
import csv, glob, os, sys
from collections import defaultdict

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw_v2"

def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

print("=" * 72)
print("TABLE IV  (e2 chainfl clients CSVs, post warm-up rounds >= 3)")
print("=" * 72)
print(f"{'attack':10s} {'ratio':5s} {'precision':>9s} {'recall':>7s} {'honFPR':>7s}")
for atk in ["labelflip", "scale"]:
    for mr in ["0.1", "0.2", "0.3"]:
        sl_mal = sl_hon = tot_mal = tot_hon = 0
        for f in glob.glob(os.path.join(RAW, f"e2_chainfl_{atk}_m{mr}_s*_clients.csv")):
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if int(r["round"]) <= 2:
                        continue
                    mal = int(r["malicious"]); sl = int(r["slashed"])
                    if mal: tot_mal += 1; sl_mal += sl
                    else:   tot_hon += 1; sl_hon += sl
        if tot_mal == 0:
            print(f"{atk:10s} {mr:5s}  MISSING"); continue
        prec = sl_mal / max(1, sl_mal + sl_hon)
        rec = sl_mal / tot_mal
        fpr = sl_hon / max(1, tot_hon)
        print(f"{atk:10s} {mr:5s} {prec:9.3f} {rec:7.3f} {fpr:7.3f}")

print()
print("=" * 72)
print("TABLE V  (cumulative token balance over rounds >= 3, mean per client)")
print("=" * 72)
for atk in ["labelflip", "scale"]:
    group = {"hon": [], "mal": []}
    for f in glob.glob(os.path.join(RAW, f"e2_chainfl_{atk}_m*_clients.csv")):
        rounds = defaultdict(list)
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rounds[int(r["round"])].append(r)
        h_tot = m_tot = 0.0
        for rnd, rows in sorted(rounds.items()):
            if rnd <= 2:
                continue
            mq = med([float(x["quality"]) for x in rows])
            for x in rows:
                d = -2.0 if int(x["slashed"]) else max(0.0, float(x["quality"]) - mq)
                if int(x["malicious"]): m_tot += d
                else: h_tot += d
        n_h = sum(1 for x in rounds[3] if not int(x["malicious"])) if 3 in rounds else 7
        n_m = sum(1 for x in rounds[3] if int(x["malicious"])) if 3 in rounds else 3
        group["hon"].append(h_tot / max(1, n_h))
        group["mal"].append(m_tot / max(1, n_m))
    if group["hon"]:
        print(f"{atk:10s} honest {sum(group['hon'])/len(group['hon']):+6.1f}  "
              f"malicious {sum(group['mal'])/len(group['mal']):+6.1f}  "
              f"(n={len(group['hon'])} runs)")

print()
print("=" * 72)
print("NORM-GATE CHECK (a): scaling adversary weight share during warm-up")
print("=" * 72)
for mr in ["0.1", "0.2", "0.3"]:
    shares = []
    for f in glob.glob(os.path.join(RAW, f"e2_chainfl_scale_m{mr}_s*_clients.csv")):
        rounds = defaultdict(list)
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(r["round"]) <= 2:
                    rounds[int(r["round"])].append(r)
        for rnd, rows in rounds.items():
            tw = sum(float(x["weight"]) for x in rows)
            mw = sum(float(x["weight"]) for x in rows if int(x["malicious"]))
            if tw > 0:
                shares.append(mw / tw)
    if shares:
        print(f"scale {mr}: mean warm-up malicious weight share = "
              f"{sum(shares)/len(shares):.4f} (max {max(shares):.4f}, n={len(shares)})")

print()
print("=" * 72)
print("NORM-GATE CHECK (b): honest zero-weight incidents in attack-free runs")
print("=" * 72)
tot_inc = tot_slots = 0
for f in glob.glob(os.path.join(RAW, "e1_*_chainfl_s*_clients.csv")):
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["round"]) <= 2 and not int(r["malicious"]):
                tot_slots += 1
                if float(r["weight"]) < 1e-6:
                    tot_inc += 1
print(f"warm-up honest client-slots: {tot_slots}, zero-weighted: {tot_inc} "
      f"(rate {tot_inc/max(1,tot_slots):.4f})")
