# -*- coding: utf-8 -*-
"""Lightweight in-process blockchain simulator for FL audit and incentives.

Models: block structure, SHA-256 hash chaining, per-transaction gas cost,
consensus latency (PoA-style with jitter), and token reward/slash settlement.
All randomness is seeded for reproducibility.
"""
import hashlib
import time

import numpy as np

GAS_PER_BYTE = 16          # calldata-style cost
GAS_BASE_TX = 21000        # base transaction cost
# post-EIP-3529 SSTORE schedule: a fresh storage slot costs 20,000 gas
# (conservative, charged by default); a warm reset of an existing slot costs
# 2,900 gas (optimistic bound, reported alongside in the paper)
GAS_STORE_UPDATE = 20000   # fresh SSTORE slot per client record (conservative)
GAS_STORE_UPDATE_WARM = 2900  # warm reset of an existing slot (optimistic)
BLOCK_INTERVAL_S = 2.0     # target block interval of the simulated PoA chain


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Block:
    def __init__(self, index, prev_hash, records, timestamp):
        self.index = index
        self.prev_hash = prev_hash
        self.records = records
        self.timestamp = timestamp
        payload = f"{index}|{prev_hash}|{timestamp}|" + "|".join(
            str(sorted(r.items())) for r in records)
        self.hash = sha256_hex(payload.encode())

    @property
    def size_bytes(self):
        return len(str(self.records).encode()) + 128


class SimChain:
    """Simulated permissioned chain with gas and latency accounting."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.blocks = [Block(0, "0" * 64, [{"genesis": True}], time.time())]
        self.balances = {}
        self.total_gas = 0
        self.total_latency = 0.0
        self.round_stats = []

    def _gas_for_records(self, n_records):
        return GAS_BASE_TX + n_records * (GAS_STORE_UPDATE + 64 * GAS_PER_BYTE)

    def commit_round(self, rnd: int, model_hash: str, client_records: list,
                     rewards: dict, slashes: dict):
        """Commit one FL round: model hash + per-client audit records."""
        t0 = time.perf_counter()
        records = [{"round": rnd, "model_hash": model_hash}]
        for cr in client_records:
            records.append(cr)
        block = Block(len(self.blocks), self.blocks[-1].hash, records, time.time())
        self.blocks.append(block)
        # token settlement
        for cid, amt in rewards.items():
            self.balances[cid] = self.balances.get(cid, 0.0) + amt
        for cid, amt in slashes.items():
            self.balances[cid] = self.balances.get(cid, 0.0) - amt
        gas = self._gas_for_records(len(records))
        # simulated consensus latency: block interval + uniform jitter + measured build time
        build_ms = (time.perf_counter() - t0) * 1000.0
        latency = BLOCK_INTERVAL_S + float(self.rng.uniform(0.05, 0.35)) + build_ms / 1000.0
        self.total_gas += gas
        self.total_latency += latency
        stat = {
            "round": rnd,
            "block_index": block.index,
            "block_hash": block.hash,
            "block_size_bytes": block.size_bytes,
            "num_records": len(records),
            "gas": gas,
            "chain_latency_s": round(latency, 4),
            "cumulative_gas": self.total_gas,
            "cumulative_chain_bytes": sum(b.size_bytes for b in self.blocks),
        }
        self.round_stats.append(stat)
        return stat

    def summary(self):
        n = max(1, len(self.round_stats))
        return {
            "blocks": len(self.blocks),
            "total_gas": self.total_gas,
            "avg_gas_per_round": self.total_gas / n,
            "total_chain_bytes": sum(b.size_bytes for b in self.blocks),
            "avg_latency_s": self.total_latency / n,
        }
