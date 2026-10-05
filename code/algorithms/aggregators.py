# -*- coding: utf-8 -*-
"""Server-side aggregation rules: FedAvg, Krum, and ChainFL-RA (ours)."""
import numpy as np
import torch

from utils.common import state_to_vec, cosine


def weighted_average(updates, weights):
    """updates: list of state-dict deltas; weights: list of floats (auto-normalized)."""
    w = torch.tensor(weights, dtype=torch.float64)
    w = (w / w.sum()).float()
    agg = {}
    for k in updates[0]:
        agg[k] = sum(wi * u[k] for wi, u in zip(w, updates))
    return agg, w.tolist()


def cap_weights(norm_w, cap):
    """Bounded-influence cap via exact proportional water-filling.

    Finds the capped set C (top-k weights) and a single scale F such that
    w_i = min(raw_i * F, cap) sums to 1. Excess mass flows to uncapped
    clients *in proportion to their current weights*, so suppressed
    (near-zero-weight) clients cannot regain influence from the cap:
    their absolute gain is bounded by F * raw_i, and the total uncapped
    share is bounded by 1 - k*cap. Input must sum to 1.
    """
    w = [float(x) for x in norm_w]
    n = len(w)
    if n == 0 or max(w) <= cap + 1e-12:
        return w
    if cap * n <= 1.0:  # cap below uniform: degenerates to uniform
        return [1.0 / n] * n
    sr = sorted(w, reverse=True)
    k = 0
    F = 1.0
    while k < n:
        rest = sum(sr[k:])
        if rest <= 1e-12:
            break
        F = (1.0 - k * cap) / rest
        if sr[k] * F <= cap + 1e-12:
            break
        k += 1
    out = []
    for wi in w:
        v = min(wi * F, cap)
        out.append(v)
    s = sum(out)
    if s <= 0.0:
        return [1.0 / n] * n
    return [x / s for x in out]


def _aggregate_with_cap(updates, norm_w, cap):
    """Recompute the weighted sum from (possibly capped) normalized weights."""
    agg = {}
    for k in updates[0]:
        agg[k] = sum(wi * u[k] for wi, u in zip(norm_w, updates))
    return agg


def krum_select(updates, num_malicious):
    """Single-Krum: pick the update minimizing sum of squared distances to its
    n - f - 2 nearest neighbours. Returns (selected_delta, index)."""
    n = len(updates)
    vecs = [state_to_vec(u) for u in updates]
    m = max(1, n - num_malicious - 2)
    d = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = ((vecs[i] - vecs[j]) ** 2).sum()
    scores = []
    for i in range(n):
        dists, _ = torch.sort(d[i])
        scores.append(dists[1:m + 1].sum().item())
    idx = int(torch.tensor(scores).argmin().item())
    return updates[idx], idx


def coordinate_median(updates):
    """Coordinate-wise median of client updates."""
    agg = {}
    for k in updates[0]:
        agg[k] = torch.stack([u[k] for u in updates]).median(dim=0).values
    return agg


# ---------------------------------------------------------------------------
# RAgg v3 components
# ---------------------------------------------------------------------------

def geometric_median(vecs, iters=4):
    """Approximate geometric median (Weiszfeld iteration) of a list of vectors."""
    st = torch.stack(vecs)
    z = st.mean(dim=0)
    for _ in range(iters):
        d = (z.unsqueeze(0) - st).norm(dim=1) + 1e-8
        z = (st * (1.0 / d).unsqueeze(1)).sum(0) / (1.0 / d).sum()
    return z


def spectral_reference(vecs, num_mal, n_subsets=8, subset_frac=0.75, gen=None):
    """Random-subsample geometric-median (RSGM) reference.

    Draws K random subsets, each of size >= ceil((N + f) / 2) * subset_frac
    (always an honest majority by pigeonhole when |B| = f*N), computes the
    geometric median of each, and averages. Theorem: each subset median lies
    within an honest-ball; averaging K subsets shrinks reference variance by
    1/K and removes the fixed-momentum lag of the v2 anchor.
    """
    n = len(vecs)
    maj = max((n + num_mal) // 2, 1)
    k = min(n, max(maj, int(round(n * subset_frac))))
    g = torch.Generator()
    if gen is not None:
        g.manual_seed(int(gen.initial_seed() % (2 ** 31)))
    meds = []
    for _ in range(n_subsets):
        idx = torch.randperm(n, generator=g)[:k].tolist()
        meds.append(geometric_median([vecs[i] for i in idx]))
    return torch.stack(meds).mean(dim=0)


def sigmoid_gate(z, temp):
    """Smooth gate sigma(z) = 1 / (1 + exp(-z / temp)); returns in (0, 1)."""
    return torch.sigmoid(z / max(temp, 1e-6))


def iqr_temp(quals, floor=0.02):
    """Adaptive temperature: proportional to the inter-quartile range of this
    round's quality scores -- wide honest spread (heterogeneity) relaxes the
    gate; tight spread (attack or convergence) sharpens it."""
    q = torch.tensor(quals, dtype=torch.float64)
    q75, q25 = torch.quantile(q, 0.75), torch.quantile(q, 0.25)
    return max(float(q75 - q25), floor)


def norm_gate_rate(norms, med_norm, kappa, sharpness=3.0):
    """Magnitude-gate pass rate g_i in [0, 1]: smooth version of the v2 hard
    gate 1[||d_i|| <= kappa * median]. g_i = sigmoid(s * (ln kappa - ln r_i))
    with r_i = ||d_i|| / median. At r=1 (honest bulk): sigma(ln kappa * s) ~ 1;
    at r = kappa (threshold): 0.5; at r >> kappa (scaling attack): ~ 0."""
    r = torch.tensor(norms) / max(med_norm, 1e-12)
    t = sigmoid_gate(torch.log(torch.tensor(float(kappa))) - torch.log(r + 1e-12),
                     1.0 / sharpness)
    return [float(x) for x in t]


def trimmed_mean(updates, trim_ratio=0.3):
    """Coordinate-wise trimmed mean, dropping the trim_ratio extreme values
    on each side (assumes an upper bound on the adversarial fraction)."""
    n = len(updates)
    k = int(n * trim_ratio)
    agg = {}
    for key in updates[0]:
        s = torch.stack([u[key] for u in updates]).sort(dim=0).values
        agg[key] = s[k:n - k].mean(dim=0) if n - 2 * k > 0 else s.mean(dim=0)
    return agg


def fltrust_aggregate(updates, server_delta):
    """FLTrust (Cao et al., NDSS'21): trust score = ReLU(cos(Δ_i, Δ_s));
    each update is normalized to the server update's norm before weighting.
    Falls back to the coordinate-wise median if all scores vanish."""
    sv = state_to_vec(server_delta)
    snorm = sv.norm()
    weights, normed = [], []
    for u in updates:
        v = state_to_vec(u)
        score = max(0.0, cosine(v, sv))
        weights.append(score)
        scale = float(snorm / (v.norm() + 1e-12))
        normed.append({k: u[k] * scale for k in u})
    if sum(weights) < 1e-12:
        return coordinate_median(updates), [0.0] * len(updates)
    agg, norm_w = weighted_average(normed, weights)
    return agg, norm_w


class ReputationBank:
    """Slash-based reputation memory with slow recovery.

    Reputation can only decrease through slashing (multiplicative) and recovers
    additively at a slow rate. It deliberately does NOT track the quality signal
    upward: EMA-tracking of raw qualities creates a rich-get-richer feedback
    (weight concentration on transiently high-quality clients, then collapse),
    which we observed empirically under non-IID data."""

    def __init__(self, client_ids, init=0.5, recover=0.05, slash_tau=0.1, slash_gamma=0.5):
        self.rep = {cid: init for cid in client_ids}
        self.recover = recover
        self.slash_tau = slash_tau
        self.slash_gamma = slash_gamma

    def update(self, cid, quality):
        self.rep[cid] = min(1.0, self.rep[cid] + self.recover)

    def slash(self, cid):
        self.rep[cid] *= self.slash_gamma


def chainfl_ra_aggregate(updates, client_ids, sample_counts, rep_bank,
                         kappa=1.0, mode="full", warmup=False,
                         momentum_ref=None, lam=0.6, eps=0.5, wcap=0.0,
                         normcap=0.0, tau_coef=0.25, variant="v2",
                         num_mal=0, n_subsets=8, subset_frac=0.75, v3_opts=""):
    """Reputation-weighted robust aggregation (proposed).

    1) robust reference: coordinate-wise median of client updates;
    2) cosine quality of each update against the reference;
    3) slash-only reputation memory (with slow recovery);
    4) tempered weights n_i * rep_i * (eps + q_i^kappa), bounding the
       max/min weight ratio and damping feedback-driven concentration.
    Returns (agg_delta, weights, stats_dict).
    """
    vecs = [state_to_vec(u) for u in updates]
    stacked = torch.stack(vecs)

    if variant == "v3":
        # ---------- RAgg v3: spectral reference + adaptive tempering +
        # dual-channel product gate (v3_opts disables components for ablation:
        # "nospec"=v2 coordinate median reference, "notemp"=hard threshold,
        # "nodual"=additive v2-style gate instead of the product) ----------
        opts = set(v3_opts.split(",")) if v3_opts else set()
        if "nospec" in opts:
            reference = stacked.median(dim=0).values
            if momentum_ref is not None:
                reference = lam * momentum_ref + (1 - lam) * reference
        else:
            rs = spectral_reference(vecs, num_mal, n_subsets=n_subsets,
                                    subset_frac=subset_frac)
            if momentum_ref is not None and "noanchor" in opts:
                reference = rs          # pure RSGM (ablation: drift-prone)
            elif momentum_ref is not None:
                # anchored spectral reference: RSGM's honest-majority guarantee
                # + the anchor's drift resistance (Theorem 2 + lag term)
                reference = lam * momentum_ref + (1 - lam) * rs
            else:
                reference = rs
        quals = [max(0.0, cosine(v, reference)) for v in vecs]
        norms = [float(v.norm()) for v in vecs]
        med_norm = float(stacked.new_tensor(norms).median().item())
        # dual-channel: adaptive-temperature sigmoid quality gate ...
        med_q = float(stacked.new_tensor(quals).median().item())
        tau_t = max(0.05, tau_coef * med_q)
        if "notemp" in opts:
            s_gate = [1.0 if q >= tau_t else 0.0 for q in quals]
        else:
            temp = iqr_temp(quals)
            s_gate = [float(x) for x in sigmoid_gate(
                torch.tensor(quals) - tau_t, temp)]
        # ... multiplied by the smooth magnitude gate
        g_gate = norm_gate_rate(norms, med_norm, normcap if normcap > 0 else 5.0)
        if "nogate" in opts:
            g_gate = [1.0] * len(norms)
        if "nodual" in opts:
            gate = [min(1.0, si + gi) for si, gi in zip(s_gate, g_gate)]
        else:
            gate = [si * gi for si, gi in zip(s_gate, g_gate)]

        rewards, slashes = {}, {}
        if mode == "avg":
            return weighted_average(updates, sample_counts)[0], \
                weighted_average(updates, sample_counts)[1], \
                {"qualities": quals, "rewards": rewards, "slashes": slashes,
                 "reputations": dict(rep_bank.rep), "gates": gate}
        if warmup:
            # magnitude half of the product gate is meaningful from round 1;
            # the quality half warms up with the reputation memory
            for cid in client_ids:
                rewards[cid] = 0.0
            weights = [n_i * gi + 1e-9
                       for n_i, gi in zip(sample_counts, g_gate)]
            agg, norm_w = weighted_average(updates, weights)
            return agg, norm_w, {"qualities": quals, "rewards": rewards,
                                 "slashes": slashes,
                                 "reputations": dict(rep_bank.rep),
                                 "gates": gate}
        if mode == "no_qual":
            for cid in client_ids:
                rewards[cid] = 0.0
            weights = [n_i * rep_bank.rep[cid] + 1e-9
                       for cid, n_i in zip(client_ids, sample_counts)]
            agg, norm_w = weighted_average(updates, weights)
            return agg, norm_w, {"qualities": quals, "rewards": rewards,
                                 "slashes": slashes,
                                 "reputations": dict(rep_bank.rep),
                                 "gates": gate}
        for cid, q, gi in zip(client_ids, quals, gate):
            if q < tau_t or gi < 0.5:
                rep_bank.slash(cid)
                slashes[cid] = 2.0
            else:
                rep_bank.update(cid, q)
                rewards[cid] = round(max(0.0, q - med_q) * gi, 4)
        # tempered weights: n_i * rep_i * q_i * (s_i * g_i)  -- the product
        # gate zeroes any client that fails EITHER channel outright, and the
        # reputation memory suppresses persistent offenders across rounds
        weights = [n_i * (1.0 if mode == "no_rep" else rep_bank.rep[cid])
                   * q * gi + 1e-9
                   for cid, q, n_i, gi in zip(client_ids, quals,
                                              sample_counts, gate)]
        agg, norm_w = weighted_average(updates, weights)
        if wcap > 0.0:
            norm_w = cap_weights(norm_w, wcap)
            agg = _aggregate_with_cap(updates, norm_w, wcap)
        return agg, norm_w, {"qualities": quals, "weights": norm_w,
                             "rewards": rewards, "slashes": slashes,
                             "reputations": dict(rep_bank.rep), "gates": gate}

    # ---------- RAgg v2 (paper v5 default) ----------
    reference = stacked.median(dim=0).values
    if momentum_ref is not None:
        # anchor the reference to the historical descent direction so that a
        # transiently consistent adversarial coalition cannot drag the median
        reference = lam * momentum_ref + (1 - lam) * reference
    quals = [max(0.0, cosine(v, reference)) for v in vecs]
    # magnitude channel: update-scaling attacks inflate ||Δ|| far beyond the
    # honest bulk; the ratio to the per-round median norm stays large even
    # near convergence, where cosine directions become poorly conditioned
    norms = [float(v.norm()) for v in vecs]
    med_norm = float(stacked.new_tensor(norms).median().item())
    norm_flags = [bool(n > normcap * med_norm) if normcap > 0.0 else False
                  for n in norms]
    rewards, slashes = {}, {}
    if mode == "avg":  # plain sample-weighted average (both modules off)
        return weighted_average(updates, sample_counts)[0], \
            weighted_average(updates, sample_counts)[1], \
            {"qualities": quals, "rewards": rewards, "slashes": slashes,
             "reputations": dict(rep_bank.rep)}
    # adaptive threshold: a fixed fraction of this round's median quality,
    # lower-bounded (tracks the drifting scale of cosine scores over training)
    med_q = float(stacked.new_tensor(quals).median().item())
    tau_t = max(0.05, tau_coef * med_q)
    if warmup:
        # early rounds: quality signal is unreliable; freeze reputation, no slash.
        # the magnitude channel still applies: it needs no warm-up, since norm
        # ratios are meaningful from round 1 (prevents warm-up poisoning by
        # update-scaling before the direction gate is trustworthy)
        for cid in client_ids:
            rewards[cid] = 0.0
        weights = [n_i * (0.0 if nflag else (q ** kappa + (eps if q >= tau_t else 0.0))) + 1e-9
                   for q, n_i, nflag in zip(quals, sample_counts, norm_flags)]
        agg, norm_w = weighted_average(updates, weights)
        if wcap > 0.0:
            norm_w = cap_weights(norm_w, wcap)
            agg = _aggregate_with_cap(updates, norm_w, wcap)
        return agg, norm_w, {"qualities": quals, "rewards": rewards,
                             "slashes": slashes, "reputations": dict(rep_bank.rep)}
    if mode == "no_qual":
        # cosine gate fully removed: no signal, so reputation cannot learn either
        for cid in client_ids:
            rewards[cid] = 0.0
        weights = [n_i * rep_bank.rep[cid] + 1e-9
                   for cid, n_i in zip(client_ids, sample_counts)]
        agg, norm_w = weighted_average(updates, weights)
        return agg, norm_w, {"qualities": quals, "rewards": rewards,
                             "slashes": slashes, "reputations": dict(rep_bank.rep)}
    for cid, q, nflag in zip(client_ids, quals, norm_flags):
        if q < tau_t or nflag:
            rep_bank.slash(cid)
            slashes[cid] = 2.0
        else:
            rep_bank.update(cid, q)
            # three-zone settlement: only above-median contributions earn
            # tokens, so sub-median (e.g., label-flipping) attackers earn
            # nothing even when they are not explicitly slashed
            rewards[cid] = round(max(0.0, q - med_q), 4)
    weights = []
    for cid, q, n_i, nflag in zip(client_ids, quals, sample_counts, norm_flags):
        rep = 1.0 if mode == "no_rep" else rep_bank.rep[cid]
        # no floor for suspected (sub-threshold or norm-flagged) updates
        floor = eps if (q >= tau_t and not nflag) else 0.0
        weights.append(n_i * rep * (q ** kappa + floor) + 1e-9)
    agg, norm_w = weighted_average(updates, weights)
    if wcap > 0.0:
        # bounded influence: cap normalized weights at wcap (water-filling),
        # so no client — however well aligned — can steer more than a fixed
        # share of the aggregate; damps noise-driven over-rotation near
        # convergence while leaving slash/reputation suppression untouched
        norm_w = cap_weights(norm_w, wcap)
        agg = _aggregate_with_cap(updates, norm_w, wcap)
    stats = {
        "qualities": quals,
        "weights": norm_w,
        "rewards": rewards,
        "slashes": slashes,
        "reputations": dict(rep_bank.rep),
    }
    return agg, norm_w, stats


# ---------------------------------------------------------------------------
# FedRoLA (Yan et al., KDD'24): layer-based robust aggregation
# ---------------------------------------------------------------------------

class FedRoLABank:
    """Cross-round Beta-Tracking state for FedRoLA: per-layer and per-client
    Beta counts drive layer selection (Thompson sampling) and the client trust
    probability. Mirrors the reference implementation's alpha/beta update."""

    def __init__(self, client_ids):
        self.alpha_client = {cid: 1.0 for cid in client_ids}
        self.beta_client = {cid: 1.0 for cid in client_ids}
        self.alpha_layer = {}
        self.beta_layer = {}
        self.round = 0

    @staticmethod
    def disc(r):
        """discount factor ramping 0 -> 1 with round count (logistic)."""
        return 2.0 / (1.0 + float(np.exp(-0.1 * r))) - 1.0

    def layer_prob(self, n_layers):
        if not self.alpha_layer:
            self.alpha_layer = {i: 1.0 for i in range(n_layers)}
            self.beta_layer = {i: 0.0 for i in range(n_layers)}
        tot = np.array([self.alpha_layer[i] + self.beta_layer[i]
                        for i in range(n_layers)])
        p = np.array([self.alpha_layer[i] / max(tot[i], 1e-12)
                      for i in range(n_layers)])
        return p / p.sum()


def _extract_layers(sd):
    """flatten each weight/bias tensor to (1, numel), preserving per-key order."""
    return [v.detach().reshape(1, -1).float()
            for k, v in sd.items() if ("weight" in k or "bias" in k)]


def fedrola_aggregate(updates, client_ids, sample_counts, bank, variant="LASI"):
    """FedRoLA (KDD'24) robust aggregation.

    LASI: cosine of each client's layer vs the mean-update layer (the variant
    used for the method's headline results).
    PCSI: pairwise cosine among clients per layer, top-2 neighbours.
    Both: Thompson-sampled layer selection (3 layers), layer-based vote (>=3
    votes), and a Beta-tracked client trust probability that discounts flagged
    clients by disc(round). Returns (agg_delta, weights).
    """
    import torch.nn.functional as F
    n = len(updates)
    # mean update (reference) and per-key flattened layers
    avg_grad, _ = weighted_average(updates, sample_counts)
    avg_layers = _extract_layers(avg_grad)
    extracted = [_extract_layers(g) for g in updates]
    n_layers = len(extracted[0])

    # 1) Layer selection via Thompson sampling over (alpha, beta)
    # (global np RNG — deterministic under set_seed, matching reference impl)
    p = bank.layer_prob(n_layers)
    chosen = np.random.choice(n_layers, 3, replace=False, p=p) if n_layers >= 3 \
        else np.arange(n_layers)

    # 2) Layer-based detection: sweep threshold from 0.9 downward
    finds = {}
    for li in chosen:
        if variant == "LASI":
            sims = [float(F.cosine_similarity(avg_layers[li], ex[li], dim=1))
                    for ex in extracted]
        else:  # PCSI: top-2 pairwise cosine
            sim_mat = torch.zeros(n, n)
            for i in range(n):
                for j in range(i + 1, n):
                    s = float(F.cosine_similarity(extracted[i][li],
                                                  extracted[j][li], dim=1))
                    sim_mat[i, j] = sim_mat[j, i] = s
            sims = [float(torch.sort(sim_mat[i], descending=True).values[:2].mean())
                    for i in range(n)]
        thr = 0.9
        while True:
            found = [i for i, s in enumerate(sims) if s >= thr]
            prop = len(found) / max(n, 1)
            if prop < 0.5:
                bad = found
                bank.alpha_layer[li] += 1
                break
            if thr <= 0.0:
                bad = []
                bank.beta_layer[li] += 1
                break
            thr -= 0.1
        for cidx in bad:
            finds[cidx] = finds.get(cidx, 0) + 1

    # 3) Layer-based vote (>=3 votes, relax until non-empty)
    flagged = []
    vote_num = 3
    while not flagged and vote_num >= 0:
        flagged = [client_ids[cidx] for cidx, c in finds.items() if c >= vote_num]
        vote_num -= 1

    # 4) update client Beta counts
    for cid in client_ids:
        if cid in flagged:
            bank.beta_client[cid] += 1
        else:
            bank.alpha_client[cid] += 1

    # 5) trust-probability weighted aggregation
    weights = []
    for i, cid in enumerate(client_ids):
        prob = bank.alpha_client[cid] / (bank.alpha_client[cid] + bank.beta_client[cid])
        if cid in flagged:
            prob *= bank.disc(bank.round)
        weights.append(sample_counts[i] * prob + 1e-9)
    bank.round += 1

    agg, norm_w = weighted_average(updates, weights)
    return agg, norm_w


# ---------------------------------------------------------------------------
# FLAME (Nguyen et al., USENIX Security'22): dynamic clustering + clipping + noise
# ---------------------------------------------------------------------------

def flame_aggregate(updates, global_sd, sample_counts=None, noise_lam=1.2e-5):
    """FLAME defense (USENIX Sec'22).

    Steps (paper Sec. 5, reference impl):
      1) HDBSCAN clustering on L2-normalized client model vectors
         (min_cluster_size = N//2+1, min_samples=1, allow_single_cluster=True);
         clients in the largest cluster (label 0) are kept.
      2) adaptive clipping: scale each kept update to the median L2 norm
         (gamma = med/norm, capped at 1).
      3) average the kept (clipped) updates, then add Gaussian noise to the
         aggregated model: std = noise_lam * med * std(param), skipping biases.
    Returns (agg_delta, weights).
    """
    import hdbscan
    n = len(updates)
    # 1) rebuild client models (global + delta) and L2-normalize
    vecs = []
    for u in updates:
        v = state_to_vec({k: global_sd[k].float() + u[k].float() for k in u})
        vecs.append(v / (v.norm() + 1e-12))
    X = torch.stack(vecs).double().numpy()
    cluster = hdbscan.HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1,
                              allow_single_cluster=True)
    cluster.fit(X)
    keep = [i for i, lab in enumerate(cluster.labels_) if lab == 0]
    if not keep:                      # fallback: no clean majority found
        keep = list(range(n))
    # 2) adaptive median-norm clipping
    norms = [float(state_to_vec(u).norm()) for u in updates]
    med = float(torch.tensor(norms).median())
    clipped = []
    for i in keep:
        g = min(1.0, med / max(norms[i], 1e-12))
        clipped.append({k: v * g for k, v in updates[i].items()})
    # 3) average + DP noise
    agg, _ = weighted_average(clipped, [1.0] * len(clipped))
    for k, v in agg.items():
        if "bias" in k:
            continue
        std = noise_lam * med * float(v.std())
        agg[k] = v + torch.randn_like(v) * std
    weights = [1.0 / len(clipped) if i in keep else 0.0 for i in range(n)]
    return agg, weights


# ---------------------------------------------------------------------------
# RFLPA (Mai et al., NeurIPS'24): plaintext robust-aggregation core
# ---------------------------------------------------------------------------

def rflpa_aggregate(updates, server_delta):
    """RFLPA (NeurIPS'24) plaintext robust-aggregation core.

    RFLPA weights each client update by the cosine similarity between the local
    update and the server update trained on a small clean root dataset — the
    same directional trust signal as FLTrust. RFLPA's own contribution is
    computing this cosine in the encrypted domain (verifiable packed Shamir
    secret sharing + a dot-product aggregation protocol); that SecAgg layer has
    no effect in a single-machine plaintext simulation, so we reproduce the
    robust-aggregation core, which reduces to FLTrust's cosine weighting. Kept
    as a separate entry point for faithful citation and for the
    root-data-free contrast with ChainFL-RA.
    """
    return fltrust_aggregate(updates, server_delta)
