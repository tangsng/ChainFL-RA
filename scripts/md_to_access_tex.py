# -*- coding: utf-8 -*-
"""Convert chainfl_ra_paper.md -> IEEE Access LaTeX (ieeeaccess.cls).
Handles: $$..$$ equations (strip manual (n) tags, use equation env),
markdown tables -> table* envs, algorithm code block, figures, bold/italic,
sections, references list, and the front-matter (authors/affiliations)."""
import re
import os

SRC = "../paper/chainfl_ra_paper.md"
DST = "../paper/ieee_access_submission/main.tex"

md = open(SRC, encoding="utf-8").read()
lines = md.split("\n")

out = []
i = 0
eq_seen = set()

HAND_ALG = r'''\begin{table}[!t]
\centering\small
\begin{tabular}{l}
\textbf{Algorithm 1: RAgg, Reputation-Weighted Robust Aggregation (round $t$)} \\
\hline
\textbf{Input:} updates $\{\Delta_i\}$, sample counts $\{n_i\}$, reputations $\{r_i^{t-1}\}$, \\
\hspace*{2em} momentum direction $m^{t-1}$; $\gamma{=}0.5$, $\rho{=}0.05$, $\lambda{=}0.6$, \\
\hspace*{2em} $\kappa{=}5.0$, $s_0{=}3$, $K{=}8$ subsets, $W{=}2$ \\
\textbf{Output:} aggregate $\Delta^t$, weights $\{\omega_i\}$, reputations $\{r_i^t\}$, $m^t$ \\
1: $\hat{\mu} \leftarrow \frac{1}{K}\sum_k$ WeiszfeldMedian$(\{\Delta_i : i \in S_k\})$ $\triangleright$ $K$ honest-majority subsets \\
2: $\bar{\Delta} \leftarrow \lambda m^{t-1} + (1{-}\lambda)\hat{\mu}$ $\triangleright$ anchored spectral ref, Eq.~(2) \\
3: \textbf{for} $i = 1, \dots, N$ \textbf{do} \\
\hspace*{1em} $q_i \leftarrow \max(0, \cos(\Delta_i, \bar{\Delta}))$ \\
\hspace*{1em} $g_i \leftarrow \sigma(s_0 \ln(\kappa \cdot \mathrm{med}(\|\Delta_j\|)/\|\Delta_i\|))$ $\triangleright$ Eq.~(4) \\
4: $\tau_t \leftarrow \max(0.05, 0.25 \cdot \mathrm{median}(\{q_j\}))$; $T_t \leftarrow \max(\mathrm{IQR}(\{q_j\}), 0.02)$ \\
5: \textbf{if} $t \le W$ \textbf{then} $\omega_i \propto n_i g_i$ $\triangleright$ warm-up: freeze reputation \\
6: \textbf{else} \\
\hspace*{1em} \textbf{for} $i = 1, \dots, N$ \textbf{do} \\
\hspace*{2em} \textbf{if} $q_i < \tau_t$ \textbf{or} $g_i < 0.5$: $r_i^t \leftarrow \gamma r_i^{t-1}$; record slash($i$) \\
\hspace*{3em} \textbf{else}: $r_i^t \leftarrow \min(1, r_i^{t-1} + \rho)$; record reward($i$) \\
7: $s_i \leftarrow \sigma((q_i - \tau_t)/T_t)$; $\omega_i \propto n_i r_i^t q_i s_i g_i$ $\triangleright$ Eq.~(6) \\
8: normalize $\{\omega_i\}$ to sum 1 \\
9: $\Delta^t \leftarrow \sum_i \omega_i \Delta_i$; $m^t \leftarrow \lambda m^{t-1} + (1{-}\lambda)\Delta^t$ \\
10: commit model hash and $(i, q_i, \omega_i, r_i^t)$ to ledger; settle rewards/slashes \\
11: \textbf{return} $\Delta^t$, $\{\omega_i\}$, $\{r_i^t\}$, $m^t$ \\
\hline
\end{tabular}
\caption{RAgg, reputation-weighted robust aggregation (round $t$).}
\label{alg:ragg}
\end{table}'''

def esc(s):
    # escape LaTeX specials outside math (conservative; md has little raw tex in prose)
    for ch, rep in [("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        if ch in s:
            s = s.replace(ch, rep)
    return s

def inline(s):
    # bold/italic/code -> tex; leave $...$ math and [n] cites as-is
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    return s

# ---- split front matter ----
# find abstract section
def find(pat):
    for j, l in enumerate(lines):
        if l.startswith(pat):
            return j
    return -1

i_title = find("# ChainFL-RA")
i_abs = find("## Abstract")
i_idx = find("**Index Terms**")
i_intro = find("## I. Introduction")
i_ref = find("## References")

abstract = "\n".join(lines[i_abs+2:i_idx]).strip()
index_terms = re.sub(r"\*\*Index Terms\*+\*?—?", "", lines[i_idx]).replace("**", "").strip()
body_lines = lines[i_intro:i_ref]
ref_lines = [l for l in lines[i_ref+1:] if l.strip()]

# ---------------- body conversion ----------------
body = []
i = 0
fig_counter = 0
while i < len(body_lines):
    l = body_lines[i]

    # display equation
    if l.startswith("$$"):
        eq = l[2:]
        while not eq.rstrip().endswith("$$") and i + 1 < len(body_lines):
            i += 1
            eq += " " + body_lines[i]
        eq = eq.rstrip()[:-2].strip()
        m = re.search(r"\\qquad\s*\((\d+)\)\s*$", eq)
        if m:
            eq = eq[:m.start()].strip()
            eq_seen.add(m.group(1))
        body.append("\\begin{equation}" + eq + "\\end{equation}")
        i += 1
        continue

    # algorithm block
    if l.startswith("```text") or (l.startswith("```") and i + 1 < len(body_lines)
                                    and body_lines[i+1].startswith("Algorithm 1")):
        j = i + 1
        block = []
        while j < len(body_lines) and not body_lines[j].startswith("```"):
            block.append(body_lines[j])
            j += 1
        alg = "\n".join(block)
        # convert common pseudo markers to LaTeX
        alg = esc(alg)
        # NOTE: math conversion no longer needed — HAND_ALG is pre-written LaTeX
        body.append(HAND_ALG)
        i = j + 1
        continue

    # generic code fence -> verbatim skip
    if l.startswith("```"):
        j = i + 1
        while j < len(body_lines) and not body_lines[j].startswith("```"):
            j += 1
        i = j + 1
        continue

    # table
    if l.startswith("|"):
        tbl = []
        j = i
        while j < len(body_lines) and body_lines[j].startswith("|"):
            tbl.append(body_lines[j])
            j += 1
        rows = []
        for r in tbl:
            if re.match(r"^\|[\s\-:|]+\|$", r):
                continue
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            rows.append(cells)
        if rows:
            ncol = max(len(r) for r in rows)
            body.append("\\begin{table}[!t]\\centering\\small\\begin{tabular}{" + "l" * ncol + "}")
            for r in rows:
                body.append(" & ".join(c for c in r) + " \\\\")
            body.append("\\end{tabular}\\end{table}")
        i = j
        continue

    # figure
    m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", l)
    if m:
        fig_counter += 1
        cap = ""
        if i + 1 < len(body_lines) and body_lines[i+1].startswith("**Fig."):
            cap = body_lines[i+1].replace("**", "")
            i += 1
        path = m.group(2).replace("../results/figures/", "figures/")
        body.append("\\begin{figure}[!t]\\centering\\includegraphics[width=\\linewidth]{%s}"
                    "\\caption{%s}\\label{fig:%d}\\end{figure}" % (path, inline(cap), fig_counter))
        i += 1
        continue

    # table caption line (attach to previously emitted table not handled; keep as bold para)
    if l.startswith("**Table"):
        body.append("\\noindent\\textbf{" + inline(l.replace("**", "")) + "}")
        i += 1
        continue

    # section headers
    m = re.match(r"^(#{1,3})\s+(.*)$", l)
    if m:
        lvl = len(m.group(1))
        title = m.group(2)
        title = re.sub(r"^[IVX]+\.\s*", "", title)
        title = re.sub(r"^[A-Z]\.\s*", "", title)
        if lvl == 1:
            body.append("\\section{%s}" % title)
        elif lvl == 2:
            body.append("\\subsection{%s}" % title)
        else:
            body.append("\\subsubsection{%s}" % title)
        i += 1
        continue

    # normal paragraph
    if l.strip():
        body.append(inline(l) + "\n")
    i += 1

# ---------------- references ----------------
refs = []
for r in ref_lines:
    m = re.match(r"^\[(\d+)\]\s+(.*)$", r.strip())
    if m:
        n, entry = m.group(1), m.group(2)
        entry = entry.replace("*", "").replace("--", "-")
        # convert to \bibitem
        refs.append((int(n), entry))
refs.sort()

bib = "\\begin{thebibliography}{99}\n"
for n, e in refs:
    e = re.sub(r"\"([^\"]+)\"", r"``\1''", e)
    e = e.replace("\u201c", "``").replace("\u201d", "''")
    bib += "\\bibitem{ref%d} %s\n" % (n, e)
bib += "\\end{thebibliography}\n"
# replace [n] cites with \cite{refn} in body
body_txt = "\n".join(body)
body_txt = re.sub(r"\[(\d+)\]", lambda m: "\\cite{ref%s}" % m.group(1), body_txt)

# ---------------- assemble ----------------
tex = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.0000.0000000}

\title{ChainFL-RA: Blockchain-Enabled Decentralized Federated Learning with Reputation-Weighted Robust Aggregation and On-Chain Auditable Incentives}

\author{\uppercase{Song Tang}\authorrefmark{1,2,3}, \uppercase{Zhigang Jin}\authorrefmark{1}, and \uppercase{Zhiqiang Wang}\authorrefmark{2,3}}
\address[1]{School of Electrical and Information Engineering, Tianjin University, Tianjin, China (e-mail: zgjin@tju.edu.cn)}
\address[2]{Institute of Applied Mathematics, Hebei Academy of Sciences, Shijiazhuang, China (e-mail: tangsng@live.cn; 574465982@qq.com)}
\address[3]{Information Security Authentication Technology Innovation Center of Hebei Province, Shijiazhuang, China}
\corresp{Corresponding author: Zhigang Jin (e-mail: zgjin@tju.edu.cn).}

\markboth{Tang et al.: ChainFL-RA: Blockchain-Enabled Decentralized Federated Learning}{Tang et al.: ChainFL-RA}

\begin{abstract}
__ABSTRACT__
\end{abstract}

\begin{keywords}
__KEYWORDS__
\end{keywords}

\titlepgskip=-15pt
\maketitle

__BODY__

__BIB__

\EOD
\end{document}
"""

abstract_tex = inline(esc(abstract)).replace("[", r"{[}").replace("]", r"{]}")
# undo bracket-escape inside math/cites not needed for abstract (no cites there)
tex = tex.replace("__ABSTRACT__", abstract_tex)
tex = tex.replace("__KEYWORDS__", index_terms.rstrip("."))
tex = tex.replace("__BODY__", body_txt)
tex = tex.replace("__BIB__", bib)

os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8").write(tex)
print("written", DST, len(tex), "chars; eq tags:", sorted(eq_seen, key=int), "; figs:", fig_counter, "; refs:", len(refs))
