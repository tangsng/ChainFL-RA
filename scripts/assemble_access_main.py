# -*- coding: utf-8 -*-
"""Assemble IEEE Access main.tex from processed body + front matter + refs."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.environ.get("TEMP", r"C:\Users\ts\AppData\Local\Temp")
DST = os.path.join(ROOT, "paper", "ieee_access_submission", "main.tex")

body = open(os.path.join(TMP, "body_final.tex"), encoding="utf-8").read()
md = open(os.path.join(ROOT, "paper", "chainfl_ra_paper.md"), encoding="utf-8").read()

# abstract + keywords from md
i_abs = md.find("## Abstract")
i_idx = md.find("**Index Terms**")
abstract = md[i_abs + len("## Abstract") + 2:i_idx].strip()
# the abstract ends before Index Terms line
idx_line = [l for l in md.split("\n") if l.startswith("**Index Terms**")][0]
keywords = idx_line.replace("**Index Terms**", "").replace("—", "").strip().rstrip(".")

# references
i_ref = md.find("## References")
refs = []
for l in md[i_ref:].split("\n"):
    m = re.match(r"^\[(\d+)\]\s+(.*)$", l.strip())
    if m:
        refs.append((int(m.group(1)), m.group(2)))
refs.sort()
bib = "\\begin{thebibliography}{99}\n\\providecommand{\\url}[1]{#1}\n"
for n, e in refs:
    e = e.replace("*", "")
    e = e.replace("\u201c", "``").replace("\u201d", "''").replace("\u2013", "--").replace("\u2014", "---")
    e = e.replace("&", "\\&")
    bib += "\\bibitem{ref%d} %s\n" % (n, e)
bib += "\\end{thebibliography}\n"

tex = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{url}
% [submission build] pdftex >= 1.40.26 removed the \pdfcolorstack primitive in
% favor of \pdfextension colorstack; pdftex.def still emits the old name.
% Map primitive-level uses to the new syntax so color commands resolve.
\ifpdf
  \makeatletter\@ifundefined{pdfcolorstack}{%
    \edef\pdfcolorstack{\noexpand\pdfextension colorstack}}{}\makeatother
\fi
\def\UrlBreaks{\do\/\do-}

\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.0000.0000000}

\title{ChainFL-RA: Blockchain-Enabled Decentralized Federated Learning With Reputation-Weighted Robust Aggregation and On-Chain Auditable Incentives}

\author{\uppercase{Song Tang}\authorrefmark{1,2,3}, \uppercase{Zhigang Jin}\authorrefmark{1}, and \uppercase{Zhiqiang Wang}\authorrefmark{2,3}}
\address[1]{School of Electrical and Information Engineering, Tianjin University, Tianjin 300072, China (e-mail: zgjin@tju.edu.cn)}
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

\section*{Acknowledgment}
The authors have nothing to report.

\section*{Funding}
This work was supported by the Hebei Provincial Science and Technology Program Project (Grant No. 25360301D): Research and Application Demonstration of Key Technologies for Public Data Authorization and Operation Based on Trusted Data Space.

\section*{Conflict of interest}
The authors declare no conflict of interest.

\section*{Data availability}
The source code and raw data that support the findings of this study are openly available in AdaTrust at \url{https://github.com/tangsng/AdaTrust}.

__BIB__

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{placeholder_photo.png}}]{Song Tang}
received the B.E. degree from Hebei University of Science and Technology. He is currently working toward the Ph.D. degree with the School of Electrical and Information Engineering, Tianjin University, Tianjin, China, and is also a researcher with the Institute of Applied Mathematics, Hebei Academy of Sciences, Shijiazhuang, China. His research interests include federated learning, blockchain, and privacy-preserving machine learning. (Corresponding ORCID: 0000-0001-9048-0738.)
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{placeholder_photo.png}}]{Zhigang Jin}
received the Ph.D. degree in electrical engineering from Tianjin University. He is currently a Professor with the School of Electrical and Information Engineering, Tianjin University. His research interests include network security, federated learning, and distributed systems. (ORCID: 0000-0001-5777-569X.)
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{placeholder_photo.png}}]{Zhiqiang Wang}
is a researcher with the Institute of Applied Mathematics, Hebei Academy of Sciences, Shijiazhuang, China, and the Information Security Authentication Technology Innovation Center of Hebei Province. His research interests include applied mathematics and data security. (ORCID: 0009-0005-5718-6867.)
\end{IEEEbiography}

\EOD
\end{document}
"""

tex = tex.replace("__ABSTRACT__", abstract)
tex = tex.replace("__KEYWORDS__", keywords)
tex = tex.replace("__BODY__", body)
tex = tex.replace("__BIB__", bib)

os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8").write(tex)
print("written", DST, len(tex), "chars")
