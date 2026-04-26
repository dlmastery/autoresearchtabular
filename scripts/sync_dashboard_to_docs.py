"""Mirror dashboard/ to docs/ for GitHub Pages.

Copies dashboard.html and data.json to docs/, then writes docs/index.html
as the GitHub Pages landing page.

Usage:
    python scripts/sync_dashboard_to_docs.py
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
DOCS = ROOT / "docs"


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AUTORESEARCHTABULAR — Higgs UCI live dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
         system-ui, sans-serif; max-width: 980px; margin: 0 auto;
         padding: 24px; line-height: 1.55; color: #1c1c1c;
         background: #fafafa; }
  h1 { color: #0d4a8a; font-size: 1.7rem; }
  h2 { color: #0d4a8a; margin-top: 2rem; font-size: 1.3rem; }
  a { color: #0d4a8a; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 0.78rem; font-family: monospace; }
  .badge.live { background: #1f7a36; color: #fff; }
  .badge.frozen { background: #707070; color: #fff; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
          margin: 1.2rem 0; }
  .card { padding: 16px; border: 1px solid #d0d7de; border-radius: 8px;
          background: #fff; }
  .card h3 { margin: 0 0 8px 0; font-size: 1.05rem; }
  code { background: #f0f1f3; padding: 2px 5px; border-radius: 3px;
         font-size: 0.92em; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
  th { background: #eef2f7; }
  hr { border: 0; border-top: 1px solid #d0d7de; margin: 2rem 0; }
  footer { text-align: center; color: #707070; font-size: 0.88rem;
           margin-top: 3rem; }
</style>
</head>
<body>

<h1>AUTORESEARCHTABULAR — Higgs UCI live dashboard <span class="badge live">LIVE</span></h1>

<p>Autonomous, audit-gated ML research loop on the
<strong>Higgs UCI tabular benchmark</strong> (Baldi 2014,
<em>Nature Communications</em>, arXiv:1402.4735) — 11 M rows,
28 features, frozen 10 M / 500 k / 500 k Baldi 2014 split,
25 experiments per backbone × 14 backbones planned.</p>

<div class="grid">
  <div class="card">
    <h3>Live dashboard</h3>
    <p>Filter, sort, search and export every experiment as it lands:</p>
    <p><a href="dashboard.html"><strong>Open dashboard →</strong></a></p>
  </div>
  <div class="card">
    <h3>Repository</h3>
    <p>Code, configs, and per-experiment artefacts:</p>
    <p><a href="https://github.com/dlmastery/autoresearchtabular">
       <strong>github.com/dlmastery/autoresearchtabular →</strong></a></p>
  </div>
  <div class="card">
    <h3>Paper</h3>
    <p>Full method, results, references:</p>
    <p><a href="https://github.com/dlmastery/autoresearchtabular/blob/main/PAPER.md">
       <strong>PAPER.md →</strong></a></p>
  </div>
  <div class="card">
    <h3>SOTA comparison</h3>
    <p>Honest comparison vs published baselines:</p>
    <p><a href="https://github.com/dlmastery/autoresearchtabular/blob/main/SOTA_COMPARISON.md">
       <strong>SOTA_COMPARISON.md →</strong></a></p>
  </div>
</div>

<h2>Three programmatic gates</h2>

<ol>
  <li><strong>Data-split audit</strong> — 7 auditors verify Baldi 2014
      frozen split: pairwise disjointness, class balance, size floors,
      no leakage, byte-identical reproducibility, feature consistency.
      Output: <code>autoresearch_results/data_split_audit.md</code> +
      SHA-256 fingerprint embedded in every result row.</li>
  <li><strong>Citation Rigor</strong> — every hyperparameter that
      differs from the registry default cites a primary source (author,
      year, venue, section/table/figure) with a stated reason for why
      the change should help on Higgs. Bare URLs and folklore are
      rejected by the parser.</li>
  <li><strong>Reasoning Blob Completeness</strong> — every experiment
      ships a 7-section blob (<code>diagnose / cite / hypothesize /
      predict / run / analyze / checkpoint</code>) with per-section
      word floors. The runner refuses to start without a passing
      pre-run blob and refuses to write the result row without a
      passing post-run blob.</li>
</ol>

<h2>Composite metric (Goodhart-fingerprinted)</h2>

<p><code>composite = min(test_auc, val_auc) - 0.1 · |test_auc − val_auc|</code></p>

<p>The formula string is SHA-256 hashed at runner boot; the hash is
embedded in every result row. If the formula is silently changed,
the runner refuses to start.</p>

<h2>Companion projects</h2>

<table>
  <tr><th>Project</th><th>Domain</th><th>Status</th></tr>
  <tr>
    <td><a href="https://github.com/dlmastery/autoresearch">autoresearch</a></td>
    <td>FX forecasting (15-min EUR/USD)</td>
    <td>79 experiments — composite 0.6181</td>
  </tr>
  <tr>
    <td><a href="https://github.com/dlmastery/autoresearchimage">autoresearchimage</a></td>
    <td>Computer vision (PathMNIST + WILDS-Camelyon17)</td>
    <td>21 experiments — composite 0.9966 — <a href="https://dlmastery.github.io/autoresearchimage/">live</a></td>
  </tr>
  <tr>
    <td><strong>autoresearchtabular</strong> <em>(this repo)</em></td>
    <td>Tabular ML (Higgs UCI)</td>
    <td>150-200 experiments planned</td>
  </tr>
</table>

<footer>
<hr>
<p>Generated by AUTORESEARCHTABULAR. License MIT. Last sync: <span id="sync-ts">—</span></p>
<script>
  fetch("data.json").then(r => r.json()).then(d => {
    if (d.generated_utc) {
      document.getElementById("sync-ts").textContent = d.generated_utc;
    }
  }).catch(() => {});
</script>
</footer>

</body>
</html>
"""


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    # copy dashboard.html
    src = DASH / "dashboard.html"
    if src.exists():
        shutil.copy2(src, DOCS / "dashboard.html")
        print(f"[sync] {src} -> docs/dashboard.html")
    # copy data.json (if present)
    data_src = DASH / "data.json"
    if data_src.exists():
        shutil.copy2(data_src, DOCS / "data.json")
        print(f"[sync] {data_src} -> docs/data.json")
    # write index.html
    idx = DOCS / "index.html"
    idx.write_text(INDEX_HTML, encoding="utf-8")
    print(f"[sync] wrote {idx}")
    # .nojekyll to bypass Jekyll on GitHub Pages
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"[sync] wrote docs/.nojekyll")
    return 0


if __name__ == "__main__":
    sys.exit(main())
