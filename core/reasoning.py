"""Citation Rigor + Reasoning Blob Completeness validators (tabular adaptation)."""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

WORD_FLOORS = {
    "diagnosis": 60, "citations_single": 40, "citations_multi": 80,
    "hypothesis": 50, "prediction": 25, "verdict": 30, "learning": 40,
}
PLACEHOLDER_TOKENS = ("TODO-REWRITE", "(auto-backfilled)", "(no explicit citation)")
REQUIRED_FIELDS = ("diagnosis", "citations", "hypothesis", "prediction", "verdict", "learning", "_manual")

_RX_AUTHORS = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:,?\s+[A-Z][a-zA-Z'’\-]+){1,}")
_RX_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RX_VENUE = re.compile(
    r"\b(NeurIPS|ICML|ICLR|AAAI|CVPR|ECCV|ICCV|MICCAI|KDD|TMLR|JMLR|"
    r"Nature|Science|Lancet|JAMA|EJOR|MIA|TPAMI|BMVC|WACV|arXiv|bioRxiv|"
    r"COLM|VLDB|SIGMOD|ICDM|UAI|AISTATS|Machine Learning)\b"
)
_RX_ARXIV = re.compile(r"arXiv:\d{4}\.\d{4,5}", re.IGNORECASE)
_RX_BIORXIV = re.compile(r"bioRxiv:[\d./a-zA-Z\-]+", re.IGNORECASE)
_RX_TITLE = re.compile(r"['‘][^'’]{8,}['’]")
_RX_RELEVANCE = re.compile(r"—|--|–|:")


@dataclass
class ReasoningEntry:
    diagnosis: str = ""
    citations: str = ""
    hypothesis: str = ""
    prediction: str = ""
    verdict: str = ""
    learning: str = ""
    _manual: bool = True
    _needs_rewrite: bool = False

    def words(self, field: str) -> int:
        s = (getattr(self, field) or "").strip()
        return 0 if not s else len(re.findall(r"\S+", s))


def _has_placeholder(s: str) -> bool:
    return any(tok in (s or "") for tok in PLACEHOLDER_TOKENS)


def validate_citation_rigor(citations: str) -> List[str]:
    issues: List[str] = []
    if not citations or not citations.strip():
        return ["citations field is empty"]
    if _has_placeholder(citations):
        issues.append("citations: placeholder token")
    if not _RX_AUTHORS.search(citations):
        issues.append("citations: missing author surnames (need ≥ 2)")
    if not _RX_YEAR.search(citations):
        issues.append("citations: missing 4-digit year")
    if not _RX_VENUE.search(citations):
        issues.append("citations: missing venue (NeurIPS/ICML/ICLR/KDD/Nature/etc.)")
    if not _RX_TITLE.search(citations):
        issues.append("citations: missing single-quoted paper title")
    if (not _RX_ARXIV.search(citations)) and (not _RX_BIORXIV.search(citations)):
        issues.append("citations: missing arXiv (or bioRxiv) ID")
    if not _RX_RELEVANCE.search(citations):
        issues.append("citations: missing — / -- / : relevance separator")
    return issues


def validate_reasoning_blob(entry: ReasoningEntry, *, post_run: bool = False) -> List[str]:
    issues: List[str] = []
    pre_fields = [("diagnosis", WORD_FLOORS["diagnosis"]),
                   ("hypothesis", WORD_FLOORS["hypothesis"]),
                   ("prediction", WORD_FLOORS["prediction"])]
    for f, floor in pre_fields:
        s = getattr(entry, f, "") or ""
        if _has_placeholder(s):
            issues.append(f"{f}: placeholder")
        if entry.words(f) < floor:
            issues.append(f"{f}: word count {entry.words(f)} < floor {floor}")
    issues.extend(validate_citation_rigor(entry.citations))
    cw = entry.words("citations")
    n_papers = max(1, entry.citations.count(";") + 1)
    floor = WORD_FLOORS["citations_multi"] if n_papers > 1 else WORD_FLOORS["citations_single"]
    if cw < floor:
        issues.append(f"citations: word count {cw} < floor {floor} (n_papers~={n_papers})")
    if post_run:
        for f, floor in (("verdict", WORD_FLOORS["verdict"]),
                          ("learning", WORD_FLOORS["learning"])):
            s = getattr(entry, f, "") or ""
            if _has_placeholder(s):
                issues.append(f"{f}: placeholder")
            if entry.words(f) < floor:
                issues.append(f"{f}: word count {entry.words(f)} < floor {floor}")
    return issues


def load_reasoning_db(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_reasoning_db(path: str, db: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def commit_pre_run(annotations_path: str, experiment_num: int,
                   entry: ReasoningEntry, *, allow_bypass: bool = False
                   ) -> Tuple[bool, List[str]]:
    violations = validate_reasoning_blob(entry, post_run=False)
    db = load_reasoning_db(annotations_path)
    key = str(experiment_num)
    if violations and not allow_bypass:
        return False, violations
    if violations and allow_bypass:
        entry._needs_rewrite = True
    db[key] = asdict(entry)
    save_reasoning_db(annotations_path, db)
    return True, violations


def commit_post_run(annotations_path: str, experiment_num: int,
                    verdict: str, learning: str, *, fallback: bool = False) -> None:
    db = load_reasoning_db(annotations_path)
    key = str(experiment_num)
    rec = db.get(key, {})
    if fallback and not (verdict or "").strip():
        verdict = "TODO-REWRITE — Claude must rewrite verdict per dashboard rules"
        rec["_needs_rewrite"] = True
    if fallback and not (learning or "").strip():
        learning = "TODO-REWRITE — Claude must rewrite learning per dashboard rules"
        rec["_needs_rewrite"] = True
    rec["verdict"] = verdict
    rec["learning"] = learning
    db[key] = rec
    save_reasoning_db(annotations_path, db)
