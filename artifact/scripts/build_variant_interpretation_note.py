#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_one(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _summarize_audit(rows: list[dict]) -> dict:
    if not rows:
        return {}
    total_duplicates = 0
    all_unique_complete = True
    all_ok_complete = True
    total_unique = 0
    total_ok = 0
    total_missing = 0
    for row in rows:
        total_duplicates += int(float(row.get("duplicate_rows", 0) or 0))
        all_unique_complete = all_unique_complete and str(row.get("complete_unique_coverage", "0")) == "1"
        all_ok_complete = all_ok_complete and str(row.get("complete_ok_coverage", "0")) == "1"
        total_unique += int(float(row.get("unique_task_ids", 0) or 0))
        total_ok += int(float(row.get("ok_unique", 0) or 0))
        total_missing += int(float(row.get("missing_unique", 0) or 0))
    return {
        "complete_unique_coverage": "1" if all_unique_complete else "0",
        "complete_ok_coverage": "1" if all_ok_complete else "0",
        "duplicate_rows": str(total_duplicates),
        "unique_task_ids_total": str(total_unique),
        "ok_unique_total": str(total_ok),
        "missing_unique_total": str(total_missing),
        "n_seed_files": str(len(rows)),
    }


def _classify_status(audit_row: dict) -> str:
    if not audit_row:
        return "rebuttal-only"
    if str(audit_row.get("complete_unique_coverage", "0")) != "1":
        return "stale-do-not-use"
    if str(audit_row.get("complete_ok_coverage", "0")) != "1":
        return "rebuttal-only"
    if int(float(audit_row.get("duplicate_rows", 0) or 0)) > 0:
        return "rebuttal-only"
    return "paper-usable"


def _reviewer_usefulness(dataset: str, variant_label: str) -> str:
    if variant_label in {"CodeExecRerank", "TrustRoute-NoLightTests"} or dataset in {"humaneval", "mbpp"}:
        return "reviewer C primary, reviewer A secondary"
    if variant_label.startswith("TrustRoute-Strongest") or variant_label == "TrustRoute-MaxConf":
        return "reviewer A primary, reviewer B secondary"
    return "reviewer B secondary / triage support"


def _signal(vs_row: dict, paired_row: dict) -> str:
    headline = str((paired_row or {}).get("headline_outcome") or (vs_row or {}).get("headline_outcome") or "")
    if headline.startswith("official_"):
        return "favorable"
    if headline.startswith("variant_"):
        return "unfavorable"
    return "mixed"


def _main_text_worth(status: str, signal: str, variant_label: str) -> str:
    if status == "stale-do-not-use":
        return "triage only"
    if signal == "favorable" and variant_label in {"CodeExecRerank", "TrustRoute-StrongestFallback", "TrustRoute-MaxConf", "TrustRoute-StrongestQueried"}:
        return "rebuttal main text"
    if signal == "favorable":
        return "rebuttal appendix"
    if signal == "mixed":
        return "rebuttal appendix / cautious mention"
    return "do not cite as supportive evidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant_label", required=True)
    ap.add_argument("--summary_dir", required=True)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    audit_csv = summary_dir / f"{args.dataset}_{args.variant_label}_audit.csv"
    vs_csv = summary_dir / f"{args.variant_label}_vs_official_summary.csv"
    paired_csv = summary_dir / f"{args.dataset}_{args.variant_label}_paired_vs_official.csv"

    audit_rows = _read_rows(audit_csv) if audit_csv.exists() else []
    audit_row = _summarize_audit(audit_rows)
    vs_rows = []
    if vs_csv.exists():
        with vs_csv.open("r", encoding="utf-8", newline="") as f:
            vs_rows = list(csv.DictReader(f))
    vs_row = next((r for r in vs_rows if r.get("dataset") == args.dataset), {})
    paired_row = _read_one(paired_csv) if paired_csv.exists() else {}

    status = _classify_status(audit_row)
    signal = _signal(vs_row, paired_row)
    usefulness = _reviewer_usefulness(args.dataset, args.variant_label)
    worth = _main_text_worth(status, signal, args.variant_label)

    note = "\n".join(
        [
            f"# Interpretation: {args.dataset} / {args.variant_label}",
            "",
            f"- Reviewer usefulness: {usefulness}.",
            f"- Evidence status: `{status}`.",
            f"- Result direction: `{signal}` relative to official TrustRoute.",
            f"- Main-text decision: {worth}.",
            (
                f"- Coverage/audit: unique coverage={audit_row.get('complete_unique_coverage', 'n/a')}, "
                f"ok coverage={audit_row.get('complete_ok_coverage', 'n/a')}, "
                f"duplicate rows={audit_row.get('duplicate_rows', 'n/a')}, "
                f"seed files={audit_row.get('n_seed_files', 'n/a')}."
            ),
            (
                f"- vs_official headline: `{vs_row.get('headline_outcome', 'n/a')}`; "
                f"paired headline: `{paired_row.get('headline_outcome', 'n/a')}`."
            ),
            (
                "- Rebuttal guidance: use only if this variant is fully audited and its claim stays within the "
                "specific reviewer question it addresses; otherwise keep it in triage/appendix."
            ),
            "",
        ]
    )
    out_md = summary_dir / f"{args.dataset}_{args.variant_label}_interpretation.md"
    out_md.write_text(note, encoding="utf-8")
    print(f"[OK] md -> {out_md}")


if __name__ == "__main__":
    main()
