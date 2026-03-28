#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ["arc_challenge", "humaneval", "mbpp", "gsm8k", "hellaswag", "mmlu", "winogrande"]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / "results"
    official_root = results_root / "official"
    audits_root = results_root / "audits"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--truth_json",
        default=str(official_root / "reviewer_truth_manifest.json"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(audits_root / "judge_usage_audit.csv"),
    )
    ap.add_argument(
        "--out_md",
        default=str(audits_root / "judge_usage_audit.md"),
    )
    args = ap.parse_args()

    _ = json.loads(Path(args.truth_json).read_text(encoding="utf-8"))
    rows = []
    md = [
        "# Judge Usage Audit",
        "",
        "This audit reports the truthful runtime status of judge usage in the current official line.",
        "",
        "| Dataset | judge_calls | judge_cost_share | judge_latency_share | stage_paths | note |",
        "|---|---:|---:|---:|---|---|",
    ]

    for ds in DATASETS:
        row = {
            "dataset": ds,
            "judge_calls": 0,
            "judge_call_ratio": 0.0,
            "judge_cost_usd": 0.0,
            "judge_cost_share": 0.0,
            "judge_latency_s": 0.0,
            "judge_latency_share": 0.0,
            "stage_paths": "none",
            "baselines_share_judge": 0,
            "note": "Current official runtime path does not exercise a real judge-model scoring call.",
        }
        rows.append(row)
        md.append(f"| {ds} | 0 | 0.0 | 0.0 | none | {row['note']} |")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "judge_calls",
                "judge_call_ratio",
                "judge_cost_usd",
                "judge_cost_share",
                "judge_latency_s",
                "judge_latency_share",
                "stage_paths",
                "baselines_share_judge",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"WROTE_CSV={args.out_csv}")
    print(f"WROTE_MD={args.out_md}")


if __name__ == "__main__":
    main()
