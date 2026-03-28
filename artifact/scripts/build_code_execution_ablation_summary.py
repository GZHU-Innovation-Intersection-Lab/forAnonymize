#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


DATASETS = ["humaneval", "mbpp"]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tex_escape(s: str):
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def load_official(path: Path):
    out = {}
    for row in read_csv(path):
        if row.get("method") != "TrustRoute":
            continue
        if row.get("row_type") and row.get("row_type") != "mean":
            continue
        if row.get("dataset") in DATASETS:
            out[row["dataset"]] = row
    return out


def load_variant(summary_dir: Path, variant_label: str):
    out = {}
    for dataset in DATASETS:
        path = summary_dir / f"{dataset}_{variant_label}_summary.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        row = next((r for r in rows if r.get("method") == variant_label), None)
        if row is not None:
            out[dataset] = row
    return out


def main():
    repo_root = _default_repo_root()
    results_root = repo_root / "results"
    official_root = results_root / "official"
    ablations_root = results_root / "ablations"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--official_csv",
        default=str(official_root / "trustroute_mainline_only.csv"),
    )
    ap.add_argument(
        "--variant_summary_dir",
        default=str(ablations_root / "TrustRoute-NoLightTests"),
    )
    ap.add_argument("--variant_label", default="TrustRoute-NoLightTests")
    ap.add_argument(
        "--out_dir",
        default=str(ablations_root / "TrustRoute-NoLightTests"),
    )
    args = ap.parse_args()

    official = load_official(Path(args.official_csv))
    variant = load_variant(Path(args.variant_summary_dir), args.variant_label)

    rows = []
    md_lines = [
        "# Code Execution Ablation: Official TrustRoute vs No-Light-Tests Variant",
        "",
        "| Dataset | Official Acc / Cost / Lat | No-Light-Tests Acc / Cost / Lat | Headline |",
        "|---|---|---|---|",
    ]
    tex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & $\\Delta$Acc & $\\Delta$Cost & $\\Delta$Lat & Headline \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        off = official.get(dataset)
        var = variant.get(dataset)
        if off is None or var is None:
            continue
        off_acc = float(off["acc"])
        off_cost = float(off["cost"])
        off_lat = float(off["lat"])
        var_acc = float(var["acc_mean"])
        var_cost = float(var["cost_mean"])
        var_lat = float(var["latency_mean"])
        d_acc = var_acc - off_acc
        d_cost = var_cost - off_cost
        d_lat = var_lat - off_lat
        if d_acc > 0 and d_cost <= 0 and d_lat <= 0:
            headline = "no_light_tests_dominates"
        elif d_acc < 0 and d_cost >= 0 and d_lat >= 0:
            headline = "official_dominates"
        elif d_acc > 0:
            headline = "no_light_tests_acc_better_tradeoff_mixed"
        elif d_acc < 0:
            headline = "official_acc_better_tradeoff_mixed"
        else:
            headline = "mixed_tie"
        rows.append(
            {
                "dataset": dataset,
                "official_source": off["source"],
                "official_acc": off_acc,
                "official_cost": off_cost,
                "official_lat": off_lat,
                "variant_acc": var_acc,
                "variant_cost": var_cost,
                "variant_lat": var_lat,
                "delta_acc_variant_minus_official": d_acc,
                "delta_cost_variant_minus_official": d_cost,
                "delta_lat_variant_minus_official": d_lat,
                "headline_outcome": headline,
            }
        )
        md_lines.append(
            "| {} | {:.4f} / {:.4e} / {:.3f} | {:.4f} / {:.4e} / {:.3f} | {} |".format(
                dataset, off_acc, off_cost, off_lat, var_acc, var_cost, var_lat, headline
            )
        )
        tex_lines.append(
            "{} & {:+.4f} & {:+.2e} & {:+.2f} & {} \\\\".format(
                tex_escape(dataset), d_acc, d_cost, d_lat, tex_escape(headline)
            )
        )

    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "code_execution_ablation_summary.csv",
        rows,
        [
            "dataset",
            "official_source",
            "official_acc",
            "official_cost",
            "official_lat",
            "variant_acc",
            "variant_cost",
            "variant_lat",
            "delta_acc_variant_minus_official",
            "delta_cost_variant_minus_official",
            "delta_lat_variant_minus_official",
            "headline_outcome",
        ],
    )
    write_text(out_dir / "code_execution_ablation_summary.md", "\n".join(md_lines) + "\n")
    write_text(out_dir / "code_execution_ablation_summary.tex", "\n".join(tex_lines) + "\n")
    print(f"WROTE_CSV={out_dir / 'code_execution_ablation_summary.csv'}")
    print(f"WROTE_MD={out_dir / 'code_execution_ablation_summary.md'}")
    print(f"WROTE_TEX={out_dir / 'code_execution_ablation_summary.tex'}")


if __name__ == "__main__":
    main()
