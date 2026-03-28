from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _default_exp_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant_label", required=True)
    ap.add_argument("--exp_root", default="")
    ap.add_argument("--summary_dir", default="")
    ap.add_argument("--official_csv", default="")
    return ap.parse_args()


def _read_csv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(x: str | None) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0


def _headline(da: float, dc: float, dl: float) -> str:
    if da >= 0 and dc <= 0 and dl <= 0:
        return "variant_3d_dominates_official"
    if da <= 0 and dc >= 0 and dl >= 0:
        return "official_3d_dominates_variant"
    if da > 0:
        return "variant_acc_better_tradeoff_mixed"
    if da < 0:
        return "official_acc_better_tradeoff_mixed"
    return "mixed_tie_like"


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve() if args.exp_root else _default_exp_root()
    summary_dir = Path(args.summary_dir).resolve() if args.summary_dir else (exp_root / "test90_summary_variant" / args.variant_label)
    official_csv = Path(args.official_csv).resolve() if args.official_csv else (exp_root / "test90_summary" / "final_report_test90_fixed.csv")

    official_rows = _read_csv_rows(official_csv)
    official_by_ds = {}
    for row in official_rows:
        if row.get("method") != "TrustRoute":
            continue
        ds = row.get("dataset", "")
        if not ds:
            continue
        official_by_ds[ds] = row

    out_rows = []
    for csv_path in sorted(summary_dir.glob(f"*_{args.variant_label}_summary.csv")):
        dataset = csv_path.name[: -len(f"_{args.variant_label}_summary.csv")]
        rows = _read_csv_rows(csv_path)
        variant_row = None
        for row in rows:
            if row.get("method") == args.variant_label:
                variant_row = row
                break
        if not variant_row:
            continue
        official = official_by_ds.get(dataset)
        if not official:
            continue

        off_acc = _float(official.get("acc"))
        off_cost = _float(official.get("cost"))
        off_lat = _float(official.get("lat"))
        var_acc = _float(variant_row.get("acc_mean"))
        var_cost = _float(variant_row.get("cost_mean"))
        var_lat = _float(variant_row.get("latency_mean"))

        da = var_acc - off_acc
        dc = var_cost - off_cost
        dl = var_lat - off_lat

        out_rows.append(
            {
                "dataset": dataset,
                "official_source": official.get("source", ""),
                "official_acc": off_acc,
                "official_cost": off_cost,
                "official_lat": off_lat,
                "variant_acc": var_acc,
                "variant_cost": var_cost,
                "variant_lat": var_lat,
                "delta_acc_variant_minus_official": da,
                "delta_cost_variant_minus_official": dc,
                "delta_lat_variant_minus_official": dl,
                "headline_outcome": _headline(da, dc, dl),
            }
        )

    out_csv = summary_dir / f"{args.variant_label}_vs_official_summary.csv"
    out_md = summary_dir / f"{args.variant_label}_vs_official_summary.md"
    fieldnames = [
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
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        f"# {args.variant_label} vs Official",
        "",
        "| Dataset | Official Acc | Official Cost | Official Lat | Variant Acc | Variant Cost | Variant Lat | Delta Acc | Delta Cost | Delta Lat | Outcome |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in out_rows:
        lines.append(
            "| {dataset} | {official_acc:.4f} | {official_cost:.4e} | {official_lat:.3f} | {variant_acc:.4f} | {variant_cost:.4e} | {variant_lat:.3f} | {delta_acc_variant_minus_official:+.4f} | {delta_cost_variant_minus_official:+.4e} | {delta_lat_variant_minus_official:+.3f} | {headline_outcome} |".format(
                **row
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] csv -> {out_csv}")
    print(f"[OK] md -> {out_md}")


if __name__ == "__main__":
    main()
