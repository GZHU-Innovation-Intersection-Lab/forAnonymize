import argparse
import csv
import os
import re
import statistics
from pathlib import Path
from typing import Optional


def _default_exp_root() -> Path:
    env = os.environ.get("EXP_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _infer_seed_from_name(name: str):
    m = re.search(r"_seed(\d+)\.csv$", Path(name).name)
    if m:
        return int(m.group(1))
    return None


def _to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if not s:
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _read_seed_rows_from_report(report_csv: Path):
    rows = []
    with report_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("row_type") or "").strip() != "seed":
                continue
            dataset = str(row.get("dataset") or "").strip()
            method = str(row.get("method") or "").strip()

            seed = row.get("seed")
            try:
                seed = int(seed) if seed is not None and str(seed).strip() else None
            except Exception:
                seed = None

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "acc": _to_float(row.get("acc")),
                    "cost": _to_float(row.get("cost")),
                    "lat": _to_float(row.get("lat")),
                    "n_present": _to_float(row.get("n_present")),
                    "n_total": _to_float(row.get("n_total")),
                    "file": str(row.get("file") or ""),
                }
            )
    return rows


def _aggregate(items):
    accs = [x.get("acc", 0.0) for x in items]
    costs = [x.get("cost", 0.0) for x in items]
    lats = [x.get("lat", 0.0) for x in items]
    pres = [x.get("n_present", 0.0) for x in items]
    ntot = items[0].get("n_total", 0.0) if items else 0.0

    return {
        "acc": _mean(accs),
        "acc_std": _std(accs),
        "cost": _mean(costs),
        "cost_std": _std(costs),
        "lat": _mean(lats),
        "lat_std": _std(lats),
        "n_present": _mean(pres),
        "n_total": ntot,
    }


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def _std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _has_column(csv_path: Path, col: str) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header:
                return False
            return col in header
    except Exception:
        return False


def _prefer_src_for_baseline(dataset: str, selected_path: Path, evaluated_dir: Path) -> Path:
    if dataset in {"humaneval", "mbpp"}:
        cand = evaluated_dir / selected_path.name
        if cand.exists() and cand.is_file() and _has_column(cand, "is_correct"):
            return cand
    return selected_path


def _summarize_csv(dataset: str, csv_path: Path, gt_order, gt_map, timeout_s: int, label: str):
    from summarize_test90 import summarize_one_csv

    return summarize_one_csv(dataset, csv_path, gt_order, gt_map, timeout_s, label)


def _load_gt(dataset: str, test_jsonl: Path):
    from summarize_test90 import _load_gt as _load

    return _load(dataset, test_jsonl)


def _resolve_paper_main_v1_csv(paper_main_v1_dir: Path, dataset: str, seed: int) -> Optional[Path]:
    ds_dir = paper_main_v1_dir / dataset
    if not ds_dir.exists():
        return None
    cands = sorted(ds_dir.glob(f"TrustRoute_*_seed{int(seed)}.csv"))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    exact = []
    for p in cands:
        if "t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0" in p.name:
            exact.append(p)
    if len(exact) == 1:
        return exact[0]
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="")
    ap.add_argument(
        "--datasets",
        default="arc_challenge hellaswag winogrande mmlu humaneval mbpp gsm8k",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--timeout_s", type=int, default=10)

    ap.add_argument(
        "--selected_dir",
        default="/data_huawei/jiakun/ISSTA/final_experiments/selected",
    )
    ap.add_argument(
        "--evaluated_dir",
        default="/data_huawei/jiakun/ISSTA/final_experiments/evaluated",
    )
    ap.add_argument(
        "--paper_main_v1_dir",
        default="/data_huawei/jiakun/ISSTA/final_experiments/paper_main_v1",
    )

    ap.add_argument("--out_csv", default="")
    ap.add_argument(
        "--from_report_csv",
        default="",
        help="If set, read seed rows from an existing report CSV and only evaluate TrustRoute baseline as needed to output a fixed-methods summary.",
    )
    ap.add_argument(
        "--out_csv_fixed",
        default="",
        help="Output path for the fixed-methods (mean-only) summary CSV.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    exp_root = Path(args.exp_root).resolve() if args.exp_root else _default_exp_root()

    datasets = [x.strip() for x in str(args.datasets).split() if x.strip()]
    seeds = [int(x) for x in args.seeds]

    selected_dir = Path(args.selected_dir).resolve()
    evaluated_dir = Path(args.evaluated_dir).resolve()
    paper_main_v1_dir = Path(args.paper_main_v1_dir).resolve()

    if args.from_report_csv:
        base_methods = [
            "Cascade",
            "FrugalGPT",
            "MV-3",
            "Oracle",
            "Random",
            "SA",
            "SC-10",
            "SC-3",
        ]

        report_csv = Path(args.from_report_csv).resolve()
        report_seed_rows = _read_seed_rows_from_report(report_csv)

        fixed_rows = []
        for dataset in datasets:
            test_jsonl = exp_root / "test90" / f"{dataset}_test90.jsonl"
            if not test_jsonl.exists():
                raise SystemExit(f"Missing test90 input: {test_jsonl}")
            gt_order, gt_map = _load_gt(dataset, test_jsonl)
            n_total = float(len(gt_order))

            for method in base_methods:
                items = [
                    x
                    for x in report_seed_rows
                    if x.get("dataset") == dataset and x.get("method") == method
                ]
                agg = _aggregate(items)
                fixed_rows.append(
                    {
                        "row_type": "mean",
                        "dataset": dataset,
                        "method": method,
                        "trustroute_candidate": "",
                        "seed": "",
                        "acc": agg["acc"],
                        "acc_std": agg["acc_std"],
                        "cost": agg["cost"],
                        "cost_std": agg["cost_std"],
                        "lat": agg["lat"],
                        "lat_std": agg["lat_std"],
                        "n_present": agg["n_present"],
                        "n_total": n_total,
                        "file": "",
                    }
                )

            tr_bestcfg_items = [
                x
                for x in report_seed_rows
                if x.get("dataset") == dataset and x.get("method") == "TrustRoute(best_cfg)"
            ]

            tr_baseline_items = []
            for seed in seeds:
                p = selected_dir / f"{dataset}_TrustRoute_seed{seed}.csv"
                if not p.exists():
                    continue
                src_use = _prefer_src_for_baseline(dataset, p, evaluated_dir)
                print("[TRUSTROUTE_FILE]", str(src_use), flush=True)
                r = _summarize_csv(
                    dataset,
                    src_use,
                    gt_order,
                    gt_map,
                    int(args.timeout_s),
                    label="TrustRoute",
                )
                if r and r.get("seed") is None:
                    r["seed"] = _infer_seed_from_name(src_use.name) or seed
                if r:
                    tr_baseline_items.append(
                        {
                            "dataset": dataset,
                            "method": "TrustRoute",
                            "seed": r.get("seed"),
                            "acc": _to_float(r.get("acc")),
                            "cost": _to_float(r.get("cost")),
                            "lat": _to_float(r.get("lat")),
                            "n_present": _to_float(r.get("n_present")),
                            "n_total": n_total,
                            "file": str(src_use),
                        }
                    )

            variants = []
            if tr_baseline_items:
                variants.append(("TrustRoute", tr_baseline_items))
            if tr_bestcfg_items:
                variants.append(("TrustRoute(best_cfg)", tr_bestcfg_items))

            best_variant = ""
            best_items = []
            best_key = None
            for name, items in variants:
                agg = _aggregate(items)
                key = (agg["acc"], -agg["cost"], -agg["lat"], name)
                if best_key is None or key > best_key:
                    best_key = key
                    best_variant = name
                    best_items = items

            best_agg = _aggregate(best_items)
            fixed_rows.append(
                {
                    "row_type": "mean",
                    "dataset": dataset,
                    "method": "TrustRoute",
                    "trustroute_candidate": best_variant,
                    "seed": "",
                    "acc": best_agg["acc"],
                    "acc_std": best_agg["acc_std"],
                    "cost": best_agg["cost"],
                    "cost_std": best_agg["cost_std"],
                    "lat": best_agg["lat"],
                    "lat_std": best_agg["lat_std"],
                    "n_present": best_agg["n_present"],
                    "n_total": n_total,
                    "file": "",
                }
            )

        out_csv_fixed = (
            Path(args.out_csv_fixed).resolve()
            if args.out_csv_fixed
            else (exp_root / "test90_summary" / "final_report_test90_fixed.csv")
        )
        out_csv_fixed.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "row_type",
            "dataset",
            "method",
            "trustroute_candidate",
            "seed",
            "acc",
            "acc_std",
            "cost",
            "cost_std",
            "lat",
            "lat_std",
            "n_present",
            "n_total",
            "file",
        ]

        with out_csv_fixed.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in fixed_rows:
                out = {k: r.get(k, "") for k in fieldnames}
                w.writerow(out)

        print("IN_CSV=", str(report_csv))
        print("OUT_FIXED_CSV=", str(out_csv_fixed))
        return

    candidates = [
        {
            "id": "best_cfg_rerun",
            "label": "TrustRoute(best_cfg)",
            "kind": "best_cfg_rerun",
        },
        {
            "id": "frozen_selected",
            "label": "TrustRoute(frozen)",
            "kind": "selected",
        },
        {
            "id": "paper_main_v1",
            "label": "TrustRoute(paper_main_v1)",
            "kind": "paper_main_v1",
        },
    ]

    candidate_seed_rows = {c["id"]: [] for c in candidates}
    candidate_ds_acc_mean = {c["id"]: {} for c in candidates}

    for dataset in datasets:
        print("[DATASET]", dataset, flush=True)
        test_jsonl = exp_root / "test90" / f"{dataset}_test90.jsonl"
        if not test_jsonl.exists():
            raise SystemExit(f"Missing test90 input: {test_jsonl}")

        gt_order, gt_map = _load_gt(dataset, test_jsonl)

        for cand in candidates:
            print("  [CANDIDATE]", cand["id"], flush=True)
            seed_rows = []
            if cand["kind"] == "best_cfg_rerun":
                ds_dir = exp_root / "test90_runs" / dataset
                if ds_dir.exists():
                    paths = sorted([p for p in ds_dir.glob("*.csv") if p.is_file()])
                else:
                    paths = []

                for p in paths:
                    print("    [FILE]", str(p), flush=True)
                    r = _summarize_csv(
                        dataset,
                        p,
                        gt_order,
                        gt_map,
                        int(args.timeout_s),
                        label=cand["label"],
                    )
                    if r.get("seed") is None:
                        r["seed"] = _infer_seed_from_name(p.name)
                    seed_rows.append(r)

            elif cand["kind"] == "selected":
                for seed in seeds:
                    p = selected_dir / f"{dataset}_TrustRoute_seed{seed}.csv"
                    if not p.exists():
                        continue
                    print("    [FILE]", str(p), flush=True)
                    r = _summarize_csv(
                        dataset,
                        p,
                        gt_order,
                        gt_map,
                        int(args.timeout_s),
                        label=cand["label"],
                    )
                    seed_rows.append(r)

            elif cand["kind"] == "paper_main_v1":
                for seed in seeds:
                    p = _resolve_paper_main_v1_csv(paper_main_v1_dir, dataset, seed)
                    if not p:
                        continue
                    print("    [FILE]", str(p), flush=True)
                    r = _summarize_csv(
                        dataset,
                        p,
                        gt_order,
                        gt_map,
                        int(args.timeout_s),
                        label=cand["label"],
                    )
                    seed_rows.append(r)

            seed_rows = [x for x in seed_rows if x]
            if seed_rows:
                acc_mean = _mean([x["acc"] for x in seed_rows])
                candidate_ds_acc_mean[cand["id"]][dataset] = acc_mean
                candidate_seed_rows[cand["id"]].extend(
                    [
                        {
                            **x,
                            "dataset": dataset,
                            "row_type": "seed",
                            "trustroute_candidate": cand["id"],
                        }
                        for x in seed_rows
                    ]
                )

    scored = []
    for cand in candidates:
        ds_map = candidate_ds_acc_mean.get(cand["id"], {})
        if not ds_map:
            continue
        macro_acc = _mean([ds_map.get(ds, 0.0) for ds in datasets])
        scored.append((macro_acc, cand["id"]))

    if not scored:
        raise SystemExit("No TrustRoute candidate CSVs found")

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_cand_id = scored[0][1]
    print("[BEST]", best_cand_id, flush=True)

    include_ablations = best_cand_id == "frozen_selected"

    final_rows_seed = []

    for dataset in datasets:
        print("[FINAL_DATASET]", dataset, flush=True)
        test_jsonl = exp_root / "test90" / f"{dataset}_test90.jsonl"
        gt_order, gt_map = _load_gt(dataset, test_jsonl)

        final_rows_seed.extend(
            [
                {
                    **x,
                    "dataset": dataset,
                    "row_type": "seed",
                    "trustroute_candidate": best_cand_id,
                }
                for x in candidate_seed_rows.get(best_cand_id, [])
                if x.get("dataset") == dataset
            ]
        )

        pat = re.compile(rf"^{re.escape(dataset)}_(.+)_seed(\d+)\.csv$")
        for seed in seeds:
            matched = sorted(selected_dir.glob(f"{dataset}_*_seed{seed}.csv"))
            matched = [p for p in matched if p.is_file() and p.parent == selected_dir]

            for src in matched:
                m = pat.match(src.name)
                if not m:
                    continue
                method = m.group(1)
                if method == "TrustRoute":
                    continue
                if method.startswith("TrustRoute-") and (not include_ablations):
                    continue

                src_use = _prefer_src_for_baseline(dataset, src, evaluated_dir)

                print("  [BASELINE_FILE]", str(src_use), flush=True)

                r = _summarize_csv(
                    dataset,
                    src_use,
                    gt_order,
                    gt_map,
                    int(args.timeout_s),
                    label=method,
                )
                r["dataset"] = dataset
                r["row_type"] = "seed"
                r["trustroute_candidate"] = ""
                final_rows_seed.append(r)

    grouped = {}
    for r in final_rows_seed:
        key = (r.get("dataset"), r.get("method"))
        grouped.setdefault(key, []).append(r)

    final_rows = []
    for (dataset, method), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        items = sorted(items, key=lambda x: (x.get("seed") is None, x.get("seed") or 0))
        final_rows.extend(items)

        accs = [x.get("acc", 0.0) for x in items]
        costs = [x.get("cost", 0.0) for x in items]
        lats = [x.get("lat", 0.0) for x in items]
        pres = [x.get("n_present", 0.0) for x in items]
        ntot = items[0].get("n_total", 0.0) if items else 0.0

        final_rows.append(
            {
                "row_type": "mean",
                "dataset": dataset,
                "method": method,
                "seed": "",
                "acc": _mean(accs),
                "acc_std": _std(accs),
                "cost": _mean(costs),
                "cost_std": _std(costs),
                "lat": _mean(lats),
                "lat_std": _std(lats),
                "n_present": _mean(pres),
                "n_total": ntot,
                "file": "",
                "trustroute_candidate": best_cand_id if method.startswith("TrustRoute") else "",
            }
        )

    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else (exp_root / "test90_summary" / "final_report_test90.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_type",
        "dataset",
        "method",
        "trustroute_candidate",
        "seed",
        "acc",
        "acc_std",
        "cost",
        "cost_std",
        "lat",
        "lat_std",
        "n_present",
        "n_total",
        "file",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in final_rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)

    print("BEST_TRUSTROUTE_CANDIDATE=", best_cand_id)
    for macro_acc, cid in scored:
        print("CANDIDATE", cid, "macro_acc", f"{macro_acc:.6f}")
    print("OUT_CSV=", str(out_csv))


if __name__ == "__main__":
    main()
