# scripts/compare_runs.py
"""
Compare two benchmark run output directories for reproducibility.
Usage: python scripts/compare_runs.py results/run_A results/run_B

Exit 0 = runs are reproducibly identical
Exit 1 = divergence detected (see output for details)
"""
import json, sys, argparse, math
from pathlib import Path

TOLERANCE = 1e-6   # max allowed metric delta (floating point noise)

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def compare_metadata(a: dict, b: dict) -> list[str]:
    """Return list of divergence messages, empty = identical."""
    issues = []
    for key in ("git_commit", "seed", "config_hash"):
        if a.get(key) != b.get(key):
            issues.append(f"  DIVERGE  {key}: {a.get(key)!r} vs {b.get(key)!r}")
    return issues

def compare_metrics(a: dict, b: dict, path: str) -> list[str]:
    issues = []
    gm_a, gm_b = a["global_metrics"], b["global_metrics"]
    for metric in ("micro_f1", "macro_f1", "precision", "recall"):
        va, vb = gm_a[metric], gm_b[metric]
        delta = abs(va - vb)
        if delta > TOLERANCE:
            issues.append(
                f"  DIVERGE  {path} / {metric}: {va:.8f} vs {vb:.8f}  (delta={delta:.2e})"
            )
    # Per-attack-type check
    for atype in set(a["per_attack_metrics"]) | set(b["per_attack_metrics"]):
        if atype not in a["per_attack_metrics"]:
            issues.append(f"  MISSING  {path} / per_attack / {atype} in run A")
            continue
        if atype not in b["per_attack_metrics"]:
            issues.append(f"  MISSING  {path} / per_attack / {atype} in run B")
            continue
        for metric in ("micro_f1", "macro_f1"):
            va = a["per_attack_metrics"][atype][metric]
            vb = b["per_attack_metrics"][atype][metric]
            if abs(va - vb) > TOLERANCE:
                issues.append(
                    f"  DIVERGE  {path} / {atype} / {metric}: {va:.6f} vs {vb:.6f}"
                )
    return issues
def compare_datasets(a: dict, b: dict) -> list[str]:
    issues = []
    if a.get("content_hash") != b.get("content_hash"):
        issues.append(
            f"  DIVERGE  dataset content_hash: {a.get('content_hash')!r}"
            f" vs {b.get('content_hash')!r}"
        )
    return issues
def run_compare(dir_a: str, dir_b: str) -> int:
    A, B = Path(dir_a), Path(dir_b)
    all_issues = []
    ok_count = 0

    print(f"\nComparing runs:")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}\n")

    # 1. Metadata / reproducibility snapshot
    meta_a = load_json(A / "run_metadata.json")
    meta_b = load_json(B / "run_metadata.json")
    issues = compare_metadata(meta_a, meta_b)
    if issues:
        all_issues.extend(issues)
        print(f"[FAIL] run_metadata: {len(issues)} divergence(s)")
    else:
        ok_count += 1
        print(f"[OK]   run_metadata: git_commit, seed, config_hash identical")
    # 2. Per-model-dataset metrics files
    metrics_a = {p.name: p for p in (A / "metrics").glob("*.json")}
    metrics_b = {p.name: p for p in (B / "metrics").glob("*.json")}
    for name in sorted(set(metrics_a) | set(metrics_b)):
        if name not in metrics_a:
            all_issues.append(f"  MISSING  metrics/{name} in run A"); continue
        if name not in metrics_b:
            all_issues.append(f"  MISSING  metrics/{name} in run B"); continue
        issues = compare_metrics(
            load_json(metrics_a[name]),
            load_json(metrics_b[name]),
            f"metrics/{name}"
        )
        if issues:
            all_issues.extend(issues)
            print(f"[FAIL] metrics/{name}: {len(issues)} divergence(s)")
        else:
            ok_count += 1
            print(f"[OK]   metrics/{name}: all metrics within tolerance {TOLERANCE:.0e}")
    # 3. Dataset content hashes
    ds_a = load_json(A / "run_metadata.json").get("dataset_hashes", {})
    ds_b = load_json(B / "run_metadata.json").get("dataset_hashes", {})
    for ds_name in set(ds_a) | set(ds_b):
        if ds_a.get(ds_name) == ds_b.get(ds_name):
            ok_count += 1
            print(f"[OK]   dataset {ds_name}: content_hash identical")
        else:
            all_issues.append(
                f"  DIVERGE  dataset {ds_name}: {ds_a.get(ds_name)} vs {ds_b.get(ds_name)}"
            )
            print(f"[FAIL] dataset {ds_name}: content_hash mismatch")

    print(f"\n{'='*50}")
    if all_issues:
        print(f"AUDIT FAILED — {len(all_issues)} divergence(s) found:\n")
        for issue in all_issues:
            print(issue)
        print(f"\nRuns are NOT reproducible. See failure guide below.")
        return 1
    else:
        print(f"AUDIT PASSED — {ok_count} checks identical")
        print(f"Runs are reproducibly identical (tolerance {TOLERANCE:.0e})")
        return 0
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_a"); p.add_argument("run_b")
    sys.exit(run_compare(p.parse_args().run_a, p.parse_args().run_b))