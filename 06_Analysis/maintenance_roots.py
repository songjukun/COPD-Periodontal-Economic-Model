"""Reproduce the four cost-neutral maintenance roots in Supplementary Table S19.

No document templates, network access or earlier revision folders are required.
Run after analysis.py: python maintenance_roots.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
import analysis as core

HERE = Path(__file__).resolve().parent


def calculate():
    rows = []
    for source, costs in [("Tianjin", {}), ("Chongqing", core.chongqing())]:
        for severe in [0, 1]:
            result = core.run(maint=0, severe_effect=severe, **costs)
            factor = sum((1-core.BASE["m"])**t /
                         (1+core.BASE["discount"])**t
                         for t in core.BASE["maint_times"])
            root = -result["dc"]/factor
            residual = core.run(maint=root, severe_effect=severe, **costs)["dc"]
            assert abs(residual) < 1e-8
            rows.append(dict(
                source=source, severe_effect=severe,
                cost_with_zero_maintenance=float(result["dc"]),
                algebraic_root=float(root), feasible=bool(root >= 0)))
    assert len(rows) == 4
    assert [r["feasible"] for r in rows] == [False, True, False, True]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = calculate()
    # Independently check the published positive roots from the main analysis.
    main_summary = args.output_dir/"summary.json"
    if main_summary.exists():
        saved = json.loads(main_summary.read_text(encoding="utf-8"))
        assert math.isclose(rows[1]["algebraic_root"],
                            saved["maintenance_threshold"], abs_tol=1e-8)
        assert math.isclose(rows[3]["algebraic_root"],
                            saved["chongqing_maintenance_threshold"], abs_tol=1e-8)
    (args.output_dir/"maintenance_roots_20261020.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    record = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        checks=["Four analytic roots evaluated",
                "Each root gives zero incremental cost within 1e-8 CNY",
                "Moderate-only roots are negative and infeasible prices",
                "Both-severity roots are positive"],
        limitation="Conditional resource thresholds, not measured treatment efficacy.")
    (args.output_dir/"maintenance_roots_run_metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
