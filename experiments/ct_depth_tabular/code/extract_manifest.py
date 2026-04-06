#!/usr/bin/env python3
"""
Parse 3D Slicer markups under data/ct_data_original → paired tabular manifest (CSV + JSON report).

Usage (repo root)::

  python experiments/ct_depth_tabular/code/extract_manifest.py \\
    --config experiments/ct_depth_tabular/config/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent


def _normalize_stem(stem: str) -> str:
    """Insert missing spaces before minimum/maximum/thoracic keywords (filename typos)."""
    s = stem.strip()
    s = re.sub(r"([0-9a-zA-Z)])(minimum)", r"\1 minimum", s, flags=re.IGNORECASE)
    s = re.sub(r"([0-9a-zA-Z)])(maximum)", r"\1 maximum", s, flags=re.IGNORECASE)
    s = re.sub(r"([0-9a-zA-Z)])(thoracic)", r"\1 thoracic", s, flags=re.IGNORECASE)
    return s


_KIND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\s+minimum\s+depth\s*$", re.IGNORECASE), "min"),
    (re.compile(r"\s+maximum\s+depth\s*$", re.IGNORECASE), "max"),
    (re.compile(r"\s+thoracic\s+depth\s*$", re.IGNORECASE), "thoracic"),
    (re.compile(r"\s+anterior\s*$", re.IGNORECASE), "anterior"),
    (re.compile(r"\s+posterior\s*$", re.IGNORECASE), "posterior"),
]


def classify_stem(stem: str) -> tuple[str | None, str | None]:
    """
    Return (case_key, kind) where kind in min|max|thoracic|anterior|posterior|None.
    """
    norm = _normalize_stem(stem)
    for pat, kind in _KIND_PATTERNS:
        m = pat.search(norm)
        if m:
            case_key = norm[: m.start()].strip()
            case_key = re.sub(r"\s+", " ", case_key)
            return case_key, kind
    return None, None


def parse_age_years(case_key: str) -> int | None:
    """Leading digits before optional y/Y (e.g. 77y -3 → 77)."""
    m = re.match(r"^\s*(\d+)", case_key.strip())
    return int(m.group(1)) if m else None


def extract_line_length_mm(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None
    for markup in data.get("markups", []):
        if markup.get("type") != "Line":
            continue
        for meas in markup.get("measurements", []):
            if meas.get("name") != "length" or not meas.get("enabled", True):
                continue
            units = str(meas.get("units", "mm")).lower()
            if units != "mm":
                continue
            try:
                return float(meas["value"])
            except (TypeError, ValueError):
                return None
    return None


def sex_from_folder(folder_name: str) -> str:
    low = folder_name.strip().lower()
    if low == "male":
        return "male"
    if low == "female":
        return "female"
    return low


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CT depth manifest from Slicer .mrk.json files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_CODE_DIR.parent / "config" / "default.yaml",
        help="YAML config (default: experiment config/default.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ext = cfg.get("extract", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    ct_root = _REPO_ROOT / str(cfg.get("ct_original_root", "data/ct_data_original"))
    if not ct_root.is_dir():
        print(f"ct_original_root not found: {ct_root}", file=sys.stderr)
        return 1

    swap_if_reversed = bool(ext.get("swap_if_reversed", True))
    drop_missing_age = bool(ext.get("drop_missing_age", False))
    include_thoracic = bool(ext.get("include_thoracic", True))

    manifest_path = _REPO_ROOT / str(out_cfg.get("manifest_csv", "experiments/ct_depth_tabular/data/ct_depth_manifest.csv"))
    report_path = _REPO_ROOT / str(out_cfg.get("report_json", "experiments/ct_depth_tabular/data/extraction_report.json"))

    # (sex, case_key) -> dict of paths and mm values
    bucket: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "min_mm": None,
            "max_mm": None,
            "thoracic_mm": None,
            "path_min": None,
            "path_max": None,
            "path_thoracic": None,
        }
    )

    skipped: list[dict[str, str]] = []
    unclassified: list[str] = []

    for sex_dir in sorted(ct_root.iterdir()):
        if not sex_dir.is_dir():
            continue
        sex = sex_from_folder(sex_dir.name)
        for f in sorted(sex_dir.glob("*.mrk.json")):
            stem = f.name[: -len(".mrk.json")]
            case_key, kind = classify_stem(stem)
            if kind is None or not case_key:
                unclassified.append(str(f.relative_to(_REPO_ROOT)))
                continue
            if kind in ("anterior", "posterior"):
                continue

            mm = extract_line_length_mm(f)
            if mm is None:
                skipped.append({"path": str(f.relative_to(_REPO_ROOT)), "reason": "no_line_length_mm"})
                continue

            key = (sex, case_key)
            rel = str(f.relative_to(_REPO_ROOT)).replace("\\", "/")
            if kind == "min":
                bucket[key]["min_mm"] = mm
                bucket[key]["path_min"] = rel
            elif kind == "max":
                bucket[key]["max_mm"] = mm
                bucket[key]["path_max"] = rel
            elif kind == "thoracic":
                bucket[key]["thoracic_mm"] = mm
                bucket[key]["path_thoracic"] = rel

    rows: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []

    for (sex, case_key), d in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        min_mm, max_mm = d["min_mm"], d["max_mm"]
        if min_mm is None or max_mm is None:
            unpaired.append(
                {
                    "sex": sex,
                    "case_key": case_key,
                    "has_min": min_mm is not None,
                    "has_max": max_mm is not None,
                }
            )
            continue

        min_cm = min_mm / 10.0
        max_cm = max_mm / 10.0
        swapped = False
        if max_cm < min_cm and swap_if_reversed:
            min_cm, max_cm = max_cm, min_cm
            swapped = True

        age = parse_age_years(case_key)
        if age is None and drop_missing_age:
            skipped.append(
                {
                    "path": f"{sex}/{case_key}",
                    "reason": "missing_age_and_drop_missing_age",
                }
            )
            continue

        row: dict[str, Any] = {
            "case_key": case_key,
            "sex": sex,
            "age_years": age if age is not None else "",
            "depth_min_cm": round(min_cm, 4),
            "depth_max_cm": round(max_cm, 4),
            "source_min_path": d["path_min"],
            "source_max_path": d["path_max"],
        }
        if age is not None:
            row["age_sq"] = round(float(age * age), 6)
            row["male_age"] = round(float(age) if sex == "male" else 0.0, 6)
        else:
            row["age_sq"] = ""
            row["male_age"] = ""
        if swapped:
            row["swapped_min_max"] = True
        if include_thoracic and d["thoracic_mm"] is not None:
            row["thoracic_depth_mm"] = round(float(d["thoracic_mm"]), 4)
            row["source_thoracic_path"] = d["path_thoracic"]
        rows.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_key",
        "sex",
        "age_years",
        "age_sq",
        "male_age",
        "depth_min_cm",
        "depth_max_cm",
        "thoracic_depth_mm",
        "source_min_path",
        "source_max_path",
        "source_thoracic_path",
        "swapped_min_max",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    report: dict[str, Any] = {
        "ct_root": str(ct_root.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "paired_rows": len(rows),
        "unpaired_case_keys": len(unpaired),
        "unclassified_files": len(unclassified),
        "skipped_files": len(skipped),
        "unpaired": unpaired[:200],
        "unpaired_truncated": len(unpaired) > 200,
        "unclassified_sample": unclassified[:50],
        "unclassified_truncated": len(unclassified) > 50,
        "skipped_sample": skipped[:100],
        "skipped_truncated": len(skipped) > 100,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"manifest": str(manifest_path.relative_to(_REPO_ROOT)), "paired": len(rows), "report": str(report_path.relative_to(_REPO_ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
