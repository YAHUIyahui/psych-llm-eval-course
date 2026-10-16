"""Reproducibility: run manifests and drift monitoring on an anchor set."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def sha256_of(obj) -> str:
    """Hash of a file path's bytes, or of any JSON-serialisable object."""
    if isinstance(obj, str) and obj.endswith((".csv", ".json", ".jsonl", ".txt")):
        with open(obj, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_manifest(config: dict, dataset) -> dict:
    """Everything needed to rerun an evaluation and know what was run."""
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_sha256": sha256_of(dataset), "config": config,
            "config_sha256": sha256_of(config), "python": platform.python_version()}


def anchor_drift(history: pd.DataFrame, baseline_runs: int = 3, z: float = 3.0):
    """history: columns run, item, score. Uses the first `baseline_runs` runs to
    set control limits (mean +- z * SE of the run mean) and flags later runs."""
    runs = history.groupby("run")["score"].agg(["mean", "std", "count"]).reset_index()
    base = runs.iloc[:baseline_runs]
    centre = base["mean"].mean()
    se = (base["std"] / np.sqrt(base["count"])).mean()
    runs["lower"], runs["upper"] = centre - z * se, centre + z * se
    runs["flag"] = (runs["mean"] < runs["lower"]) | (runs["mean"] > runs["upper"])
    return runs
