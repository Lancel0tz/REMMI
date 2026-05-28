#!/usr/bin/env python3
"""Cheap sanity check for the staged ATM-Bench data files.

Run after `bash scripts/download_data.sh`:

    python scripts/check_data_schema.py

This does NOT validate that every retrieval result is correct; it just
confirms the files are present and shaped the way the eval scripts expect.

Exits non-zero if any required file is missing or malformed, so it can be
piped into `bash scripts/setup_local_mac.sh` as a final sanity gate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ok(msg: str) -> None:
    print(f"  \033[0;32mOK  \033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[1;33mWARN\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[0;31mFAIL\033[0m {msg}")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        _fail(f"missing: {path.relative_to(REPO_ROOT)}")
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {path.relative_to(REPO_ROOT)}: {exc}")
        return None


def check_qa_file(path: Path) -> bool:
    print(f"\n[QA] {path.relative_to(REPO_ROOT)}")
    data = _load_json(path)
    if data is None:
        return False
    if isinstance(data, dict) and "qas" in data:
        items = data["qas"]
    elif isinstance(data, list):
        items = data
    else:
        _fail("expected list or {'qas': [...]} at top level")
        return False

    if not items:
        _fail("QA list is empty")
        return False

    required = {"id", "question", "answer", "evidence_ids"}
    sample = items[0]
    missing = required - sample.keys()
    if missing:
        _fail(f"first QA missing keys: {sorted(missing)}")
        return False

    n = len(items)
    n_with_evidence = sum(
        1 for it in items if isinstance(it.get("evidence_ids"), list) and it["evidence_ids"]
    )
    n_with_niah = sum(1 for it in items if it.get("niah_evidence_ids"))
    n_abstain = sum(
        1
        for it in items
        if not it.get("evidence_ids") or it.get("answer", "").strip().lower() == "unknown"
    )
    _ok(f"{n} QA items; {n_with_evidence} have evidence; "
        f"{n_with_niah} have niah_evidence_ids; ~{n_abstain} look like abstain cases")
    _ok(f"sample id = {sample['id']}, question = {sample['question'][:80]!r}")
    return True


def check_email_file(path: Path) -> bool:
    print(f"\n[Email] {path.relative_to(REPO_ROOT)}")
    if not path.exists():
        _warn(f"missing (optional): {path.relative_to(REPO_ROOT)}")
        return True
    data = _load_json(path)
    if not isinstance(data, list):
        _fail("expected a JSON list of email objects")
        return False
    if not data:
        _warn("email list is empty")
        return True
    sample = data[0]
    expected = {"id", "timestamp"}
    missing = expected - sample.keys()
    if missing:
        _fail(f"first email missing required keys: {sorted(missing)}")
        return False
    has_summary = "short_summary" in sample
    has_detail = "detail" in sample
    _ok(f"{len(data)} emails; sample id = {sample.get('id')!r}; "
        f"has short_summary={has_summary}, has detail={has_detail}")
    return True


def check_batch_file(path: Path, kind: str) -> bool:
    print(f"\n[{kind} batch] {path.relative_to(REPO_ROOT)}")
    if not path.exists():
        _fail(f"missing: {path.relative_to(REPO_ROOT)}")
        return False
    data = _load_json(path)
    if not isinstance(data, list):
        _fail("expected a JSON list of batch entries")
        return False
    if not data:
        _warn("batch list is empty")
        return True
    sample = data[0]
    path_key = "image_path" if kind == "image" else "video_path"
    if path_key not in sample:
        _fail(f"first entry missing {path_key}")
        return False
    sgm_fields = [k for k in ("timestamp", "location_name", "short_caption", "caption", "ocr_text", "tags") if k in sample]
    _ok(f"{len(data)} entries; sample {path_key} = {sample.get(path_key)!r}")
    _ok(f"SGM fields present in first entry: {sgm_fields}")
    return True


def check_niah_dir(niah_dir: Path) -> bool:
    print(f"\n[NIAH] {niah_dir.relative_to(REPO_ROOT)}")
    if not niah_dir.exists():
        _warn(f"missing (optional): {niah_dir.relative_to(REPO_ROOT)}")
        return True
    pool_files = sorted(niah_dir.glob("*.json"))
    if not pool_files:
        _warn("NIAH directory exists but contains no .json files")
        return True
    for pf in pool_files[:5]:
        sz = pf.stat().st_size
        _ok(f"{pf.name}  ({_human_size(sz)})")
    if len(pool_files) > 5:
        _ok(f"... and {len(pool_files) - 5} more pool files")
    return True


def main() -> int:
    print(f"Checking ATM-Bench data under {REPO_ROOT}")
    everything_ok = True

    qa_files = [
        REPO_ROOT / "data/atm-bench/atm-bench.json",
        REPO_ROOT / "data/atm-bench/atm-bench-hard.json",
    ]
    for qa in qa_files:
        if not check_qa_file(qa):
            everything_ok = False

    if not check_email_file(REPO_ROOT / "data/raw_memory/email/emails.json"):
        everything_ok = False

    for kind, batch in (
        ("image", REPO_ROOT / "output/image/qwen3vl2b/batch_results.json"),
        ("video", REPO_ROOT / "output/video/qwen3vl2b/batch_results.json"),
    ):
        if not check_batch_file(batch, kind):
            everything_ok = False

    check_niah_dir(REPO_ROOT / "data/atm-bench/niah")

    print()
    if everything_ok:
        print("\033[0;32mAll required schema checks PASSED.\033[0m")
        return 0
    print("\033[0;31mOne or more required files are missing/malformed.\033[0m")
    print("Re-run `bash scripts/download_data.sh` (HF login may be required).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
