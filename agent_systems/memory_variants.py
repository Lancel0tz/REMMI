#!/usr/bin/env python3
"""Build memory input variants for agent_systems sandboxes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


VALID_MEMORY_MODES = ("sgm", "raw", "descriptive", "org_heuristic", "org_static", "org_dynamic", "org_hybrid", "org_remmi")


def normalize_memory_mode(mode: str) -> str:
    normalized = (mode or "sgm").strip().lower().replace("-", "_")
    aliases = {
        "baseline": "sgm",
        "full": "sgm",
        "description": "descriptive",
        "dm": "descriptive",
        "descriptive_memory": "descriptive",
        "raw_entries": "raw",
        "raw_media": "raw",
        "orgh": "org_heuristic",
        "organized_heuristic": "org_heuristic",
        "orgs": "org_static",
        "organized_static": "org_static",
        "orgd": "org_dynamic",
        "organized_dynamic": "org_dynamic",
        "orgx": "org_hybrid",
        "organized_hybrid": "org_hybrid",
        "orgr": "org_remmi",
        "organized_remmi": "org_remmi",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_MEMORY_MODES:
        raise ValueError(f"Unsupported memory mode: {mode!r}. Expected one of: {', '.join(VALID_MEMORY_MODES)}")
    return normalized


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def dump_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def item_id(item: dict[str, Any], path_key: str) -> str:
    path_value = item.get(path_key, "")
    if not path_value:
        raise ValueError(f"Missing {path_key} in metadata item")
    return Path(str(path_value)).stem


def media_name(item: dict[str, Any], path_key: str) -> str:
    path_value = item.get(path_key, "")
    if not path_value:
        raise ValueError(f"Missing {path_key} in metadata item")
    return Path(str(path_value)).name


def raw_entries(records: list[dict[str, Any]], path_key: str, media_dir_name: str) -> list[dict[str, str]]:
    return [
        {
            "id": item_id(item, path_key),
            path_key: f"memory/{media_dir_name}/{media_name(item, path_key)}",
        }
        for item in records
    ]


def descriptive_entries(records: list[dict[str, Any]], path_key: str) -> list[dict[str, str]]:
    return [
        {
            "id": item_id(item, path_key),
            "caption": str(item.get("caption", "")),
        }
        for item in records
    ]


def hardlink_media_files(
    records: list[dict[str, Any]],
    path_key: str,
    source_dir: Path,
    dst_dir: Path,
) -> dict[str, Any]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Raw media source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Raw media source is not a directory: {source_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing: list[str] = []
    for item in records:
        name = media_name(item, path_key)
        src = source_dir / name
        dst = dst_dir / name
        if not src.exists():
            missing.append(name)
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError as exc:
            raise OSError(
                f"Could not hardlink raw media {src} -> {dst}. "
                "Set AGSYS_RAW_IMAGE_DIR/AGSYS_RAW_VIDEO_DIR to media on the same filesystem, "
                "or extend memory_variants.py with an explicit copy mode."
            ) from exc
        linked += 1

    return {
        "source_dir": str(source_dir),
        "dst_dir": str(dst_dir),
        "linked": linked,
        "missing": missing,
    }


def build_organized_index(
    *,
    mode: str,
    image_source: Path,
    video_source: Path,
    emails_source: Path,
    out_dir: Path,
    organized_source: Path | None,
) -> dict[str, Any]:
    """Write ``organized_memory.json`` next to the SGM files.

    - ``org_heuristic`` builds the event index deterministically via
      ``remmi.organize.heuristic`` (no LLM calls).
    - ``org_static`` consumes a prebuilt index (produced offline by
      ``python -m remmi.organize.agent_static``), passed via
      ``--organized-source`` / ``AGSYS_ORGANIZED_MEMORY``.
    """
    if mode == "org_heuristic":
        import sys

        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from remmi.organize.heuristic import build_organized_memory

        payload = build_organized_memory(image_source, video_source, emails_source)
        dump_json(out_dir / "organized_memory.json", payload)
    elif mode in ("org_static", "org_hybrid"):
        if organized_source is None or not Path(organized_source).exists():
            raise FileNotFoundError(
                f"{mode} requires a prebuilt organized memory index. Generate one with "
                "`python -m remmi.organize.agent_static ... --out <file>` and pass it via "
                "--organized-source or AGSYS_ORGANIZED_MEMORY."
            )
        shutil.copy2(organized_source, out_dir / "organized_memory.json")
        with open(out_dir / "organized_memory.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        raise AssertionError(f"Not an organized mode: {mode}")
    return {
        "organized_memory": "organized_memory.json",
        "generated_by": payload.get("generated_by", ""),
        "event_count": payload.get("event_count", 0),
        "trip_count": payload.get("trip_count", 0),
    }


def build_memory_variant(
    *,
    mode: str,
    image_source: Path,
    video_source: Path,
    emails_source: Path,
    out_dir: Path,
    raw_image_dir: Path | None = None,
    raw_video_dir: Path | None = None,
    organized_source: Path | None = None,
) -> dict[str, Any]:
    mode = normalize_memory_mode(mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_records = load_json_list(image_source)
    video_records = load_json_list(video_source)

    media_manifest: dict[str, Any] = {}
    if mode in ("sgm", "org_heuristic", "org_static", "org_dynamic", "org_hybrid", "org_remmi"):
        # Organized modes keep the full SGM per-item files (recall questions
        # must still answer with exact item ids); org_heuristic/org_static/
        # org_hybrid add a compact event index on top, org_dynamic changes only
        # the prompt (org_hybrid = index + dynamic organize-then-answer prompt),
        # org_remmi ships the REMMI search tool + compact corpus projection.
        shutil.copy2(image_source, out_dir / "image_metadata.json")
        shutil.copy2(video_source, out_dir / "video_metadata.json")
        if mode in ("org_heuristic", "org_static", "org_hybrid"):
            media_manifest["organized"] = build_organized_index(
                mode=mode,
                image_source=image_source,
                video_source=video_source,
                emails_source=emails_source,
                out_dir=out_dir,
                organized_source=organized_source,
            )
        if mode == "org_remmi":
            import sys

            repo_root = Path(__file__).resolve().parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from remmi.organize.search_pack import build_search_corpus

            corpus = build_search_corpus(image_source, video_source, emails_source)
            dump_json(out_dir / "search_corpus.json", corpus)
            shutil.copy2(repo_root / "agent_systems" / "tools" / "search.py", out_dir / "search.py")
            media_manifest["search_tool"] = {
                "search_tool": "search.py",
                "search_corpus": "search_corpus.json",
                "corpus_items": len(corpus),
            }
    elif mode == "raw":
        dump_json(out_dir / "image_metadata.json", raw_entries(image_records, "image_path", "raw_images"))
        dump_json(out_dir / "video_metadata.json", raw_entries(video_records, "video_path", "raw_videos"))
        if raw_image_dir is None or raw_video_dir is None:
            raise ValueError("raw mode requires raw_image_dir and raw_video_dir")
        media_manifest["raw_images"] = hardlink_media_files(
            image_records,
            "image_path",
            raw_image_dir,
            out_dir / "raw_images",
        )
        media_manifest["raw_videos"] = hardlink_media_files(
            video_records,
            "video_path",
            raw_video_dir,
            out_dir / "raw_videos",
        )
    elif mode == "descriptive":
        dump_json(out_dir / "image_metadata.json", descriptive_entries(image_records, "image_path"))
        dump_json(out_dir / "video_metadata.json", descriptive_entries(video_records, "video_path"))
    else:
        raise AssertionError(f"Unhandled memory mode: {mode}")

    shutil.copy2(emails_source, out_dir / "emails.json")

    manifest = {
        "memory_mode": mode,
        "image_metadata": "image_metadata.json",
        "video_metadata": "video_metadata.json",
        "emails": "emails.json",
        "image_count": len(image_records),
        "video_count": len(video_records),
        "emails_source": str(emails_source),
        "media": media_manifest,
    }
    dump_json(out_dir / "memory_variant.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an agent_systems memory variant")
    parser.add_argument("--mode", default="sgm", choices=VALID_MEMORY_MODES)
    parser.add_argument("--image-source", required=True, type=Path)
    parser.add_argument("--video-source", required=True, type=Path)
    parser.add_argument("--emails-source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--raw-image-dir", type=Path)
    parser.add_argument("--raw-video-dir", type=Path)
    parser.add_argument(
        "--organized-source",
        type=Path,
        default=os.environ.get("AGSYS_ORGANIZED_MEMORY") and Path(os.environ["AGSYS_ORGANIZED_MEMORY"]),
        help="Prebuilt organized_memory.json (required for org_static; from remmi.organize.agent_static)",
    )
    args = parser.parse_args()

    manifest = build_memory_variant(
        mode=args.mode,
        image_source=args.image_source,
        video_source=args.video_source,
        emails_source=args.emails_source,
        out_dir=args.out_dir,
        raw_image_dir=args.raw_image_dir,
        raw_video_dir=args.raw_video_dir,
        organized_source=args.organized_source,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
