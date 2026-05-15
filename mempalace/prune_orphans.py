#!/usr/bin/env python3
"""
prune_orphans.py — Delete drawers whose source files are gone from disk.

This is the surgical sibling of mempalace_mine(prune_deleted=True): it
ONLY removes orphan drawers, never mines anything new. Use when you
want to clean stale entries without re-walking the project tree (e.g.
post-split aftermath, files moved to a different repo).

The mine-driven prune only catches orphans inside a target project's
tree because that's where the mine walks. This tool can prune
project-wide (any wing) without requiring a mempalace.yaml at the
wing root.

Usage:
  python -m mempalace.prune_orphans --dry-run
  python -m mempalace.prune_orphans --apply
  python -m mempalace.prune_orphans --apply --wing tuontirengas
  python -m mempalace.prune_orphans --apply --no-backup

Always writes a JSON record of (drawer_id, source_file) tuples before
deleting, plus a full chroma-dir snapshot (unless --no-backup).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import chromadb

from .config import MempalaceConfig


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup_chroma_dir(palace_path: str) -> str:
    src = Path(palace_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Palace dir not found: {src}")
    dst = src.with_name(src.name + f"-prune-bak-{_ts()}")
    shutil.copytree(src, dst)
    return str(dst)


def _connect(palace_path: str | None = None):
    cfg = MempalaceConfig()
    palace_path = palace_path or cfg.palace_path
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection(cfg.collection_name)
    return col, palace_path


def _scan(col, wing: str | None) -> tuple[dict[str, list[str]], int]:
    """
    Return (grouped, total_drawers_in_wing) where grouped maps
    source_file -> list of drawer ids. Drawers with empty source_file
    are excluded — they're typically diary entries written via MCP.
    """
    where = {"wing": wing} if wing else None
    kwargs = {"include": ["metadatas"]}
    if where:
        kwargs["where"] = where
    r = col.get(**kwargs)
    grouped: dict[str, list[str]] = defaultdict(list)
    for id_, meta in zip(r.get("ids", []) or [], r.get("metadatas", []) or []):
        src = (meta or {}).get("source_file", "")
        if not isinstance(src, str) or not src:
            continue
        grouped[src].append(id_)
    return grouped, len(r.get("ids", []) or [])


def find_orphans(wing: str | None = None,
                 palace_path: str | None = None) -> dict:
    """
    Identify orphan drawers (source_file doesn't exist on disk).
    Returns:
      {
        "wing": str | "all",
        "total_drawers": int,
        "total_unique_files": int,
        "orphan_files": int,
        "orphan_drawers": int,
        "orphans": [{"source_file": str, "drawer_count": int}, ...],
        "sample": first 10,
      }
    """
    col, _ = _connect(palace_path)
    grouped, total_drawers = _scan(col, wing)

    orphans = []
    orphan_drawer_count = 0
    for src, ids in grouped.items():
        if not os.path.exists(src):
            orphans.append({"source_file": src, "drawer_count": len(ids)})
            orphan_drawer_count += len(ids)

    orphans.sort(key=lambda x: x["source_file"])
    return {
        "wing": wing or "all",
        "total_drawers": total_drawers,
        "total_unique_files": len(grouped),
        "orphan_files": len(orphans),
        "orphan_drawers": orphan_drawer_count,
        "sample": orphans[:10],
        # Full list omitted from dry-run output; --apply has it via the JSON backup.
    }


def apply(wing: str | None = None, palace_path: str | None = None,
          backup: bool = True, backup_json_path: str | None = None) -> dict:
    """
    Delete orphan drawers. Writes a JSON record of removed entries
    (always) plus a chroma-dir snapshot (default; --no-backup skips it).
    """
    col, palace_path = _connect(palace_path)
    grouped, total_drawers = _scan(col, wing)

    pending_ids: list[str] = []
    pending_records: list[dict] = []
    for src, ids in grouped.items():
        if not os.path.exists(src):
            pending_ids.extend(ids)
            pending_records.append({"source_file": src, "drawer_ids": list(ids)})

    if not pending_ids:
        return {"deleted_drawers": 0, "deleted_files": 0,
                "note": "No orphans found — nothing to do."}

    # JSON record of what we're about to delete (always, even with --no-backup)
    if backup_json_path is None:
        backup_json_path = str(Path(palace_path).parent / f"prune-orphans-backup-{_ts()}.json")
    with open(backup_json_path, "w") as f:
        json.dump({
            "wing": wing or "all",
            "palace_path": palace_path,
            "records": pending_records,
        }, f, indent=2)

    # Optional full chroma dir snapshot
    chroma_backup_path = None
    if backup:
        chroma_backup_path = _backup_chroma_dir(palace_path)

    # Delete in batches (chroma handles big batches but be conservative)
    BATCH = 500
    deleted = 0
    for start in range(0, len(pending_ids), BATCH):
        batch = pending_ids[start:start + BATCH]
        col.delete(ids=batch)
        deleted += len(batch)

    return {
        "wing": wing or "all",
        "deleted_drawers": deleted,
        "deleted_files": len(pending_records),
        "total_drawers_before": total_drawers,
        "total_drawers_after": total_drawers - deleted,
        "json_backup": backup_json_path,
        "chroma_dir_backup": chroma_backup_path,
    }


def _cli_main():
    p = argparse.ArgumentParser(
        description="Delete drawers whose source files are missing from disk"
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only, no writes (default)")
    p.add_argument("--apply", action="store_true", help="Delete orphan drawers")
    p.add_argument("--wing", default=None,
                   help="Limit to one wing (default: scan all wings)")
    p.add_argument("--palace", default=None,
                   help="Palace path (default: from config)")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip chroma-dir snapshot backup (JSON record still written). NOT recommended.")
    args = p.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    if not args.apply:
        result = find_orphans(wing=args.wing, palace_path=args.palace)
        print(json.dumps(result, indent=2, default=str))
        return

    result = apply(wing=args.wing, palace_path=args.palace, backup=not args.no_backup)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _cli_main()
