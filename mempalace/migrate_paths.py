#!/usr/bin/env python3
"""
migrate_paths.py — Rewrite drawer source_file metadata after a project move.

The deployed palace at ~/.mempalace/palace was populated when projects
lived under ~/Claude/Asiakkaat/<Client>/<Client>/. After the workspace
split (22.04.2026 / Decision #22) the canonical layout became
~/Claude/<client>/ — direct siblings, lowercase. Existing drawers
still carry the stale source_file paths, which means:

  - mempalace_mine cannot recognize them as already-filed (because
    file_already_mined() compares source_file strings exactly), so
    re-mining the same files at the new path would create duplicate
    drawers.
  - Search results show paths that no longer exist on disk.

This tool rewrites source_file in-place via ChromaDB's update API.
Two modes:

  --dry-run   : count + sample matches, no writes (default)
  --apply     : rewrite + write a JSON backup + the chroma dir is
                also copied to a timestamped sibling before any
                update is issued.

Usage:
  python -m mempalace.migrate_paths --dry-run
  python -m mempalace.migrate_paths --apply
  python -m mempalace.migrate_paths --apply --old-prefix X --new-prefix Y

The default mapping handles the Asiakkaat → tuontirengas case. Pass
--old-prefix / --new-prefix for other client splits as the workspace
evolves.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import chromadb

from .config import MempalaceConfig

DEFAULT_OLD_PREFIX = "/Users/joni/Claude/Asiakkaat/Tuontirengas/Tuontirengas/"
DEFAULT_NEW_PREFIX = "/Users/joni/Claude/tuontirengas/"

BATCH_SIZE = 500


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup_chroma_dir(palace_path: str) -> str:
    """
    Copy the entire chroma directory to a sibling with -migrate-bak-TS suffix.
    Returns the backup path. Raises if source missing.
    """
    src = Path(palace_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Palace dir not found: {src}")
    dst = src.with_name(src.name + f"-migrate-bak-{_ts()}")
    shutil.copytree(src, dst)
    return str(dst)


def _scan_candidates(col, old_prefix: str) -> list[tuple[str, dict, str]]:
    """Return list of (id, existing_metadata, new_source_file)."""
    r = col.get(include=["metadatas"], limit=200000)
    out = []
    for id_, meta in zip(r["ids"], r["metadatas"]):
        src = (meta or {}).get("source_file", "")
        if isinstance(src, str) and src.startswith(old_prefix):
            new_src = (DEFAULT_NEW_PREFIX if old_prefix == DEFAULT_OLD_PREFIX
                       else _replacement_prefix(old_prefix)) + src[len(old_prefix):]
            # Override with actual --new-prefix when called via CLI
            out.append((id_, meta, new_src))
    return out


def _replacement_prefix(old: str) -> str:
    """Internal helper; CLI overrides this when --new-prefix is given."""
    return DEFAULT_NEW_PREFIX


def plan(old_prefix: str = DEFAULT_OLD_PREFIX, new_prefix: str = DEFAULT_NEW_PREFIX,
         palace_path: str | None = None) -> dict:
    """
    Compute the migration plan without writing anything.
    Returns a summary dict with counts, file existence at new path,
    and a small sample.
    """
    cfg = MempalaceConfig()
    palace_path = palace_path or cfg.palace_path
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection(cfg.collection_name)

    r = col.get(include=["metadatas"], limit=200000)
    candidates = []
    unique_files_old = set()
    unique_files_new = set()
    new_path_exists = 0
    new_path_missing = 0
    sample = []

    for id_, meta in zip(r["ids"], r["metadatas"]):
        src = (meta or {}).get("source_file", "")
        if not (isinstance(src, str) and src.startswith(old_prefix)):
            continue
        new_src = new_prefix + src[len(old_prefix):]
        candidates.append((id_, src, new_src))
        unique_files_old.add(src)
        unique_files_new.add(new_src)
        if len(sample) < 6:
            sample.append({"id": id_, "old": src, "new": new_src})

    for p in unique_files_new:
        if os.path.exists(p):
            new_path_exists += 1
        else:
            new_path_missing += 1

    return {
        "old_prefix": old_prefix,
        "new_prefix": new_prefix,
        "palace_path": palace_path,
        "total_drawers_in_palace": len(r["ids"]),
        "drawers_to_rewrite": len(candidates),
        "unique_source_files_old": len(unique_files_old),
        "unique_source_files_new": len(unique_files_new),
        "files_existing_at_new_path": new_path_exists,
        "files_missing_at_new_path": new_path_missing,
        "missing_note": (
            "Drawers whose new path does not exist remain in the palace as "
            "orphans (file probably moved to another repo, e.g. himmeli). "
            "Their source_file is rewritten anyway so future tooling can "
            "consistently distinguish 'orphan' from 'on-disk' via a single "
            "os.path.exists() check."
        ),
        "sample": sample,
    }


def apply(old_prefix: str = DEFAULT_OLD_PREFIX, new_prefix: str = DEFAULT_NEW_PREFIX,
          palace_path: str | None = None, backup: bool = True,
          backup_json_path: str | None = None) -> dict:
    """
    Apply the migration. Returns a summary dict including paths to the
    chroma-dir backup and the JSON backup of (id, old, new) tuples.
    """
    cfg = MempalaceConfig()
    palace_path = palace_path or cfg.palace_path
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection(cfg.collection_name)

    # 1) Read all matching drawers up-front (avoid mutating while iterating)
    r = col.get(include=["metadatas"], limit=200000)
    pending = []  # list of (id, full_new_metadata_dict, old_src, new_src)
    for id_, meta in zip(r["ids"], r["metadatas"]):
        src = (meta or {}).get("source_file", "")
        if not (isinstance(src, str) and src.startswith(old_prefix)):
            continue
        new_src = new_prefix + src[len(old_prefix):]
        new_meta = dict(meta)
        new_meta["source_file"] = new_src
        # Record the migration timestamp on each rewritten drawer so the
        # history is greppable later.
        new_meta["migrated_at"] = datetime.now().isoformat()
        new_meta["migrated_from_prefix"] = old_prefix
        pending.append((id_, new_meta, src, new_src))

    if not pending:
        return {"updated": 0, "note": "No drawers matched old_prefix — already migrated?"}

    # 2) JSON backup of (id, old, new) records — for rollback / audit
    if backup_json_path is None:
        backup_json_path = str(Path(palace_path).parent / f"migrate-paths-backup-{_ts()}.json")
    with open(backup_json_path, "w") as f:
        json.dump(
            [{"id": id_, "old_source_file": old_src, "new_source_file": new_src}
             for id_, _, old_src, new_src in pending],
            f,
            indent=2,
        )

    # 3) Full chroma-dir backup (snapshot)
    chroma_backup_path = None
    if backup:
        chroma_backup_path = _backup_chroma_dir(palace_path)

    # 4) Apply updates in batches
    n_total = len(pending)
    n_done = 0
    for start in range(0, n_total, BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        ids = [b[0] for b in batch]
        metas = [b[1] for b in batch]
        col.update(ids=ids, metadatas=metas)
        n_done += len(batch)

    return {
        "updated": n_done,
        "total_drawers_scanned": len(r["ids"]),
        "json_backup": backup_json_path,
        "chroma_dir_backup": chroma_backup_path,
        "old_prefix": old_prefix,
        "new_prefix": new_prefix,
    }


def _cli_main():
    p = argparse.ArgumentParser(description="Migrate source_file paths in MemPalace drawers")
    p.add_argument("--dry-run", action="store_true", help="Plan only, no writes (default)")
    p.add_argument("--apply", action="store_true", help="Apply the migration")
    p.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX,
                   help=f"Source path prefix to replace (default: {DEFAULT_OLD_PREFIX})")
    p.add_argument("--new-prefix", default=DEFAULT_NEW_PREFIX,
                   help=f"Replacement path prefix (default: {DEFAULT_NEW_PREFIX})")
    p.add_argument("--palace", default=None, help="Palace path (default: from config)")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip the chroma-dir snapshot backup (JSON record still written). NOT recommended.")
    args = p.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    if not args.apply:
        # default = dry run
        result = plan(old_prefix=args.old_prefix, new_prefix=args.new_prefix,
                      palace_path=args.palace)
        print(json.dumps(result, indent=2, default=str))
        return

    result = apply(old_prefix=args.old_prefix, new_prefix=args.new_prefix,
                   palace_path=args.palace, backup=not args.no_backup)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _cli_main()
