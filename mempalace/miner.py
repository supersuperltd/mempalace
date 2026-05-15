#!/usr/bin/env python3
"""
miner.py — Files everything into the palace.

Reads mempalace.yaml from the project directory to know the wing + rooms.
Routes each file to the right room based on content.
Stores verbatim chunks as drawers. No summaries. Ever.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import chromadb

READABLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".sh",
    ".csv",
    ".sql",
    ".toml",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
}

CHUNK_SIZE = 800  # chars per drawer
CHUNK_OVERLAP = 100  # overlap between chunks
MIN_CHUNK_SIZE = 50  # skip tiny chunks


# =============================================================================
# CONFIG
# =============================================================================


def load_config(project_dir: str) -> dict:
    """Load mempalace.yaml from project directory (falls back to mempal.yaml)."""
    import yaml

    config_path = Path(project_dir).expanduser().resolve() / "mempalace.yaml"
    if not config_path.exists():
        # Fallback to legacy name
        legacy_path = Path(project_dir).expanduser().resolve() / "mempal.yaml"
        if legacy_path.exists():
            config_path = legacy_path
        else:
            print(f"ERROR: No mempalace.yaml found in {project_dir}")
            print(f"Run: mempalace init {project_dir}")
            sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# FILE ROUTING — which room does this file belong to?
# =============================================================================


def detect_room(filepath: Path, content: str, rooms: list, project_path: Path) -> str:
    """
    Route a file to the right room.
    Priority:
    1. Folder path matches a room name
    2. Filename matches a room name or keyword
    3. Content keyword scoring
    4. Fallback: "general"
    """
    relative = str(filepath.relative_to(project_path)).lower()
    filename = filepath.stem.lower()
    content_lower = content[:2000].lower()

    # Priority 1: folder path contains room name
    path_parts = relative.replace("\\", "/").split("/")
    for part in path_parts[:-1]:  # skip filename itself
        for room in rooms:
            if room["name"].lower() in part or part in room["name"].lower():
                return room["name"]

    # Priority 2: filename matches room name
    for room in rooms:
        if room["name"].lower() in filename or filename in room["name"].lower():
            return room["name"]

    # Priority 3: keyword scoring from room keywords + name
    scores = defaultdict(int)
    for room in rooms:
        keywords = room.get("keywords", []) + [room["name"]]
        for kw in keywords:
            count = content_lower.count(kw.lower())
            scores[room["name"]] += count

    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

    return "general"


# =============================================================================
# CHUNKING
# =============================================================================


def chunk_text(content: str, source_file: str) -> list:
    """
    Split content into drawer-sized chunks.
    Tries to split on paragraph/line boundaries.
    Returns list of {"content": str, "chunk_index": int}
    """
    # Clean up
    content = content.strip()
    if not content:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))

        # Try to break at paragraph boundary
        if end < len(content):
            newline_pos = content.rfind("\n\n", start, end)
            if newline_pos > start + CHUNK_SIZE // 2:
                end = newline_pos
            else:
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start + CHUNK_SIZE // 2:
                    end = newline_pos

        chunk = content[start:end].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(
                {
                    "content": chunk,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        start = end - CHUNK_OVERLAP if end < len(content) else end

    return chunks


# =============================================================================
# PALACE — ChromaDB operations
# =============================================================================


def get_collection(palace_path: str):
    os.makedirs(palace_path, exist_ok=True)
    client = chromadb.PersistentClient(path=palace_path)
    try:
        return client.get_collection("mempalace_drawers")
    except Exception:
        return client.create_collection("mempalace_drawers")


def file_already_mined(collection, source_file: str) -> bool:
    """Fast check: has this file been filed before?"""
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        return len(results.get("ids", [])) > 0
    except Exception:
        return False


def get_filed_mtime(collection, source_file: str) -> float | None:
    """
    Return the most recent mtime stored on drawers for this source_file,
    or None if no drawer carries an mtime field (legacy data, pre-mtime
    schema). Callers treat None as "unknown, assume unchanged" — they
    must NOT auto-re-mine legacy drawers, that would create duplicates.
    """
    try:
        r = collection.get(where={"source_file": source_file}, include=["metadatas"])
    except Exception:
        return None
    best = None
    for meta in r.get("metadatas", []) or []:
        mt = (meta or {}).get("mtime")
        if mt is None:
            continue
        try:
            mt = float(mt)
        except (TypeError, ValueError):
            continue
        if best is None or mt > best:
            best = mt
    return best


def delete_drawers_for_source(collection, source_file: str) -> int:
    """
    Remove every drawer with this source_file. Returns the number removed.
    Used by the updated-mode path (file changed, replace its drawers) and
    by the deleted-mode path (file removed from disk, prune its drawers).
    """
    try:
        r = collection.get(where={"source_file": source_file})
    except Exception:
        return 0
    ids = r.get("ids", []) or []
    if not ids:
        return 0
    try:
        collection.delete(ids=ids)
    except Exception:
        return 0
    return len(ids)


def wing_source_files(collection, wing: str) -> set[str]:
    """
    Return the set of unique source_file values stored under a wing.
    Backs the orphan-detection pass: anything in this set that no
    longer exists on disk is an orphan candidate.
    """
    try:
        r = collection.get(where={"wing": wing}, include=["metadatas"])
    except Exception:
        return set()
    out = set()
    for meta in r.get("metadatas", []) or []:
        src = (meta or {}).get("source_file", "")
        if isinstance(src, str) and src:
            out.add(src)
    return out


def add_drawer(
    collection, wing: str, room: str, content: str, source_file: str, chunk_index: int, agent: str,
    mtime: float | None = None,
):
    """
    Add one drawer to the palace.

    mtime: optional file-system mtime (epoch seconds) recorded with the
        drawer so future mines can detect updated files. Older drawers
        without this field are treated as 'unknown mtime' and skipped on
        re-mine (see get_filed_mtime / process_file).
    """
    drawer_id = f"drawer_{wing}_{room}_{hashlib.md5((source_file + str(chunk_index)).encode()).hexdigest()[:16]}"
    metadata = {
        "wing": wing,
        "room": room,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "added_by": agent,
        "filed_at": datetime.now().isoformat(),
    }
    if mtime is not None:
        metadata["mtime"] = float(mtime)
    try:
        collection.add(
            documents=[content],
            ids=[drawer_id],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            return False
        raise


# =============================================================================
# PROCESS ONE FILE
# =============================================================================


def process_file(
    filepath: Path,
    project_path: Path,
    collection,
    wing: str,
    rooms: list,
    agent: str,
    dry_run: bool,
) -> dict:
    """
    Read, chunk, route, and file one file. Returns a structured result:

        {
            "status":          "added" | "updated" | "unchanged" | "skipped",
            "drawers_added":   int,    # new drawers written
            "drawers_removed": int,    # drawers replaced (only on 'updated')
            "reason":          str,    # short tag explaining 'skipped' / 'unchanged'
        }

    Status semantics:
      added     — first time seen, fresh drawers written
      updated   — file already filed and file mtime is newer than stored
                  mtime; old drawers were deleted and replaced with fresh
                  chunks. drawers_added is the new count, drawers_removed
                  is what was purged.
      unchanged — file already filed; either mtime matches stored mtime,
                  or the drawer has no recorded mtime (legacy schema —
                  we conservatively skip rather than risk duplicates).
      skipped   — read error, file too small, or dry-run on a fresh file.
    """
    source_file = str(filepath)

    try:
        file_mtime = filepath.stat().st_mtime
    except OSError:
        return {"status": "skipped", "drawers_added": 0, "drawers_removed": 0,
                "reason": "stat_failed"}

    if not dry_run and file_already_mined(collection, source_file):
        stored = get_filed_mtime(collection, source_file)
        if stored is None:
            # Legacy drawer without mtime — treat as unchanged to avoid
            # duplicating content. Future re-mine after mtime fields
            # backfill will pick up real updates.
            return {"status": "unchanged", "drawers_added": 0, "drawers_removed": 0,
                    "reason": "legacy_no_mtime"}
        if file_mtime <= stored + 0.5:  # 0.5s slack for FS resolution
            return {"status": "unchanged", "drawers_added": 0, "drawers_removed": 0,
                    "reason": "mtime_match"}
        # File is newer → updated path: purge old drawers, re-mine fresh
        removed = delete_drawers_for_source(collection, source_file)
    else:
        removed = 0

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"status": "skipped", "drawers_added": 0, "drawers_removed": removed,
                "reason": "read_failed"}

    content = content.strip()
    if len(content) < MIN_CHUNK_SIZE:
        return {"status": "skipped", "drawers_added": 0, "drawers_removed": removed,
                "reason": "too_small"}

    room = detect_room(filepath, content, rooms, project_path)
    chunks = chunk_text(content, source_file)

    if dry_run:
        print(f"    [DRY RUN] {filepath.name} → room:{room} ({len(chunks)} drawers)")
        return {"status": "added", "drawers_added": len(chunks), "drawers_removed": 0,
                "reason": "dry_run"}

    drawers_added = 0
    for chunk in chunks:
        added = add_drawer(
            collection=collection,
            wing=wing,
            room=room,
            content=chunk["content"],
            source_file=source_file,
            chunk_index=chunk["chunk_index"],
            agent=agent,
            mtime=file_mtime,
        )
        if added:
            drawers_added += 1

    status = "updated" if removed > 0 else "added"
    return {"status": status, "drawers_added": drawers_added, "drawers_removed": removed,
            "reason": "mtime_newer" if status == "updated" else "fresh"}


# =============================================================================
# SCAN PROJECT
# =============================================================================


def scan_project(project_dir: str) -> list:
    """Return list of all readable file paths."""
    project_path = Path(project_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            filepath = Path(root) / filename
            if filepath.suffix.lower() in READABLE_EXTENSIONS:
                # Skip config files
                if filename in (
                    "mempalace.yaml",
                    "mempalace.yml",
                    "mempal.yaml",
                    "mempal.yml",
                    ".gitignore",
                    "package-lock.json",
                ):
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MAIN: MINE
# =============================================================================


def mine(
    project_dir: str,
    palace_path: str,
    wing_override: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
):
    """Mine a project directory into the palace."""

    project_path = Path(project_dir).expanduser().resolve()
    config = load_config(project_dir)

    wing = wing_override or config["wing"]
    rooms = config.get("rooms", [{"name": "general", "description": "All project files"}])

    files = scan_project(project_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Rooms:   {', '.join(r['name'] for r in rooms)}")
    print(f"  Files:   {len(files)}")
    print(f"  Palace:  {palace_path}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'─' * 55}\n")

    if not dry_run:
        collection = get_collection(palace_path)
    else:
        collection = None

    total_drawers = 0
    files_skipped = 0
    room_counts = defaultdict(int)

    for i, filepath in enumerate(files, 1):
        result = process_file(
            filepath=filepath,
            project_path=project_path,
            collection=collection,
            wing=wing,
            rooms=rooms,
            agent=agent,
            dry_run=dry_run,
        )
        drawers = result["drawers_added"]
        if drawers == 0 and not dry_run:
            files_skipped += 1
        else:
            total_drawers += drawers
            room = detect_room(filepath, "", rooms, project_path)
            room_counts[room] += 1
            if not dry_run:
                tag = result["status"].upper()
                print(f"  ✓ [{i:4}/{len(files)}] [{tag:9}] {filepath.name[:40]:40} +{drawers}")

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Drawers filed: {total_drawers}")
    print("\n  By room:")
    for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {room:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


# =============================================================================
# STATUS
# =============================================================================


def status(palace_path: str):
    """Show what's been filed in the palace."""
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        return

    # Count by wing and room
    r = col.get(limit=10000, include=["metadatas"])
    metas = r["metadatas"]

    wing_rooms = defaultdict(lambda: defaultdict(int))
    for m in metas:
        wing_rooms[m.get("wing", "?")][m.get("room", "?")] += 1

    print(f"\n{'=' * 55}")
    print(f"  MemPalace Status — {len(metas)} drawers")
    print(f"{'=' * 55}\n")
    for wing, rooms in sorted(wing_rooms.items()):
        print(f"  WING: {wing}")
        for room, count in sorted(rooms.items(), key=lambda x: x[1], reverse=True):
            print(f"    ROOM: {room:20} {count:5} drawers")
        print()
    print(f"{'=' * 55}\n")
