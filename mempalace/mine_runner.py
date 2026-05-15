#!/usr/bin/env python3
"""
mine_runner.py — Structured mine orchestration for MCP.

Wraps the lower-level miner primitives (scan_project, process_file, get_collection)
to return structured counts instead of printing. Supports:

  - Sync runs (mempalace_mine MCP tool, wait=True)
  - Async runs via subprocess (wait=False) with job-file status tracking
  - Cross-process flock-based locking on ~/.mempalace/mine.lock
  - Auto-discovery of mempalace.yaml targets under a scan root
  - Single-target runs via project_dir

Job state lives in ~/.mempalace/jobs/<job_id>.json. A heartbeat field is
refreshed each file; mempalace_mine_status flags status="stale" if the
heartbeat is older than 5 minutes on a "running" job (process likely died).

Subprocess entry point:
    python -m mempalace.mine_runner --job-id <uuid> [--project-dir P | --scan-root R]
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import io
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import MempalaceConfig

logger = logging.getLogger("mempalace_mine_runner")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMPALACE_DIR = Path(os.path.expanduser("~/.mempalace"))
LOCK_PATH = MEMPALACE_DIR / "mine.lock"
JOBS_DIR = MEMPALACE_DIR / "jobs"

# How old a heartbeat can be before mempalace_mine_status calls a job stale.
HEARTBEAT_STALE_SECONDS = 300

# Default scan root for auto-discovery. The legacy daily re-mine documented
# ~/Claude/Asiakkaat, but the real layout has yaml files scattered under
# ~/Claude (e.g. ~/Claude/tuontirengas/TOOLS/mempalace.yaml). Walking the
# whole ~/Claude tree is fine — we hard-stop at project boundaries and skip
# heavy directories.
DEFAULT_SCAN_ROOT = "~/Claude"

DISCOVERY_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".mempalace",
    ".claude",
    ".vscode",
    ".idea",
    "Library",
    "Pictures",
    # ~/Claude/git holds ephemeral worktrees created by Agent isolation —
    # they are short-lived copies of the canonical sibling repos and would
    # cause duplicate mining if walked.
    "git",
}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LockHeldError(RuntimeError):
    def __init__(self, active_job_id: Optional[str]):
        super().__init__(f"mine.lock held by job {active_job_id}")
        self.active_job_id = active_job_id


# ---------------------------------------------------------------------------
# Job file helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat()


def _jobs_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _write_job(job_id: str, **fields) -> None:
    """Merge fields into the job JSON file, atomic via os.replace."""
    path = _job_path(job_id)
    data: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data["job_id"] = job_id
    data.update(fields)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _read_job(job_id: str) -> Optional[dict]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _find_active_job() -> Optional[str]:
    """Scan jobs dir for the freshest job marked 'running'."""
    if not JOBS_DIR.exists():
        return None
    best_id = None
    best_started = ""
    for f in JOBS_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") == "running":
            started = data.get("started_at", "")
            if started > best_started:
                best_started = started
                best_id = data.get("job_id") or f.stem
    return best_id


# ---------------------------------------------------------------------------
# Lock helpers (flock-based, cross-process safe)
# ---------------------------------------------------------------------------


@contextmanager
def mine_lock():
    """
    Exclusive non-blocking flock on ~/.mempalace/mine.lock.
    Raises LockHeldError(active_job_id) if held by another process.
    Lock is released on context exit OR process death (OS-managed).
    """
    MEMPALACE_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as e:
            if e.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                os.close(fd)
                raise
            os.close(fd)
            raise LockHeldError(_find_active_job())
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_targets(scan_root: str = DEFAULT_SCAN_ROOT) -> list[Path]:
    """
    Walk scan_root and return every directory that contains mempalace.yaml
    (or legacy mempal.yaml). Stops descending once a target is found:
    a project is a single mining unit.
    """
    root = Path(scan_root).expanduser().resolve()
    if not root.exists():
        return []

    targets: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune heavy / irrelevant dirs in-place
        dirnames[:] = [d for d in dirnames if d not in DISCOVERY_SKIP_DIRS and not d.startswith(".")]
        if "mempalace.yaml" in filenames or "mempal.yaml" in filenames:
            targets.append(Path(dirpath))
            dirnames[:] = []  # project boundary; do not recurse further
    return sorted(targets)


# ---------------------------------------------------------------------------
# Structured mine — runs the real miner with stdout suppressed and returns counts
# ---------------------------------------------------------------------------


def _mine_one_target(target: Path, palace_path: str, heartbeat_cb=None) -> dict:
    """
    Mine a single target directory. Returns counts:
      {
        "path": str(target),
        "wing": str,
        "added": int,           # drawers newly filed
        "files_processed": int, # files that contributed >=1 drawer
        "unchanged": int,       # files skipped (already filed / too small / unreadable)
        "files_total": int,
      }
    Raises on hard failure (missing yaml, etc.). Caller decides whether to
    accumulate as error or abort.
    """
    # Late imports keep module load fast for status-only callers
    import yaml
    from .miner import (
        SKIP_DIRS,
        READABLE_EXTENSIONS,
        get_collection,
        process_file,
    )

    yaml_path = target / "mempalace.yaml"
    if not yaml_path.exists():
        yaml_path = target / "mempal.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No mempalace.yaml in {target}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    wing = cfg.get("wing") or target.name
    rooms = cfg.get("rooms") or [{"name": "general", "description": "All project files"}]

    # Collect files (replicates miner.scan_project but inline so we can heartbeat)
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in ("mempalace.yaml", "mempalace.yml", "mempal.yaml", "mempal.yml",
                      ".gitignore", "package-lock.json"):
                continue
            p = Path(dirpath) / fn
            if p.suffix.lower() in READABLE_EXTENSIONS:
                files.append(p)

    collection = get_collection(palace_path)

    added = 0
    files_processed = 0
    unchanged = 0
    for i, filepath in enumerate(files, 1):
        try:
            drawers = process_file(
                filepath=filepath,
                project_path=target,
                collection=collection,
                wing=wing,
                rooms=rooms,
                agent="mempalace_mcp_mine",
                dry_run=False,
            )
        except Exception as e:
            logger.warning(f"process_file failed for {filepath}: {e}")
            unchanged += 1
            continue

        if drawers > 0:
            added += drawers
            files_processed += 1
        else:
            unchanged += 1

        if heartbeat_cb and (i % 25 == 0 or i == len(files)):
            heartbeat_cb()

    return {
        "path": str(target),
        "wing": wing,
        "added": added,
        "files_processed": files_processed,
        "unchanged": unchanged,
        "files_total": len(files),
    }


def run_mine(
    project_dir: Optional[str] = None,
    scan_root: Optional[str] = None,
    palace_path: Optional[str] = None,
    heartbeat_cb=None,
) -> dict:
    """
    Run a mine across one project_dir or all auto-discovered targets under
    scan_root. Returns the structured result dict.

    Note: callers should hold `mine_lock()` for the duration of this call
    if they need mutual exclusion. This function does not acquire the lock
    itself, so it can be called both from sync and async (subprocess) paths.
    """
    started = time.monotonic()
    started_at = _now_iso()

    config = MempalaceConfig()
    if palace_path is None:
        palace_path = config.palace_path

    if project_dir:
        targets = [Path(project_dir).expanduser().resolve()]
    else:
        targets = discover_targets(scan_root or DEFAULT_SCAN_ROOT)

    target_results: list[dict] = []
    errors: list[dict] = []
    wings_agg: dict[str, dict[str, int]] = defaultdict(lambda: {"added": 0, "unchanged": 0, "files_processed": 0})

    for t in targets:
        # Suppress miner.process_file's own logging side effects but keep ours.
        try:
            buf = io.StringIO()
            with _redirect_stdout(buf):
                tr = _mine_one_target(t, palace_path=palace_path, heartbeat_cb=heartbeat_cb)
            target_results.append(tr)
            w = wings_agg[tr["wing"]]
            w["added"] += tr["added"]
            w["unchanged"] += tr["unchanged"]
            w["files_processed"] += tr["files_processed"]
        except Exception as e:
            errors.append({
                "path": str(t),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })

    total_added = sum(tr["added"] for tr in target_results)
    total_unchanged = sum(tr["unchanged"] for tr in target_results)

    return {
        "added": total_added,
        "updated": 0,      # miner does not currently re-mine modified files
        "deleted": 0,      # miner does not currently prune orphaned drawers
        "unchanged": total_unchanged,
        "targets": target_results,
        "wings": dict(wings_agg),
        "duration_seconds": round(time.monotonic() - started, 2),
        "started_at": started_at,
        "finished_at": _now_iso(),
        "errors": errors,
        "palace_path": palace_path,
    }


@contextmanager
def _redirect_stdout(buf):
    """Temporarily redirect sys.stdout to buf — miner prints a progress banner."""
    old = sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# High-level callable used by MCP server (sync) and __main__ (async)
# ---------------------------------------------------------------------------


def run_with_lock(
    job_id: str,
    mode: str,
    project_dir: Optional[str] = None,
    scan_root: Optional[str] = None,
) -> dict:
    """
    Acquire mine.lock and run a mine end-to-end, persisting state to the job
    file. Returns the same result dict that run_mine() returns, plus the
    job_id and final status. If the lock is held, returns
    {"status": "already_running", "job_id": <active>}.

    Used by both sync MCP calls and the subprocess CLI.
    """
    try:
        with mine_lock():
            _write_job(
                job_id,
                status="running",
                mode=mode,
                started_at=_now_iso(),
                heartbeat=_now_iso(),
                project_dir=project_dir,
                scan_root=scan_root,
            )

            def _heartbeat():
                _write_job(job_id, heartbeat=_now_iso())

            try:
                result = run_mine(
                    project_dir=project_dir,
                    scan_root=scan_root,
                    heartbeat_cb=_heartbeat,
                )
                _write_job(
                    job_id,
                    status="done",
                    finished_at=_now_iso(),
                    heartbeat=_now_iso(),
                    result=result,
                )
                return {"status": "done", "job_id": job_id, **result}
            except Exception as e:
                err = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                _write_job(
                    job_id,
                    status="failed",
                    finished_at=_now_iso(),
                    heartbeat=_now_iso(),
                    **err,
                )
                raise
    except LockHeldError as e:
        return {"status": "already_running", "job_id": e.active_job_id}


def spawn_background(
    mode: str,
    project_dir: Optional[str] = None,
    scan_root: Optional[str] = None,
) -> dict:
    """
    Spawn a detached subprocess that runs the mine. Returns immediately with
    the new job_id. The subprocess is responsible for acquiring the lock,
    writing job state, and exiting cleanly.
    """
    # Best-effort early reject if lock is currently held
    try:
        with mine_lock():
            pass
    except LockHeldError as e:
        return {"status": "already_running", "job_id": e.active_job_id}

    job_id = str(uuid.uuid4())
    _write_job(
        job_id,
        status="queued",
        mode=mode,
        started_at=_now_iso(),
        heartbeat=_now_iso(),
        project_dir=project_dir,
        scan_root=scan_root,
    )

    log_path = _jobs_dir() / f"{job_id}.log"
    cmd = [sys.executable, "-m", "mempalace.mine_runner", "--job-id", job_id, "--mode", mode]
    if project_dir:
        cmd += ["--project-dir", project_dir]
    if scan_root:
        cmd += ["--scan-root", scan_root]

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return {
        "status": "spawned",
        "job_id": job_id,
        "pid": proc.pid,
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# Status reader (used by the mempalace_mine_status MCP tool)
# ---------------------------------------------------------------------------


def read_status(job_id: str) -> dict:
    data = _read_job(job_id)
    if data is None:
        return {"error": f"Job not found: {job_id}", "job_id": job_id}

    status = data.get("status")
    if status == "running":
        last_hb = data.get("heartbeat") or data.get("started_at")
        if last_hb:
            try:
                last_dt = datetime.fromisoformat(last_hb)
                age = (datetime.now() - last_dt).total_seconds()
                if age > HEARTBEAT_STALE_SECONDS:
                    data = dict(data)
                    data["status"] = "stale"
                    data["heartbeat_age_seconds"] = round(age, 1)
                    data["stale_reason"] = (
                        f"No heartbeat for {round(age)}s (>{HEARTBEAT_STALE_SECONDS}s) — "
                        "job process likely died (sleep, crash, kill)."
                    )
            except ValueError:
                pass
    return data


# ---------------------------------------------------------------------------
# CLI for subprocess execution
# ---------------------------------------------------------------------------


def _cli_main():
    parser = argparse.ArgumentParser(description="Background mine runner (internal)")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--scan-root", default=None)
    args = parser.parse_args()

    try:
        run_with_lock(
            job_id=args.job_id,
            mode=args.mode,
            project_dir=args.project_dir,
            scan_root=args.scan_root,
        )
    except Exception:
        # run_with_lock already persisted the failed status; surface trace to log
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
