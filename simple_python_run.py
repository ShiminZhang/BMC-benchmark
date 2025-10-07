import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple


def trim_log_file(path: Path, head_lines: int, tail_lines: int) -> bool:
    """Trim the log file to keep only the first `head_lines` and last `tail_lines`.

    Returns True if the file was modified, False otherwise.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    if len(lines) <= head_lines + tail_lines:
        return False

    trimmed = lines[:head_lines] + lines[-tail_lines:]
    path.write_text("".join(trimmed), encoding="utf-8")
    return True


def process_logs(
    root_dir: Path,
    head_lines: int = 40,
    tail_lines: int = 50,
    max_workers: Optional[int] = None,
) -> None:
    """Traverse `root_dir` and trim every `.log` file found recursively in parallel."""

    log_files = list(root_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files found under {root_dir}")
        return

    workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

    def process_file(path: Path) -> Tuple[Path, bool, Optional[Exception]]:
        try:
            modified = trim_log_file(path, head_lines, tail_lines)
            return path, modified, None
        except OSError as error:
            return path, False, error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, modified, error in executor.map(process_file, log_files):
            if error:
                print(f"Failed to process {path}: {error}")
            elif modified:
                print(f"Trimmed {path}")


def main() -> None:
    logs_root = Path("results/solving_logs")
    if not logs_root.exists():
        print(f"Log directory not found: {logs_root}")
        return

    process_logs(logs_root)


if __name__ == "__main__":
    main()
