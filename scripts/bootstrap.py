"""One-command setup for a team member's machine. Safe to re-run.

    python scripts/bootstrap.py            # uses the Python you run it with (3.13 recommended)

1. creates .venv with that interpreter (if missing)
2. installs the exact pins from requirements-lock.txt, then this package in editable mode
3. points git at the shared hooks in .githooks/ (no attribution lines, no data commits), only in a
   git checkout that has them: the public release (#76) ships without the hooks, and a download of
   it is not a git checkout
4. runs `python -m aaaag doctor`

Standard library only, so it runs on a fresh machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*args: str | Path) -> None:
    print("+", " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def wants_hooks(root: Path) -> bool:
    """True in a git checkout that carries the team hooks (`.git` is a file in a worktree)."""
    return (root / ".githooks").is_dir() and (root / ".git").exists()


def main() -> int:
    if sys.version_info < (3, 12):  # noqa: UP036 -- this script must run on old Pythons to reject them
        print(f"Python {sys.version.split()[0]} is too old: use 3.13 (3.12 minimum).")
        return 1
    if sys.version_info[:2] != (3, 13):
        print(
            f"note: Python {sys.version.split()[0]}; submissions are built and checked with 3.13."
        )

    if not venv_python().exists():
        print(f"creating {VENV}")
        venv.EnvBuilder(with_pip=True).create(VENV)
    py = venv_python()
    run(py, "-m", "pip", "install", "--upgrade", "pip", "--quiet")
    run(py, "-m", "pip", "install", "-r", ROOT / "requirements-lock.txt", "--quiet")
    run(py, "-m", "pip", "install", "-e", ROOT, "--no-deps", "--quiet")

    if wants_hooks(ROOT):
        run("git", "config", "core.hooksPath", ".githooks")
        if os.name != "nt":
            for hook in (ROOT / ".githooks").iterdir():
                hook.chmod(0o755)
    else:
        print(
            "no .githooks/ or not a git checkout: the team hooks are skipped (fine for a release)"
        )

    print()
    subprocess.run([str(py), "-m", "aaaag", "doctor"], cwd=ROOT, check=False)
    activate = r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(f"\nDone. Activate the environment with:  {activate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
