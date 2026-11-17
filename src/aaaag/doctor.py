"""`python -m aaaag doctor`: is this machine set up to reproduce the team's results?

Prints one OK / WARN / FAIL line per check, aggregates only (file sizes, never file contents).
Exit code 1 if anything FAILs. Paste the output into the onboarding issue when asked.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import shutil
import subprocess
import sys

from .paths import REPO_ROOT

REFERENCE_PYTHON = (3, 13)
REQUIRED = ("numpy", "pandas", "scipy", "yaml")
OPTIONAL = ("sklearn", "lightgbm", "kloppy", "matplotlib", "pytest", "ruff")


def run_doctor() -> int:
    results: list[tuple[str, str, str]] = []

    def report(status: str, name: str, detail: str = "") -> None:
        results.append((status, name, detail))

    version = sys.version_info[:2]
    if version < (3, 12):
        report("FAIL", "python", f"{sys.version.split()[0]} < 3.12")
    elif version != REFERENCE_PYTHON:
        report(
            "WARN", "python", f"{sys.version.split()[0]}; the reference for the submission is 3.13"
        )
    else:
        report("OK", "python", sys.version.split()[0])

    for name in REQUIRED:
        report(*_import(name, required=True))
    for name in OPTIONAL:
        report(*_import(name, required=False))

    drift = _lock_drift()
    report("WARN" if drift else "OK", "versions match requirements-lock.txt", "; ".join(drift[:6]))

    from .official import verify_vendor

    bad = verify_vendor()
    report("FAIL" if bad else "OK", "vendored official code is byte-exact", ", ".join(bad[:5]))

    hooks = _run("git", "config", "core.hooksPath")
    report(
        "OK" if hooks == ".githooks" else "WARN",
        "git hooks (core.hooksPath=.githooks)",
        hooks or "not set",
    )

    if shutil.which("gh") is None:
        report("WARN", "GitHub CLI", "gh not installed")
    else:
        authed = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, check=False
        )
        report("OK" if authed.returncode == 0 else "WARN", "GitHub CLI authenticated")

    from .sources.bar_ath import asset_paths

    for name, path in sorted(asset_paths().items()):
        if path.is_file():
            report("OK", f"data: {name}", f"{path.stat().st_size / 1e6:,.1f} MB")
        else:
            report("WARN", f"data: {name}", "missing (docs/ONBOARDING.md section 4)")

    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        print(f"{status:<4}  {name:<{width}}  {detail}".rstrip())
    failed = sum(status == "FAIL" for status, _, _ in results)
    print(f"\n{failed} FAIL, {sum(s == 'WARN' for s, _, _ in results)} WARN")
    return 1 if failed else 0


def _import(name: str, required: bool) -> tuple[str, str, str]:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return ("FAIL" if required else "WARN"), f"import {name}", "not installed"
    return "OK", f"import {name}", getattr(module, "__version__", "")


def _lock_drift() -> list[str]:
    drift = []
    lock = REPO_ROOT / "requirements-lock.txt"
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, pinned = line.split("==", 1)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            drift.append(f"{name} missing")
            continue
        if installed != pinned:
            drift.append(f"{name} {installed}!={pinned}")
    return drift


def _run(*args: str) -> str:
    done = subprocess.run(list(args), cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else ""
