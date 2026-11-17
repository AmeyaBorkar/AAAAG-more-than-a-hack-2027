"""Repository policy checks, shared by the git hooks (.githooks/) and CI (.github/workflows/ci.yml).

    python scripts/policy.py commit-msg FILE    # commit-msg hook: subject format, no attribution
    python scripts/policy.py staged             # pre-commit hook: not on main, no data, no secrets
    python scripts/policy.py pre-push           # pre-push hook (refs on stdin): never push to main
    python scripts/policy.py tree               # CI: no data or secrets in the tracked tree
    python scripts/policy.py commits A..B       # CI: every commit message in a range
    python scripts/policy.py title "TEXT"       # CI: the PR title (it becomes history)
    python scripts/policy.py text FILE          # CI: no attribution in a PR description

The rules (AGENTS.md section 2), all non-negotiable:
1. No co-author, "generated with", or other tool/AI attribution in commits, PR titles/bodies or
   comments. Commits are authored by the team member alone.
2. Commit subjects are Conventional Commits: `type(scope): summary`, at most 100 characters.
3. Nothing that is, or could reconstruct, the gated match data is ever committed -- the repository
   goes public at submission time and git history is forever. Same for secrets and agent chat
   transcripts (*.jsonl).
4. Nobody commits to or pushes main; everything lands through a PR. The coordinator may set
   AAAAG_ALLOW_MAIN=1 for repository administration only.

Standard library only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ATTRIBUTION = [
    re.compile(r"^\s*co-authored-by\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"generated\s+(with|by|using)\s+\[?\s*(claude|cursor|copilot|chatgpt|codex|gemini|devin|"
        r"windsurf|aider|openai|anthropic|an?\s+ai)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(signed-off-by|assisted-by|ai-assisted-by|co-developed-by|made-with|written-by)\s*:"
        r".*(claude|anthropic|openai|cursor|copilot|gemini|codex|chatgpt|gpt-|noreply@)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"\U0001f916\s*generated", re.IGNORECASE),
    re.compile(r"claude\.ai/code/session", re.IGNORECASE),
]

TYPES = "feat|fix|perf|refactor|test|docs|analysis|exp|chore|ci|build|style|revert"
SUBJECT = re.compile(rf"^({TYPES})(\([a-z0-9._/-]+\))?!?: \S")
SUBJECT_EXEMPT = re.compile(r'^(Revert "|fixup! |squash! |amend! )')
MAX_SUBJECT = 100

VENDOR = "vendor/mtah2027/"
ALLOWED = {"data/README.md"}
FORBIDDEN_DIRS = ("data/", "cache/", "outputs/", "external_data/")
FORBIDDEN_SUFFIXES = (
    ".mp4", ".avi", ".mov", ".mkv", ".npz", ".npy", ".parquet", ".feather",
    ".pkl", ".pickle", ".h5", ".hdf5", ".joblib", ".ipynb", ".jsonl",
)  # fmt: skip
FORBIDDEN_NAMES = [
    re.compile(r"_atd\.(txt|xml)$", re.IGNORECASE),
    re.compile(r"_homography\.csv$", re.IGNORECASE),
    re.compile(r"_smart_tagging\.xml$", re.IGNORECASE),
    re.compile(r"_possession_gt(_meta|_reference)?\.(csv|json)$", re.IGNORECASE),
    re.compile(r"(^|/)AAAAG\.csv$", re.IGNORECASE),
]
MAX_BYTES = 5_000_000
MAX_CSV_LINES = 5_000  # a per-frame dump of the match has ~199k lines
SECRET_SCAN_MAX_BYTES = 2_000_000
SECRETS = [
    re.compile(rb"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(rb"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{50,}"),
    re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
]

FIX_ATTRIBUTION = (
    "Remove the attribution line(s) and commit again (git commit --amend for the last commit). "
    "Team rule: commits carry no co-author or tool/AI attribution (AGENTS.md section 2)."
)
FIX_SUBJECT = (
    "Use a Conventional Commits subject: type(scope): summary -- e.g. "
    f"'feat(decoding): HSMM with duration priors (#14)'. Types: {TYPES.replace('|', ', ')}."
)
FIX_MAIN = (
    "main only changes through pull requests (AGENTS.md section 3). Create a branch: "
    "git switch -c <member>/<issue>-<topic>"
)


def attribution_problems(text: str) -> list[str]:
    hits = []
    for pattern in ATTRIBUTION:
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            hits.append(text[line_start : line_end if line_end != -1 else len(text)].strip())
    return sorted(set(hits))


def subject_problems(message: str) -> list[str]:
    subject = next(
        (line for line in message.splitlines() if line.strip() and not line.startswith("#")), ""
    )
    if SUBJECT_EXEMPT.match(subject):
        return []
    problems = []
    if not SUBJECT.match(subject):
        problems.append(f"subject is not 'type(scope): summary': {subject[:80]!r}")
    if len(subject) > MAX_SUBJECT:
        problems.append(f"subject is {len(subject)} characters (max {MAX_SUBJECT})")
    return problems


def path_problems(path: str, size: int | None = None, csv_lines: int | None = None) -> list[str]:
    path = path.replace("\\", "/")
    if path.startswith(VENDOR):
        return []
    problems = []
    if path not in ALLOWED and path.startswith(FORBIDDEN_DIRS):
        problems.append("lives in a data/cache/output directory")
    if path.lower().endswith(FORBIDDEN_SUFFIXES):
        problems.append("binary array, video, notebook or transcript file type")
    if any(pattern.search(path) for pattern in FORBIDDEN_NAMES):
        problems.append("named like a match-data asset or a submission")
    if size is not None and size > MAX_BYTES:
        problems.append(f"{size / 1e6:.1f} MB > {MAX_BYTES / 1e6:.0f} MB")
    if csv_lines is not None and csv_lines > MAX_CSV_LINES:
        problems.append(f"CSV with {csv_lines} lines looks like a per-frame dump")
    return problems


def secret_problems(data: bytes) -> list[str]:
    return [pattern.pattern.decode()[:40] for pattern in SECRETS if pattern.search(data)]


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _blob(spec: str) -> bytes:
    return subprocess.run(["git", "cat-file", "-p", spec], capture_output=True, check=False).stdout


def check_files(paths: list[str], spec_prefix: str) -> int:
    failures = 0
    for path in paths:
        spec = f"{spec_prefix}{path}"
        try:
            size = int(_git("cat-file", "-s", spec).strip())
        except RuntimeError:
            continue  # deleted or unreadable: nothing is being added
        data = (
            _blob(spec) if size <= SECRET_SCAN_MAX_BYTES or path.lower().endswith(".csv") else b""
        )
        lines = data.count(b"\n") if path.lower().endswith(".csv") else None
        problems = path_problems(path, size, lines)
        if not path.startswith(VENDOR) and size <= SECRET_SCAN_MAX_BYTES:
            problems += [f"looks like a secret ({hit})" for hit in secret_problems(data)]
        if problems:
            failures += 1
            print(f"POLICY  {path}: {'; '.join(problems)}")
    if failures:
        print(
            "\nThese files must not be committed: the repository goes public and git history is "
            "forever (docs/COMPLIANCE.md). Unstage them (git restore --staged <file>) and make "
            "sure .gitignore covers them."
        )
    return 1 if failures else 0


def check_message(message: str, what: str, *, subject: bool = True) -> int:
    problems = [f"attribution line: {hit}" for hit in attribution_problems(message)]
    if subject:
        problems += subject_problems(message)
    for problem in problems:
        print(f"POLICY  {what}: {problem}")
    if any(p.startswith("attribution") for p in problems):
        print("\n" + FIX_ATTRIBUTION)
    if any(p.startswith("subject") for p in problems):
        print("\n" + FIX_SUBJECT)
    return 1 if problems else 0


def _main_allowed() -> bool:
    return os.environ.get("AAAAG_ALLOW_MAIN") == "1"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "commit-msg":
        message = Path(rest[0]).read_text(encoding="utf-8", errors="replace")
        return check_message(message, "commit message")
    if command == "text":
        text = Path(rest[0]).read_text(encoding="utf-8", errors="replace")
        return check_message(text, "PR description", subject=False)
    if command == "title":
        return check_message(" ".join(rest), "PR title")
    if command == "staged":
        branch = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if branch == "main" and not _main_allowed():
            print(f"POLICY  committing on main is not allowed. {FIX_MAIN}")
            return 1
        names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
        return check_files([n for n in names if n], ":")
    if command == "pre-push":
        for line in sys.stdin.read().splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2] == "refs/heads/main" and not _main_allowed():
                print(f"POLICY  pushing to main is not allowed. {FIX_MAIN}")
                return 1
        return 0
    if command == "tree":
        names = _git("ls-files").splitlines()
        return check_files([n for n in names if n], "HEAD:")
    if command == "commits":
        revision = rest[0]
        base = revision.split("..")[0] if ".." in revision else ""
        if not base or set(base) == {"0"}:
            shas = [revision.split("..")[-1]]
        else:
            shas = _git("rev-list", revision).split()
        failures = sum(
            check_message(_git("log", "-1", "--format=%B", sha), f"commit {sha[:10]}")
            for sha in shas
        )
        return 1 if failures else 0
    print(f"unknown command {command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
