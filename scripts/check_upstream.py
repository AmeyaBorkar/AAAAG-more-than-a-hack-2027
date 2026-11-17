"""Watch the organisers' repository for changes to the rules, data docs or scorer.

    python scripts/check_upstream.py             # compare upstream HEAD with the vendored commit
    python scripts/check_upstream.py --verify    # check vendored bytes against UPSTREAM.json
    python scripts/check_upstream.py --sync      # re-vendor from upstream HEAD (opens a diff to review)

Exit codes: 0 up to date / intact, 3 upstream has new commits, 1 vendored files modified.
The organisers clarified the rules twice in the week before data release; a rule change we miss
can disqualify us, so CI runs this twice a day (.github/workflows/upstream-watch.yml) and opens an
issue when upstream moves. Standard library + git only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "mtah2027"
MANIFEST = VENDOR / "UPSTREAM.json"


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if args.verify:
        bad = [
            rel
            for rel, digest in manifest["files"].items()
            if not (VENDOR / rel).is_file()
            or hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() != digest
        ]
        for rel in bad:
            print(f"MODIFIED  vendor/mtah2027/{rel}")
        print(
            "vendored files intact"
            if not bad
            else f"{len(bad)} vendored file(s) differ from upstream"
        )
        return 1 if bad else 0

    head = git("ls-remote", manifest["upstream_repo"], "HEAD").split()[0]
    vendored = manifest["upstream_commit"]
    if head == vendored and not args.sync:
        print(f"up to date with upstream {head[:10]}")
        return 0
    if not args.sync:
        print(f"UPSTREAM CHANGED: vendored {vendored[:10]}, upstream HEAD {head[:10]}")
        print(f"compare: {manifest['upstream_repo']}/compare/{vendored}...{head}")
        return 3

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "upstream"
        git("clone", "--quiet", manifest["upstream_repo"], str(clone))
        changed = []
        for rel in manifest["files"]:
            try:
                blob = subprocess.run(
                    ["git", "cat-file", "-p", f"HEAD:{rel}"],
                    cwd=clone,
                    capture_output=True,
                    check=True,
                ).stdout
            except subprocess.CalledProcessError:
                print(f"REMOVED upstream: {rel} (kept locally; decide by hand)")
                continue
            digest = hashlib.sha256(blob).hexdigest()
            if digest != manifest["files"][rel]:
                changed.append(rel)
                (VENDOR / rel).write_bytes(blob)
                manifest["files"][rel] = digest
        tracked = set(git("ls-files", cwd=clone).split())
        new = sorted(p for p in tracked - set(manifest["files"]) if not p.endswith(".ipynb"))
        manifest["upstream_commit"] = git("rev-parse", "HEAD", cwd=clone).strip()
        manifest["upstream_commit_date"] = git("log", "-1", "--format=%cs", cwd=clone).strip()
    MANIFEST.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    print(f"synced to {manifest['upstream_commit'][:10]}; changed: {changed or 'none'}")
    if new:
        print("new upstream files not vendored (review, add to UPSTREAM.json if needed):", new)
    print(
        "Review `git diff vendor/`, then open a PR labelled upstream that summarises what changed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
