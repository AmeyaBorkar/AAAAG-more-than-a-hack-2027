"""Team roster, member identity, and `orient`: the session-start view of GitHub for agents.

`python -m aaaag orient` answers, in one screen, what an agent must know before choosing work:
where its branch stands against main, what is assigned to its human, who is waiting on its lane
(`needs:<lane>`), which PRs want its review, what merged recently, and the latest coordination
notes. GitHub is the team's shared memory; this is the read side of it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from .paths import HANDOVER_DIR, REPO_ROOT, STATUS_DIR, TEAM_DIR

ROSTER = TEAM_DIR / "roster.yaml"
# docs/handover/YYYY-MM-DD_HHMMZ_<member>_<topic>.md, as written by `python -m aaaag new handover`
HANDOVER_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}Z_[a-z0-9-]+_.+\.md$")


def load_roster(path: Path = ROSTER) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def current_member(explicit: str | None = None) -> str:
    """Roster key of whoever is running this: --member, $AAAAG_MEMBER, GitHub login, git email."""
    roster = load_roster()
    members = roster["members"]
    wanted = explicit or os.environ.get("AAAAG_MEMBER")
    if wanted:
        if wanted not in members:
            raise KeyError(f"unknown member {wanted!r}; roster has {sorted(members)}")
        return wanted
    login = (gh_text("api", "user", "--jq", ".login") or "").lower()
    email = _git("config", "user.email").lower()
    for key, member in members.items():
        if login and (member.get("github") or "").lower() == login:
            return key
        if email and email in [e.lower() for e in member.get("emails") or []]:
            return key
    return "unknown"


def gh_text(*args: str) -> str | None:
    if shutil.which("gh") is None:
        return None
    done = subprocess.run(["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def gh_json(*args: str) -> Any:
    text = gh_text(*args)
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def orient(member: str | None = None) -> int:
    roster = load_roster()
    key = current_member(member)
    me = roster["members"].get(key, {})
    lane = me.get("lane", "?")
    print(f"== orient: {key} (lane: {lane}) ==")

    _git("fetch", "--quiet", "origin")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    counts = _git("rev-list", "--left-right", "--count", "origin/main...HEAD").split()
    if len(counts) == 2:
        print(f"branch {branch}: {counts[0]} behind / {counts[1]} ahead of origin/main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        print("working tree has uncommitted changes")
    if _git("config", "core.hooksPath") != ".githooks":
        print("WARNING: git hooks are off -- run: git config core.hooksPath .githooks")

    _files_view(key)

    if shutil.which("gh") is None or gh_text("auth", "status") is None:
        print(
            "\n`gh` is missing or not authenticated -- run `gh auth login`; GitHub sections skipped."
        )
        return 1

    fields = "number,title,labels,assignees,updatedAt"
    _section(
        "Assigned to you",
        gh_json("issue", "list", "--assignee", "@me", "--state", "open", "--json", fields),
    )
    _section(
        f"Requests to your lane (needs:{lane})",
        gh_json("issue", "list", "--label", f"needs:{lane}", "--state", "open", "--json", fields),
    )
    unassigned = [
        issue
        for issue in gh_json(
            "issue",
            "list",
            "--label",
            f"lane:{lane}",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            fields,
        )
        or []
        if not issue["assignees"]
    ]
    _section(f"Unclaimed in lane:{lane} (pick the highest priority)", _by_priority(unassigned)[:8])
    _section(
        "PRs waiting for your review",
        gh_json(
            "pr",
            "list",
            "--search",
            "review-requested:@me",
            "--json",
            "number,title,author,updatedAt",
        ),
    )
    _section(
        "Your open PRs",
        gh_json("pr", "list", "--author", "@me", "--json", "number,title,isDraft,updatedAt"),
    )
    _section(
        "Recently merged",
        gh_json(
            "pr",
            "list",
            "--state",
            "merged",
            "--limit",
            "8",
            "--json",
            "number,title,author,mergedAt",
        ),
    )

    number = roster.get("coordination_issue")
    if number:
        thread = gh_json("issue", "view", str(number), "--json", "comments") or {}
        comments = thread.get("comments") or []
        print(f"\n-- Coordination #{number}: last {min(3, len(comments))} notes --")
        for comment in comments[-3:]:
            first_line = (comment.get("body") or "").strip().splitlines()[:1]
            print(
                f"  {comment.get('author', {}).get('login', '?')}: {first_line[0] if first_line else ''}"[
                    :160
                ]
            )

    champion = REPO_ROOT / "configs" / "submission.yaml"
    if champion.is_file():
        print(f"\nchampion: {yaml.safe_load(champion.read_text(encoding='utf-8')).get('pipeline')}")
    print(
        "\nNext: unblock others first (reviews, needs:), then your in-progress issue, then claim one."
    )
    return 0


def _files_view(key: str) -> None:
    """The repo side of the shared memory: your status file and the newest handovers (origin/main
    is what the team has agreed on; read files there, not only on your branch)."""
    status = STATUS_DIR / f"{key}.md"
    if status.is_file():
        age_h = (time.time() - status.stat().st_mtime) / 3600
        print(
            f"\nyour status file: {status.relative_to(REPO_ROOT).as_posix()} (edited {age_h:.0f} h ago)"
        )
    else:
        print(f"\nno status file yet: create docs/status/{key}.md from docs/status/_TEMPLATE.md")
    handovers = sorted(p for p in HANDOVER_DIR.glob("*.md") if HANDOVER_NAME.match(p.name))
    mine = [p for p in handovers if f"_{key}_" in p.name]
    others = [p for p in handovers if f"_{key}_" not in p.name]
    print("-- newest handovers from the team (read them before choosing work) --")
    for path in others[-3:]:
        print(f"  {path.relative_to(REPO_ROOT).as_posix()}")
    if not others:
        print("  (none)")
    if mine:
        print(f"-- your last handover: {mine[-1].relative_to(REPO_ROOT).as_posix()}")


def _section(title: str, items: list[dict] | None) -> None:
    print(f"\n-- {title} --")
    if not items:
        print("  (none)")
        return
    for item in items:
        labels = ",".join(label["name"] for label in item.get("labels", []))
        who = ",".join(a["login"] for a in item.get("assignees", [])) or (
            item.get("author") or {}
        ).get("login", "")
        draft = " [draft]" if item.get("isDraft") else ""
        print(f"  #{item['number']:<4} {item['title'][:70]}{draft}  {labels}  {who}".rstrip())


def _by_priority(issues: list[dict]) -> list[dict]:
    def rank(issue: dict) -> int:
        names = {label["name"] for label in issue.get("labels", [])}
        return next(
            (i for i, p in enumerate(("priority:P0", "priority:P1", "priority:P2")) if p in names),
            3,
        )

    return sorted(issues, key=lambda issue: (rank(issue), issue["number"]))


def _git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return done.stdout.strip() if done.returncode == 0 else ""
