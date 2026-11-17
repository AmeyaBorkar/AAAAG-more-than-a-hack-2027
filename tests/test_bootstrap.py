"""bootstrap.py sets the team hooks only in a git checkout that has them (#76: the release has none)."""

from __future__ import annotations

import importlib.util

from aaaag.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("bootstrap", REPO_ROOT / "scripts" / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_hooks_only_in_a_git_checkout_that_has_them(tmp_path):
    assert not bootstrap.wants_hooks(tmp_path)  # a release download: no .git, no .githooks
    (tmp_path / ".githooks").mkdir()
    assert not bootstrap.wants_hooks(tmp_path)  # hooks but not a checkout
    (tmp_path / ".git").write_text("gitdir: elsewhere\n")  # a worktree's .git is a file
    assert bootstrap.wants_hooks(tmp_path)


def test_a_release_clone_without_hooks_skips_them(tmp_path):
    (tmp_path / ".git").mkdir()  # a clone of the public release: git, but no .githooks/
    assert not bootstrap.wants_hooks(tmp_path)
