# Contributing

Thank you for your interest. This repository is a **frozen snapshot** of team AAAAG's submission to
More than a Hack 2027, Phase 1. Its job is to rebuild the submitted `AAAAG.csv` byte for byte, so a
change that alters that file cannot be merged.

## Welcome
- **Questions and bug reports:** open an issue.
- **Reproduction reports**, whether your rebuild matched or not. Please include:
  - your OS and CPU;
  - your Python version;
  - the `sha256` line and the `reproduction check` line printed by `python -m aaaag submit`;
  - the output of `python -m aaaag doctor`.
- **Small fixes** to documentation, tests or tooling that leave the output unchanged.

## Please never post the match data
The organisers' videos, tracking, Smart Tagging and homography are licensed to participants for the
challenge only. Do not attach them, or any per-frame rows, exports or images derived from them, to
an issue or a pull request. Counts, rates and scores are fine.

## Pull requests
1. Install with `python scripts/bootstrap.py`. It uses the exact pins of `requirements-lock.txt`.
2. Keep the change small and say why it is needed.
3. Before you open the PR, run:

   ```bash
   ruff check . && ruff format --check .
   pytest -m "not data"
   ```

   `tests/test_champion_reproducible.py` must still pass: it pins the champion's output on a
   synthetic match.
4. Do not edit `vendor/mtah2027/`. It is the organisers' code, and its bytes are checked against
   their repository.

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems as
described in [SECURITY.md](SECURITY.md), not in a public issue.
