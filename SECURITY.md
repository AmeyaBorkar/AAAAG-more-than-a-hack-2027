# Security policy

## Supported versions

This repository is a frozen snapshot of team AAAAG's submission to More than a Hack 2027,
Phase 1.

| Version | Supported |
|---|---|
| v15, the release of 17 Nov 2026 (`configs/submission.yaml`) | yes |
| anything else | no |

A fix may come out as a new release. It will not change the submitted `AAAAG.csv`, which v15
rebuilds byte for byte.

## Reporting a vulnerability

Please report privately, not in a public issue:
1. **GitHub private vulnerability reporting.** Use the repository's **Security** tab, then
   **Report a vulnerability**. Only the maintainers see the report.
2. **If that option is not available,** open an issue titled "Security contact request" with no
   details in it. A maintainer will reply with a private channel.

A useful report says:
- the version or commit;
- the file and function concerned;
- how to reproduce it;
- what an attacker could do with it.

Please do not put credentials, personal data or any of the organisers' match data in a report. We
are a small volunteer team: we will acknowledge a report as soon as we can, and credit you in the
fix unless you prefer otherwise.

## What this repository does and does not contain

- **No credentials.** There are no tokens, keys or passwords in the repository, and the release is
  scanned for secret-like tokens before it is published. If you find one, report it as above.
- **No match data.** The organisers' videos, tracking, Smart Tagging and homography are licensed to
  participants for the challenge only. They are not here and are not redistributed. Nor is any
  per-frame export of them. If you find anything that looks like match data, please report it
  privately.
- **No third-party datasets.** The external datasets in [docs/EXTERNAL_DATA.md](docs/EXTERNAL_DATA.md)
  are downloaded by each user and are not in the repository.

## Attack surface

- **The rebuild** (`python -m aaaag submit`) reads local files only: the data bundle, the configs
  and the weights. It opens no network connection.
- **Weights** are JSON files. Before use, each is checked against the SHA-256 pin in its pipeline
  config. Inference loads no pickles.
- **Training** loads torch checkpoints with `weights_only=True`. Load only checkpoints you trust.
- **Network access** happens only when you ask for it:
  - `scripts/bootstrap.py` installs the pinned versions of `requirements-lock.txt` from PyPI;
  - the IDSSE and Metrica readers download from a pinned revision and check every file's SHA-256;
  - `scripts/check_upstream.py` compares the vendored organisers' code with their repository.
- **`vendor/mtah2027/`** is the organisers' code, kept byte-exact. Please report problems in it to
  the organisers as well.
