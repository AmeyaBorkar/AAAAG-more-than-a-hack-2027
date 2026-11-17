# AAAAG: frame-level ball possession from automatic tracking

Team **AAAAG**'s submission to **More than a Hack 2027, Phase 1**, the ball-possession challenge
organised by Metrica Sports, FC Barcelona and Mobile World Capital Barcelona.

For each of the 199,319 frames (25 fps) of FC Barcelona vs Athletic Club, the pipeline says which
team has the ball (`H` home, `A` away), or that the ball is dead (`D`). Its only input is the
organisers' automatic tracking, which is uncorrected, has no player identities and often loses the
ball. A compact network, trained only on other matches, gives per-frame probabilities. A
semi-Markov decoder and three rules turn them into possessions. The build runs on a laptop CPU
with numpy alone, and it reproduces the submitted `AAAAG.csv` byte for byte.

| | |
|---|---|
| **Submission** | `AAAAG.csv`, built by champion **v15** ([`configs/submission.yaml`](configs/submission.yaml) → [`possession_learned_joint_v15.yaml`](configs/pipelines/possession_learned_joint_v15.yaml)) |
| **Expected SHA-256** | `c57e6e7762864ca736cc339bdc49af1b399c9c259dfcfb93eca98e67d8e30cf2` |
| **Ranking metric** | spellF1: possessions matched one to one by temporal IoU, F1 averaged over IoU 0.3–0.7 (macroF1 breaks ties) |
| **Runtime** | CPU only, no GPU and no torch. About 6–12 min on a recent laptop CPU, 25–50 min on a 15 W laptop CPU |
| **Technical document** | [docs/technical_report/REPORT.md](docs/technical_report/REPORT.md) |
| **Licence** | MIT for our code and docs ([LICENSE](LICENSE)). The organisers' code in `vendor/` and all third-party data are excluded |

This repository is a frozen public snapshot of the code, configs, weights, tests and technical
document behind the submission. The team's working records (plans, experiment logs, discussions)
stay in a private repository.

## Contents
- [Results on external data](#results-on-external-data)
- [Reproduce AAAAG.csv](#reproduce-aaaagcsv)
- [How it works](#how-it-works)
- [Training and validation data](#training-and-validation-data)
- [What is in this repository](#what-is-in-this-repository)
- [Compliance](#compliance)
- [Citation, licence and team](#citation-licence-and-team)

## Results on external data

There was no leaderboard and no feedback before the deadline, and the public five minutes of
ground truth were used only to check the file's format and alignment. Every number below is
therefore measured on **other** matches, and no network was scored on a match it was trained on:
Metrica and the PFF test split were never trained on. The IDSSE row uses fold models that leave
out the match they are scored on.

| Check | Matches | Labels | v15 |
|---|---|---|---|
| Metrica Sports sample data | 3 | built by our code from Metrica's events | spellF1 **0.788** (v14: 0.773) |
| Metrica, ball smoothed (Gaussian, sd 8 frames) | 3 | as above | spellF1 **0.740** (v14: 0.739) |
| PFF FC World Cup 2022, sealed test split | 16 | built from PFF's events | **+0.016** spellF1 over v14, better on 14 of 16 games (v14: 0.614) |
| DFL / Sportec IDSSE, leave-one-match-out | 5 | the provider's per-frame flags | **+0.003** on the pre-declared score over v14's networks, ahead on 3 of 3 seed pairs |

How to read these numbers:
- **Metrica** produced the target match's tracking, so it is the closest provider we have. Its
  labels come from hand-tagged events, so they are approximate.
- The **PFF test split** (every 4th game id, fixed before any PFF result) was never trained,
  tuned or selected on. The pretraining did use other games from the same tournament and provider,
  so this check rules out a regression rather than estimating the gain on the target.
- **IDSSE** is the selection suite. Fold models are trained on four matches and scored on the
  fifth, with the tracking degraded to look like the target's. The pre-declared score combines
  several label granularities and a smoothed-ball stress.
- The gains are small next to the spread between matches. Read the table as "no regression,
  small consistent gains", not as a forecast of the score on the target match.

## Reproduce AAAAG.csv

You need:
- **CPython 3.13.** 3.12 and 3.14 give the same bytes.
- **The organisers' data bundle**, which is not in this repository.
- Nothing else: no GPU, no torch and no network access after the install.

```bash
git clone https://github.com/AmeyaBorkar/AAAAG-more-than-a-hack-2027.git aaaag && cd aaaag
python scripts/bootstrap.py              # creates .venv with the exact pins of requirements-lock.txt
# Unpack the bundle OUTSIDE the repository, keeping its tactical/ and broadcast/ folders.
export MTAH_DATA_DIR=/path/to/bundle     # PowerShell: $env:MTAH_DATA_DIR = "C:\path\to\bundle"
.venv/bin/python -m aaaag submit         # Windows: .venv\Scripts\python -m aaaag submit
```

`submit` builds the pipeline named in [`configs/submission.yaml`](configs/submission.yaml) cold,
without any cache. Then it:
1. writes `outputs/submission/AAAAG.csv` (199,319 rows);
2. runs the organisers' validator on the file;
3. prints the file's SHA-256 and compares it with `expected_sha256` in `configs/submission.yaml`.

When the bytes match, the last line reads:

```
reproduction check: MATCH
```

**Why the bytes are stable:**
- **Dependencies** are pinned in [`requirements-lock.txt`](requirements-lock.txt), and
  `bootstrap.py` installs from the lock rather than from `pyproject.toml`'s ranges.
- **Weights** are `models/*.json` files (no pickles), each checked against a SHA-256 pin in the
  pipeline config before use. Inference runs in float64 numpy with BLAS held to one thread.
- **Line endings** are normalised by `.gitattributes`, so a Windows clone hashes the same.
- **Checked on other machines:** the v15 file was rebuilt on a second machine at 1 and 8 threads
  with the same SHA-256. For an earlier champion with the same inference path, a synthetic match
  gave identical bytes on Linux x86 and ARM, macOS ARM and Windows, with AVX2 and AVX-512 CPUs.
  `tests/test_champion_reproducible.py` pins v15's output on a synthetic match.

**Tests.** `pytest -m "not data"` runs the synthetic suite without the bundle. With
`MTAH_DATA_DIR` set, `pytest -m data` adds the checks on the real files.

## How it works

```
 automatic tracking (players + ball, 25 fps; no identities, ball often missing)
        │
        ▼
 learned_team: 3 networks (seeds 2, 3, 4) x 2 mirrored views, numpy
   per frame: a set encoder over the nearest players of each team and the ball
   over time:  a non-causal dilated temporal convolution stack
   outputs:    P(home), P(away), P(dead), and a "possession switches here" logit
        │
        ▼
 factorised fusion          P(dead), and P(home | alive), kept as separate factors
        │
        ▼
 dead-gated semi-Markov     stoppages where P(dead) >= 0.30, then H/A spells with a
 decoder                    duration prior and the switch logit (weight 1.5), per period
        │
        ▼
 rules                      D before each opening kick-off (the organisers' anchors)
                            turnovers snapped to the gaining team's touch within 0.25 s
                            controlled possession: a spell under 1.5 s is kept only if it
                            shows control, otherwise it merges into its neighbours
        │
        ▼
 AAAAG.csv                  H / A / D for every graded frame, X elsewhere
```

**The networks.** Each of the three networks is a *weight soup*: the weighted average of four
networks that share their first two training steps and differ in the last one.

> v15 network (seed *s*) = ½ **W** + ⅙ (**G** + **G′** + **G″**)

- **W** is a TeamNet with `d_model` 128, trained in three steps:
  1. pretrained on 44 PFF FC World Cup 2022 games from the train split;
  2. trained on the seven DFL / Sportec IDSSE matches;
  3. fine-tuned on stoppages where the ball stays visible.
- **G, G′ and G″** start from the same step 2. Their step 3 is W's fine-tune plus an airborne-ball
  projection and spell-balanced loss weights. They differ only in the training seed.
- **Degradation.** Training degrades the clean external tracking to look like the target's. The
  ball-gap simulator is fitted on the three Metrica games, and the player-track noise comes from
  physical priors.

Alone, G is better on balls in flight but worse on Metrica. The soup keeps most of that gain at
little cost on Metrica.

**Why this design:**
- **The metric scores possessions, not frames.** A wrong extra possession costs as much as a missed
  one. The decoder therefore works on whole spells with a duration prior, and it looks both ways in
  time (a lag of 8 frames costs about 10% of spellF1).
- **A missing ball is not a dead ball.** On this provider's tracking (Metrica's) the ball also goes
  missing in open play. The networks are trained with the ball masked at random, so a gap alone
  does not end a possession.
- **The organisers' possession changes only on *control*.** The controlled-possession rule absorbs
  a deflection or a contested touch that never becomes control.

The technical document covers the details, what was tried and dropped, and the limitations.

## Training and validation data

All training and validation data comes from other matches. Nothing was trained, tuned or selected
on the target match or its public ground truth.

| Dataset | Used for | Licence and credit |
|---|---|---|
| **PFF FC** World Cup 2022 tracking | pretraining (step 1): 44 train-split games; the 16 test-split games are a sealed check | PFF FC's terms. Not redistributed here. Thanks to PFF FC. |
| **DFL / Sportec IDSSE** (Bassek et al. 2025, *Scientific Data*) | training (all 7 matches) and selection (leave-one-match-out on 5) | CC BY 4.0, doi:10.6084/m9.figshare.28196177 |
| **Metrica Sports** sample data | independent validation (3 games), and fitting the ball-gap simulator; never trained on as labels | No formal licence; used with acknowledgement and not redistributed. Thanks to Metrica Sports. |

Sources, versions, labellers and adapters are in [docs/EXTERNAL_DATA.md](docs/EXTERNAL_DATA.md).
None of these datasets is in the repository. The readers in `src/aaaag/sources/` fetch or read them
from a local directory (`AAAAG_EXTERNAL_DIR`). The training code is in `src/aaaag/ml/`. Training
needs torch, ideally with a GPU (`requirements-dl.txt`); the rebuild needs neither.

## What is in this repository

```
src/aaaag/          the package: sources -> evidence -> fusion -> decoders -> postprocess -> submission
  contracts.py      the data types every stage exchanges
  ml/               the network, its training code and the numpy inference
configs/            pipelines (YAML), validation suites, degradation models, the champion (submission.yaml)
models/             the pinned network weights (JSON)
vendor/mtah2027/    the organisers' scorer, validator, rules and docs, byte-exact (their MIT licence)
tests/              synthetic tests (pytest -m "not data"); pytest -m data needs the bundle
docs/               the technical document and the external data sources
scripts/            bootstrap.py (environment), policy.py and check_upstream.py (repository checks)
```

## Compliance

The team worked under the challenge's terms and rules
([vendor/mtah2027/RULES.md](vendor/mtah2027/RULES.md)):
- **The match data never reached an AI model.** The videos, tracking, tags and homography stayed on
  the members' own machines. AI coding assistants saw code and aggregate statistics only.
- **Every label comes from an algorithm.** There are no hand labels or hand corrections, no
  human-made data about this match, and no code or threshold tied to a moment of it. The only
  match-specific constants are the organisers' own metadata in `vendor/mtah2027/games.py`, such as
  the graded periods and the kick-off anchors.
- **The public five minutes were a smoke test only.** They checked the file format and alignment.
  Nothing was trained, tuned or selected on them.
- **No match data is redistributed.** This repository holds no videos, tracking, tags,
  predictions or per-frame exports of the match, and `.gitignore` keeps them out.

## Citation, licence and team

**Citing.** [CITATION.cff](CITATION.cff) has the citation metadata, and GitHub's "Cite this
repository" button reads it. Please also cite the challenge (`vendor/mtah2027/CITATION.cff`) and the
datasets above.

**Licence.** Our code and documentation are MIT ([LICENSE](LICENSE)). That licence does not cover:
- `vendor/mtah2027/`, the organisers' code, docs and public ground truth, under their own MIT
  licence ([vendor/mtah2027/LICENSE](vendor/mtah2027/LICENSE));
- the match data, which is licensed to participants for the challenge only and is not here;
- the external datasets, which keep their own terms.

**Contributing.** This is a frozen competition snapshot, so changes that alter `AAAAG.csv` cannot
be merged. Issues and questions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Everyone
taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). To report a vulnerability, follow
[SECURITY.md](SECURITY.md), not a public issue.

**Team AAAAG:** Ameya, Arnav, Bakshi and Ayush.
