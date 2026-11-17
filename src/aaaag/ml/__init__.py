"""Learned team-in-possession harness (issue #82): token windows, a small model, LOMO training,
deterministic inference, and a registered evidence producer.

What is here and what needs torch:

| module | needs torch | role |
|---|---|---|
| `spec`, `geometry`, `windows`, `augment`, `data` | no (numpy only) | turn a MatchBundle into token tensors, degrade and mask on the fly, sample windows |
| `model`, `train`, `infer` | yes | the network, the leave-one-match-out trainer, single-thread CPU inference |
| `evaluate`, `cli` | inference needs torch | fold reports against the champion; `python -m aaaag.ml` |

The registered producer `learned_team` lives in `aaaag/evidence/possession/learned_team.py`. It
imports torch only when it has weights to run, so the repository's normal environment (no torch,
`requirements-lock.txt`) loads and tests it. The DL environment is `requirements-dl.txt`; see
docs/ml/README.md.

Data rules (docs/COMPLIANCE.md): training and tuning use external matches only. The IDSSE dev
matches are the leave-one-match-out roster. `train.extra_matches` may add other labelled sources;
`idsse` and `bar-ath` are refused there, so the exam stays out and BAR-ATH is never read. The
IDSSE source still refuses the exam without `exam=True`, which nothing here passes.
"""
