# Configs

| Folder / file | What | Who edits |
|---|---|---|
| `pipelines/*.yaml` | Pipelines: evidence producers, fusion, decoder, all parameters. New idea = new file. | Anyone may add a file. The author edits their own file. |
| `pipelines/official_baseline_*.yaml` | Exact reproductions of the organisers' four baselines (parity references) | Nobody. They are frozen. |
| `suites/*.yaml` | Validation suites (lists of labelled matches) | The data lane |
| `submission.yaml` | The champion and its expected SHA-256 | The captain, through the champion protocol |
| `matches/bar-ath.yaml` | Team names and Smart Tagging spellings only | The data lane. Never labels, frames or thresholds. |

Schema details are in `docs/ARCHITECTURE.md` (section "configs", in the team repository). Every parameter
that could be tuned belongs in a pipeline YAML, so experiments are fully described by their
config digest.
