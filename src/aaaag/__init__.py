"""Team AAAAG -- More than a Hack 2027, Phase 1: frame-level ball possession.

The code is a pipeline of pluggable stages that only talk through `aaaag.contracts`:

    source -> MatchBundle -> evidence (per-frame H/A/D probabilities) -> fusion -> decoder -> labels

Every stage is registered by name (`aaaag.registry`) and composed from YAML (`configs/pipelines`),
so a new idea is a new file plus a config, never an edit to someone else's module.
See docs/ARCHITECTURE.md.
"""

__version__ = "0.1.0"
