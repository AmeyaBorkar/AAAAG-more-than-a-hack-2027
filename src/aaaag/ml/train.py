"""Training: one fold of leave-one-match-out, degradation on the fly, bf16 on the GPU.

    run = RunConfig.from_yaml("configs/ml/team_baseline.yaml")
    train(run, matches, out_dir)           # matches: the TrainMatch list of the fold

What a step does:
- the training matches are degraded afresh every `refresh_steps` steps (`data.build_set`), built
  on a background thread while the GPU works on the previous set, so a set is a pure function of
  (run seed, set index, match order);
- a batch is `batch_size` windows of `window` frames drawn inside periods, from the coupled or
  from the decoupled variants (`decouple_prob`: the missing-ball shortcut breaker);
- each window may be mirrored in y and/or have its teams swapped with H and A exchanged;
- the loss is the class-weighted 3-class cross entropy plus `aux_weight` times a Huber loss on the
  ball position the network reads from the players, on frames where the ball is alive and hidden
  from the input; both only on the frames whose temporal context is complete (the interior,
  `receptive_half` frames away from each window edge). When `train.aux_displaced_m` is positive,
  the Huber term also covers alive frames where the input ball is seen but more than that many
  metres from the clean provider ball it regresses to: the variant's fed ball after degrade, the
  coupled-ball steps (airborne projection included), the augmentations and the decoupled hiding,
  against the clean ball, both in pitch metres before attack normalisation. The mask is built
  per match and per variant with the visibility masks, so it follows `refresh_steps`. 0 builds
  no mask and leaves the tensors and the random stream unchanged;
- when `train.possessor_weight` is positive, the same interior frames also train the possessor
  head with a cross entropy against an algorithmic slot target. Weight 0 does not build those
  targets and leaves the loss and the random stream unchanged;
- when `train.switch_weight` is positive, those interior frames also train the switch head with a
  BCE-with-logits against direct H↔A changes in the window labels. Weight 0 does not build that
  target and leaves the loss and the random stream unchanged. A team swap and a y flip leave the
  target unchanged, because a switch has no direction;
- when `train.scrf_weight` is positive, those interior frames also train the team log-odds and the
  switch logit by the negative log-likelihood of the true block segmentation under the shipped
  dead-gated HSMM (scrf.py). A short H/A spell (under 38 frames, measured on the full match) is
  free in the constrained term. The mask is built once per match, like the touch target, and a
  team swap or a y flip leaves it unchanged. Weight 0 does not build it and leaves the loss and
  the random stream unchanged. The D logit gets no gradient from this term;
- when `train.tmse_weight` is positive, those interior frames also add the truncated temporal MSE
  of MS-TCN (Farha and Gall, CVPR 2019) on the class head: the squared step of each class's
  log-probability between consecutive frames, against the earlier frame detached, capped at
  `tmse_tau**2` and averaged over the frame pairs where neither label is X. It discourages
  frame-to-frame flicker; a step past the cap passes no gradient, so a sharp change costs a
  bounded amount. Pretraining has no class loss and leaves it out. Weight 0 leaves the loss, the
  logged terms and the random stream unchanged;
- when `train.boundary_weight` is set (E155), each frame of that cross entropy also carries a weight
  of `factor` within `window_s` seconds of a change of the training target label (H, A or D on
  both sides, inside one period) and 1 elsewhere, built once per match on the full target labels
  after the label rewrite (boundaryweight.py). It multiplies the class weights and the spell
  weights, and the cross entropy stays the mean over the product of the weights. A team swap and
  a y flip leave it unchanged. Null (the default) does not build it and leaves the loss, the
  random stream, the saved config and the checkpoint unchanged;
- when `train.touch_weight` is positive, those interior frames also train the touch head with a
  BCE-with-logits against a target built once per match from the clean ball, before degradation
  or augmentation: frames within `touch_tol_frames` of a `detect_touches` event (any team) are 1,
  other alive frames with a seen clean ball are 0, and the rest are ignored. Weight 0 does not
  build that target and leaves the loss, the random stream and the checkpoint unchanged. A team
  swap and a y flip leave it unchanged, because a touch has no team. `train.touch_source`
  "events" (default "tracking", today's target) takes the touch frames from the provider's
  on-ball events instead (`aaaag.sources.idsse_events`, DFL events on IDSSE), with the same
  tolerance and the same supervised frames; every training match must come from a source with
  events (`data.EVENT_SOURCES`), or the run raises before training. It needs `touch_weight` > 0;
- when `train.event_switch_weight` is positive, those interior frames also train the switch head
  with a second BCE-with-logits (`switch_pos_weight`), against the possession changes of the
  provider events (`idsse_events.possession_changes`): H/A frames within `switch_tol_frames` of
  a change are 1, other H/A frames 0, D and X ignored. Built once per match, like the touch
  target, from a source with events; reported as "esw". Weight 0 does not build it and leaves the
  loss, the random stream and the checkpoint unchanged. A team swap and a y flip leave it
  unchanged;
- when `train.init_add_touch_head` is true, `init_from` may be a checkpoint without the touch head
  while this run turns `model.touch_head` on: its weights load, and the touch layer starts from
  its seeded initialisation. False (the default) keeps the strict match;
- when `train.distill` is set, that cross entropy is mixed with a temperature-scaled KL toward the
  mean logits of frozen teachers (the same average inference uses), on those same frames and class
  weights. The teachers see the batch after the team swap and the y flip. `teacher_input`
  `clean_ball` shows them tokens from that same draw with ball smoothing skipped: one extra token
  array per variant, resident on the device. `same` (the default) shows them the student tokens
  and leaves the tensors and the random stream unchanged. `switch_alpha` above 0 also trains the
  switch head toward the teachers' switch probability, and only when both sides have that head.
  `distill` null or absent does not load a teacher and leaves the loss and the random stream
  unchanged;
- when `train.ema_decay` is above 0, every floating parameter and floating buffer is averaged after
  the optimizer step, in float32 on the device and outside autocast, starting from the initial
  weights: decay * average + (1 - decay) * weight. Decay 0 does not allocate that copy and leaves
  the step, the checkpoints and the saved config unchanged;
- when `train.extra_matches` is non-empty, those `source/match_id` matches are appended to the
  list passed in (leave-one-match-out appends the same ones to every fold and never holds one
  out). They are sampled by played frames times `extra_weight`. `DegradeMix` is not applied to
  them unless `extra_degrade` (default false, today's behaviour); `train.augment` still is.
  `extra_source_kwargs` is a source name mapped to loader keywords (null is the same as
  omitting it) and is passed whenever those matches are loaded. Empty leaves the tensors, the random
  stream and the checkpoint as they are. The opening log line names how many there are;
- E119's airborne projection (`train.airborne_p`) reaches only the IDSSE matches. With
  `train.airborne_extra` true it also reaches every extra match whose source has a ball height
  (PFF's raw-ball z, `aaaag.sources.pff_ballz`, train split only), with the same camera range
  and at the same point of the draw. Other sources are left as they are. Every height is read
  once before training, and a run where no extra match has one raises. With `aux_displaced_m`
  the projected frames also enter the auxiliary ball loss, in pretraining as in any other run.
  False (the default) leaves the tensors, the random stream and the saved config as they are;
- when `train.pretrain` is true, the run trains only on those extras and every one of them must be
  label-free (`extra_labels: none`). No IDSSE match is loaded. The class, switch and touch losses
  and distillation are left out of the step (their weights are treated as 0); the auxiliary ball
  loss and the possessor loss are the only terms, so one of those weights must be positive. Class
  weights are not computed. False leaves the roster, the loss and the checkpoint as they are.
  Leave-one-match-out refuses a pretraining run: it has no folds;
- `--extra-only` on the train command is not that flag. It passes no IDSSE match, so the normal
  loss, degradation, augmentation and checkpoint run on `train.extra_matches` alone. An empty
  extra list raises. A held-out dev match can still be monitored and is not trained on.
  Without the flag the roster is unchanged;
- when `train.matches_per_set` is a positive int smaller than the roster, each variant set
  tokenises that many matches, drawn without replacement from a generator seeded by the run
  seed and the set index, and kept in roster order. Each match's degradation seed stays its
  index in the full roster. Class weights stay those of the full roster. Null, or a cap at
  least as large as the roster, tokenises every match, so the tensors, the random stream and
  the checkpoint stay as they are. A 146k-frame match is about 94 MB of float16 tokens per
  variant; 48 matches and both variants are about 9 GB, and the next set is built while those
  tensors are still resident;
- when `train.init_from` is a checkpoint path, those weights are loaded after the model is built
  and before the optimizer is created. The architecture and the token spec must match, and the
  optimizer starts fresh. None leaves the initial weights as they are. A path with no `{fold}`
  is loaded as written, by a single run and by every leave-one-match-out fold. A path with
  `{fold}` is only valid in leave-one-match-out: each fold replaces every `{fold}` with its
  held-out match id, so fold J03WMX loads `.../fold-J03WMX/final.pt` and does not start from a
  model that was trained on J03WMX. A single training run whose path contains `{fold}` raises
  ValueError. The configured path, placeholder included, is kept in the saved run config. A
  resume from `last.pt` still continues that file and does not reload `init_from`;
- when `train.init_from_partial` is true, that checkpoint may lack the `model.slot_temporal`
  branch (and `token.slot_index`): every other config key, tensor name, shape and dtype must still
  match, and the branch keeps its initial weights, whose zero output projection makes step 0 the
  parent's network. False (the default) leaves the strict load, the log line and the saved config
  as they are.

Seeds: numpy generators come from (run seed, set index); torch is reseeded per set, so a resumed
run continues exactly like an uninterrupted one (up to GPU kernel non-determinism). There is no
early stopping and no model selection inside a fold: the schedule is fixed, and the held-out match
is only ever scored afterwards (docs/ml/README.md). With `ema_decay` at 0 the last weights are the
model; above 0, `final.pt` is the moving average and `final_raw.pt` is the last iterate.

Where it fails: four training matches are four sequences of play, so it overfits quickly; the
degradations cover what degrade models, not what the competition tracker does on top.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from ..contracts import A, H
from ..postprocess.controlled import controlled_possession
from ..postprocess.min_spell import absorb_short
from .augment import AugmentMix
from .boundaryweight import BoundaryWeight, boundary_weights
from .checkpoint import load_checkpoint, load_payload, save_checkpoint
from .data import (
    EXTRA_LABEL_MODES,
    TOUCH_SOURCES,
    DegradeMix,
    SetEntry,
    TrainMatch,
    build_set,
    class_weights,
    include_extra_matches,
    load_extra_matches,
    split_extra_entry,
    with_event_switch_target,
    with_touch_target,
)
from .geometry import symmetric
from .model import ModelConfig, build_model, count_parameters, get_architecture
from .scrf import scrf_nll, short_spell_mask
from .spec import TokenSpec, slot_input_columns
from .spellweight import SpellWeight, match_spell_seconds, spell_weights
from .windows import sample_starts

HUBER_DELTA = 0.25  # in units of ref_scale_m (5 m)
IGNORE = 3  # the X label


_TEACHER_INPUTS = ("same", "clean_ball")


@dataclass(frozen=True)
class DistillConfig:
    """Mix the hard-label cross entropy with a KL toward a frozen teacher ensemble.

    `teacher_runs` are run directories, not checkpoint files: `teacher_paths` picks the file for
    the fold. `alpha` weights the distillation term and `1 - alpha` stays on the hard labels.
    `temperature` softens both softmaxes; the KL is multiplied by `temperature` squared.
    `teacher_input` `same` shows the teachers the student tokens. `clean_ball` shows them the
    same windows with ball smoothing skipped. `switch_alpha` weights a BCE of the student's
    switch logit toward the teachers' switch probability; 0 leaves that head's loss unchanged.
    """

    teacher_runs: tuple[str, ...]
    alpha: float = 0.7
    temperature: float = 1.0
    teacher_input: str = "same"
    switch_alpha: float = 0.0

    def __post_init__(self) -> None:
        runs = self.teacher_runs
        if isinstance(runs, str) or not isinstance(runs, tuple) or not runs:
            raise ValueError("distill.teacher_runs must list at least one run directory")
        if any(not str(path) for path in runs):
            raise ValueError("distill.teacher_runs contains an empty path")
        if not math.isfinite(self.alpha) or not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError(f"distill.alpha must be between 0 and 1, got {self.alpha}")
        if not math.isfinite(self.temperature) or float(self.temperature) <= 0.0:
            raise ValueError(f"distill.temperature must be positive, got {self.temperature}")
        mode = self.teacher_input
        if not isinstance(mode, str) or mode not in _TEACHER_INPUTS:
            raise ValueError(f"distill.teacher_input must be 'same' or 'clean_ball', got {mode!r}")
        weight = self.switch_alpha
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise ValueError(f"distill.switch_alpha must be a finite number >= 0, got {weight!r}")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"distill.switch_alpha must be a finite number >= 0, got {weight}")
        if not isinstance(self.switch_alpha, float):
            object.__setattr__(self, "switch_alpha", weight)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DistillConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"distill must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - {
            "teacher_runs",
            "alpha",
            "temperature",
            "teacher_input",
            "switch_alpha",
        }
        if unknown:
            raise ValueError(f"unknown distill keys: {sorted(unknown)}")
        runs = raw.get("teacher_runs")
        if isinstance(runs, str) or not isinstance(runs, list | tuple) or not runs:
            raise ValueError("distill.teacher_runs must be a non-empty list of run directories")
        return cls(
            teacher_runs=tuple(str(path) for path in runs),
            alpha=float(raw.get("alpha", 0.7)),
            temperature=float(raw.get("temperature", 1.0)),
            teacher_input=str(raw.get("teacher_input", "same")),
            switch_alpha=float(raw.get("switch_alpha", 0.0)),
        )


def teacher_paths(runs: Sequence[str | Path], holdout: str | None) -> list[Path]:
    """Final checkpoints of `runs` for one fold. Missing files raise before any training step.

    Held-out match ``M`` reads ``<run>/fold-M/final.pt`` (that fold never trained on M).
    ``holdout`` None or ``"none"`` reads ``<run>/final.pt`` (trained on every match).
    """
    if holdout in (None, "none"):
        relative = Path("final.pt")
        where = "holdout none"
    else:
        if Path(str(holdout)).name != str(holdout):
            raise ValueError(f"holdout must be a match id, got {holdout!r}")
        relative = Path(f"fold-{holdout}") / "final.pt"
        where = f"holdout {holdout}"
    found: list[Path] = []
    missing: list[str] = []
    for run_dir in runs:
        path = Path(run_dir) / relative
        if path.is_file():
            found.append(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"missing teacher checkpoint(s) for {where}: {', '.join(missing)}")
    return found


def _normalise_extra_source_kwargs(raw) -> dict | None:
    """A source name mapped to loader keywords, or None when unset or empty."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"train.extra_source_kwargs must be a mapping or null, got {type(raw).__name__}"
        )
    out: dict = {}
    for source, params in raw.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"train.extra_source_kwargs keys must be source names, got {source!r}")
        if not isinstance(params, dict):
            raise ValueError(
                f"train.extra_source_kwargs[{source!r}] must be a mapping, "
                f"got {type(params).__name__}"
            )
        out[source] = dict(params)
    return out or None


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 1500
    refresh_steps: int = 100
    batch_size: int = 16
    window: int = 1000
    lr: float = 2e-3
    final_lr_frac: float = 0.02
    warmup_steps: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    aux_weight: float = 0.2
    aux_displaced_m: float = 0.0  # >0: the aux loss also covers a seen fed ball this far (m) off
    decouple_prob: float = 0.5
    flip_y: bool = True
    swap_teams: bool = True
    amp: bool = True  # bf16 autocast on CUDA
    degrade: DegradeMix = field(default_factory=DegradeMix)
    augment: AugmentMix = field(default_factory=AugmentMix)  # every p == 0 unless the YAML sets it
    distill: DistillConfig | None = None  # null: the loss and the random stream are unchanged
    spell_weight: SpellWeight | None = None  # null or gamma 0: the plain class-weighted loss
    boundary_weight: BoundaryWeight | None = None  # E155: null leaves the loss as it is
    possessor_weight: float = 0.0  # 0: the loss, the targets and the random stream stay as they are
    possessor_reach_m: float = 2.0  # a player farther than this from the clean ball is not a target
    switch_weight: float = 0.0  # 0: the loss, the target and the random stream stay as they are
    switch_tol_frames: int = 3  # H/A frames within this many frames of a direct H↔A change
    switch_pos_weight: float = 20.0  # BCE pos_weight; a switch frame is rare
    touch_weight: float = 0.0  # 0: the loss, the target and the random stream stay as they are
    touch_tol_frames: int = 2  # supervised frames within this many frames of a clean-ball touch
    touch_pos_weight: float = 10.0  # BCE pos_weight; a touch frame is the minority
    touch_source: str = "tracking"  # "tracking": detect_touches on the clean ball; "events": DFL
    event_switch_weight: float = (
        0.0  # 0: the loss, the target and the random stream stay as they are
    )
    init_add_touch_head: bool = False  # True: init_from may lack the touch head this run adds
    ema_decay: float = 0.0  # 0: no shadow and no second checkpoint; the run stays as it is
    restart_blank_p: float = 0.0  # E56: >0 hides the ball past true restarts (coupled variant)
    restart_blank_before_s: float = 1.0  # up to this long before the restart (uniform)
    restart_blank_after_s: float = 3.0  # up to this long after the restart (uniform)
    stoppage_carry_p: float = 0.0  # E59: >0 has the restart taker carry the ball (coupled variant)
    stoppage_carry_max_s: float = 6.0  # up to this long before the restart (uniform)
    dead_visible_p: float = 0.0  # E71: >0 shows the uncarried dead ball (coupled variant)
    airborne_p: float = (
        0.0  # E119: >0 projects the coupled ball as an airborne ball seen by a camera
    )
    airborne_cz_min: float = 15.0  # camera height range (m)
    airborne_cz_max: float = 25.0
    airborne_extra: bool = False  # True: also project extra matches with a ball height (PFF z)
    frag_swap_p: float = 0.0  # E66: >0 cuts player tracks into fragments, 3% to the wrong team
    drop_dup_p: float = 0.0  # E66: >0 drops 10% of slot-frames and duplicates 5% (2 s blocks)
    long_gap_p: float = 0.0  # E66: >0 hides the coupled ball 6 s, 1 per minute of alive play
    alive_gap_target: float = 0.0  # E158: >0 label-free coupled-ball gaps up to this alive share
    extra_matches: tuple[str, ...] = ()  # "source/match_id"; empty leaves the run unchanged
    extra_degrade: bool = False  # False: extra matches stay as delivered; augment still runs
    extra_source_kwargs: dict | None = (
        None  # source -> loader kwargs; null leaves the load unchanged
    )
    extra_weight: float = 1.0  # sampling multiplier on played frames, extra matches only
    extra_labels: str = "all"  # "all", "team" (D -> X), "core" or "interior" (extra_label_view)
    target_min_s: float = (
        0.0  # >0: absorb possessions shorter than this (E40); hold_s if target_rule is ctrl
    )
    target_rule: str = ""  # "": no control rule; "ctrl": controlled_possession (E40c)
    pretrain: bool = False  # True: label-free extras only; class, switch, touch and distill off
    matches_per_set: int | None = None  # None: tokenise every match; else that many per variant set
    init_from: str | None = None  # checkpoint before the optimizer; {fold} is per lomo fold
    scrf_weight: float = 0.0  # E69: >0 trains team and switch logits through the shipped HSMM
    scrf_lam_team: float = 0.6  # the dead-gated decoder's lam_team
    scrf_switch_weight: float = 2.0  # the dead-gated decoder's switch bonus scale
    tmse_weight: float = 0.0  # MS-TCN T-MSE on the class head; 0: the loss stays as it is
    tmse_tau: float = 4.0  # truncation: a squared log-prob step counts at most tau**2
    init_from_partial: bool = False  # True: init_from may lack the slot_temporal branch

    def __post_init__(self) -> None:
        target = self.target_min_s
        if isinstance(target, bool) or not isinstance(target, int | float) or not target >= 0.0:
            raise ValueError(f"train.target_min_s must be a number >= 0, got {target!r}")
        rule = self.target_rule
        if not isinstance(rule, str) or rule not in ("", "ctrl"):
            raise ValueError(f"train.target_rule must be '' or 'ctrl', got {rule!r}")
        if self.extra_labels not in EXTRA_LABEL_MODES:
            raise ValueError(
                f"train.extra_labels must be one of {EXTRA_LABEL_MODES}, got {self.extra_labels!r}"
            )
        decay = self.ema_decay
        if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError(f"train.ema_decay must be between 0 and 1, got {decay}")
        carry_p, carry_s = self.stoppage_carry_p, self.stoppage_carry_max_s
        if not math.isfinite(carry_p) or not 0.0 <= carry_p <= 1.0:
            raise ValueError(f"train.stoppage_carry_p must be between 0 and 1, got {carry_p}")
        if not math.isfinite(carry_s) or not carry_s > 0.0:
            raise ValueError(f"train.stoppage_carry_max_s must be above 0, got {carry_s}")
        if not 0.0 <= self.airborne_p <= 1.0:
            raise ValueError(f"train.airborne_p must be between 0 and 1, got {self.airborne_p}")
        if not 0.0 < self.airborne_cz_min <= self.airborne_cz_max:
            raise ValueError("train.airborne_cz_min / _max must satisfy 0 < min <= max")
        if not isinstance(self.airborne_extra, bool):
            raise ValueError(f"train.airborne_extra must be a bool, got {self.airborne_extra!r}")
        if self.airborne_extra and not self.airborne_p > 0.0:
            raise ValueError(
                "train.airborne_extra projects extra matches with train.airborne_p; "
                "it needs airborne_p > 0"
            )
        if not 0.0 <= self.tmse_weight:
            raise ValueError(f"train.tmse_weight must be >= 0, got {self.tmse_weight}")
        if not 0.0 < self.tmse_tau:
            raise ValueError(f"train.tmse_tau must be above 0, got {self.tmse_tau}")
        if self.touch_source not in TOUCH_SOURCES:
            raise ValueError(
                f"train.touch_source must be one of {TOUCH_SOURCES}, got {self.touch_source!r}"
            )
        if self.touch_source != "tracking" and not float(self.touch_weight) > 0.0:
            raise ValueError(
                f"train.touch_source {self.touch_source!r} has no effect with touch_weight 0"
            )
        if not isinstance(self.init_add_touch_head, bool):
            raise ValueError(
                f"train.init_add_touch_head must be a bool, got {self.init_add_touch_head!r}"
            )
        visible_p = self.dead_visible_p
        if not math.isfinite(visible_p) or not 0.0 <= visible_p <= 1.0:
            raise ValueError(f"train.dead_visible_p must be between 0 and 1, got {visible_p}")
        for key in ("frag_swap_p", "drop_dup_p", "long_gap_p"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"train.{key} must be between 0 and 1, got {value}")
        gap = self.alive_gap_target
        if isinstance(gap, bool) or not isinstance(gap, int | float) or not 0.0 <= gap < 1.0:
            raise ValueError(f"train.alive_gap_target must be in [0, 1), got {gap!r}")
        if not isinstance(gap, float):
            object.__setattr__(self, "alive_gap_target", float(gap))
        for key in (
            "scrf_weight",
            "scrf_lam_team",
            "scrf_switch_weight",
            "aux_displaced_m",
            "event_switch_weight",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"train.{key} must be a finite number >= 0, got {value!r}")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"train.{key} must be a finite number >= 0, got {value!r}")
            if not isinstance(getattr(self, key), float):
                object.__setattr__(self, key, value)
        entries = self.extra_matches
        if isinstance(entries, str) or not isinstance(entries, tuple | list):
            raise ValueError(
                "train.extra_matches must be a list of 'source/match_id', "
                f"got {type(entries).__name__}"
            )
        normalised = tuple(entries)
        for entry in normalised:
            split_extra_entry(entry)
        if not isinstance(entries, tuple):
            object.__setattr__(self, "extra_matches", normalised)
        if not isinstance(self.extra_degrade, bool):
            raise ValueError(f"train.extra_degrade must be a bool, got {self.extra_degrade!r}")
        source_kwargs = _normalise_extra_source_kwargs(self.extra_source_kwargs)
        if source_kwargs != self.extra_source_kwargs:
            object.__setattr__(self, "extra_source_kwargs", source_kwargs)
        weight = self.extra_weight
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise ValueError(f"train.extra_weight must be a finite number >= 0, got {weight!r}")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"train.extra_weight must be a finite number >= 0, got {weight}")
        if not isinstance(self.extra_weight, float):
            object.__setattr__(self, "extra_weight", weight)
        distill = self.distill
        if distill is not None and distill.teacher_input == "clean_ball":
            if not distill.teacher_runs:
                raise ValueError("distill.teacher_input 'clean_ball' requires distill.teacher_runs")
            if self.augment.ball_smooth.p <= 0.0:
                raise ValueError(
                    "distill.teacher_input 'clean_ball' requires "
                    "train.augment.ball_smooth with p > 0"
                )
        if not isinstance(self.pretrain, bool):
            raise ValueError(f"train.pretrain must be a bool, got {self.pretrain!r}")
        init_from = self.init_from
        if init_from is not None and (not isinstance(init_from, str) or init_from == ""):
            raise ValueError(
                f"train.init_from must be a checkpoint path or null, got {init_from!r}"
            )
        if not isinstance(self.init_from_partial, bool):
            raise ValueError(
                f"train.init_from_partial must be a bool, got {self.init_from_partial!r}"
            )
        if self.init_from_partial and init_from is None:
            raise ValueError("train.init_from_partial needs train.init_from")
        # `{fold}` is legal in the config. Leave-one-match-out replaces it at train time;
        # a single train() raises. Rejecting it here would make that YAML fail to load.
        if self.pretrain:
            if self.extra_labels != "none":
                raise ValueError(
                    "train.pretrain requires extra_labels 'none' so every extra match is "
                    f"label-free, got {self.extra_labels!r}"
                )
            if not self.extra_matches:
                raise ValueError(
                    "train.pretrain trains only on train.extra_matches, and that list is empty"
                )
            if not (float(self.aux_weight) > 0.0 or float(self.possessor_weight) > 0.0):
                raise ValueError(
                    "train.pretrain needs aux_weight > 0 or possessor_weight > 0; "
                    "the class, switch and touch losses are off"
                )
        per_set = self.matches_per_set
        if per_set is not None and (
            isinstance(per_set, bool) or not isinstance(per_set, int) or per_set < 1
        ):
            raise ValueError(f"train.matches_per_set must be an int >= 1 or null, got {per_set!r}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        raw = dict(raw)
        if "degrade" in raw:
            raw["degrade"] = DegradeMix.from_dict(raw["degrade"])
        if "augment" in raw:
            raw["augment"] = AugmentMix.from_dict(raw["augment"])
        if "distill" in raw:
            distill = raw["distill"]
            raw["distill"] = None if distill is None else DistillConfig.from_dict(distill)
        if "spell_weight" in raw:
            spell = raw["spell_weight"]
            raw["spell_weight"] = None if spell is None else SpellWeight.from_dict(spell)
        if "boundary_weight" in raw:
            near = raw["boundary_weight"]
            raw["boundary_weight"] = None if near is None else BoundaryWeight.from_dict(near)
        if "extra_matches" in raw:
            entries = raw["extra_matches"]
            if entries is None:
                raw["extra_matches"] = ()
            elif isinstance(entries, str) or not isinstance(entries, list | tuple):
                raise ValueError(
                    "train.extra_matches must be a list of 'source/match_id', "
                    f"got {type(entries).__name__}"
                )
            else:
                raw["extra_matches"] = tuple(entries)
        return cls(**raw)


# Architectures built on TeamNet's heads (possessor, switch, touch): `team_net` and its E81 variant
# with attention over time (archs/team_net_tx.py). Other designs cannot use those heads' losses.
TEAM_NET_ARCHS = ("team_net", "team_net_tx")


@dataclass(frozen=True)
class RunConfig:
    """Token spec, model and training settings of one run: the YAML under configs/ml/."""

    token: TokenSpec = field(default_factory=TokenSpec)
    model: Any = field(default_factory=ModelConfig)  # the config object of `arch`
    train: TrainConfig = field(default_factory=TrainConfig)
    arch: str = "team_net"  # a registered architecture (model.py, archs/)

    def __post_init__(self) -> None:
        if getattr(self.model, "tokens_per_team", self.token.tokens_per_team) != (
            self.token.tokens_per_team
        ):
            raise ValueError("model.tokens_per_team must equal token.tokens_per_team")
        if self.train.possessor_weight > 0 and (
            self.arch not in TEAM_NET_ARCHS or not getattr(self.model, "possessor_head", False)
        ):
            raise ValueError(
                "train.possessor_weight > 0 requires a team_net model with model.possessor_head on"
            )
        if self.train.switch_weight > 0 and (
            self.arch not in TEAM_NET_ARCHS or not getattr(self.model, "switch_head", False)
        ):
            raise ValueError(
                "train.switch_weight > 0 requires a team_net model with model.switch_head on"
            )
        if self.train.scrf_weight > 0 and (
            self.arch not in TEAM_NET_ARCHS or not getattr(self.model, "switch_head", False)
        ):
            raise ValueError(
                "train.scrf_weight > 0 requires a team_net model with model.switch_head on"
            )
        if self.train.touch_weight > 0 and (
            self.arch not in TEAM_NET_ARCHS or not getattr(self.model, "touch_head", False)
        ):
            raise ValueError(
                "train.touch_weight > 0 requires a team_net model with model.touch_head on"
            )
        slot = getattr(self.model, "slot_temporal", None)
        slot_on = bool(slot is not None and slot.enabled)
        if slot_on != self.token.slot_index:
            raise ValueError(
                "model.slot_temporal.enabled and token.slot_index go together: the slot branch "
                "reads the SLOT_ID and SLOT_RUN columns, and nothing else does "
                f"(got enabled {slot_on}, slot_index {self.token.slot_index})"
            )
        if slot_on and self.model.n_features != self.token.model_features:
            raise ValueError(
                f"model.n_features {self.model.n_features} must equal the embedded token columns "
                f"{self.token.model_features} (token.slot_index adds two that are not embedded)"
            )
        if self.train.event_switch_weight > 0 and (
            self.arch != "team_net" or not getattr(self.model, "switch_head", False)
        ):
            raise ValueError(
                "train.event_switch_weight > 0 requires a team_net model with model.switch_head on"
            )
        if self.train.init_add_touch_head and (
            not self.train.init_from or not getattr(self.model, "touch_head", False)
        ):
            raise ValueError(
                "train.init_add_touch_head needs train.init_from and model.touch_head on"
            )
        if self.train.init_add_touch_head and self.train.init_from_partial:
            raise ValueError(
                "train.init_add_touch_head and train.init_from_partial cannot be combined"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        unknown = set(raw) - {"arch", "token", "model", "train"}
        if unknown:
            raise ValueError(f"unknown run config keys: {sorted(unknown)}")
        token = TokenSpec.from_dict(raw.get("token", {}))
        arch = raw.get("arch", "team_net")
        config_cls, _ = get_architecture(arch)
        reads_layout = "tokens_per_team" in {f.name for f in fields(config_cls)}
        model = {**({"tokens_per_team": token.tokens_per_team} if reads_layout else {})}
        model.update(raw.get("model", {}))
        slot = model.get("slot_temporal")
        if isinstance(slot, dict) and "inputs" in slot:  # names such as "accel" need the spec
            model["slot_temporal"] = {**slot, "inputs": slot_input_columns(slot["inputs"], token)}
        return cls(
            token=token,
            model=config_cls.from_dict(model),
            train=TrainConfig.from_dict(raw.get("train", {})),
            arch=arch,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> dict[str, Any]:
        train = asdict(self.train)
        train["degrade"] = {
            k: list(v) if isinstance(v, tuple) else v for k, v in train["degrade"].items()
        }
        if train.get("distill") is None:
            train.pop("distill", None)
        else:
            # Defaults match a distill block written before these keys existed.
            distill = train["distill"]
            if distill.get("teacher_input", "same") == "same":
                distill.pop("teacher_input", None)
            if not distill.get("switch_alpha"):
                distill.pop("switch_alpha", None)
        if train.get("spell_weight") is None:
            train.pop("spell_weight", None)
        elif train["spell_weight"].get("length_view", "provider") == "provider":
            # The default matches a spell_weight block written before the key existed.
            train["spell_weight"].pop("length_view", None)
        if train.get("boundary_weight") is None:
            train.pop("boundary_weight", None)
        if not train.get("possessor_weight"):
            train.pop("possessor_weight", None)
            train.pop("possessor_reach_m", None)
        if not train.get("switch_weight"):
            train.pop("switch_weight", None)
            train.pop("switch_tol_frames", None)
            train.pop("switch_pos_weight", None)
        if not train.get("touch_weight"):
            train.pop("touch_weight", None)
            train.pop("touch_tol_frames", None)
            train.pop("touch_pos_weight", None)
        # Defaults match a config written before these keys existed.
        if train.get("touch_source", "tracking") == "tracking":
            train.pop("touch_source", None)
        if not train.get("event_switch_weight"):
            train.pop("event_switch_weight", None)
        if not train.get("init_add_touch_head"):
            train.pop("init_add_touch_head", None)
        if not train.get("ema_decay"):
            train.pop("ema_decay", None)
        if not train.get("restart_blank_p"):
            train.pop("restart_blank_p", None)
            train.pop("restart_blank_before_s", None)
            train.pop("restart_blank_after_s", None)
        if not train.get("stoppage_carry_p"):
            train.pop("stoppage_carry_p", None)
            train.pop("stoppage_carry_max_s", None)
        if not train.get("dead_visible_p"):
            train.pop("dead_visible_p", None)
        if not train.get("airborne_p"):
            train.pop("airborne_p", None)
            train.pop("airborne_cz_min", None)
            train.pop("airborne_cz_max", None)
        if not train.get("airborne_extra"):
            train.pop("airborne_extra", None)
        if not train.get("aux_displaced_m"):
            train.pop("aux_displaced_m", None)
        for key in ("frag_swap_p", "drop_dup_p", "long_gap_p"):
            if not train.get(key):
                train.pop(key, None)
        if not train.get("alive_gap_target"):
            train.pop("alive_gap_target", None)
        # Defaults match a config written before these keys existed.
        if not train.get("extra_matches"):
            train.pop("extra_matches", None)
        elif isinstance(train["extra_matches"], tuple):
            train["extra_matches"] = list(train["extra_matches"])
        if not train.get("extra_degrade"):
            train.pop("extra_degrade", None)
        if not train.get("extra_source_kwargs"):
            train.pop("extra_source_kwargs", None)
        if train.get("extra_weight", 1.0) == 1.0:
            train.pop("extra_weight", None)
        if train.get("extra_labels", "all") == "all":
            train.pop("extra_labels", None)
        if not train.get("target_min_s"):
            train.pop("target_min_s", None)
        if not train.get("target_rule"):
            train.pop("target_rule", None)
        if not train.get("pretrain"):
            train.pop("pretrain", None)
        if not train.get("init_from"):
            train.pop("init_from", None)
        if not train.get("init_from_partial"):
            train.pop("init_from_partial", None)
        if train.get("matches_per_set") is None:
            train.pop("matches_per_set", None)
        if not train.get("scrf_weight"):
            train.pop("scrf_weight", None)
            train.pop("scrf_lam_team", None)
            train.pop("scrf_switch_weight", None)
        if not train.get("tmse_weight"):
            train.pop("tmse_weight", None)
            train.pop("tmse_tau", None)
        return {
            "arch": self.arch,
            "token": self.token.to_dict(),
            "model": self.model.to_dict(),
            "train": train,
        }


# -- data on the device ---------------------------------------------------------------------------


class DeviceSet:
    """One variant set resident on the device, with a batch sampler.

    When a variant carries clean-ball teacher tokens, those arrays sit on the device beside the
    student tokens: one extra token array per variant. A batch windows them with the student
    tokens, and the team swap and the y flip touch both.
    """

    def __init__(
        self,
        entries: list[SetEntry],
        matches: list[TrainMatch],
        device: torch.device,
        *,
        extra_weight: float = 1.0,
    ) -> None:
        self.matches = matches
        self.device = device
        self.tokens = {
            "coupled": [torch.from_numpy(e.coupled).to(device) for e in entries],
            "decoupled": [torch.from_numpy(e.decoupled).to(device) for e in entries],
        }
        self.visible = {
            "coupled": [torch.from_numpy(e.visible_coupled).to(device) for e in entries],
            "decoupled": [torch.from_numpy(e.visible_decoupled).to(device) for e in entries],
        }
        self.labels = [torch.from_numpy(m.labels).to(device) for m in matches]
        self.ball = [torch.from_numpy(m.ball_target).to(device) for m in matches]
        self.ball_ok = [torch.from_numpy(m.target_ok).to(device) for m in matches]
        self.displaced = None
        if entries and entries[0].displaced_coupled is not None:
            # Rebuilt with the variant tokens on every set, so it follows `refresh_steps`.
            self.displaced = {
                "coupled": [torch.from_numpy(e.displaced_coupled).to(device) for e in entries],
                "decoupled": [torch.from_numpy(e.displaced_decoupled).to(device) for e in entries],
            }
        self.possessor = None
        self.touch = None
        self.short = None
        self.spell_s = None
        if matches and getattr(matches[0], "spell_s", None) is not None:
            # Match-level spell lengths (E154 length_view). Windowing slices them; a swap does not.
            self.spell_s = [
                torch.from_numpy(np.ascontiguousarray(m.spell_s)).to(device) for m in matches
            ]
        if matches and getattr(matches[0], "short_spell", None) is not None:
            # Built once on the full-match labels. Windowing slices it; a swap does not.
            self.short = [
                torch.from_numpy(np.ascontiguousarray(m.short_spell)).to(device) for m in matches
            ]
        if matches and matches[0].touch is not None:
            self.touch = [torch.from_numpy(np.asarray(m.touch)).to(device) for m in matches]
        self.event_switch = None
        if matches and getattr(matches[0], "event_switch", None) is not None:
            # Built once per match from the provider events; windows slice it, a swap does not.
            self.event_switch = [
                torch.from_numpy(np.asarray(m.event_switch)).to(device) for m in matches
            ]
        self.boundary = None
        if matches and matches[0].boundary is not None:
            # Built once on the full-match labels. Windowing slices it; a swap does not.
            self.boundary = [
                torch.from_numpy(np.ascontiguousarray(m.boundary)).to(device) for m in matches
            ]
        if entries and entries[0].possessor_coupled is not None:
            self.possessor = {
                "coupled": [torch.from_numpy(e.possessor_coupled).to(device) for e in entries],
                "decoupled": [torch.from_numpy(e.possessor_decoupled).to(device) for e in entries],
            }
        self.teacher = None
        if entries and entries[0].teacher_coupled is not None:
            # One extra token array per variant, same dtype and device as the student tokens.
            self.teacher = {
                "coupled": [torch.from_numpy(e.teacher_coupled).to(device) for e in entries],
                "decoupled": [torch.from_numpy(e.teacher_decoupled).to(device) for e in entries],
            }
        played = np.array([m.played.sum() for m in matches], dtype=np.float64)
        # Played frames, times extra_weight on extra matches. Weight 1 leaves the shares as they
        # were: the division below is the only arithmetic on that path.
        if extra_weight != 1.0 and any(match.extra for match in matches):
            scale = np.ones(len(matches), dtype=np.float64)
            for i, match in enumerate(matches):
                if match.extra:
                    scale[i] = float(extra_weight)
            played *= scale
        # The same division as before when any frame is played. A roster with no played frame
        # would be 0/0; a uniform share keeps the sampler from seeing that.
        if played.sum() == 0:
            n = len(played)
            self.share = played if n == 0 else np.full(n, 1.0 / n, dtype=np.float64)
        else:
            self.share = played / played.sum()
        self.info = [e.info for e in entries]

    def aux_mask(self, kind: str, m: int, start: int, stop: int) -> torch.Tensor:
        """(stop - start,) bool: the frames of match `m`, variant `kind`, the aux ball loss covers.

        The target exists and the ball is alive, and the fed ball is hidden from the input. With
        `aux_displaced_m` above 0 a seen fed ball farther than that from the target counts too.
        """
        ok = self.ball_ok[m][start:stop]
        hidden = ~self.visible[kind][m][start:stop]
        if self.displaced is None:
            return ok & hidden
        return ok & (hidden | self.displaced[kind][m][start:stop])

    def batch(
        self, rng: np.random.Generator, batch: int, window: int, *, decoupled: bool
    ) -> dict[str, torch.Tensor]:
        kind = "decoupled" if decoupled else "coupled"
        which = rng.choice(len(self.matches), size=batch, p=self.share)
        starts = np.zeros(batch, dtype=np.int64)
        for m in np.unique(which):
            rows = np.flatnonzero(which == m)
            starts[rows] = sample_starts(
                self.matches[m].bundle.meta.periods, window, len(rows), rng
            )
        picks = [(int(m), int(s)) for m, s in zip(which, starts, strict=True)]
        tokens = torch.stack([self.tokens[kind][m][s : s + window] for m, s in picks]).float()
        labels = torch.stack([self.labels[m][s : s + window] for m, s in picks]).long()
        ball = torch.stack([self.ball[m][s : s + window] for m, s in picks])
        ok = torch.stack([self.aux_mask(kind, m, s, s + window) for m, s in picks])
        out = {"tokens": tokens, "labels": labels, "ball": ball, "ball_ok": ok}
        if self.teacher is not None:
            out["teacher_tokens"] = torch.stack(
                [self.teacher[kind][m][s : s + window] for m, s in picks]
            ).float()
        if self.possessor is not None:
            out["possessor"] = torch.stack(
                [self.possessor[kind][m][s : s + window] for m, s in picks]
            ).long()
        if self.touch is not None:
            # The clean-ball target, the same window as the labels. A team swap does not move it.
            out["touch"] = torch.stack([self.touch[m][s : s + window] for m, s in picks])
        if self.event_switch is not None:
            out["event_switch"] = torch.stack(
                [self.event_switch[m][s : s + window] for m, s in picks]
            )
        if self.short is not None:
            out["short"] = torch.stack([self.short[m][s : s + window] for m, s in picks])
        if self.spell_s is not None:
            out["spell_s"] = torch.stack([self.spell_s[m][s : s + window] for m, s in picks])
        if self.boundary is not None:
            out["boundary"] = torch.stack([self.boundary[m][s : s + window] for m, s in picks])
        return out


def device_index(indices: np.ndarray, device: torch.device) -> torch.Tensor:
    """Row indices on the device. On CUDA the copy goes from pinned memory and does not block: a
    copy from pageable memory first waits for every kernel queued before it, which stalls the whole
    training step and, on a GPU shared by several runs, costs a time slice each. Same values."""
    tensor = torch.from_numpy(indices)
    if device.type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def _swap_possessor(index: torch.Tensor, tokens_per_team: int) -> torch.Tensor:
    """Home slots and away slots exchange under a team swap. The nobody node and -1 stay."""
    k = tokens_per_team
    home = (index >= 0) & (index < k)
    away = (index >= k) & (index < 2 * k)
    return torch.where(home, index + k, torch.where(away, index - k, index))


def augment(
    batch: dict[str, torch.Tensor], spec: TokenSpec, cfg: TrainConfig, rng: np.random.Generator
) -> dict[str, torch.Tensor]:
    """Per-window y mirror and team swap (with H and A exchanged). Changes `batch` in place.

    A team swap maps a possessor target in [0, K) to t+K and one in [K, 2K) to t-K, so it still
    points at the same player after `symmetric(..., swap=True)`. 2K and -1 stay. A y mirror does
    not change the target. A touch target, when the batch carries one, is left as it is: a touch
    has no team and a y mirror does not move it. Neither branch draws a random number of its own.
    Teacher tokens, when the batch carries them, take the same rows and the same swap and flip.
    """
    n = batch["tokens"].shape[0]
    device = batch["tokens"].device
    if cfg.swap_teams:
        rows = device_index(np.flatnonzero(rng.random(n) < 0.5), device)
        if len(rows):
            batch["tokens"][rows] = symmetric(batch["tokens"][rows], spec, swap=True)
            if "teacher_tokens" in batch:
                batch["teacher_tokens"][rows] = symmetric(
                    batch["teacher_tokens"][rows], spec, swap=True
                )
            labels = batch["labels"][rows]
            batch["labels"][rows] = torch.where(labels == 0, 1, torch.where(labels == 1, 0, labels))
            batch["ball"][rows] = -batch["ball"][rows]
            if "possessor" in batch:
                batch["possessor"][rows] = _swap_possessor(
                    batch["possessor"][rows], spec.tokens_per_team
                )
    if cfg.flip_y:
        rows = device_index(np.flatnonzero(rng.random(n) < 0.5), device)
        if len(rows):
            batch["tokens"][rows] = symmetric(batch["tokens"][rows], spec, flip_y=True)
            if "teacher_tokens" in batch:
                batch["teacher_tokens"][rows] = symmetric(
                    batch["teacher_tokens"][rows], spec, flip_y=True
                )
            batch["ball"][rows, :, 1] = -batch["ball"][rows, :, 1]
    return batch


# -- loss and schedule ----------------------------------------------------------------------------


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to `final_lr_frac` of the peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.final_lr_frac + (1.0 - cfg.final_lr_frac) * cosine)


def mean_teacher_outputs(
    teachers: Sequence[torch.nn.Module], tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`(mean class logits, mean switch logit or None)`, one forward per teacher.

    The class-logit reduction matches what training used before the switch term: a single teacher
    is returned undivided, and several are summed and divided. The switch logit is that same
    reduction when every teacher returns `"switch"`, and None when any teacher has no switch head.
    Both tensors are detached. Inference averages class logits the same way.
    """
    total = None
    switch_total = None
    all_switch = True
    for teacher in teachers:
        out = teacher(tokens)
        logits = out["logits"]
        total = logits if total is None else total + logits
        if all_switch and "switch" in out:
            switch = out["switch"]
            switch_total = switch if switch_total is None else switch_total + switch
        else:
            all_switch = False
            switch_total = None
    if total is None:
        raise ValueError("mean_teacher_logits needs at least one teacher")
    if len(teachers) == 1:
        logits = total.detach()
        switch = None if switch_total is None else switch_total.detach()
    else:
        logits = (total / len(teachers)).detach()
        switch = None if switch_total is None else (switch_total / len(teachers)).detach()
    return logits, switch


def mean_teacher_logits(teachers: Sequence[torch.nn.Module], tokens: torch.Tensor) -> torch.Tensor:
    """Mean of the teachers' class logits on one batch, one forward each.

    Inference averages logits too. The switch logit, when every teacher has a switch head, is
    the second value of `mean_teacher_outputs`; this function returns only the class logits.
    """
    return mean_teacher_outputs(teachers, tokens)[0]


def _teacher_batch_tokens(batch: dict[str, torch.Tensor], distill: DistillConfig) -> torch.Tensor:
    """Tokens the teachers score. `same` is the student batch; `clean_ball` is the other array."""
    if distill.teacher_input == "same":
        return batch["tokens"]
    tokens = batch.get("teacher_tokens")
    if tokens is None:
        raise ValueError("distill.teacher_input 'clean_ball' requires teacher tokens on the batch")
    return tokens


def _distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Class-weighted mean KL(softmax(teacher/T) || softmax(student/T)).

    Same frames as the cross entropy: `labels` already drops the window edges, and the X label
    (ignore index 3) is dropped here. Each kept frame is weighted by its class weight, then the
    weights are normalised, matching ``cross_entropy(..., weight=, ignore_index=)``.
    """
    log_p = F.log_softmax(teacher_logits / temperature, dim=-1)
    log_q = F.log_softmax(student_logits / temperature, dim=-1)
    per_frame = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)
    valid = labels != IGNORE
    index = labels.clamp(min=0, max=weights.shape[0] - 1)
    class_w = weights.to(dtype=per_frame.dtype)
    frame_w = torch.where(
        valid, class_w[index], torch.zeros((), dtype=per_frame.dtype, device=per_frame.device)
    )
    return (per_frame * frame_w).sum() / frame_w.sum().clamp_min(torch.finfo(per_frame.dtype).tiny)


def switch_targets(labels: torch.Tensor, tol: int, margin: int = 0) -> torch.Tensor:
    """(B, W) float32 target for a direct home/away switch. 1 positive, 0 negative, NaN ignored.

    A change is two consecutive frames labelled H then A, or A then H. Frame i is positive when it
    is H or A and its distance to either frame of such a pair is at most `tol`. D and X are
    ignored, and so is a change that only exists because D or X sits between two team labels.
    `margin` frames at each end of the window are ignored too. Pairs are read from the whole
    window first, so a change inside the margin can still mark an interior frame. A team swap
    exchanges H and A and leaves this target unchanged; a y flip does not touch the labels.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be (B, W), got {tuple(labels.shape)}")
    if tol < 0:
        raise ValueError(f"switch_tol_frames must be >= 0, got {tol}")
    # 0 and 1 are H and A, the same codes augment() exchanges on a team swap.
    home, away = labels == 0, labels == 1
    pair = (home[:, :-1] & away[:, 1:]) | (away[:, :-1] & home[:, 1:])
    marked = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
    if labels.shape[1] >= 2:
        marked[:, :-1] |= pair
        marked[:, 1:] |= pair
    if tol > 0 and labels.shape[1] > 0:
        # max_pool pads with -inf, so the dilation does not invent a change past the window.
        pooled = F.max_pool1d(
            marked.float().unsqueeze(1), kernel_size=2 * tol + 1, stride=1, padding=tol
        )
        near = pooled.squeeze(1) > 0.5
    else:
        near = marked
    team = home | away
    target = torch.full(labels.shape, torch.nan, dtype=torch.float32, device=labels.device)
    target = torch.where(team & near, torch.ones_like(target), target)
    target = torch.where(team & ~near, torch.zeros_like(target), target)
    if margin > 0:
        target[:, :margin] = torch.nan
        target[:, -margin:] = torch.nan
    return target


def _switch_distill_bce(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mean BCE-with-logits of `student` toward sigmoid(`teacher`), over every frame of both."""
    student = student.float().reshape(-1)
    target = torch.sigmoid(teacher.float().reshape(-1))
    if student.numel() == 0:
        return student.new_zeros(())
    return F.binary_cross_entropy_with_logits(student, target)


def _with_switch_distill(
    result: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    teacher_switch: torch.Tensor | None,
    distill: DistillConfig | None,
    inner: slice,
) -> dict[str, torch.Tensor]:
    """Add `switch_alpha` times the switch BCE when both sides have a switch head.

    Weight 0, a missing teacher switch logit, or a student without the head returns `result`
    with its loss tensor unchanged. The target is the sigmoid of the teachers' mean switch logit
    on the same interior frames as the other terms, including D and X: it is a dense soft target,
    not the hard switch label.
    """
    if distill is None or not float(distill.switch_alpha) or teacher_switch is None:
        return result
    if "switch" not in out:
        return result
    if teacher_switch.shape != out["switch"].shape:
        raise ValueError(
            f"teacher switch {tuple(teacher_switch.shape)} != student {tuple(out['switch'].shape)}"
        )
    swd = _switch_distill_bce(out["switch"][:, inner], teacher_switch[:, inner])
    result["loss"] = result["loss"] + float(distill.switch_alpha) * swd
    result["swd"] = swd.detach()
    return result


def loss_terms(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: torch.Tensor,
    margin: int,
    aux_weight: float,
    *,
    teacher_logits: torch.Tensor | None = None,
    teacher_switch: torch.Tensor | None = None,
    distill: DistillConfig | None = None,
    spell: SpellWeight | None = None,
    possessor_weight: float = 0.0,
    switch_weight: float = 0.0,
    switch_tol_frames: int = 3,
    switch_pos_weight: float = 20.0,
    touch_weight: float = 0.0,
    touch_pos_weight: float = 10.0,
    class_loss: bool = True,
    scrf_weight: float = 0.0,
    scrf_lam_team: float = 0.6,
    scrf_switch_weight: float = 2.0,
    tmse_weight: float = 0.0,
    tmse_tau: float = 4.0,
    event_switch_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Total loss, its parts and the team accuracy on alive interior frames.

    Without a teacher (or with `alpha` 0) the loss is the class-weighted cross entropy plus the
    auxiliary ball term, unchanged. Otherwise the cross entropy is replaced by
    `(1 - alpha) * CE + alpha * T^2 * KL(teacher || student)` on the same interior frames; the
    ball term is still added as it is. With `spell` (gamma > 0) each alive frame also carries its
    spell weight (spellweight.py) and the mean is taken over the product of the two weights. A
    `length_view` other than provider reads the match-level lengths the batch carries as "spell_s".
    When the batch carries "boundary" (train.boundary_weight, boundaryweight.py), each frame also
    carries that weight, multiplied into the same product; without it the loss is unchanged.
    `possessor_weight` above 0, when both sides carry "possessor", adds that weight times a
    cross entropy over the 2K+1 logits (ignore index -1) and reports it as "pos". Weight 0 leaves
    the loss tensor and the dict as they are without the term. An interior with no target is 0.
    `switch_weight` above 0, when "switch" is in the forward, adds that weight times a
    BCE-with-logits on the interior frames (D, X and the window margin ignored) and reports it
    as "sw". Weight 0 leaves the loss tensor and the dict as they are without the term.
    `touch_weight` above 0, when "touch" is in the forward and the batch carries the clean-ball
    target, adds that weight times a BCE-with-logits on the interior frames (NaN targets ignored)
    and reports it as "touch". Weight 0 does not read the head or the target.
    `switch_alpha` above 0, when `teacher_switch` is set and the student has a switch head, adds
    that weight times a BCE of the student's switch logit toward the sigmoid of the teachers'
    mean switch logit, on the same interior frames, and reports it as "swd". Weight 0 leaves the
    loss tensor unchanged.
    `class_loss` False (a label-free pretraining run) leaves the cross entropy and the teacher KL
    out of the loss and out of the dict, so a window of X labels never divides by a zero count of
    kept frames. The auxiliary term, and the possessor, switch, touch and semi-Markov terms, still
    follow their weights.
    `scrf_weight` above 0, when "switch" is in the forward, adds that weight times the semi-Markov
    NLL of the interior labels (scrf.py) and reports it as "scrf". The short-spell mask, when the
    batch carries one, is sliced with the interior. Weight 0 does not read the head, the labels or
    the mask, and leaves the loss tensor unchanged.
    `tmse_weight` above 0, with the class loss on, adds that weight times the truncated temporal
    MSE of the class log-probabilities on the interior (`_tmse_loss`) and reports it as "tmse".
    Weight 0, or `class_loss` False, leaves the loss tensor and the dict as they are.
    `event_switch_weight` above 0, when "switch" is in the forward and the batch carries the
    event target (`batch["event_switch"]`, `data.event_switch_target`), adds that weight times a
    BCE-with-logits of the switch head against it on the interior (NaN ignored, `switch_pos_weight`)
    and reports it as "esw". It is added last. Weight 0 does not read the head or the target.
    """
    inner = slice(margin, batch["tokens"].shape[1] - margin)
    logits = out["logits"][:, inner].float()
    labels = batch["labels"][:, inner]
    near = batch.get("boundary")
    spell_on = spell is not None and float(spell.gamma) != 0.0
    if not class_loss:
        ce = None
    elif not spell_on and near is None:
        ce = F.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1), weight=weights, ignore_index=IGNORE
        )
    else:
        frame_w = None
        if spell_on:
            frame_w = spell_weights(batch["labels"], spell, batch.get("spell_s"))
            frame_w = frame_w[:, inner].reshape(-1)
        if near is not None:
            near_w = near[:, inner].reshape(-1).float()
            frame_w = near_w if frame_w is None else frame_w * near_w
        flat = labels.reshape(-1)
        per_frame = F.cross_entropy(
            logits.reshape(-1, 3), flat, weight=weights, ignore_index=IGNORE, reduction="none"
        )
        total_w = (weights[flat.clamp(max=weights.shape[0] - 1)] * frame_w * (flat != IGNORE)).sum()
        ce = (per_frame * frame_w).sum() / total_w.clamp_min(torch.finfo(per_frame.dtype).tiny)
    ok = batch["ball_ok"][:, inner]
    if aux_weight > 0 and "ball" in out and bool(ok.any()):
        aux = F.huber_loss(
            out["ball"][:, inner].float()[ok], batch["ball"][:, inner][ok], delta=HUBER_DELTA
        )
    else:
        aux = logits.new_zeros(())
    alive = labels < 2
    hit = (logits[..., :2].argmax(-1) == labels) & alive
    team_acc = hit.sum() / alive.sum().clamp(min=1)
    pos = _possessor_loss(out, batch, inner, possessor_weight)
    sw = _switch_loss(
        out, batch, inner, switch_weight, switch_tol_frames, switch_pos_weight, margin
    )
    touch = _touch_loss(out, batch, inner, touch_weight, touch_pos_weight)
    scrf = _scrf_loss(out, batch, inner, scrf_weight, scrf_lam_team, scrf_switch_weight)
    # A smoothing term on the class head, so it follows the class loss: off without it.
    tmse = _tmse_loss(logits, labels, tmse_weight, tmse_tau) if class_loss else None
    esw = _event_switch_loss(out, batch, inner, event_switch_weight, switch_pos_weight)

    def finish(result: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return _with_event_switch(
            _with_tmse(
                _with_scrf(
                    _with_touch(
                        _with_switch(
                            _with_possessor(result, pos, possessor_weight), sw, switch_weight
                        ),
                        touch,
                        touch_weight,
                    ),
                    scrf,
                    scrf_weight,
                ),
                tmse,
                tmse_weight,
            ),
            esw,
            event_switch_weight,
        )

    if not class_loss:
        # The cross entropy mean divides by the number of labelled frames. A label-free window
        # has none, so the term is omitted instead of evaluated.
        return finish({"loss": aux_weight * aux, "aux": aux.detach(), "acc": team_acc})

    # alpha 0 takes this return, so the loss tensor is the one a run without distillation builds.
    # switch_alpha 0 does not touch that tensor either.
    if teacher_logits is None or distill is None or float(distill.alpha) == 0.0:
        return _with_switch_distill(
            finish(
                {
                    "loss": ce + aux_weight * aux,
                    "ce": ce.detach(),
                    "aux": aux.detach(),
                    "acc": team_acc,
                }
            ),
            out,
            teacher_switch,
            distill,
            inner,
        )
    if teacher_logits.shape != out["logits"].shape:
        raise ValueError(
            f"teacher logits {tuple(teacher_logits.shape)} != student {tuple(out['logits'].shape)}"
        )
    temperature = float(distill.temperature)
    kl = _distill_kl(logits, teacher_logits[:, inner].float(), labels, weights, temperature)
    mixed = (1.0 - float(distill.alpha)) * ce + (float(distill.alpha) * temperature**2) * kl
    return _with_switch_distill(
        finish(
            {
                "loss": mixed + aux_weight * aux,
                "ce": ce.detach(),
                "aux": aux.detach(),
                "acc": team_acc,
                "kl": kl.detach(),
            }
        ),
        out,
        teacher_switch,
        distill,
        inner,
    )


def _possessor_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    inner: slice,
    weight: float,
) -> torch.Tensor | None:
    """Cross entropy of the possessor logits on `inner`, or None when the term is off.

    The mean over the kept frames (target != -1), written as a masked sum so that it needs no host
    sync (a sync costs a time slice per step on a shared GPU). An interior whose every target is -1
    returns 0, where `F.cross_entropy`'s own mean would be NaN and wipe the run.
    """
    if not weight or weight <= 0.0 or "possessor" not in out or "possessor" not in batch:
        return None
    logits = out["possessor"][:, inner].float()
    flat = batch["possessor"][:, inner].long().reshape(-1)
    keep = flat != -1
    per_frame = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), flat, ignore_index=-1, reduction="none"
    )
    return (per_frame * keep).sum() / keep.sum().clamp_min(1)


def _with_possessor(
    result: dict[str, torch.Tensor], pos: torch.Tensor | None, weight: float
) -> dict[str, torch.Tensor]:
    """Add `weight * pos` when the term was computed. Weight 0 returns `result` unchanged."""
    if pos is None:
        return result
    result["loss"] = result["loss"] + weight * pos
    result["pos"] = pos.detach()
    return result


def _switch_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    inner: slice,
    weight: float,
    tol: int,
    pos_weight: float,
    margin: int,
) -> torch.Tensor | None:
    """BCE-with-logits of the switch head on the interior, or None when the term is off.

    Ignored frames (D, X, the window margin) drop out of the mean. An interior with no H/A frame
    returns 0, so an empty target cannot wipe the run with a NaN. Weight 0 does not read the head.
    """
    if not weight or weight <= 0.0 or "switch" not in out:
        return None
    logits = out["switch"][:, inner].float().reshape(-1)
    target = switch_targets(batch["labels"], tol, margin)[:, inner].reshape(-1)
    if logits.numel() == 0:
        return logits.new_zeros(())
    valid = torch.isfinite(target)
    filled = torch.where(valid, target, torch.zeros_like(target))
    per = F.binary_cross_entropy_with_logits(
        logits, filled, pos_weight=logits.new_full((1,), float(pos_weight)), reduction="none"
    )
    return (per * valid).sum() / valid.sum().clamp_min(1)


def _with_switch(
    result: dict[str, torch.Tensor], sw: torch.Tensor | None, weight: float
) -> dict[str, torch.Tensor]:
    """Add `weight * sw` when the term was computed. Weight 0 returns `result` unchanged."""
    if sw is None:
        return result
    result["loss"] = result["loss"] + weight * sw
    result["sw"] = sw.detach()
    return result


def _touch_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    inner: slice,
    weight: float,
    pos_weight: float,
) -> torch.Tensor | None:
    """BCE-with-logits of the touch head on the interior, or None when the term is off.

    The target is the clean-ball array sliced with the window (`batch["touch"]`): 1, 0 or NaN.
    NaN frames (unseen ball, dead, X, the part of the window the slice already dropped) leave the
    mean. An interior with no supervised frame returns 0, so an empty target cannot wipe the run.
    Weight 0 does not read the head or the target.
    """
    if not weight or weight <= 0.0 or "touch" not in out:
        return None
    if "touch" not in batch:
        raise ValueError("touch_weight > 0 requires the clean-ball touch target on the batch")
    logits = out["touch"][:, inner].float().reshape(-1)
    target = batch["touch"][:, inner].reshape(-1).float()
    if logits.numel() == 0:
        return logits.new_zeros(())
    valid = torch.isfinite(target)
    filled = torch.where(valid, target, torch.zeros_like(target))
    per = F.binary_cross_entropy_with_logits(
        logits, filled, pos_weight=logits.new_full((1,), float(pos_weight)), reduction="none"
    )
    return (per * valid).sum() / valid.sum().clamp_min(1)


def _event_switch_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    inner: slice,
    weight: float,
    pos_weight: float,
) -> torch.Tensor | None:
    """BCE-with-logits of the switch head against the event target, or None when the term is off.

    The target is the match's event-switch array sliced with the window (1, 0 or NaN); NaN frames
    leave the mean, and an interior with none supervised returns 0. Weight 0 reads nothing.
    """
    if not weight or weight <= 0.0 or "switch" not in out:
        return None
    if "event_switch" not in batch:
        raise ValueError("event_switch_weight > 0 requires the event-switch target on the batch")
    logits = out["switch"][:, inner].float().reshape(-1)
    target = batch["event_switch"][:, inner].reshape(-1).float()
    if logits.numel() == 0:
        return logits.new_zeros(())
    valid = torch.isfinite(target)
    filled = torch.where(valid, target, torch.zeros_like(target))
    per = F.binary_cross_entropy_with_logits(
        logits, filled, pos_weight=logits.new_full((1,), float(pos_weight)), reduction="none"
    )
    return (per * valid).sum() / valid.sum().clamp_min(1)


def _with_event_switch(
    result: dict[str, torch.Tensor], term: torch.Tensor | None, weight: float
) -> dict[str, torch.Tensor]:
    """Add `weight * term` when the event-switch loss was computed. Weight 0 returns `result`."""
    if term is None:
        return result
    result["loss"] = result["loss"] + weight * term
    result["esw"] = term.detach()
    return result


def _with_touch(
    result: dict[str, torch.Tensor], term: torch.Tensor | None, weight: float
) -> dict[str, torch.Tensor]:
    """Add `weight * term` when the touch loss was computed. Weight 0 returns `result` unchanged."""
    if term is None:
        return result
    result["loss"] = result["loss"] + weight * term
    result["touch"] = term.detach()
    return result


def _scrf_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    inner: slice,
    weight: float,
    lam_team: float,
    switch_weight: float,
) -> torch.Tensor | None:
    """Semi-Markov NLL on the interior, or None when the term is off.

    Weight 0 does not read the switch head or the labels. Above 0 the head is required: the bonus
    is part of the energy. The value is the mean over scored interior frames (scrf.py).
    """
    if not weight or weight <= 0.0:
        return None
    if "switch" not in out:
        raise ValueError("scrf_weight > 0 requires the switch head")
    short = batch.get("short")
    return scrf_nll(
        out["logits"][:, inner],
        out["switch"][:, inner],
        batch["labels"][:, inner],
        lam_team=lam_team,
        switch_weight=switch_weight,
        short=None if short is None else short[:, inner],
    )


def _with_scrf(
    result: dict[str, torch.Tensor], term: torch.Tensor | None, weight: float
) -> dict[str, torch.Tensor]:
    """Add `weight * term` when the semi-Markov loss was computed. Weight 0 returns `result`."""
    if term is None:
        return result
    result["loss"] = result["loss"] + weight * term
    result["scrf"] = term.detach()
    return result


def _tmse_loss(
    logits: torch.Tensor, labels: torch.Tensor, weight: float, tau: float
) -> torch.Tensor | None:
    """Truncated temporal MSE of the class log-probabilities, or None when the term is off.

    The MS-TCN smoothing loss (Farha and Gall, CVPR 2019). `logits` (B, W, 3) and `labels` (B, W)
    are the interior frames. Between consecutive frames, each class's log-probability step is
    squared and capped at `tau**2`, then averaged over the classes. The earlier frame is detached,
    so the gradient pulls each frame toward the one before it. The mean is over the frame pairs
    whose two labels are both kept (not X), written as a masked sum so it needs no host sync; an
    interior with no such pair returns 0. A capped step passes no gradient, so a sharp change
    costs at most `tau**2`. Where it fails: a step under the cap is penalised, so with tau 4 a
    confident H to A change (a log-probability step near 3) is still pulled toward the earlier
    frame and can blur a boundary by a few frames. Weight 0 does not read the logits.
    """
    if not weight or weight <= 0.0:
        return None
    logp = F.log_softmax(logits.float(), dim=-1)
    step = logp[:, 1:] - logp[:, :-1].detach()
    per_pair = torch.clamp(step**2, max=float(tau) ** 2).mean(dim=-1)
    keep = (labels[:, 1:] != IGNORE) & (labels[:, :-1] != IGNORE)
    return (per_pair * keep).sum() / keep.sum().clamp_min(1)


def _with_tmse(
    result: dict[str, torch.Tensor], term: torch.Tensor | None, weight: float
) -> dict[str, torch.Tensor]:
    """Add `weight * term` when the T-MSE was computed. Weight 0 returns `result` unchanged."""
    if term is None:
        return result
    result["loss"] = result["loss"] + weight * term
    result["tmse"] = term.detach()
    return result


def _frozen_teachers(
    paths: Sequence[str | Path],
    device: torch.device,
    spec: TokenSpec,
    arch: str,
    receptive_half: int,
) -> list[torch.nn.Module]:
    """Load each teacher once, on `device`, eval mode, parameters frozen."""
    models: list[torch.nn.Module] = []
    for path in paths:
        loaded = load_checkpoint(path)
        if loaded.spec != spec:
            raise ValueError(
                f"teacher {path} token spec does not match the student: "
                f"{loaded.spec.to_dict()} vs {spec.to_dict()}"
            )
        teacher_arch = getattr(loaded.model, "arch", arch)
        if teacher_arch != arch:
            raise ValueError(
                f"teacher {path} is architecture {teacher_arch!r}, the student is {arch!r}"
            )
        if loaded.model.receptive_half != receptive_half:
            raise ValueError(
                f"teacher {path} receptive field +-{loaded.model.receptive_half} "
                f"does not match the student +-{receptive_half}"
            )
        model = loaded.model.to(device)
        model.eval()
        model.requires_grad_(False)
        models.append(model)
    return models


# -- the loop -------------------------------------------------------------------------------------


def _ema_members(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Floating parameters, then floating buffers, in module order.

    Integer buffers are left out. TeamNet's only buffer is the integer relation matrix, a constant
    mask that is not in the checkpoint (`persistent=False`).
    """
    members = [
        (name, value) for name, value in model.named_parameters() if value.is_floating_point()
    ]
    members.extend(
        (name, value) for name, value in model.named_buffers() if value.is_floating_point()
    )
    return members


class _WeightEma:
    """Float32 exponential moving average of the floating parameters and buffers.

    The shadow is a clone of the initial weights and stays on their device. `update` runs under
    `no_grad` with autocast off, in place on that storage:
    ``shadow = d * shadow + (1 - d) * value`` with the warm-up ``d = min(decay, (1 + n) / (10 + n))``
    after n earlier updates. Without the warm-up, decay 0.998 over 2000 steps would leave
    0.998**2000 = 1.8% of the random initial weights in the average; with it the initial weights
    are gone after the first few steps. `updates` is saved in `last.pt` so a resume continues the
    same schedule. Decay 0 never builds this object.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, value in _ema_members(model):
                self.shadow[name] = value.detach().to(dtype=torch.float32).clone()

    def current_decay(self) -> float:
        """The decay the next update uses (the warm-up, capped at `decay`)."""
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """One step on device. No host sync, and not inside autocast."""
        if not self.shadow:
            return
        live = dict(_ema_members(model))
        decay = self.current_decay()
        self.updates += 1
        complement = 1.0 - decay
        device_type = next(iter(self.shadow.values())).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            for name, shadow in self.shadow.items():
                value = live[name].detach()
                if value.dtype != torch.float32:
                    value = value.to(dtype=torch.float32)
                # Same storage as the shadow allocated at init. No .item(), no host copy.
                shadow.copy_(decay * shadow + complement * value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """CPU float32 clones for `last.pt`. Later updates do not change the snapshot."""
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(state))
            extra = sorted(set(state) - set(self.shadow))
            raise ValueError(
                f"EMA state does not match the model (missing {missing}, extra {extra})"
            )
        with torch.no_grad():
            for name, value in state.items():
                shadow = self.shadow[name]
                shadow.copy_(value.detach().to(device=shadow.device, dtype=torch.float32))

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Write the average into `model`. Integer buffers are left as they are."""
        live = dict(model.named_parameters())
        live.update(model.named_buffers())
        for name, shadow in self.shadow.items():
            target = live[name]
            target.copy_(shadow.to(dtype=target.dtype))


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_init_from(path: str | None, fold: str | None) -> str | None:
    """Checkpoint path to load. ``{fold}`` becomes the held-out match id.

    A path with no placeholder is returned unchanged, whether or not `fold` is set.
    Leave-one-match-out passes the held-out match id as `fold`, so
    ``runs/earlier/fold-{fold}/final.pt`` loads that fold's own `final.pt`. A single
    training run passes `fold` None (or ``"none"``); a placeholder there raises
    ValueError, because one path would have to name every held-out match. The
    configured string is not rewritten: callers keep it in the saved run config and
    load the path this returns.
    """
    if path is None or "{fold}" not in path:
        return path
    if not isinstance(path, str):
        raise ValueError(f"train.init_from must be a checkpoint path or null, got {path!r}")
    if fold is None or fold == "" or fold == "none":
        raise ValueError(
            "train.init_from contains '{fold}', which leave-one-match-out replaces "
            "with the held-out match id; a single training run must name one checkpoint, "
            f"got {path!r}"
        )
    bad_fold = (
        not isinstance(fold, str)
        or Path(fold).name != fold
        or fold in {".", ".."}
        or "/" in fold
        or "\\" in fold
        or "{" in fold
    )
    if bad_fold:
        raise ValueError(f"fold must be a match id, got {fold!r}")
    return path.replace("{fold}", fold)


def _roster_for_set(
    matches: list[TrainMatch], per_set: int | None, seed: int, set_index: int
) -> tuple[list[TrainMatch], list[int] | None]:
    """Matches tokenised for one variant set, and their indexes in `matches`.

    None, or a cap at least as large as the roster, returns `(matches, None)`. The caller
    then seeds each match by its position, which is that index, so the draws stay as they
    are. A smaller cap draws `per_set` indexes without replacement from a generator seeded
    by `(seed, set_index)`, separate from the batch sampler and the per-match degradation
    generator, sorts them so roster order is kept, and returns those matches with the
    indexes. A 146k-frame match is about 94 MB of float16 tokens per variant, so a 48-match
    roster with both variants resident is about 9 GB before the next set is built.
    """
    if per_set is None or per_set >= len(matches):
        return matches, None
    # 901 is this draw's stream. The batch sampler uses 17; build_set uses the roster index.
    rng = np.random.default_rng([seed, set_index, 901])
    picked = [int(i) for i in np.sort(rng.choice(len(matches), size=per_set, replace=False))]
    return [matches[i] for i in picked], picked


def train(
    run: RunConfig,
    matches: list[TrainMatch],
    out_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    log: Callable[[str], None] = print,
    resume: bool = True,
    meta: dict[str, Any] | None = None,
    threads: int | None = None,
    monitor: Callable[[torch.nn.Module], dict[str, float]] | None = None,
    teachers: Sequence[str | Path] | None = None,
    fold: str | None = None,
) -> Path:
    """Train on `matches` and write `final.pt` (and `last.pt` after every variant set) to
    `out_dir`. `meta` (fold, match ids, ...) is stored in the checkpoint. `threads` caps the CPU
    threads torch uses (the team shares the CPU; numpy follows OMP_NUM_THREADS). `monitor`, if
    given, is called with the model after every set and its numbers are logged and kept in the
    history: a learning curve on data the run is not allowed to select on (it never changes the
    schedule or picks a checkpoint). Returns `final.pt`.

    `teachers` are the checkpoints from `teacher_paths` for this fold. They are required when
    `run.train.distill` is set, loaded once, and run under ``no_grad`` in the student's autocast.
    With `teacher_input` `clean_ball` they score the batch's teacher tokens (the same windows,
    swap and flip, ball smoothing skipped); `same` scores the student tokens. With distillation
    off, or with `alpha` and `switch_alpha` both 0, `teachers` is ignored and the step is the
    undistilled step.

    With `train.ema_decay` above 0 the returned `final.pt` carries the exponential moving average
    and `final_raw.pt` the last iterate, both in the inference checkpoint format. `last.pt` keeps
    the raw weights plus the average, so a resume matches an uninterrupted run. The monitor sees
    the raw weights. Decay 0 writes neither the average nor `final_raw.pt`.

    `train.pretrain` keeps only the label-free extras and skips the class, switch, touch and
    distillation terms. `--extra-only` does not set that flag: the command trains on the extras
    with the normal loss and loads no IDSSE match. `matches_per_set`, when smaller than the
    roster, tokenises only that many matches in each variant set; null tokenises every match
    and does not change the draws. `train.init_from` loads that checkpoint into the new model
    before the optimizer exists; a resume from `last.pt` does not load it again.
    `fold` replaces every `{fold}` in that path (leave-one-match-out passes the held-out
    match id). A single run leaves `fold` unset, and a placeholder then raises ValueError.
    The saved config keeps the path as configured.
    """
    cfg = run.train
    # Before any match is read. A path with no `{fold}` is unchanged.
    init_path = resolve_init_from(cfg.init_from, fold)
    pretrain = cfg.pretrain
    # These equal the configured weights when pretrain is false, so that run's loss is unchanged.
    switch_weight = 0.0 if pretrain else cfg.switch_weight
    touch_weight = 0.0 if pretrain else cfg.touch_weight
    tmse_weight = 0.0 if pretrain else cfg.tmse_weight  # it smooths the class head
    event_switch_weight = 0.0 if pretrain else cfg.event_switch_weight
    teacher_list: list[Path] = []
    if cfg.distill is not None and not pretrain:
        teacher_list = [Path(path) for path in (teachers or ())]
        if not teacher_list:
            raise ValueError(
                "distill is set but no teacher checkpoints were passed; "
                "resolve run directories with teacher_paths(runs, holdout) and pass them to train"
            )
        missing = [str(path) for path in teacher_list if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing teacher checkpoint(s): " + ", ".join(missing))
    matches, meta = _with_extra_matches(matches, cfg, run.token, meta)
    # Off: nothing is read. On: every extra match's height is read here, once and in order, so a
    # missing or misaligned one fails before training. It draws no random number.
    if cfg.airborne_extra:
        _check_extra_heights(matches, log)
    # Before the seed. With target_rule "" and target_min_s 0 this does not touch `matches`,
    # so the tensors, the random stream and the checkpoint stay bit-identical.
    if cfg.target_rule == "ctrl":
        matches = [control_targets(match, _ctrl_hold_s(cfg)) for match in matches]
    elif cfg.target_min_s > 0:
        matches = [coarsen_targets(match, float(cfg.target_min_s)) for match in matches]
    # Weight 0 does not build the target, so the loss, the stream and the checkpoint stay as they
    # are. Above 0 the target is the clean ball, after the label rewrite and before degradation.
    # Pretrain treats the touch weight as 0, so the target is not built.
    # `touch_source` "events" takes the frames from the provider events instead; a source without
    # events raises here, before any training.
    if touch_weight > 0:
        matches = [
            with_touch_target(match, int(cfg.touch_tol_frames), cfg.touch_source)
            for match in matches
        ]
    # Weight 0 does not build it. Above 0 it is the provider events' possession changes.
    if event_switch_weight > 0:
        matches = [with_event_switch_target(match, int(cfg.switch_tol_frames)) for match in matches]
    # Null does not build the weights, so the loss, the stream and the checkpoint stay as they are.
    # Built on the full-match target labels, after the label rewrite, per period. New TrainMatch
    # objects (before the short-spell mask, which `replace` would drop): the caller's
    # list is not touched.
    if cfg.boundary_weight is not None and not pretrain:
        matches = [with_boundary_weight(match, cfg.boundary_weight) for match in matches]
    # Weight 0 does not build the mask, so this stays off the random stream. The length is the
    # full-match spell, after the label rewrite and before any window is cut.
    if cfg.scrf_weight > 0 and not pretrain:
        for match in matches:
            match.short_spell = short_spell_mask(match.labels)
    # E154: the provider view (the default) reads the window and builds nothing here, so the
    # tensors, the random stream and the checkpoint stay as they are. Another view measures each
    # spell once on the full-match labels, after the label rewrite, for the weights only.
    spell = cfg.spell_weight
    if spell is not None and spell.match_level and not pretrain:
        for match in matches:
            match.spell_s = match_spell_seconds(
                match.labels, float(match.bundle.meta.fps), spell.length_view
            )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    seed_everything(cfg.seed)
    model = build_model(run.arch, run.model).to(device)
    last = out_dir / "last.pt"
    # After the seed and the build, before the optimizer. None does not touch the model.
    # A resume overwrites both the weights and the optimizer from last.pt, so it does not reload.
    # `init_path` is the configured path with `{fold}` already replaced; the saved config is not.
    if init_path and not (resume and last.is_file()):
        if cfg.init_from_partial:
            kept = _load_init_weights(model, init_path, run, partial=True)
            log(f"init_from {init_path} (partial: {len(kept)} new tensors keep their init)")
        else:
            _load_init_weights(model, init_path, run)
            log(f"init_from {init_path}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if pretrain:
        # No labelled frame: inverse-frequency weights would divide by zero, and nothing reads them.
        weights = torch.ones(3, dtype=torch.float32, device=device)
    else:
        weights = torch.tensor(class_weights(matches), dtype=torch.float32, device=device)
    n_sets = math.ceil(cfg.steps / cfg.refresh_steps)
    # From the initial weights. A resume then replaces both the raw weights and this shadow.
    ema = _WeightEma(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    first_set, history = 0, []
    if resume and last.is_file():
        state = load_payload(last)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        first_set, history = int(state["set_index"]) + 1, list(state["history"])
        if ema is not None:
            saved_ema = state.get("ema")
            if saved_ema is None:
                raise ValueError(
                    f"{last} has no EMA state to resume a run with ema_decay {cfg.ema_decay}"
                )
            ema.load_state_dict(saved_ema)
            ema.updates = int(state.get("ema_updates", 0))
        log(f"resumed from {last} at set {first_set}/{n_sets}")
    n_extra = sum(match.extra for match in matches)
    # No suffix when there are none, so this line stays the one a run without extras prints.
    extra_bit = f" ({n_extra} extra)" if n_extra else ""
    mode_bit = "" if not pretrain else " pretrain"
    log(
        f"train {len(matches)} matches{extra_bit}{mode_bit}, {count_parameters(model)} parameters, "
        f"device {device}, {cfg.steps} steps in {n_sets} sets, "
        f"receptive field +-{model.receptive_half} frames"
    )
    if cfg.matches_per_set is not None and cfg.matches_per_set < len(matches):
        log(
            f"each set builds {cfg.matches_per_set} of {len(matches)} matches "
            "so the token tensors stay bounded"
        )
    if ema is not None:
        log(f"ema decay {cfg.ema_decay}: final.pt is the average, final_raw.pt the last iterate")
    # After the student is built: constructing a teacher draws torch random numbers, and the
    # student's initial weights must not depend on that. Each set reseeds torch before its steps.
    teacher_models: list[torch.nn.Module] = []
    if teacher_list:
        teacher_models = _frozen_teachers(
            teacher_list, device, run.token, run.arch, model.receptive_half
        )
    # alpha 0 and switch_alpha 0 skip the teacher forward, so that step matches a run with no
    # distill term. The class KL is logged only while alpha is non-zero.
    kl_on = bool(teacher_models) and cfg.distill is not None and float(cfg.distill.alpha) != 0.0
    wants_switch_distill = (
        bool(teacher_models)
        and cfg.distill is not None
        and float(cfg.distill.switch_alpha) > 0.0
        and bool(getattr(run.model, "switch_head", False))
        and all(getattr(item.cfg, "switch_head", False) for item in teacher_models)
    )
    distill_on = kl_on or wants_switch_distill
    if teacher_models:
        assert cfg.distill is not None
        detail = (
            f"distill: {len(teacher_models)} teachers, alpha {cfg.distill.alpha}, "
            f"temperature {cfg.distill.temperature}"
        )
        if cfg.distill.teacher_input != "same":
            detail += f", teacher_input {cfg.distill.teacher_input}"
        if cfg.distill.switch_alpha:
            detail += f", switch_alpha {cfg.distill.switch_alpha}"
        log(detail)
        if cfg.distill.switch_alpha and not wants_switch_distill:
            log(
                "distill: switch_alpha is set but the switch term is off "
                "(the student or a teacher has no switch head)"
            )
    # Pretrain leaves distillation off, including the extra token array clean_ball would store.
    teacher_input = "same" if pretrain or cfg.distill is None else cfg.distill.teacher_input

    def make(index: int) -> tuple[list[SetEntry], list[TrainMatch]]:
        roster, seed_index = _roster_for_set(matches, cfg.matches_per_set, cfg.seed, index)
        entries = build_set(
            roster,
            index,
            cfg.seed,
            run.token,
            cfg.degrade,
            augment=cfg.augment,
            possessor_weight=cfg.possessor_weight,
            possessor_reach_m=cfg.possessor_reach_m,
            extra_degrade=cfg.extra_degrade,
            teacher_input=teacher_input,
            restart_blank=(
                (cfg.restart_blank_p, cfg.restart_blank_before_s, cfg.restart_blank_after_s)
                if cfg.restart_blank_p > 0
                else None
            ),
            stoppage_carry=(
                (cfg.stoppage_carry_p, cfg.stoppage_carry_max_s)
                if cfg.stoppage_carry_p > 0
                else None
            ),
            dead_visible=cfg.dead_visible_p if cfg.dead_visible_p > 0 else None,
            track_stress=(
                (cfg.frag_swap_p, cfg.drop_dup_p, cfg.long_gap_p)
                if max(cfg.frag_swap_p, cfg.drop_dup_p, cfg.long_gap_p) > 0
                else None
            ),
            seed_index=seed_index,
            airborne=(
                (cfg.airborne_p, cfg.airborne_cz_min, cfg.airborne_cz_max)
                if cfg.airborne_p > 0
                else None
            ),
            aux_displaced_m=cfg.aux_displaced_m,
            airborne_extra=cfg.airborne_extra,
            alive_gap=cfg.alive_gap_target if cfg.alive_gap_target > 0 else None,
        )
        return entries, roster

    margin = model.receptive_half
    if cfg.window <= 2 * margin:
        raise ValueError(f"window {cfg.window} leaves no interior for +-{margin} frames of context")
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(make, first_set) if first_set < n_sets else None
        for set_index in range(first_set, n_sets):
            built = time.perf_counter()
            entries, roster = pending.result()
            waited = time.perf_counter() - built
            pending = pool.submit(make, set_index + 1) if set_index + 1 < n_sets else None
            data = DeviceSet(entries, roster, device, extra_weight=cfg.extra_weight)
            rng = np.random.default_rng([cfg.seed, set_index, 17])
            torch.manual_seed(cfg.seed * 1009 + set_index)
            model.train()
            began, frames = time.perf_counter(), 0
            # "kl" only while the class KL is in the loss, so an undistilled history stays as it was.
            # Pretrain has no class term, so "ce" is absent from the history.
            if pretrain:
                sum_keys = ("loss", "aux", "acc")
            else:
                sum_keys = (
                    ("loss", "ce", "aux", "acc", "kl") if kl_on else ("loss", "ce", "aux", "acc")
                )
            if cfg.possessor_weight > 0:
                sum_keys = (*sum_keys, "pos")
            if switch_weight > 0:
                sum_keys = (*sum_keys, "sw")
            if touch_weight > 0:
                sum_keys = (*sum_keys, "touch")
            if cfg.scrf_weight > 0:
                sum_keys = (*sum_keys, "scrf")
            if wants_switch_distill:
                sum_keys = (*sum_keys, "swd")
            if tmse_weight > 0:
                sum_keys = (*sum_keys, "tmse")
            if event_switch_weight > 0:
                sum_keys = (*sum_keys, "esw")
            sums = {k: torch.zeros((), device=device) for k in sum_keys}
            first_step = set_index * cfg.refresh_steps
            last_step = min(cfg.steps, first_step + cfg.refresh_steps)
            for step in range(first_step, last_step):
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, cfg)
                batch = data.batch(
                    rng,
                    cfg.batch_size,
                    cfg.window,
                    decoupled=bool(rng.random() < cfg.decouple_prob),
                )
                batch = augment(batch, run.token, cfg, rng)
                with autocast:
                    # Teachers are eval and frozen, so this forward draws no random numbers.
                    # alpha 0 and switch_alpha 0 skip it and build the undistilled loss tensor.
                    if distill_on:
                        assert cfg.distill is not None
                        with torch.no_grad():
                            teacher_logits, teacher_switch = mean_teacher_outputs(
                                teacher_models, _teacher_batch_tokens(batch, cfg.distill)
                            )
                    out = model(batch["tokens"])
                if distill_on:
                    terms = loss_terms(
                        out,
                        batch,
                        weights,
                        margin,
                        cfg.aux_weight,
                        teacher_logits=teacher_logits,
                        teacher_switch=teacher_switch,
                        distill=cfg.distill,
                        spell=cfg.spell_weight,
                        possessor_weight=cfg.possessor_weight,
                        switch_weight=switch_weight,
                        switch_tol_frames=cfg.switch_tol_frames,
                        switch_pos_weight=cfg.switch_pos_weight,
                        touch_weight=touch_weight,
                        touch_pos_weight=cfg.touch_pos_weight,
                        class_loss=not pretrain,
                        scrf_weight=cfg.scrf_weight,
                        scrf_lam_team=cfg.scrf_lam_team,
                        scrf_switch_weight=cfg.scrf_switch_weight,
                        tmse_weight=tmse_weight,
                        tmse_tau=cfg.tmse_tau,
                        event_switch_weight=event_switch_weight,
                    )
                else:
                    terms = loss_terms(
                        out,
                        batch,
                        weights,
                        margin,
                        cfg.aux_weight,
                        spell=cfg.spell_weight,
                        possessor_weight=cfg.possessor_weight,
                        switch_weight=switch_weight,
                        switch_tol_frames=cfg.switch_tol_frames,
                        switch_pos_weight=cfg.switch_pos_weight,
                        touch_weight=touch_weight,
                        touch_pos_weight=cfg.touch_pos_weight,
                        class_loss=not pretrain,
                        scrf_weight=cfg.scrf_weight,
                        scrf_lam_team=cfg.scrf_lam_team,
                        scrf_switch_weight=cfg.scrf_switch_weight,
                        tmse_weight=tmse_weight,
                        tmse_tau=cfg.tmse_tau,
                        event_switch_weight=event_switch_weight,
                    )
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                frames += cfg.batch_size * cfg.window
                for key in sums:
                    sums[key] += terms[key].detach()
            n = last_step - first_step
            means = {k: float(v) / n for k, v in sums.items()}  # the one sync of the set
            seconds = time.perf_counter() - began
            row = {"set": set_index, "step": last_step, **means}
            row.update(frames_per_s=frames / seconds, wait_s=waited, set_s=seconds)
            if monitor is not None:
                row.update({f"monitor_{k}": float(v) for k, v in monitor(model).items()})
            history.append(row)
            watched = "".join(f" {k} {v:.3f}" for k, v in row.items() if k.startswith("monitor_"))
            kl_bit = f" kl {row['kl']:.4f}" if "kl" in row else ""
            pos_bit = f" pos {row['pos']:.4f}" if "pos" in row else ""
            sw_bit = f" sw {row['sw']:.4f}" if "sw" in row else ""
            touch_bit = f" touch {row['touch']:.4f}" if "touch" in row else ""
            scrf_bit = f" scrf {row['scrf']:.4f}" if "scrf" in row else ""
            swd_bit = f" swd {row['swd']:.4f}" if "swd" in row else ""
            tmse_bit = f" tmse {row['tmse']:.4f}" if "tmse" in row else ""
            esw_bit = f" esw {row['esw']:.4f}" if "esw" in row else ""
            ce_text = f"ce {row['ce']:.4f}" if "ce" in row else ""
            log(
                f"set {set_index + 1}/{n_sets} step {last_step}: loss {row['loss']:.4f} "
                f"{ce_text}{kl_bit} aux {row['aux']:.4f}{pos_bit}{sw_bit}{touch_bit}"
                f"{scrf_bit}{swd_bit}{tmse_bit}{esw_bit} "
                f"team-acc {row['acc']:.3f} "
                f"{row['frames_per_s'] / 1000:.0f}k frames/s, waited {waited:.1f}s for data"
                f"{watched}"
            )
            del data, entries
            extra = {
                "optimizer": optimizer.state_dict(),
                "set_index": set_index,
                "history": history,
            }
            if ema is not None:
                extra["ema"] = ema.state_dict()
                extra["ema_updates"] = ema.updates
            save_checkpoint(
                last,
                model,
                run.token,
                {**(meta or {}), "run": run.to_dict()},
                extra=extra,
            )
    final_meta = {
        **(meta or {}),
        "run": run.to_dict(),
        "history": history,
        "torch": str(torch.__version__),
    }
    if ema is not None:
        save_checkpoint(out_dir / "final_raw.pt", model, run.token, final_meta)
        ema.copy_to(model)
    final = save_checkpoint(out_dir / "final.pt", model, run.token, final_meta)
    if ema is None:
        log(f"wrote {final}")
    else:
        log(f"wrote {final} (ema) and {out_dir / 'final_raw.pt'} (last iterate)")
    return final


def with_defaults(got: Any, current: Any) -> Any:
    """`got` (a saved config dict) with every key it lacks filled from the defaults of
    `current`'s class. Anything that is not a dict, or a class without a no-argument
    constructor, is returned unchanged."""
    if not isinstance(got, dict):
        return got
    try:
        defaults = type(current)().to_dict()
    except (TypeError, AttributeError):
        return got
    return {**{k: v for k, v in defaults.items() if k not in got}, **got}


# What `train.init_from_partial` lets the parent lack: the slot branch (#86, roadmap B). The config
# keys are left out of the comparison, and only parameters under these prefixes may be missing.
_PARTIAL_MODEL_KEYS = ("slot_temporal",)
_PARTIAL_TOKEN_KEYS = ("slot_index",)
_PARTIAL_PREFIXES = ("slot_temporal.",)


def _without(raw: Any, keys: Sequence[str]) -> Any:
    """A copy of the dict `raw` without `keys`; anything else unchanged."""
    if not isinstance(raw, dict):
        return raw
    return {k: v for k, v in raw.items() if k not in keys}


def _load_init_weights(
    model: torch.nn.Module, path: str, run: RunConfig, *, partial: bool = False
) -> list[str]:
    """Copy `path`'s state dict into `model`. The optimizer in the file is not read.

    The architecture name, the model-config dict and the token-spec dict must equal this run.
    A key the checkpoint lacks (a field added after it was written, e.g. `gap_bracket`) counts as
    that field's default, so an older checkpoint whose missing fields this run leaves at their
    defaults still matches (#86, E84). A mismatch raises before any parameter is replaced.

    With `partial` (`train.init_from_partial`), the parent may lack the slot branch. The configs
    are compared without `model.slot_temporal` and `token.slot_index`, and everything else must
    still be equal. Every tensor of the parent must exist in `model` with the same shape and dtype.
    The only tensors the parent may lack are `slot_temporal.*`; they keep their initial values, and
    the branch's zero output projection makes the first step compute the parent's network. Any
    other difference raises before a parameter is replaced. Returns the names of the tensors that
    kept their initial values (empty without `partial`).

    `train.init_add_touch_head` (off by default) also accepts a checkpoint without the touch head
    when this run turns `model.touch_head` on and nothing else differs: every weight the
    checkpoint has is loaded, and the touch layer keeps its seeded fresh initialisation. Off, the
    comparison and the strict load are exactly as above.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"init_from checkpoint not found: {path}")
    payload = load_payload(file)
    arch = str(payload.get("arch", "team_net"))
    got_model = with_defaults(payload.get("model_config"), run.model)
    got_token = with_defaults(payload.get("token_spec"), run.token)
    expect_model = run.model.to_dict()
    expect_token = run.token.to_dict()
    if partial:
        got_model, expect_model = (
            _without(got_model, _PARTIAL_MODEL_KEYS),
            _without(expect_model, _PARTIAL_MODEL_KEYS),
        )
        got_token, expect_token = (
            _without(got_token, _PARTIAL_TOKEN_KEYS),
            _without(expect_token, _PARTIAL_TOKEN_KEYS),
        )
    adding = (
        run.train.init_add_touch_head
        and isinstance(got_model, dict)
        and not got_model.get("touch_head", False)
        and bool(expect_model.get("touch_head", False))
    )
    if adding:
        expect_model = {k: v for k, v in expect_model.items() if k != "touch_head"}
        got_model = {k: v for k, v in got_model.items() if k != "touch_head"}
    mismatches: list[str] = []
    if arch != run.arch or got_model != expect_model:
        mismatches.append(f"architecture {arch} {got_model} != {run.arch} {expect_model}")
    if got_token != expect_token:
        mismatches.append(f"token spec {got_token} != {expect_token}")
    if mismatches:
        raise ValueError(
            f"init_from checkpoint does not match this run ({'; '.join(mismatches)}): {path}"
        )
    state = payload["state_dict"]
    if adding:
        result = model.load_state_dict(state, strict=False)
        if sorted(result.missing_keys) != ["touch.bias", "touch.weight"] or result.unexpected_keys:
            raise ValueError(
                f"init_add_touch_head: expected only the touch layer missing from {path}, got "
                f"missing {sorted(result.missing_keys)} and unexpected {sorted(result.unexpected_keys)}"
            )
        return []
    if not partial:
        model.load_state_dict(state)
        return []
    own = model.state_dict()
    unexpected = sorted(name for name in state if name not in own)
    missing = [name for name in own if name not in state]
    not_new = [name for name in missing if not name.startswith(_PARTIAL_PREFIXES)]
    differ = [
        f"{name} {tuple(state[name].shape)} {state[name].dtype} != "
        f"{tuple(own[name].shape)} {own[name].dtype}"
        for name in state
        if name in own
        and (state[name].shape != own[name].shape or state[name].dtype != own[name].dtype)
    ]
    problems = []
    if unexpected:
        problems.append(f"tensors this model does not have: {unexpected}")
    if not_new:
        problems.append(f"missing tensors outside {list(_PARTIAL_PREFIXES)}: {not_new}")
    if differ:
        problems.append(f"shape or dtype differs: {differ}")
    if problems:
        raise ValueError(
            f"init_from_partial: checkpoint does not fit this model ({'; '.join(problems)}): {path}"
        )
    model.load_state_dict(state, strict=False)
    return missing


def _check_extra_heights(matches: list[TrainMatch], log: Callable[[str], None]) -> None:
    """Read the ball height of every extra match (`airborne.ball_height`) and log how many have one.

    `train.airborne_extra` with no extra match that has a height would change nothing, so it
    raises. A PFF test-split game raises in `pff_ballz` (it is refused at load already).
    """
    from .airborne import HEIGHT_SOURCES, ball_height

    extras = [match for match in matches if match.extra]
    with_height = sum(
        ball_height(match.match_id, match.bundle, extra=True) is not None for match in extras
    )
    if not with_height:
        raise ValueError(
            "train.airborne_extra is on, but no extra match has a ball height "
            f"(sources with one: {', '.join(HEIGHT_SOURCES)})"
        )
    log(f"airborne_extra: {with_height} of {len(extras)} extra matches have a ball height")


def _pretrain_roster(
    matches: list[TrainMatch],
    cfg: TrainConfig,
    spec: TokenSpec,
    meta: dict[str, Any] | None,
) -> tuple[list[TrainMatch], dict[str, Any] | None]:
    """The label-free extras only. A non-extra match passed in is not trained on.

    Nothing here loads an IDSSE match. Extras already on `matches` are kept when their count
    matches the config; otherwise they are loaded with `extra_labels` none.
    """
    if cfg.extra_labels != "none":
        raise ValueError(
            "train.pretrain requires extra_labels 'none' so every extra match is "
            f"label-free, got {cfg.extra_labels!r}"
        )
    extras = [match for match in matches if match.extra]
    if extras:
        if len(extras) != len(cfg.extra_matches):
            raise ValueError(
                f"{len(extras)} extra matches were passed in but train.extra_matches "
                f"lists {len(cfg.extra_matches)}"
            )
        roster = extras
    else:
        roster = load_extra_matches(
            cfg.extra_matches,
            spec,
            labels="none",
            extra_source_kwargs=cfg.extra_source_kwargs,
        )
    if not roster:
        raise ValueError(
            "train.pretrain trains only on train.extra_matches, and that list is empty"
        )
    labelled = [match.match_id for match in roster if not match.label_free]
    if labelled:
        raise ValueError(
            "train.pretrain requires every extra match to be label-free "
            f"(extra_labels 'none'); {labelled} still carry labels"
        )
    if meta is None:
        return roster, None
    return roster, {**meta, "train_matches": [match.match_id for match in roster]}


def _with_extra_matches(
    matches: list[TrainMatch],
    cfg: TrainConfig,
    spec: TokenSpec,
    meta: dict[str, Any] | None,
) -> tuple[list[TrainMatch], dict[str, Any] | None]:
    """Append `cfg.extra_matches` when the caller has not already. Empty config returns both as
    given, so a checkpoint's match list is the one the caller stored.

    `pretrain` keeps only the label-free extras. The caller does not pass an IDSSE roster.
    """
    if cfg.pretrain:
        return _pretrain_roster(matches, cfg, spec, meta)
    updated = include_extra_matches(
        matches,
        cfg.extra_matches,
        spec,
        labels=cfg.extra_labels,
        extra_source_kwargs=cfg.extra_source_kwargs,
    )
    if updated is matches or meta is None:
        return updated, meta
    return updated, {**meta, "train_matches": [match.match_id for match in updated]}


def coarsen_targets(match: TrainMatch, min_s: float) -> TrainMatch:
    """`match` training on its labels with every possession shorter than `min_s` absorbed into the
    previous one (`absorb_short`, the function behind the decoder's `min_spell` and the k-second
    truths). It is the organisers' rule that an uncontrolled touch does not change possession, as a
    training target (E40). Dead runs and X are left as they are; `target_ok` is recomputed."""
    labels = absorb_short(match.labels, match.bundle.meta, min_s).astype(np.int8)
    return _with_target_labels(match, labels)


def _ctrl_hold_s(cfg: TrainConfig) -> float:
    """`target_min_s` when it is set, otherwise the 1.5 s controlled-possession default."""
    return float(cfg.target_min_s or 1.5)


def control_targets(match: TrainMatch, hold_s: float) -> TrainMatch:
    """`match` training on `controlled_possession` (E40c), with `hold_s` the duration that counts
    without a pass. `target_ok` is recomputed as in `coarsen_targets`. With `target_rule` ctrl and
    `target_min_s` both set, training uses this and `target_min_s` as `hold_s`, not both rules."""
    labels = controlled_possession(match.labels, match.bundle.tracking, hold_s=hold_s).astype(
        np.int8
    )
    return _with_target_labels(match, labels)


def with_boundary_weight(match: TrainMatch, cfg: BoundaryWeight) -> TrainMatch:
    """`match` with its E155 frame weights (`boundary_weights`) on its training target labels,
    per period at the match's frame rate. A new TrainMatch; `match` is not changed."""
    meta = match.bundle.meta
    return replace(match, boundary=boundary_weights(match.labels, meta.runs(), meta.fps, cfg))


def _with_target_labels(match: TrainMatch, labels: np.ndarray) -> TrainMatch:
    seen = ~np.isnan(match.bundle.tracking.ball_xy[:, 0])
    ok = seen & np.isin(labels, (H, A)) & match.played
    return replace(match, labels=labels, target_ok=ok, bundle=replace(match.bundle, labels=labels))


def replace_train(run: RunConfig, **changes: Any) -> RunConfig:
    """`run` with some TrainConfig fields changed (a CLI override or a test)."""
    return replace(run, train=replace(run.train, **changes))
