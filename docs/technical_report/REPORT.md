# AAAAG: frame-level ball possession from automatic tracking

_Technical document for More than a Hack 2027, Phase 1. It describes the final submission, v15
(`possession_learned_joint_v15`)._

- **Team AAAAG:** Ameya, Arnav, Bakshi, Ayush.
- **Submitted as** `AAAAG.pdf` (or `.md`). The sections follow the organisers' SUBMISSION.md.
- **Numbers.** Every score here is an aggregate over external matches: IDSSE, Metrica, PFF FC and
  SkillCorner. Nothing measured on the competition match (BAR-ATH) is reported, except the run time
  of our build and the SHA-256 of the file it writes. Facts about the competition data come from the
  organisers' own documents (DATA.md, FAQ.md, EVALUATION.md, `games.py`).
- **Evidence.** Every experiment was declared, with its success rule, before it ran, in issue #86
  of the repository. The ids E0 to E158 refer to that log. Decision records for each champion are in
  `docs/decisions/`.

## 1. Approach, and why

**The task.** Label every frame of the BAR-ATH tactical video H, A, D or X from automatic tracking.
- The ranking metric, spellF1, matches whole possessions one-to-one by temporal IoU. So the score
  rests on **how many possessions we find and where their edges fall**, not on frame accuracy.
- The organisers' sensitivity table makes the point: a perfect labelling 1 s late scores 0.654.

**What we built.** A small learned network reads the tracking. A structured decoder turns its
per-frame posterior into stoppages and possessions. Three deterministic steps then apply the
organisers' metadata and their possession convention. Every stage is a registered component behind
fixed data contracts (`src/aaaag/contracts.py`).

| stage | component (v15 setting) | what it does |
|---|---|---|
| Evidence | `learned_team`, three **TeamNet** networks × two mirrored views | Per frame: (H, A, D) logits and a "possession changes here" switch logit. The six passes are averaged. |
| Fusion | `factorised` | P(dead) and P(home \| alive) both come from the network's joint head. No rule-based ball state. |
| Decoding | `dead_gated_hsmm`, `dead_threshold` 0.30, `switch_weight` 1.5 | The dead mask comes first: P(D) ≥ 0.30 is a stoppage; stoppages under 2 s are dropped and live gaps under 1 s between two stoppages are filled. Then, inside each live run, an explicit-duration semi-Markov segmentation of H/A on 5-frame blocks. It uses a lognormal spell-length prior fitted on IDSSE labels and a bonus, at weight 1.5, where the switch logit fires. |
| Post-process 1 | `pre_kickoff_dead` | D from each half's first graded frame to its kick-off, the frame of the organisers' kick-off anchor in `games.py`. |
| Post-process 2 | `touch_snap`, window 0.25 s | Each H↔A boundary moves to the nearest ball touch of the gaining team within ±0.25 s. |
| Post-process 3 | `controlled_spell`, hold 1.5 s, `keep_restart` | A possession shorter than 1.5 s is kept only if two teammates each have the ball within 1 m during it (a completed pass), or if it is the first possession after a stoppage. Otherwise it is absorbed into its neighbour (Section 5). |
| Output | | X outside the organisers' two graded halves; H, A or D everywhere else. |

### The network

**TeamNet** at `d_model` 128, about 0.82M parameters (`experiments/ameya/e99/e99wa.yaml`, `model`).
- **Tokens.** Each frame has 23 tokens: the 11 observed slots of each team nearest the ball (else
  the team centroid), plus the ball. A missing slot is a zero token with a validity flag. Slots are
  anonymous, so no slot identity is used. Positions are rotated per period so that home attacks +x;
  the direction comes from the slot geometry, without labels.
- **Inputs, 20 columns per token:**
  - position, velocity, speed, team and ball flags, validity, offset and distance to the ball;
  - on the ball token: log time since the ball was last seen and until it is next seen;
  - **control memory**: which team last controlled the ball (nearest slot within 1 m for 3 frames)
    and how long ago;
  - **contact columns**: the ball's velocity change, a decayed touching-team sign, a live control
    flag and player acceleration.
- **Encoder.** Per frame, self-attention over the 23 tokens with a learned bias for each token
  relation, then attention pooling per team. Then a dilated non-causal temporal convolution stack
  (dilations 1 to 64), whose receptive field is ±127 frames (±5.1 s).
- **Heads:**
  - a joint H/A/D head;
  - a **possessor head**: a softmax over the 22 player tokens plus "nobody", summed per team and fed
    back into the frame embedding;
  - a **switch head**: one logit per frame for "an H↔A change happens here";
  - an auxiliary ball-position head, trained on live frames where the ball is hidden.

### How the shipped weights are made

Each of the three shipped networks (seeds s = 2, 3, 4) is a **weight soup** of four networks that
share one pretrained ancestor. All training uses external data only.

| step | what it does | data | key settings |
|---|---|---|---|
| 1. Pretraining | Learns possession from a second provider and tournament | 44 PFF FC World Cup 2022 train-split games | 6000 steps, raw ball, event-derived `control` labels, seed s+200 |
| 2. IDSSE recipe (E56a) | The full training recipe on the target-like provider data, starting from step 1 | all 7 IDSSE matches, degraded (Section 3) | 2000 steps, lr 2e-3, seed s |
| 3W. Visible-dead-ball fine-tune (E92 C) | Teaches that a visible, uncarried ball can still be dead | as step 2 | from step 2, 2000 steps, lr 5e-4, `dead_visible_p` 0.2, seed s+100 |
| 3G. Airborne and spell-balanced fine-tune (A25G25) | Step 3W plus two changes: the airborne-ball projection and spell-balanced loss weights | as step 2 | `airborne_p` 0.25 (camera 15–25 m high), spell weight gamma 0.25; three seeds G, G′, G″ = s+900, s+1100, s+1700 |
| 4. Soup | Averages the float tensors of the four step-3 networks | none | **½ W + ⅙ (G + G′ + G″)**; integer buffers come from W |

- W is v14's network. The soup is called SGG″ in the log (E144).
- Every window is 1000 frames (40 s), 16 windows per step, with AdamW (weight decay 0.05), linear
  warmup and cosine decay to 2% of the peak rate. Each step-3 fine-tune takes about 2 GPU-hours per
  seed.
- **Why a soup.** The G fine-tune alone recovers most of a simulated airborne-ball weakness, but it
  costs accuracy on Metrica. Averaging it with W, its sibling from the same step-2 parent, keeps most
  of the robustness and drops most of the Metrica cost (E129b). This is the "model soup" idea
  (Wortsman et al., 2022). Averaging three G seeds instead of one reduces the variance of the G side
  (E141, E144); six seeds add nothing more (E152).

**Inference** is torch-free: a numpy float64 forward pass from float32 weights stored as JSON
(base64, 4.35 MB per network). The second view is the y-mirrored pitch.

### Why this design rather than another

- **Learned evidence beats our rules on external data.**
  - The network's dead head calls 0.3% of live frames dead, against 10.9% for our rule-based
    ball-state HMM. On Metrica, which it never trained on, it gains +0.060 spellF1 (E9).
  - The learned team factor gains +0.073 over the last-controller rule (E0).
- **A small network, because the labelled data is small.** Larger networks lost (E16: 0.95M
  parameters −0.003, 2.42M −0.020), and so did temporal attention (E81: −0.027, 0/5 matches). The
  shipped `d_model` 128 network comes with pretraining on 44 extra games (E99); without extra data,
  larger networks lost.
- **Physical inputs carry over to a new provider.** On Metrica the contact columns keep 77% of their
  development gain; the switch head keeps 34% and the possessor head 19%.
- **Structure goes in the decoder.** spellF1 punishes fragmentation. The semi-Markov decoder
  enforces plausible possession lengths, and the dead gate stops a confident team posterior from
  moving the stoppage edges (E1c).
- **Learn the fine labels, apply the convention afterwards.** Training on convention-coarsened
  targets lost on every truth (E40, E40c), so the convention is a deterministic step after decoding.
- **Each hard regime gets its own fine-tune, and the soup keeps them together.** Under stress, a
  visible dead ball costs the network 0.054 and an airborne ball 0.083 (E71, E118), far more than
  any clean-input change gains. Training from scratch on them failed (E71, E71b) and interpolating separately trained
  networks hit a loss barrier (E84 arm S, E88). Fine-tuning from a shared parent, then averaging,
  worked (E84, E92, E129b).
- **Every change passed a declared rule** on paired fold models, on fresh seed pairs, with Metrica
  as a guard (Section 3). Ties went to the simpler option.

## 2. Assets used and dropped

| Asset | Used? | How, or why it was dropped |
|---|---|---|
| Tactical ATD | **Yes, the only input** | Ball and player positions in pitch-centred metres and each slot's team. Slot identities are never used: slots fragment and swap. |
| Organisers' metadata (`games.py`) | Yes | The frame ranges of the two graded halves (X is written outside them) and the two kick-off anchors (`kickoff`, `kickoff_2nd`), before which D is written. |
| Smart Tagging | No | Not a pipeline input. A restart prior built from tag windows was tested on simulated tags (E78) and failed its rule. |
| Homography | No | The ATD is already in pitch coordinates. |
| Tactical video and its audio | No | Three video-derived inputs were tested and none passed: an off-the-shelf ball detector (E91), a ball detector trained on external clips (E150) and a referee-whistle detector on the audio (E156). Watching the video to understand what counts as possession is allowed; nothing seen was recorded or used. |
| Broadcast video and ATD | No | Redundant with the tactical tracking for this task. |
| Third-party pretrained models | None | The published possession checkpoint we found (PathCRF) was trained on IDSSE and would leak our validation matches. Our own pretraining (step 1) uses PFF FC data, credited in Section 8. We use no code, data or weights under a non-commercial or GPL/AGPL licence. |

## 3. Training and validation data

**All training and validation data is external. None comes from the provided game.**

| dataset | source and licence | who labelled it | used for | amount |
|---|---|---|---|---|
| **IDSSE** (DFL / Sportec open data, Bassek et al. 2025, *Scientific Data*) | figshare doi:10.6084/m9.figshare.28196177, mirrored on Hugging Face `pysport/idsse-data` at a pinned revision, sha256-checked; **CC BY 4.0** | The provider: official per-frame `BallStatus` and `BallPossession` flags. Our label is D where the ball is dead, else the owning team. | Steps 2 and 3 of every shipped network; all selection; the decoder's spell-length prior; the airborne projection (it uses IDSSE's ball height) | 7 full matches at 25 Hz, about 145,000 frames each. Five (J03WMX, J03WOH, J03WOY, J03WPY, J03WR9) are the development set. Two (J03WQQ, J03WN1) were a sealed exam, run once on 9 Nov; the production networks were then trained on all seven. |
| **PFF FC World Cup 2022** | PFF FC's public 2022 World Cup dataset (tracking, metadata, rosters, events); used under PFF FC's terms, never redistributed | Our code builds the labels from PFF's events (the `control` version, chosen on 8 train games before any training as the one closest to IDSSE's possession signal) | Step 1 pretraining of every shipped network; a sealed **test split** that checks each champion against the previous one | 64 games of broadcast tracking at 29.97 fps, resampled to 25. Pretraining uses 44 train-split games. The 16 games with index % 4 == 3 are the test split, fixed before any PFF result and never trained, tuned or selected on. |
| **Metrica Sports sample data** | github.com/metrica-sports/sample-data, pinned commit `e706dd5`, sha256-checked. No formal licence; the README asks for responsible use and acknowledgement. Not redistributed. | Our code builds the labels from Metrica's hand-tagged events, so they are approximate | Fitting the degradation simulator; a cross-provider **guard** on every change. It never trains a model and never selects one. | 3 anonymised games at 25 fps |
| **SkillCorner open data** | github.com/SkillCorner/opendata, pinned commit `4340d27`; **MIT** | SkillCorner's per-frame `possession.group` | One training arm (E35), **dropped**, so it is in no shipped network; a label-granularity comparison (Section 5) | 20 A-League matches at 10 fps, resampled to 25 |

**Why Metrica matters.** The organisers' DATA.md says the competition tracking is the output of
Metrica's computer-vision pipeline, so Metrica's sample games show the same provider's behaviour.
In the sample games the ball is missing on 32–41% of played frames and blanked through 96% of dead
frames, and its track is not smoothed.

**Why PFF.** Its broadcast tracking has natural gaps (the ball is flagged visible on about 33–38% of
frames, and 57–71% of player positions are estimated), so it shows a network many more possessions
under imperfect tracking than seven IDSSE matches can.

### The degradation simulator (degrade-v5)

IDSSE's tracking is clean (the ball is never missing), so we degrade it into BAR-ATH-like inputs.
The simulator is fitted on the three Metrica games only. Earlier versions calibrated on aggregates of
the provided tracking were retired when the organisers ruled, on 2026-11-05, that tuned parameters
must come from external data.
- **Dead ball.** The ball is hidden through every stoppage. It vanishes 1 frame after the stoppage
  starts and returns at the restart; 1.2% of stoppages keep it. The edges are jittered in training
  (0, ±0.5 s or ±2 s).
- **Live gaps.** They start at a rate that depends on the context, from 0.00047 per seen live frame
  in open play to 0.00261 near a line in a crowd. Their lengths are resampled from Metrica (median
  0.08 s, p90 1.46 s).
- **Players** (the v5.1 ladder, physics priors). Slot dropout in runs of about 2 s, plus position
  noise smoothed over 0.5 s: mild (5% dropped, σ 0.10 m), medium (10%, 0.25 m), severe (20%,
  0.50 m), and a medium level concentrated near the ball.

| realism | Metrica | degrade-v5 |
|---|---|---|
| ball missing, played frames | 0.373 | 0.445 |
| ball missing, dead frames | 0.962 | 0.991 |
| live gaps per live minute | 0.98 | 1.06 |

### Training augmentations

| augmentation | setting | why |
|---|---|---|
| Label-independent ball mask | half of each batch | The ball is hidden independently of the labels, so "missing ball means dead" cannot pay. |
| Ball smoothing | Gaussian, sd 2–10 frames, probability 0.65 | Providers differ in how much they smooth the ball; IDSSE's and Metrica's sample balls are not smoothed. |
| Restart blanking (E56a) | probability 0.5, ball hidden from 1 s before to 3 s after a true restart | A restart with a hidden ball must not read as an ongoing stoppage. |
| Visible dead ball (step 3) | probability 0.2 per stoppage of 4 s or more | A stoppage where the ball stays visible and uncarried must still read as dead. |
| Airborne-ball projection (step 3G) | probability 0.25 per match; one camera on the halfway line ±20 m, 20 m behind a touchline, 15–25 m high | A single tactical camera places an airborne ball where the camera ray through it meets the grass. TRACAB records the true ground position, so the network had never seen this. |
| Spell-balanced weights (step 3G) | each live frame weighted (6 s / max(L, 0.5 s))^0.25, L its spell's length | spellF1 counts every possession once, however short. |
| Mirroring | y-flip and team swap | Symmetry. |

E157 checked that the simulator does not leak D through ball presence beyond what the provider
does. In Metrica's raw tracking P(ball missing | dead) is 0.962 and P(missing | live) 0.122; the
training ball gives 0.955 and 0.043 on the coupled half, and 0.317 and 0.321 on the label-independent
half.

### Truths and evaluation levels

BAR-ATH's label convention is unknown, so every change is scored against several truths built from
the same external labels:
- **provider**: the IDSSE labels as delivered (DFL's flag flips on touches);
- **k1, k2**: the provider labels with every possession shorter than 1 s or 2 s absorbed;
- **ctrl**: the organisers' controlled-possession rule, applied to the clean tracking;
- **aba1.75** (from 16 Nov): inside each live run, an A–B–A with B shorter than 1.75 s becomes A.
  The organisers' delay table (EVALUATION.md: 1.0000 / 0.9818 / 0.8933 / 0.6539 at 1 / 3 / 8 / 25
  frames late) depends only on the truth's possession lengths. Shifting each candidate truth against
  itself, aba1.75 reproduces that table best on IDSSE (RMS 0.025 against 0.053 for provider; E125).

Each truth is scored at several levels of input damage:

| level | what it is |
|---|---|
| clean | IDSSE dev degraded by degrade-v5 (seeds 5–9) |
| S8@8 | the same with the ball smoothed by an 8-frame Gaussian (seeds 5 and 6) |
| hr | S8@8 with the ball also hidden ±2 s around every true restart |
| dv | S8@8 with the ball visible and uncarried for the last 3 s of every true stoppage of 4 s or more |
| mix50, mix90 | the two regimes in one match: 50% or 90% of restarts hidden as in hr, the other stoppages as in dv |
| rec18 | S8@8 with IDSSE's real ball height projected through a camera 18 m high (E118) |
| Metrica clean, S8@8 | the three Metrica games on their event-derived labels |

The **declared score** is (clean three-truth mean delta + S8@8 delta) / 2, over provider, k1 and k2.

### Validation discipline

- **Pre-registration.** Every experiment was posted on issue #86 before it ran: hypothesis, arms,
  levels, the rule and its bars. Every result was posted, including failures. When a rule changed,
  the change was posted before any affected result was read.
- **Fold models and fold-seed pairs.** Every comparison uses leave-one-match-out fold models on the
  five development matches. A "pair" is two training seeds (for example 2+3), ensembled as in
  production. A candidate and its control are scored on the same pairs, folds and degradation seeds,
  and the delta is paired. A selection pair picks the arm; fresh pairs confirm it. From E99 on, the
  three pairs are 2+3, 4+5 and 12+13.
- **The adoption rule.** Until 15 Nov each experiment declared its own bars, typically a pooled gain
  of at least +0.003 that is positive on 3/3 pairs, with guards. On 16 Nov the team adopted **R16**,
  before any affected result was read: a gain counts if it is consistently positive.
  - Gain: the declared score (or its aba1.75 analogue) positive pooled and on 3/3 pairs, and mix50
    ≥ 0 pooled; for airborne arms, rec18 positive pooled and on 3/3.
  - Guards: Metrica clean and S8@8 ≥ −0.002 pooled and ≥ −0.005 on every pair; clean provider and k1
    ≥ −0.005; hr and dv ≥ −0.003.
  - Between two passing arms: the larger pooled hr+dv, by more than 0.001.
  - Decode-only changes also had to pass guards G1–G4 against the nearest simpler decode, because
    decode variants share the same fold models and so pass 3/3 too easily.
- **Metrica is a guard, never an objective.** It never trains and never chooses between two arms
  that both pass.
- **The PFF test split** (16 sealed games) checks each champion against the previous one, on
  production weights. It caught v13 (Section 6), which had passed on IDSSE and Metrica.
- **The sealed IDSSE exam** was run once, on v10, on 9 Nov. It was then spent: the production
  networks were retrained on all seven matches.
- **Noise.** Re-running v14's recipe on new seeds moves the pooled declared score by +0.0003 and the
  degraded levels by ±0.0005, but single pairs by up to ±0.004 (E138). For v11b, six fresh seed pairs
  gave the same level as the tuned pairs (0.7344 against 0.7348 declared), so there was no
  seed-level optimism.
- **BAR-ATH never selects anything.** The public five minutes are touched only by a smoke test that
  prints PASS or FAIL. No parameter is fitted to labels, to the video, or to specific frames or
  moments of the match.

## 4. Missing and noisy tracking

| problem (organisers' DATA.md) | how it is handled |
|---|---|
| **The ball is missing on 41% of graded frames** | The network reads the time since and until the ball is seen, and learns P(dead) from context. The label-independent mask stops it from equating "missing" with "dead". The auxiliary head learns where a hidden ball is. Through live gaps the control memory carries the last controller. |
| **Hidden ball at restarts** | Restart blanking in training (E56a). The first possession after a stoppage is kept (`keep_restart`). |
| **Visible dead ball** | The step-3 fine-tune shows stoppages with a visible, uncarried ball (E84, E92). |
| **Airborne ball placed on the ground by the camera** | The step-3G projection, kept in the soup (E129b, E144). |
| **46 anonymous slots for 22 players; slots fragment and swap** | Tokens are the nearest observed slots, sorted by distance, with no identity. A player velocity above 12 m/s is a swap and is dropped. The set encoder is permutation-equivariant within a team. |
| **Players missing near the ball** | Training on the v5.1 dropout ladder, including a near-ball level. With nobody in reach the control memory holds the last team. |
| **Ball jumps** | A ball velocity above 45 m/s is treated as a jump and dropped before the inputs are built. |
| **A smoothed ball** | Wide smoothing augmentation, and the possessor head, which reads player proximity rather than sharp ball kinks (E29b). |

**Robustness checks** (spellF1 on IDSSE fold models unless marked):

| check | result |
|---|---|
| Ball smoothed at 4 / 6 / 8 / 10 frames (E29b, v4 decoder) | 0.689 / 0.694 / 0.687 / 0.678. The previous network (E19): 0.654 / 0.648 / 0.626 / 0.609. The rule fall-back: 0.447 / 0.400 / 0.373 at 2 / 4 / 6 frames. |
| Long live gaps, 18% and 21% of live frames unseen (E9b) | Live play called dead: network 2% and 1%, rule 29% and 22%. Team accuracy does not fall with gap length (0.84–0.92). |
| Airborne ball, 18 m camera (E118, D1) | v14's networks lose 0.083 under this stress. v15 scores 0.7156 on it against v14's 0.6650 (+0.051; three pairs). |
| Ball carried through stoppages, 6 s (E101) | Cost against S8@8: v12 −0.053, v14 −0.039. |
| Tracker errors at S8@8 (E66) | Fragmentation and ghost slots cost nothing measurable. Wrong-team fragments cost −0.009, dropouts with duplicates −0.011, long open-play ball gaps −0.014. Training on them did not recover the cost (−0.0045 to −0.0083 declared). |
| Player error, slot dropout, duplicates, wrong team labels (S1–S6, an earlier network, E14) | At least +0.17 ahead of the rule fall-back at every level. Smoothed player error of 1 m or more is a cliff (Section 6). |

## 5. Possession rules implemented

| situation | organisers' rule (CHALLENGE.md) | what the pipeline does |
|---|---|---|
| **Ball out of play** | D | The network's dead head, trained on the provider's referee-driven `BallStatus`. P(D) ≥ 0.30 is a stoppage, which lasts at least 2 s; live play between two stoppages lasts at least 1 s. 0.30 replaced 0.40 after gaining +0.0028 to +0.0031 on four independent pairs (E72, E72b). |
| **Restarts** | play restarts with a clear possessor | No restart rule. The network sees the restart in its ±5 s context. The first spell after a stoppage is never absorbed by the convention step (`keep_restart`, E61). A rule that named the restarting side was wrong on about 15% of restarts; dropping it gained +0.014 (E11). |
| **Loose or contested ball** | stays with the last team | A new spell opens only on sustained evidence (duration prior plus switch bonus). The control memory tells the network who last had control. |
| **Touch without control, deflection** | not a change | The touch snap places real changes on the gaining team's first touch. A short spell without a pass is then absorbed. |
| **Aerial duel won without control** | not a change | the same |
| **A short controlled possession** (a quick pass between teammates) | a change | Kept when at least two distinct teammates have the ball within 1 m during it. |
| **Before kick-off, half-time, after the whistle** | not graded | X, from the organisers' half boundaries. Every graded frame gets H, A or D, because spellF1 forward-fills an X and an X never helps. |
| **Graded frames before a half's kick-off** | D (the organisers' FAQ notes that the second half is graded from before its kick-off) | `pre_kickoff_dead` writes D up to the organisers' kick-off anchor. IDSSE halves start at the kick-off, so the network never saw such frames. A match without anchors is unchanged, so this step changes no external score. Since v8 it runs first, so a possession left at the cut meets the other steps like any other (E55, E55b). |

**The convention step.** It runs after the touch snap, inside each live stretch; stoppages are never
rewritten. The shortest short spell merges first, into the previous spell, which then joins the next
spell of the same team. The stretch is re-checked after every merge. A spell under 1.5 s survives
only with pass evidence or as the first spell after a stoppage. With no pass evidence it equals a
plain 1.5 s minimum, so it fails safe on a sparse or smoothed ball. The reach (1 m, no minimum pass
length) was confirmed by a declared six-cell sweep (E40d).

**Why 1.5 s.** Four checks point to the same hold:
- **Minimax regret over plausible truths** (E7, E34): 1.5 s had the smallest worst-case regret on
  IDSSE.
- **Metrica, on production weights** (E106; mean spellF1 over three games, clean and S8@8):

  | hold | provider | k1 | k2 | ctrl | aba3 | worst regret | mean |
  |---|---|---|---|---|---|---|---|
  | off | 0.758 | 0.788 | 0.714 | 0.763 | 0.683 | 0.110 | 0.741 |
  | 1.0 s | **0.768** | **0.827** | 0.763 | 0.810 | 0.726 | 0.068 | 0.779 |
  | **1.5 s** | 0.753 | 0.822 | 0.803 | **0.840** | 0.747 | 0.047 | **0.793** |
  | 2.0 s | 0.722 | 0.795 | **0.824** | 0.824 | 0.768 | **0.046** | 0.786 |
  | 3.0 s | 0.678 | 0.749 | 0.789 | 0.785 | **0.794** | 0.090 | 0.759 |

  2.0 s wins the worst regret by 0.0007, below the declared 0.01 margin, and 1.5 s has the best mean.
- **The organisers' delay table** points to aba1.75. Scored against aba1.75 on v14's fold models,
  no other hold passed the declared rule: on clean input 2.0 s loses 0.005 to 0.006 and 1.0 s about
  0.02 on every pair, and 2.0 s also loses on Metrica (E125).
- **Independent conventions that require control** land at the rate of a 1.5 s rule:

  | labels | spells per 10 live minutes | spells under 2 s |
  |---|---|---|
  | IDSSE provider | 81 | 30% |
  | IDSSE k1.5 | 59 | 8% |
  | SkillCorner | 61 | 21% |
  | Metrica | 57 | 27% |

  Both external conventions keep more short spells than a pure duration rule, and keeping the short
  spells that show control is what the pass rule adds.

## 6. Results and limitations

### Results on external data

**v15 against v14** (fold models; three pairs, pooled; deltas in spellF1):

| step | declared | mix50 | hr / dv | rec18 | Metrica clean / S8@8 | k2 |
|---|---|---|---|---|---|---|
| SGG″ networks against W, v14's decode | **+0.0029** (3/3) | +0.0047 | +0.0058 / +0.0044 | **+0.046** (3/3) | −0.0012 / +0.0019 | −0.0026 |
| v15 decode against v14 decode, on SGG″ | **+0.0055** (mean of provider declared and its aba1.75 analogue) | | | +0.0048 (3/3) | | |

On the earlier SG networks the decode step alone gave provider declared +0.0042 and aba1.75 +0.0071
(3/3 each), with Metrica clean +0.0098.

**Production weights**, on data the networks never trained on:

| champion | Metrica 1–3, clean | Metrica 1–3, S8@8 | PFF test split: provider / k1 / k2, against the previous champion |
|---|---|---|---|
| v11b | 0.7585 | 0.7251 | |
| v12 | 0.7721 | 0.7338 | +0.0191 / +0.0194 / +0.0146 (14 games) |
| v13 (not adopted) | | | −0.0064 / −0.0080 / −0.0111 against v12 (14 games) |
| v14 | 0.7727 | 0.7386 | +0.0654 / +0.0653 / +0.0392, 16/16 games¹ |
| **v15 (submitted)** | **0.7881** | **0.7399** | **+0.0155 / +0.0158 / +0.0081** (14, 14 and 13 of 16 games) |

¹ v14's pretraining saw other games of the same tournament and provider, so this split is
in-domain for v14 and v15. It rules out a regression; it does not estimate the gain on BAR-ATH.

**Level.** On one fold-seed pair (16+17), v14's networks score 0.7407 declared, 0.734 / 0.806 / 0.788
/ 0.837 on clean provider / k1 / k2 / ctrl, and 0.695 / 0.709 / 0.702 on hr / dv / mix50 (E113).
v15 adds the deltas above.

**The sealed exam** (two IDSSE matches never used before, run once on v10): declared 0.7079 against
0.7318 on the development folds (J03WQQ 0.7212, J03WN1 0.6947). Most of the gap comes from the
development set's mix of match tempos, but it says the development numbers are optimistic by about
0.02.

**Where the stack started** (8 Nov; fold models, two seeds × two views):

| pipeline | provider | k1 | k2 | ctrl | S8@8 | Metrica 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| organisers' baseline (`official_baseline_smooth50`)² | 0.348 | | | | | |
| rule champion (`possession_snap_05s`)² | 0.516 | | | | | 0.497 (mean) |
| first network with switch head (E19) | 0.716 | 0.799 | 0.820 | | 0.626 | 0.741 / 0.732 / 0.772 |
| v5: E29b and the controlled-possession rule | 0.732 | 0.805 | 0.797 | 0.839 | 0.689 | 0.754 / 0.750 / 0.763 |

² On the earlier `idsse-dev-v5` suite: the same matches and simulator, degradation seeds 0–4.

### What each adopted step added

Each step was measured against the stack of its time, so the gains overlap and do not sum.

| version | step | experiment | gain (spellF1, external data) |
|---|---|---|---|
| v1–v3 | learned team factor instead of the last-controller rule | E0 | +0.073, 4/5 matches |
| | the network's dead head instead of the rule-based ball state | E9 | +0.050 dev, +0.060 Metrica |
| | dead-gated semi-Markov decoder instead of a 0.5 s vote | E1c | +0.018 [+0.012, +0.023], 5/5 |
| | three seeds; two mirrored views | E2, E15 | +0.015; +0.0046, 5/5 |
| | gaining-team touch snap | E10 | +0.0125, 5/5; any team's touch is a tie (−0.0007) |
| | control memory; contact columns | E14, E17 | +0.0075; +0.026 provider, 5/5, Metrica +0.019 to +0.022 |
| | switch head and decoder bonus | E19 | +0.012 and +0.015 (two pairs) |
| v4 | possessor head and wide ball-smoothing training | E29b | clean −0.0004; **S8@8 +0.062**, confirmed +0.057 |
| v4 | 1.5 s minimum possession | E7, E34 | a hedge: −0.041 on provider, +0.07 on the 2 s truth, smallest worst-case regret |
| v5 | controlled-possession rule | E40c | provider +0.014, k1 +0.006, k2 −0.019; Metrica +0.006 to +0.015 |
| v7, v8 | D before each kick-off; the step moved first | E55, E55b | no external change by construction |
| v9 | keep the first spell after a stoppage | E61 | +0.0037 on fresh seeds (provider +0.0054, S8@8 +0.0022) |
| v10 | restart blanking in training | E56a | hr +0.0131 pooled over three pairs; declared −0.0017; adopted by its declared adjudication |
| v11 | retrain on all seven IDSSE matches after the exam | E65 | a fourth training match gained +0.0084 declared over three |
| v11b | dead threshold 0.30 | E72, E72b | +0.0028 to +0.0031 declared on four independent pairs; Metrica +0.001 to +0.004 |
| v12 | visible-dead-ball fine-tune from the network's own weights (WiSE-FT style) | E84 F | dv +0.057 to +0.062, declared +0.002 to +0.007, mix50 +0.031 pooled |
| v14 | PFF pretraining, `d_model` 128, then the IDSSE recipe and a 2000-step fine-tune | E99 | declared +0.0036, mix50 +0.0155, hr +0.0152, dv +0.0138, Metrica +0.0034 / +0.0109; clean provider −0.0046 |
| v15 | the weight soup ½ W + ⅙ (G + G′ + G″) | E129b, E141, E144 | declared +0.0029, mix50 +0.0047, hr+dv +0.0102, rec18 +0.046 (3/3 each) |
| v15 | `switch_weight` 1.5, touch-snap window 0.25 s | E130, E139, E136 | +0.0055 over v14's decode on SGG″; Metrica clean about +0.01 |

### Tried and not adopted

Each line failed its declared rule unless marked.

| group | idea (experiment) | result |
|---|---|---|
| Architecture | Bigger networks (E16); temporal attention (E81); distance-aware set attention (E86) | −0.003 and −0.020; −0.027 at 4× the CPU time; −0.0026 |
| | Longer context, ±10 s (E31, E33, E70) | ties or losses |
| | Per-slot temporal encoder (E148, E153) | Alone, +0.0033 on its selection pair but +0.0010 pooled. Inside the soup, −0.0015 and +0.0004 hr+dv against SGG″. |
| Inputs | Rule-based stoppage cues (E36), a Laws-of-the-game trigger probe (E67), team-shape features | under 0.002 AUC beyond what the network already reads |
| Training data | SkillCorner as training data (E35) | −0.033, 0/5. Broadcast tracking sees 6–11 players and labels differently. |
| | PFF mixed into step 3 (E103); other PFF pretraining variants (E95, E96) | no gain; Metrica guard failures |
| | PFF ball height for the airborne projection (E147) | stopped: PFF's ball height is essentially flat |
| | Event-derived training labels (E80, E146) | DFL event-timed labels score 0.42–0.48 against the provider flags; event changes fall within 3 frames of a label change only 5% of the time |
| Targets and losses | Convention-coarsened targets (E40, E40c) | −0.008 to −0.016, even on their own truths |
| | Spell-balanced weights alone (E5a, E113) | E113 failed by 0.0001 on one Metrica pair. They pay inside the soup. |
| | Airborne fine-tune alone (E119 at p 0.5, E120 at p 0.25) | recovers about +0.06 of the airborne stress, but costs Metrica (−0.0048 clean pooled at p 0.5) |
| | Dead-edge weight (E85), HSMM-energy loss (E69), temporal-smoothness loss (E133), displaced-ball auxiliary target (E132), boundary-weighted loss (E155), aba-view spell weights (E154), Metrica-matched live gaps (E158) | each failed a guard, or lost to SGG″ on 2 of 3 pairs |
| Augmentations | Formation warp, ball offset, ghost slots (E6); a ball carried through stoppages (E59, E102); tracker errors (E66) | each cost more on normal input than it recovered |
| | Visible dead ball trained from scratch (E71, E71b) | full recovery, but −0.010 everywhere else; solved instead by fine-tuning (v12) |
| Ensembles and weight space | 5 or 6 seeds, 4 views, 6 production networks (E38, E57, E94); six G seeds (E152) | saturated: +0.0002 for six G seeds over three |
| | Distillation into the smaller network (E97, E98) | mix50 +0.0028 against a +0.003 bar |
| | Interpolating separately trained networks (E84 arm S); layer stitching and direct weight edits (E88, E90) | loss barrier, worse everywhere |
| | Weight EMA (E62, E138) | ties |
| | Other soups: ½ W + ½ G25 (E137), ½ W + ½ X (E137b), ¾ G (E140), a second W seed (E142), Model Stock toward the step-2 parent (Jang et al., 2024; E143) | Metrica guards fail, or a pass with lower hr+dv |
| | Extrapolation 1.2·SGG″ − 0.2·(step-2 parent) (EXPO) | hr+dv −0.0074 on the first pair; stopped by its kill rule |
| Decoding and post-process | Boundary refinement (E42); decoder re-tuning under smoothing (E30a); expected-spellF1 decoding (E83) | −0.005; flat within 0.004; +0.0003 |
| | Dead threshold 0.20 or 0.25 (E134, E136) | gains on clean input only; costs S8@8 Metrica, hr and mix50 |
| | Restart hold (E131); restart back-extrapolation (E68); restart-team states (E128) | rh2 −0.0048; rejected; the reachable gain is about +0.004 |
| | Ball-flight and jump gates (E127) | Metrica clean −0.065 and −0.073: on a real ball the jumps sit at real kicks |
| | Other holds, re-fitted to the delay table (E106, E125, E145) | 1.5 s stays |
| Other inputs | Smart Tagging restart prior (E78); a ball-gap specialist network (E82) | fails its kill rule on simulated tags; +0.0036 against a +0.004 bar |
| | Off-the-shelf video ball detector (E91) | NO-GO at its declared bars |
| | TrackNet-style ball detector trained on external TeamTrack clips (E150) | top-1 recall within 5 px 0.62 (val) and 0.83 (test), but 0.26 and 0.11 false-confident against a 0.05 cap |
| | Referee whistle from the tactical audio (E156) | stopped at its declared sanity check, with no retry; no candidate |

### Limitations: where it fails, and what we would fix with more time

- **The label convention is the largest uncertainty, about ±0.05.**
  - On one network, the best decoder for each truth scores from 0.78 (provider) to 0.855 (2 s)
    (E34). The 1.5 s hold costs about 0.06 if the competition truth follows the provider's per-touch flag.
  - The organisers' delay table points to aba1.75, which supports the 1.5 s hold. But under aba1.75
    with the shipped decode, the SGG″ networks are about **−0.001 hr+dv against W**. v15's network
    step rests on the airborne-ball recovery and the provider truths, not on aba1.75.
- **Simulated stresses are not measurements.** rec18, hr, dv and the mix levels simulate failure
  modes; they do not measure them. Where a real camera places airborne balls differently from the
  18 m projection, or rarely, most of the rec18 gain will not appear.
- **v15's costs.** On fold models the soup costs k2 (−0.0026) and Metrica clean (−0.0012). On
  production weights, v15's Metrica gain is large on clean input (+0.0154) but **small on the
  smoothed ball (+0.0014)**.
- **The provider shift.** A smoothed ball costs about 0.03. The 2-frame touch detector goes almost
  silent (1.9 touches per 10 minutes against 206 on the sharp ball), so the touch snap and the pass
  rule do little there.
- **The hidden ball is an information limit.** A players-only network calls the team as well as
  ours inside long ball gaps (E82), and training on long gaps did not help (E66). A ball carried
  through a stoppage still costs v14 −0.039 (E101).
- **The team call dominates the remaining error.** The failure anatomy finds no systematic boundary
  bias (E143b). Wrong-team stretches inside long possessions, and short true spells held through, are
  the largest losses. The decoder also emits about 7.5% fewer possessions than its own posterior
  expects (the E[N] diagnostic).
- **Training data.** Seven IDSSE matches from one provider (five share a home team) plus World Cup
  broadcast tracking. BAR-ATH's teams and league are in no training set.
- **Selection on five matches.** The decode gains are measured on the same five matches in every
  pair, so 3/3 pairs is not independent evidence for them; Metrica is the independent check, and it
  agrees. Seed noise on one pair (±0.004) is close to the per-pair Metrica guard. The sealed exam
  suggests the development numbers are about 0.02 optimistic. The organisers put the noise band of
  spellF1 on one full match at about ±0.04, wider than the gaps between our own options.
- **Smoothed player error of 1 m or more is a cliff** (S1, an earlier network): 0.72 → 0.66 → 0.50 at
  1 m and 2 m. Training covers player noise up to 0.5 m; Metrica's real tracking scores close to
  clean IDSSE, which suggests real error is inside that range.
- **Training is not bit-reproducible across machines.** GPU training with mixed precision differs
  between devices, so a retrain gives slightly different weights (Section 7). The build is
  byte-reproducible because the weights are committed.
- **Run time.** The build takes 6 to 12 minutes on a fast laptop CPU and up to 50 minutes on a 15 W
  one (Section 7), above our own 5-minute target; the competition sets no limit.
- **With more time** we would add labelled full-pitch matches from other providers and leagues,
  look for a contact cue that survives a smoothed ball, and test a ball detector on footage with
  annotated airborne balls.

## 7. Reproduction

**Rebuilding `AAAAG.csv`** (CPython 3.13; 3.12 and 3.14 give the same bytes; no GPU, no torch):

```bash
git clone <this repository> aaaag && cd aaaag
python scripts/bootstrap.py               # creates .venv with the exact pins of requirements-lock.txt
export MTAH_DATA_DIR=/path/to/bundle      # the organisers' bundle, tactical/ and broadcast/ as delivered
.venv/bin/python -m aaaag submit          # Windows: .venv\Scripts\python -m aaaag submit
```

**What `submit` does:**
1. It runs the pipeline named in `configs/submission.yaml` (`possession_learned_joint_v15.yaml`),
   with no cache, and writes `outputs/submission/AAAAG.csv`.
2. It runs the organisers' `validate_submission.py` on the file.
3. It prints the public-window smoke test (PASS/FAIL only), the label-free diagnostics and the
   file's SHA-256, and compares it with `expected_sha256`.

**The submitted file's SHA-256 is**
`c57e6e7762864ca736cc339bdc49af1b399c9c259dfcfb93eca98e67d8e30cf2`.

**Why it is deterministic:**
- **Inputs.** The build needs only the data bundle and the repository. The three networks
  (`models/v15sggpp_all7_seed{2,3,4}.json`) are committed and pinned by SHA-256 prefix in the
  pipeline YAML (`cd80e3c0118b300c`, `c473e3cadde35ecd`, `d21efe73f7dbc32d`). Every other value is
  in that YAML, in metres, seconds and probabilities.
- **No randomness, no wall-clock dependence.** Inference is float64 numpy. It differs from the torch
  float32 pass by about 7e-7 in the posterior, with no label flips on a 146,000-frame external match.
  Moving every float32 evidence value by one ulp changes no label (IDSSE J03WMX, three trials).
- **Threads and machines.** Three cold builds of v15 on one laptop gave the hash above, at 1 and at
  8 threads. A second laptop, building from a clean checkout, gave the same hash at 1 and at 8
  threads. An earlier champion gave the same bytes on Linux x86 and ARM, macOS ARM and Windows, with
  AVX2 and AVX-512 CPUs.
- **Build time.** About 6 to 12 minutes on a 12-core laptop CPU (AMD Ryzen AI 9 HX 370; 354 s,
  642 s and 742 s, the spread being other load on the machine), and 25 minutes at 8 threads or
  50 minutes at 1 thread on a 15 W laptop CPU. The six network passes take almost all of it: the
  decoder and post-process take about a second. Doubling `d_model` from 64 to 128 (v14) roughly
  doubled the time.

**Reproducing the weights** (a separate environment with torch, `requirements-dl.txt`; external data
only; one laptop GPU). For each seed s in 2, 3, 4:

```bash
export AAAAG_EXTERNAL_DIR=/path/to/external AAAAG_CACHE_DIR=/path/to/cache PYTHONPATH=src
python -m aaaag.sources.idsse                     # all 7 matches, pinned revision, sha256-checked
# PFF FC World Cup 2022: download it from PFF FC into $AAAAG_EXTERNAL_DIR/pff/ (not redistributed)
E=experiments/ameya; T="python analysis/scripts/train_after_exam.py --confirm-exam-spent"
# step 1: PFF pretraining (44 train-split games)
python -m aaaag.ml train --run $E/e99/e99wa.yaml --holdout none --extra-only --seed $((s+200)) \
  --out runs/e99wa_seed$s
# step 2 and step 3W (set init_from in each YAML to the previous step's final.pt)
$T --run $E/e99/e99wb_prod_s$s.yaml --seed $s         --out runs/e99wb_prod7_seed$s
$T --run $E/e99/e99wc_prod_s$s.yaml --seed $((s+100)) --out runs/e99w_prod7_seed$s
# step 3G, three seeds, each from runs/e99wb_prod7_seed$s
$T --run $E/v15/v15a25g25_prod_s$s.yaml --seed $((s+900))  --out runs/v15a25g25_prod7_seed$s
$T --run $E/v15/e141g_prod_s$s.yaml     --seed $((s+1100)) --out runs/e141g_prod7_seed$s
$T --run $E/v15/e144g_prod_s$s.yaml     --seed $((s+1700)) --out runs/e144g_prod7_seed$s
# step 4: the soup with analysis/scripts/e129_soup.py (exact order: experiments/ameya/v15/README.md)
python -m aaaag.ml.export --weights runs/v15sggpp_prod7_seed$s/final.pt \
  --out models/v15sggpp_all7_seed$s.json --encoding base64
```

The soup is built in three averaging steps (½ W + ½ G, then with G′, then with G″), which equal
½ W + ⅙ (G + G′ + G″). A retrain on another GPU gives networks that differ in the last bits, so it
reproduces the scores, not the bytes. The submission's byte-exactness rests on the committed JSON
weights.

**Reproducing the validation** (fold models, then the grids):

```bash
# W fold models for fold seed s (2, 3, 4, 5, 12, 13): step 1 as above, then steps 2 and 3 per fold
python -m aaaag.ml lomo --run $E/e99/e99wb_s$s.yaml --folds all --seeds 0 --pipeline joint \
  --seed $s --out runs/e99wb_seed$s
python -m aaaag.ml lomo --run $E/e99/e99wc_s$s.yaml --folds all --seeds 0 --pipeline joint \
  --seed $((s+100)) --out runs/e99w_seed$s
# fold soups: analysis/scripts/e129_soup.py --a RUN_A --b RUN_B --w 0.5 --out RUN_SOUP
# grids: clean, S8@8, hr, dv and mix levels, Metrica, against the first variant
python analysis/scripts/postprocess_grid.py --config configs/pipelines/possession_learned_joint_v15.yaml \
  --runs runs/e99w_seed2 runs/e99w_seed3 --variants VARIANTS.yaml --seeds 5 6 7 8 9 --s8-seeds 5 6 \
  --levels 8 --hidden-restarts 2 --dead-visible --mix 0.5 0.9 --metrica
python analysis/scripts/e118_airborne_ball.py stress --config configs/pipelines/possession_learned_joint_v15.yaml \
  --runs runs/e99w_seed2 runs/e99w_seed3 --cz 18 --out rec18.json            # rec18
python analysis/scripts/pff_prod.py configs/pipelines/possession_learned_joint_v14.yaml \
  configs/pipelines/possession_learned_joint_v15.yaml --labels control --out pff.json   # PFF test split
python analysis/scripts/e125_lag_table.py                                    # truths against the delay table
```

**Where the evidence lives:**
- the declarations and results of every experiment: issue #86;
- the champion decision records: `docs/decisions/*champion-v*.md`;
- the production recipes: `experiments/ameya/e99/README.md` and `experiments/ameya/v15/README.md`;
- experiment records: `experiments/`; analyses: `analysis/findings/`.

## 8. Credits and licences

- **PFF FC.** The shipped networks are pretrained on PFF FC's 2022 World Cup tracking, metadata
  and event data (44 train-split games). We thank PFF FC. The files are used under PFF FC's terms
  and are not redistributed.
- **IDSSE, DFL / Sportec Solutions.** Bassek et al. 2025, *Scientific Data*;
  doi:10.6084/m9.figshare.28196177; CC BY 4.0. Every shipped network is trained on these seven
  matches.
- **Metrica Sports sample data** (github.com/metrica-sports/sample-data). Used to fit the
  degradation simulator and as a validation guard; no network is trained on it. We thank Metrica
  Sports.
- **SkillCorner open data** (github.com/SkillCorner/opendata, MIT). Used in one dropped training
  arm and a label comparison; it is in no shipped network.
- **TeamTrack** video clips: used only to train the closed ball detector (E150); nothing from it
  ships.
- **The organisers' scorer** (`vendor/mtah2027/evaluation/possession_metrics.py`, vendored
  byte-exact) is the reference metric; the evaluation code calls it through `aaaag.evaluation`.
- **Ideas from other public entries** were tested where noted (an A–B–A truth family, restart-team
  states, whistle detection). No code was copied.
- **Methods** we build on: model soups and WiSE-FT (Wortsman et al., 2022), Model Stock (Jang et al.,
  2024), TrackNet-style ball detection.
