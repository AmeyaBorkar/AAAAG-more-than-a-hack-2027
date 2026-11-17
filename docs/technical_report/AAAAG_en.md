---
lang: en
anchor: en
outline: English
header: AAAAG · Frame-level ball possession from automatic tracking
footer: AAAAG · More than a Hack 2027 · Page {page} of {total}
title: Frame-level ball possession from automatic tracking
meta: Team AAAAG: Ameya Borkar, Arnav, Bakshi, Ayush · 17 November 2026
meta: More than a Hack 2027, Phase 1
meta: Repository: REPO_URL
crosslink: es | Versión en español: página {page:es}
---

::: summary
**Summary**

1. **What it does.** Small neural networks read the automatic tracking frame by frame. A decoder turns their output into stoppages and whole possessions.
2. **How well.** On Metrica's three public games, against labels our code derives from their events, it scores spellF1 **0.788**; the organisers' baseline algorithm scores **0.407**. We never trained on these games, but they shaped our design, so our score is optimistic and not comparable with the leaderboard.
3. **Key idea.** Evidence learned from labelled external matches, decoded with a prior on possession lengths; learned dead-ball detection alone added +0.060 on Metrica.
4. **Main limitation.** The truth's convention for short possessions is unknown; across five plausible conventions our score moves by about ±0.05.
5. **Reproduction.** One command, `python -m aaaag submit`, rebuilds the submitted file byte for byte in 6 to 12 minutes on a laptop CPU.
:::

## 1. Learned evidence and structured decoding beat rules

**Task.** Every graded frame needs H, A or D, yet the ball is missing on 41% of them (organisers). The ranking metric, spellF1, matches whole possessions one to one by overlap in time; a perfect labelling one second late scores only 0.6539. So the score rests on finding each possession and its edges.

**Pipeline.** We scale the organisers' 0–1 coordinates to metres with the pitch size in their games.py, 105.3 × 68.0 m. Networks give each frame P(H, A, D), the probabilities of home, away and dead ball. A decoder turns them into possessions; three fixed rules finish the labels (Figure 1).

::: figure
Tracking | players and ball, in metres
Networks | 3 × 2 views | P(H, A, D)
Dead-ball gate | P(D) ≥ 0.30 is a stoppage
Decoder | H or A per live stretch
Fixed rules | kick-off, touch | 1.5 s hold
group: 1 | input
group: 2 | learned
group: 3-4 | structured decoding
group: 5 | fixed
caption: **Figure 1.** Learned evidence first, fixed rules last; the decoder sees each half whole.
:::

**Network.** Three networks of about 0.8 million parameters each read the match twice, once with the pitch mirrored across its long axis. Each frame becomes 23 tokens: each team's 11 players nearest the ball, and the ball. Missing players become empty tokens; track identities are never used. Attention across tokens, then a temporal convolution, gives ±5.1 s of context.

**Inputs and outputs.** Inputs are positions, velocities, ball visibility and the team that last held the ball. Besides P(H, A, D), it names the player on the ball (possessor output) and scores a change of possession (change output).

**Decision 1: learned evidence.** It replaced hand-written rules because it won clearly (Table 3). On Metrica's three public games (Section 2), its dead-ball output gained **+0.060 spellF1**. On IDSSE, seven German matches with provider labels on every frame, its team call gained **+0.073**.

**Decision 2: a possession decoder.** The metric punishes fragments, so the decoder enforces plausible lengths. A dead-ball gate marks stoppages; a semi-Markov segmentation then splits live play into H and A. It beat a ±0.5 s majority vote by **+0.018** on 5 of 5 IDSSE matches.

**Decoder scores.** Each possession scores its team evidence plus the log-probability of its length. The prior is lognormal in seconds, fitted on the five IDSSE development matches (median 3.7 s). Each change adds 1.5 times the change output's log-odds (capped at ±6.9): an unseen hand-over costs up to 10 log units.

**Decision 3: specialists, then averaging.** Two faults each cost over 0.05: a ball visible in a stoppage, and a ball in the air. One camera places an airborne ball where its line of sight meets the grass. Training on visible dead balls from scratch cost 0.010 on normal input, so we fine-tune copies of one network. A base copy learns visible dead balls; three specialists also learn airborne balls.

**Weight averaging.** Each shipped network averages the weights of the base copy (½) and the three specialists (⅙ each), which share one parent (Wortsman et al., 2022). The parent was pretrained on PFF FC tracking, then trained on IDSSE.

## 2. All labels come from other matches

**Excluded.** No label comes from the provided match, and no parameter is fitted to its labels, video or specific moments. Its public five minutes run only through a smoke test that prints PASS or FAIL.

Table: **Table 1.** Every labelled dataset, its labeller, amount and role.
| Dataset (licence) | Labelled by | Amount | Role |
|---|---|---|---|
| [IDSSE](https://doi.org/10.6084/m9.figshare.28196177) (CC BY 4.0) | DFL's provider, per frame | 7 matches, about 145,000 frames each | Training; 5 for design, 2 sealed |
| [PFF FC World Cup 2022](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) (PFF terms) | Our code, from PFF FC's events | 44 + 16 games | Pretraining; 16 held out |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) (credit asked) | Our code, from hand-tagged events | 3 games | Simulator fit; testing |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) (MIT) | SkillCorner, per frame | 20 matches | Training: −0.033 on IDSSE, dropped |
Note: We thank PFF FC (data used under its terms, not redistributed), DFL and Sportec Solutions for IDSSE (Bassek et al., 2025), Metrica Sports, SkillCorner and the organisers. Nothing under a non-commercial or GPL licence enters the pipeline.

**Event labels.** Our code turns Metrica's and PFF FC's events into frame labels, which are therefore approximate. Metrica: a stoppage runs from the end of the event before a set piece to the set piece. Otherwise the latest pass, carry, recovery, shot or ball loss names the team. PFF FC: an out-of-play event starts a stoppage, the next on-ball event or kick-off ends it; only on-ball events change the team.

**Conventions.** A convention is the rule for which short runs of one team the truth keeps. We test five: every touch, at least 1 s, at least 2 s, 1.5 s or a pass, and interruptions under 3 s ignored. The time and pass rules follow CHALLENGE.md's proxies: holding the ball, completing passes.

**Degradation.** IDSSE's ball is never missing, so a simulator fitted on the three Metrica games degrades it. It hides the ball through stoppages, opens live gaps at Metrica's rates, drops 5–20% of player positions and adds 0.10–0.50 m of noise. Its ball is missing on 44.5% of played frames, against Metrica's 37.3%.

**Validation.** Every experiment was declared, with its success rule, before it ran. Each IDSSE match is scored by networks trained without it. Metrica was never trained on, but the simulator copies its gap rates and it confirmed the dead-ball output and the 1.5 s hold. So its scores are optimistic.

## 3. It roughly doubles the baseline on external games

**Results.** On Metrica's three games we score spellF1 0.788 and the organisers' baseline algorithm 0.407, on the same labels. Their published 0.2433 is that baseline on the provided match, so the figures are not comparable. We lead the baseline on all four benchmarks by 0.28 to 0.39 (Table 2), about seven times the organisers' ±0.04 noise for one match.

**Benchmarks.** IDSSE has simulated ball gaps; we average three network pairs, each pair two networks from different seeds, averaged as in production. Metrica is also scored with the ball smoothed over 8 frames, since providers smooth the ball differently. PFF FC's 16 held-out games are labelled from events.

Table: **Table 2.** spellF1, mean (worst game), on games no system here trained on.
| System | IDSSE | Metrica | Metrica, smoothed | PFF FC |
|---|---:|---:|---:|---:|
| Organisers' baseline | 0.348 | 0.407 (0.391) | 0.384 (0.364) | 0.354 (0.285) |
| Hand-written rules | 0.516 | 0.498 (0.458) | 0.371 (0.339) | 0.359 (0.310) |
| **Our system** | **0.742** | **0.788 (0.771)** | **0.740 (0.724)** | **0.632 (0.568)** |
Note: 5, 3, 3 and 16 games; IDSSE per-match values were not kept for our system. Baseline: nearest player, ±2 s vote. Rules: last team within 1 m of the ball for 3 frames, a hand-built ball-in-play model, a ±0.5 s vote. IDSSE also chose our design: about 0.03 high (L4).

Table: **Table 3.** What each idea adds, in spellF1, against a simpler alternative.
| Idea | Simpler alternative | Benchmark | Gain |
|---|---|---|---:|
| Learned team call | Last-controller rule | IDSSE, 5 matches | +0.073 |
| Learned dead-ball call | Hand-built ball-in-play model | Metrica, 3 games | +0.060 |
| Semi-Markov decoder | Majority vote over ±0.5 s | IDSSE, 5 matches | +0.018 |
| Weight averaging | Base copy alone | IDSSE, airborne ball | +0.046 |
Note: Gains overlap; each was measured when its idea was added. The team call won on 4 of 5 matches and tied on the one whose teams were in no training match. The decoder won on 5 of 5. Averaging won on 3 of 3 network pairs; no other mean score fell by over 0.003.

## 4. Tracking is our only input

**Inputs.** We kept only the tactical tracking and the organisers' metadata (Table 4). Every other asset was redundant or failed a pre-declared test.

Table: **Table 4.** Assets used and dropped.
| Asset | Status | Use, or the test it failed |
|---|---|---|
| Tactical tracking | Used | Scaled to metres; team per track; no identities |
| Metadata, games.py | Used | Pitch size, graded halves, kick-off frames (R6) |
| Smart Tagging | Dropped | Prior from simulated tags: +0.003, bar +0.005 |
| Video and audio | Dropped | Ball and whistle detectors missed declared bars |
| Homography | Not needed | The tracking is already on the pitch |
| Broadcast feed | Not used | Not synchronised (organisers) |
| PathCRF model | Not used | Trained on IDSSE ([Kim et al., 2026](https://arxiv.org/abs/2602.12080)): a leak |
Note: Tags were simulated on IDSSE; the detectors ran on the provided video, so no figures.

## 5. Simulated faults teach noisy tracking

**Missing ball.** A rule reading a missing ball as dead calls live gaps dead. We hide the ball independently of the labels in half of each training batch, breaking that shortcut. In long simulated gaps the network calls 1–2% of live play dead, the hand-built model 22–29%.

Table: **Table 5.** Failures of automatic tracking and how the pipeline handles them.
| Failure | Handling | Effect (spellF1) |
|---|---|---|
| Ball hidden at restarts | Hidden around restarts in training | +0.013 |
| Ball visible in a stoppage | Fine-tune on visible dead balls | +0.057 to +0.062 |
| Airborne ball on the grass | Specialists, camera 15–25 m high | +0.046 at 18 m |
| Smoothed ball track | Smoothing; possessor output | +0.062 at 8 frames |
| Fragmented tracks | Nearest players, no identities | −0.0004: no cost |
| Fragments on the wrong team | None found | −0.009 |
Note: Gains compare training with and without the handling, costs clean and faulty input, on IDSSE. Airborne: IDSSE's real ball height seen by an 18 m camera. The organisers report 46 track slots for 22 players.

## 6. Possession changes only on sustained evidence

**R1. Dead ball.** The network's dead-ball output, trained on the provider's referee-driven ball status, marks a stoppage where P(D) ≥ 0.30. Stoppages last at least 2 s, live stretches at least 1 s. Against 0.40 the gain is small, +0.003, but it held on all four network pairs and every clean convention, so we kept 0.30.

**R2. Restarts.** No rule names the restarting team: the network sees each restart within ±5.1 s. The first possession after a stoppage is exempt from R5: a restart has a clear possessor (CHALLENGE.md). A restarting-team rule cost 0.014.

**R3. Loose balls.** A loose or contested ball stays with the last team (CHALLENGE.md). The decoder changes team only where the team evidence and the change output agree, so a contested ball does not flip possession. The network knows which team last had a player within 1 m of the ball for 3 frames.

**R4. Touches without control.** Deflections and aerial duels won without control change nothing. A real change moves to the gaining team's nearest touch within ±0.25 s; a touch is a sharp change of ball velocity beside a player.

**R5. Short possessions.** A possession under 1.5 s survives only if two teammates each have the ball within 1 m during it, a completed pass. Otherwise it takes the previous possession's team, shortest first, merging with its neighbours. On Metrica, with an earlier network version, the 1.5 s hold had the best mean over the five conventions: 0.793 against 0.741 without it.

**R6. Ungraded frames.** X goes outside the organisers' graded halves (games.py), and D on graded frames before a half's kick-off, as their FAQ notes. Every graded frame gets H, A or D, because spellF1 replaces an X with the previous label.

## 7. The labelling convention is the largest risk

**L1. Convention, about ±0.05.** How the truth treats short possessions matters more than any model change. On IDSSE, the best decoder for the touch, 1 s and 2 s conventions scores 0.78 to 0.855. If the truth flips on every touch, our 1.5 s hold costs about 0.06; we accept that to limit the worst case.

**L2. Simulated stresses.** The airborne-ball gain comes from a simulated 18 m camera; a real one may differ. A smoothed ball still costs about 0.05 on Metrica (Table 2). With more time: footage with annotated airborne balls.

**L3. Hidden ball.** In long ball gaps, a players-only network calls the team as well as ours; training on such gaps did not help. Wrong-team stretches in long possessions remain the main error. With more time: a touch cue robust to smoothing.

**L4. Narrow data.** All seven IDSSE matches share one provider; five share a home team. Two sealed IDSSE matches gave an earlier version 0.710, against 0.736 in development (Table 2's measure). With more time: other leagues.

## 8. One command rebuilds the exact file

**Steps.** Clone REPO_URL and run three commands with CPython 3.13; 3.12 and 3.14 give the same bytes. The trained networks are in models/, pinned by SHA-256: no GPU, no training.

```
python scripts/bootstrap.py
export MTAH_DATA_DIR=/path/to/bundle
.venv/bin/python -m aaaag submit
# Windows PowerShell: the last two lines become
$env:MTAH_DATA_DIR = "C:\path\to\bundle"
.venv\Scripts\python -m aaaag submit
```

1. **Environment.** A .venv pinned by requirements-lock.txt.
2. **Data.** MTAH_DATA_DIR is the organisers' bundle, as delivered.
3. **Build.** submit writes and validates outputs/submission/AAAAG.csv.
4. **Run time.** About **6 to 12 minutes** on a laptop CPU.
5. **Checksum.** SHA-256, matched on a second laptop at 1 and 8 threads: c57e6e7762864ca736cc339bdc49af1b399c9c259dfcfb93eca98e67d8e30cf2

**Retraining.** REPORT.md, section 7, lists the commands: torch, external data, about 2 hours per fine-tune on a 12 GB laptop GPU. A retrain reproduces the scores, not the bytes.

**References.** Bassek et al. (2025), IDSSE, Scientific Data. Kim et al. (2026), PathCRF, [arXiv:2602.12080](https://arxiv.org/abs/2602.12080). Wortsman et al. (2022), model soups, ICML.
