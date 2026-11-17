# External data: labelled OTHER matches

The rules allow any amount of data about **other** games, including human labels. It must be
accessible to the organisers, and the technical document must state its source, licence, who
labelled it and how much was used. Downloads go to `external_data/` (gitignored; `$AAAAG_EXTERNAL_DIR`).
Never commit them. Each dataset becomes a registered source in `src/aaaag/sources/`, emitting a
`MatchBundle` **with labels**, resampled to 25 fps and in our contract units.

Status as researched on 2026-10-28. Lines marked *unverified* need confirmation by the data lane.

| Dataset | Matches | Rate | Labels → H/A/D | Licence | Access | kloppy |
|---|---|---|---|---|---|---|
| **DFL / Sportec IDSSE** (Bassek et al. 2025, *Scientific Data*) | 7 (2 Bundesliga, 5 2. Bundesliga, 2022/23) | 25 Hz (TRACAB) | Per frame: `BallStatus` (1 alive / 0 dead) and `BallPossession` (1 home / 2 away). **D where dead, else the owning team**: the same mechanism as our ground truth. | CC BY 4.0 | figshare doi:10.6084/m9.figshare.28196177 (kloppy 3.18 download blocked); Hugging Face `pysport/idsse-data` (used, pinned); about 2.6 GB of positions | `sportec.load_open_tracking_data(match_id=...)` since 3.16. Frames carry `ball_state` and `ball_owning_team`. The download moved to Hugging Face in 3.19. |
| **Metrica Sports sample data** | 3 (anonymised) | 25 fps | Built from events: RECOVERY / BALL LOST change the team; BALL OUT or a FAULT to the next SET PIECE is dead; the set-piece team owns after the restart. *Approximate.* | no formal licence; credit Metrica | github.com/metrica-sports/sample-data | `metrica.load_open_data(match_id=1..3)`, plus events. Game 3 is EPTS with **11 layout blocks** (our parser handles them). |
| **SkillCorner open data** | 20 (A-League 2024/25; README says 10) | 10 fps (broadcast) | Per frame `possession.group`. Dead ball is a heuristic: null group means dead (*unverified* as a referee signal). | MIT; credit SkillCorner | github.com/SkillCorner/opendata (Git LFS) | `skillcorner.load_open_data(match_id=...)` (≥ 3.18) |
| **PFF FC World Cup 2022** | 64 | 29.97 fps broadcast, resampled to 25 | Event-derived from `game_event` / `home_ball` (approximate, like Metrica). **Training on the 48 train-split games is allowed. The 16 test games are never trained on.** The fixed test split is every 4th id (`SPLIT_RULE` in `sources/pff.py`). | PFF terms. Ameya authorised training on 11 Nov (decision record `docs/decisions/2026-11-12-pff-world-cup-data-for-training.md` in the team repository). Never redistributed; credit PFF FC | blog.fc.pff.com enhanced-2022-world-cup-dataset | own reader `aaaag.sources.pff` (kloppy's loader is not used) |

IDSSE match ids:
- J03WMX (Köln v Bayern)
- J03WN1 (Bochum v Leverkusen)
- J03WPY, J03WOH, J03WQQ, J03WOY, J03WR9 (Düsseldorf home games)

### IDSSE adapter (`idsse` source, #17)
- **Fetch:** `python -m aaaag.sources.idsse [MATCH_ID ...]` (all seven when no id is given). It
  downloads the matchinformation and positions files from the Hugging Face mirror at a pinned
  revision into `$AAAAG_EXTERNAL_DIR/idsse/` and checks each file's sha256. It needs no kloppy:
  kloppy 3.18's figshare download is blocked by a bot challenge (#14).
- **Load:** `{name: idsse, params: {match_id: J03WMX}}` in a suite. It parses the positions XML
  directly (about 17 s for a match the first time, under 0.1 s from `cache/<match_id>/`).
- **Units:** the positions are already in pitch-centred metres. Speed is converted from km/h to
  m/s. The halves are placed back to back with a 25-frame X gap between them.
- **Labels:** D where `BallStatus` = 0, otherwise H/A from `BallPossession`. An alive frame with no
  owner is labelled D and counted in `extras["provenance"]`.
- **Checked on J03WMX:**
  - labels and ball positions match kloppy's reader exactly on the first 3,000 frames;
  - the dead share is 41.5%, as in #14;
  - there are no alive frames without an owner.
- **The ball is never missing** and freezes during stoppages. Degrade it before using a match for
  selection.
- **Event touches:** `python -m aaaag.sources.idsse_events --fetch` downloads the DFL
  `events_raw` files (under 1 MB each) from the same pinned revision. The mirror keeps them out of
  LFS, so each is pinned by its size and git blob id from the Hub listing. Then
  `aaaag.sources.idsse_events.touch_events(match_id)` gives the PFF contract
  (`pff_events.touch_events`) on the `idsse` frame axis. The default `sync="elastic"` follows the
  drift of the DFL clock inside a half and assigns the on-ball events to tracked ball contacts in
  their logged order (ELASTIC/ETSY style); `sync="lag"` keeps the first version, one clock lag
  per half fitted to kinematic touches. `--probe` prints aggregates only, for both. Training reads it only when
  `train.touch_source: events` or `train.event_switch_weight > 0`.

The data lane fixes the dev/exam split in a decision record **before** anyone sees results
(`docs/VALIDATION.md` in the team repository).

### PFF adapter (`pff` source)
- **Training on the train split only.** `game_ids("test")` is index `% 4 == 3` after sorting ids as integers (16 of 64). That split was fixed before any PFF result and is never trained, tuned or selected on: it only checks a candidate against the champion. `analysis/scripts/pff_prod.py` defaults to it and to the raw ball.
- **Licence:** PFF terms. Ameya checked them and authorised training on 11 Nov; see the decision record `docs/decisions/2026-11-12-pff-world-cup-data-for-training.md` in the team repository. Files are never committed or redistributed, issues and LLM contexts get aggregates only, and PFF FC is credited if its data contributes to a submitted model.
- **Files:** `$AAAAG_EXTERNAL_DIR/pff/` as PFF delivers them (tracking `{id}.jsonl.bz2`, metadata, rosters, optional `Event Data/{id}.json`). Not in git.
- **Labels** are event-derived and approximate. `ballsSmoothed` is built from those same events, so variant `smoothed` leaks the labels. Score `raw`.
- **Ball height (B3, #86):** the bundle keeps no z. `aaaag.sources.pff_ballz.ball_z(game_id, bundle)` reads the raw ball's z on the bundle's 25 fps axis. It is cached under `cache/pff_ballz/` and versioned by `pff.BALL_Z_VERSION`, so the bundle caches are unchanged. It covers the train split only; test ids raise. `train.airborne_extra` uses it for E119's projection on PFF extras. `python -m aaaag.sources.pff_ballz --probe` is the stage-0a probe (aggregates only).
- **Probe:** `python -m aaaag.sources.pff --probe` (aggregates only). `--list` prints ids.
- **Event touches:** `aaaag.sources.pff_events.touch_events(game_id)` gives pass, reception, shot, tackle, clearance, other_touch, set_piece, foul and out events on the resampled 25 fps index, from `Event Data/{id}.json` aligned through the tracking rows' event ids. Train split only (test ids raise). `python -m aaaag.sources.pff --probe-events` prints aggregate counts. Nothing consumes it yet.

## Weaker options (noted, not planned)
- **SoccerTrack v2**: 10 university matches, panoramic video, actions without a team.
- **StatsBomb 360**: event freeze frames, not continuous in time.
- **Google Research Football**: a simulator with `ball_owned_team`; unrealistic motion.
- **SoccerNet-GSR**: 30 s clips.
- **Last Row**: 19 goal clips.

## Adapter checklist (each source)
- [ ] Coordinates in centred metres, with the pitch size from the source's metadata.
- [ ] 25 fps on the frame axis (resample on time). Periods match the source's halves.
- [ ] Labels as int8 codes: D where dead, H/A from the owning team, X outside the periods.
- [ ] Team-level only: drop player identities, or keep them only for the degradation simulator.
- [ ] Deterministic. Cached under `cache/<match_id>/`. Give the registration an `example` only if
      it runs offline (e.g. a tiny fixture); otherwise leave it empty (the contract tests skip it)
      and add a local test marked `data`.
- [ ] Provenance recorded for the technical document: dataset, version or DOI, licence, labeller,
      matches used.
