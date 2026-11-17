"""Ball-state evidence (lane: ballstate) -- is the ball alive or dead, and who restarts?

Producers here mostly speak to the D column. A producer that only knows alive/dead returns
Evidence.from_factors(p_dead, 0.5) and leaves the team split to the possession producers.

- `ballstate` (model.py): the lane's main output. Per-frame cues (cues.py) -- collective player
  slowdown, ball off the pitch, ball at rest or in flight, ball at a restart spot, kick-off
  formations (kickoff.py) -- decoded per period by a two-state alive/dead HMM (hmm.py), plus the
  team that restarts after each stoppage.
- `ballstate_kickoff`: kick-off set-ups only (D before, kicking team after), NaN elsewhere.
- `ballstate_off_pitch`: D where the ball is seen off the pitch, NaN elsewhere.

A missing ball is never dead-ball evidence here: ball cues are NaN where the ball is unseen.
"""
