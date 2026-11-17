"""Temporal decoders, registered in DECODERS: (T, 3) probabilities -> (T,) int8 labels.

This is where spellF1 is won or lost: a decoder decides how many possessions there are and where
they start and end. Decoders must work period by period (never smooth across half-time), must be
non-causal (use the future as well as the past -- causal filters add lag, and 8 frames of lag
costs ~10% of spellF1), and must write X outside the periods unless told otherwise.
"""
