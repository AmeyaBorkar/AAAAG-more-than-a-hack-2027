"""Frame-level evidence producers, registered in EVIDENCE.

Each returns an `Evidence`: (T, 3) float32 probabilities over (H, A, D), NaN rows where it has no
opinion. Abstaining honestly (NaN) is better than guessing: fusion and decoding can only weigh
what they are told is uncertain. Document in the docstring which signals the producer uses and
where it fails.
"""
