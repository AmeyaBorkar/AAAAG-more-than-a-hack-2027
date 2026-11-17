"""Post-processing of decoded labels: label -> label steps that may read the match bundle.

A step is registered with `@POSTPROCESS.register(name, version=..., example=...)` and has the
signature `fn(labels, bundle, **params) -> labels` (int8, same length, X outside the periods).
A pipeline config lists them after the decoder under `postprocess:`. They exist for corrections
that need the tracking (a decoder only sees the fused posterior), such as `touch_snap`.
"""
