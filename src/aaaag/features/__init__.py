"""Shared, vectorised feature computations used by more than one evidence producer.

Pure functions of a Tracking (or MatchBundle) returning (T,) or (T, k) arrays: ball kinematics,
nearest-slot distances per team, team centroids and spread, attacking direction per period, etc.
Put a computation here as soon as a second producer needs it; add new modules rather than editing
someone else's, and keep every function O(T * S) with no Python loop over frames.
"""
