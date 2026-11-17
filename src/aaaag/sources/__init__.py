"""Match sources: functions registered in SOURCES that return a MatchBundle.

Add a source (e.g. an external dataset adapter) by adding a module here:

    @SOURCES.register("dfl-open", version="1")
    def load_dfl_open(*, match_id: str, degrade: dict | None = None) -> MatchBundle: ...

Rules: external sources must attach `labels` (they are what validation scores against); the
competition source never may (contracts.MatchBundle).
"""
