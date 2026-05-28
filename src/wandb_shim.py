"""Minimal wandb-API shim backed by trackio.

The original codebase uses a handful of wandb features that trackio doesn't
expose directly: `wandb.summary[k] = v`, `wandb.Artifact`, `wandb.log_artifact`,
`wandb.Image`, and `wandb.run.name`. We forward `init`/`log`/`finish`/`config`
to trackio and stub out the rest so probing scripts run unmodified.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import trackio as _trackio


def _to_jsonable(v: Any) -> Any:
    """trackio.log serializes via json.dumps; numpy scalars/arrays explode there."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


class _Summary(dict):
    """`wandb.summary[k] = v` — we just collect into a dict and forward to trackio.log."""

    def __setitem__(self, key: str, value: Any) -> None:
        coerced = _to_jsonable(value)
        super().__setitem__(key, coerced)
        try:
            _trackio.log({key: coerced})
        except Exception:
            pass


class _Image:
    """`wandb.Image("path.png")` — trackio.log natively accepts file paths in many
    cases, but to stay safe we just record the path."""

    def __init__(self, path_or_obj: Any, *args: Any, **kwargs: Any) -> None:
        self.path = path_or_obj


class _Artifact:
    """No-op artifact: we keep added file paths locally; nothing is uploaded."""

    def __init__(self, name: str, type: str = "dataset", **kwargs: Any) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, local_path: str, name: str | None = None) -> None:
        self.files.append(local_path)


class _Run:
    """Stand-in for `wandb.run`. We only need `.name`."""

    def __init__(self, name: str) -> None:
        self.name = name


class _WandbShim:
    def __init__(self) -> None:
        self.summary = _Summary()
        self.run = _Run(name=f"run-{int(time.time())}")
        self.Image = _Image
        self.Artifact = _Artifact
        # config is set by init(); start with a dummy namespace
        self.config = type("Config", (), {})()

    def init(self, *args: Any, **kwargs: Any) -> Any:
        # trackio rejects `entity`, `mode`, `tags`, etc. — strip them.
        allowed = {"project", "name", "config", "resume", "id", "group", "dir", "settings"}
        clean = {k: v for k, v in kwargs.items() if k in allowed}
        # Reset summary for a new run.
        self.summary = _Summary()
        name = clean.get("name") or f"run-{int(time.time())}"
        self.run = _Run(name=name)
        # Default project name if caller didn't supply one
        clean.setdefault("project", os.environ.get("TRACKIO_PROJECT", "llmsknow"))
        try:
            r = _trackio.init(**clean)
            # trackio Run exposes a `name` attr; reuse if available
            run_name = getattr(r, "name", None) or name
            self.run = _Run(name=run_name)
            self.config = getattr(r, "config", clean.get("config") or type("Config", (), {})())
        except Exception as e:
            print(f"[wandb_shim] trackio.init failed ({e}); running offline.")
        return self.run

    def log(self, data: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        try:
            _trackio.log(_to_jsonable(data))
        except Exception:
            pass

    def log_artifact(self, artifact: _Artifact, *args: Any, **kwargs: Any) -> None:
        # No remote storage — files remain on local disk where the script wrote them.
        pass

    def finish(self, *args: Any, **kwargs: Any) -> None:
        try:
            _trackio.finish()
        except Exception:
            pass


wandb = _WandbShim()
