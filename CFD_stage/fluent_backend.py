"""Explicit adapter for ``Fish_UDF_Finial.c``; no invented case setup.

Real sessions require an ``initializer(solver, work_dir)`` callback on every
reset. It must load the supplied case/data, compile and load the UDF for the
installed Fluent version, establish RP variables ``at``/``tc``, attach correct
grid-motion, zone-motion and execute-at-end hooks, reset ALL UDF globals
(including the 40-entry action arrays), and restore flow time to zero. It must
return the actual initial 7D state. The archive has no verified journal for this.

The solver's working directory must be ``work_dir`` and visible locally. This
adapter launches local Fluent with ``cwd=work_dir`` if no solver is injected;
an injected solver's working directory is the caller's responsibility. Python's
process cwd never changes. Remote-file transfer is not implicitly performed.

Output.txt contains time, pose, FORCES, moment and direction. Velocities come
from the existing ``save_fish_state`` hook, never from those force columns.
This creates one UDF snapshot per CFD step and adds disk I/O; the integration
path has not been benchmarked against a real Fluent session.

API references:
https://fluent.docs.pyansys.com/version/stable/api/launcher/launcher.html
https://fluent.docs.pyansys.com/version/stable/user_guide/legacy/tui.html
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np


class FluentConfigurationError(ValueError):
    """Required real-case initialization is absent or incomplete."""


class FluentBackend:
    def __init__(self, work_dir, *, solver=None, initializer=None, launch_kwargs=None):
        if not callable(initializer):
            raise FluentConfigurationError(
                "An initializer(solver, work_dir) callback is required. The archive has "
                "no verified case/UDF compile, hook and full-reset journal."
            )
        self.work_dir = Path(work_dir).resolve()
        self.solver = solver
        self.initializer = initializer
        self.launch_kwargs = dict(launch_kwargs or {})
        if "cwd" in self.launch_kwargs:
            raise FluentConfigurationError("Set work_dir, not launch_kwargs['cwd'].")
        self._output_offset = 0
        self._time = 0.0
        self._actions = 0
        self._ready = False
        self._closed = False

    def reset(self):
        if self._closed:
            raise RuntimeError("The Fluent backend has been closed.")
        self._ready = False
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.solver is None:
            import ansys.fluent.core as pyfluent  # Optional until a real session is requested.
            arguments = {"precision": "double", "processor_count": 6, "dimension": 2,
                         "mode": "solver", "ui_mode": "no_gui"}
            arguments.update(self.launch_kwargs)
            self.solver = pyfluent.launch_fluent(cwd=str(self.work_dir), **arguments)
        state = np.asarray(self.initializer(self.solver, self.work_dir), dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all() or not np.isclose(state[6], 0.0, atol=1e-10):
            raise FluentConfigurationError(
                "initializer must return finite [x,y,yaw,vx_world,vy_world,wz,time] with time=0."
            )
        self._time = float(state[6])
        self._actions = 0
        output = self.work_dir / "Output.txt"
        # Ignore pre-reset records even if the initialization keeps old output.
        self._output_offset = output.stat().st_size if output.exists() else 0
        self._ready = True
        return state.copy()

    def set_action(self, period, turning, time_step):
        if not self._ready:
            raise RuntimeError("Reset the Fluent backend before setting an action.")
        if self._actions >= 40:
            raise RuntimeError("The supplied UDF buffer holds only 40 actions; reset or provide a validated UDF update.")
        self.solver.execute_tui(f"/solve/set/time-step {time_step:.17g}")
        self.solver.execute_tui(f"(rpsetvar 'tc {period:.17g})")
        self.solver.execute_tui(f"(rpsetvar 'at {turning:.17g})")
        self.solver.execute_tui('/define/user-defined/execute-on-demand "add_action_from_console::libudf"')
        self._actions += 1

    def advance(self, time_step):
        if not self._ready:
            raise RuntimeError("Reset the Fluent backend before advancing.")
        self.solver.execute_tui("/solve/dual-time-iterate 1 10")
        output = self.work_dir / "Output.txt"
        with output.open("rb") as stream:
            if output.stat().st_size < self._output_offset:
                raise RuntimeError("Output.txt was truncated during the episode.")
            stream.seek(self._output_offset)
            appended = stream.read()
            new_offset = stream.tell()
        lines = [line for line in appended.splitlines() if line.strip()]
        if len(lines) != 1 or not appended.endswith(b"\n"):
            raise RuntimeError("Expected one fresh complete UDF Output.txt record per CFD step; verify execute-at-end hooks.")
        fields = np.array([float(value) for value in lines[0].split()], dtype=np.float64)
        if fields.shape != (8,) or not np.isfinite(fields).all():
            raise RuntimeError("Output.txt must contain eight finite values from Fish_UDF_Finial.c.")
        if not np.isclose(fields[0], self._time + time_step, rtol=0.0, atol=max(1e-7, time_step * 0.01)):
            raise RuntimeError("UDF time did not advance by one configured CFD time step.")
        # The hook writes this marker only after writing its state file. Requiring
        # a new marker prevents silently reusing a previous episode's snapshot.
        latest_path = self.work_dir / "fish_latest_state.txt"
        latest_path.unlink(missing_ok=True)
        self.solver.execute_tui('/define/user-defined/execute-on-demand "save_fish_state::libudf"')
        latest = latest_path.read_text(encoding="utf-8").strip()
        matched = re.fullmatch(r"fish_state_t([0-9]+\.[0-9]{6})\.txt", latest)
        if matched is None:
            raise RuntimeError("Unexpected UDF state filename; only local fish_state_t*.txt snapshots are accepted.")
        snapshot_time = float(matched.group(1))
        if not np.isclose(snapshot_time, fields[0], rtol=0.0, atol=max(1e-6, time_step * 0.01)):
            raise RuntimeError("The UDF state snapshot is stale or does not match Output.txt.")
        saved = {}
        with (self.work_dir / latest).open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.split()
                if len(parts) == 2:
                    saved[parts[0]] = float(parts[1])
        names = ("XDISP", "YDISP", "THETADISP", "XVEL", "YVEL", "THETAVEL")
        if any(name not in saved for name in names):
            raise RuntimeError("The UDF state snapshot is missing position or velocity fields.")
        state = np.array([saved[name] for name in names] + [snapshot_time], dtype=np.float64)
        if not np.isfinite(state).all():
            raise RuntimeError("The UDF state contains non-finite values.")
        if not np.allclose(state[:3], fields[1:4], rtol=1e-4, atol=1e-7):
            raise RuntimeError("The UDF snapshot pose does not agree with Output.txt.")
        self._output_offset = new_offset
        self._time = snapshot_time
        return state

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._ready = False
        if self.solver is not None:
            try:
                self.solver.exit()
            except Exception as exc:
                warnings.warn(f"Could not close Fluent solver: {type(exc).__name__}: {exc}", RuntimeWarning)
