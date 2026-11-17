from __future__ import annotations

from isaaclab.managers import ManagerTermBase


class AssistForceCurriculum(ManagerTermBase):
    """Lower the helper-force scale by ``step`` every ``update_interval`` env steps
    (1200 = 50 iterations of 24 steps) if at least ``timeout_threshold`` of the
    episodes that ended in that window ended by time-out."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Envs terminated by timeout
        self._timeouts = 0
        # Total envs terminated
        self._resets = 0
        # Time of last update in env steps
        self._last_update = 0

    def __call__(self, env, env_ids, step, timeout_threshold, term_name="assist_force", min_scale=0.0,
                 update_interval=1200):
        term_cfg = env.event_manager.get_term_cfg(term_name)

        # Skip the initial reset of all environments at step 0
        if env.common_step_counter > 0:
            self._timeouts += int(env.termination_manager.time_outs[env_ids].sum())
            self._resets += len(env_ids)

        time_for_update: bool = env.common_step_counter - self._last_update >= update_interval
        above_threshold: bool = self._timeouts >= timeout_threshold * self._resets

        if time_for_update:
            if above_threshold:
                term_cfg.params["scale"] = max(term_cfg.params["scale"] - step, min_scale)
                env.event_manager.set_term_cfg(term_name, term_cfg)
            self._timeouts = 0
            self._resets = 0
            self._last_update = env.common_step_counter
        return term_cfg.params["scale"]
