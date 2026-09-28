"""Simple 2-D point-mass navigation environment.

This is the simplest possible goal-conditioned task: a 2-D particle with
continuous velocity control navigates to a random target.  There is no
physics engine, no joint dynamics, and no chance of "dying".  It exists
purely to verify that goal-conditioned baselines (MA-GCBC, MA-GCIQL,
MADDPG+HER, MA-SAC+HER) are correctly implemented — if they fail here,
the code is broken; if they succeed, the ant failures are due to task
difficulty, not implementation bugs.

Observation (6-D): [x, y, vx, vy, target_x, target_y]
Action (2-D):       [ax, ay]  ∈  [-1, 1]
Goal encoding:
  achieved goal = obs[0:2]  (current position)
  desired  goal = obs[-2:]  (target position)
Success threshold: ||pos - target|| < 0.5
"""

from typing import Tuple
import jax
from jax import numpy as jp
from brax.envs.base import Env, State


class Point2D(Env):
    def __init__(
        self,
        target_dist: float = 5.0,
        max_speed: float = 3.0,
        dt: float = 0.05,
        damping: float = 0.85,
        force_scale: float = 0.5,
    ):
        self._target_dist = target_dist
        self._max_speed = max_speed
        self._dt = dt
        self._damping = damping
        self._force_scale = force_scale

        # Attrs expected by training scripts
        self.state_dim = 6
        self.goal_indices = jp.array([0, 1])

    # ------------------------------------------------------------------
    # Brax Env interface
    # ------------------------------------------------------------------
    @property
    def observation_size(self) -> int:
        return 6

    @property
    def action_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "positional"

    @property
    def dt(self) -> float:
        return self._dt

    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> State:
        rng, r1, r2 = jax.random.split(rng, 3)

        # Agent starts near origin
        pos = jax.random.uniform(r1, (2,), minval=-0.1, maxval=0.1)
        vel = jp.zeros(2)

        # Target at random angle, fixed radius
        ang = jp.pi * 2.0 * jax.random.uniform(r2)
        target = jp.array([jp.cos(ang), jp.sin(ang)]) * self._target_dist

        obs = jp.concatenate([pos, vel, target])

        metrics = {
            "dist":                   jp.zeros(()),
            "success":                jp.zeros(()),
            "success_easy":           jp.zeros(()),
            # Dummy keys expected by some training scripts
            "reward_forward":         jp.zeros(()),
            "reward_survive":         jp.zeros(()),
            "reward_ctrl":            jp.zeros(()),
            "reward_contact":         jp.zeros(()),
            "x_position":            jp.zeros(()),
            "y_position":            jp.zeros(()),
            "distance_from_origin":  jp.zeros(()),
            "x_velocity":            jp.zeros(()),
            "y_velocity":            jp.zeros(()),
            "forward_reward":         jp.zeros(()),
        }
        state = State(
            pipeline_state=None,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics=metrics,
        )
        state.info.update({"seed": 0})
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pos    = state.obs[:2]
        vel    = state.obs[2:4]
        target = state.obs[4:6]

        action = jp.clip(action, -1.0, 1.0)
        vel    = vel * self._damping + action * self._force_scale
        vel    = jp.clip(vel, -self._max_speed, self._max_speed)
        pos    = pos + vel * self._dt

        obs  = jp.concatenate([pos, vel, target])
        dist = jp.linalg.norm(pos - target)

        # Dense reward by default; training scripts override with _goal_reward
        reward = -dist
        done   = jp.zeros(())   # never terminates

        success      = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 1.0, dtype=float)

        state.metrics.update(
            dist=dist,
            success=success,
            success_easy=success_easy,
            x_position=pos[0],
            y_position=pos[1],
            distance_from_origin=jp.linalg.norm(pos),
            x_velocity=vel[0],
            y_velocity=vel[1],
        )
        state.info.update({"seed": state.info.get("seed", 0)})
        return state.replace(obs=obs, reward=reward, done=done)
