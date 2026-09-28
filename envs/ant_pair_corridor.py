# Derived from the environments of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

import os
from typing import Tuple
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco


class AntPairCorridor(PipelineEnv):
    """
    Two ants in a narrow corridor that must swap positions.

    Ant A starts at the left end, must reach the right end.
    Ant B starts at the right end, must reach the left end.
    The corridor is narrow enough that both ants cannot pass simultaneously,
    requiring coordination (turn-taking, yielding).

    Ant-ant collision is ENABLED (contype=1, conaffinity=1 on all body geoms).

    Observation (full): [ant_a_body(29), ant_b_body(29), target_a_xy(2), target_b_xy(2)] = 62 dims
    Observation (partial, per agent): [own_body(29), own_target_xy(2)] = 31 dims

    Success: both ants within 0.5m of their respective targets.
    """

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 2.0),
        reset_noise_scale=0.1,
        corridor_half_length=6.0,
        backend="generalized",
        **kwargs,
    ):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'assets', 'ant_pair_corridor.xml')
        sys = mjcf.load(path)

        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._corridor_half_length = corridor_half_length

        # Body indices
        self._torso_b_idx = self.sys.link_names.index('torso_b')

        # Obs dims
        # Full: ant_a(29) + ant_b(29) + target_a_xy(2) + target_b_xy(2) = 62
        self.obs_dim_full = 62
        self.obs_dim_agent = 31  # own_body(29) + own_target(2)
        self.goal_dim = 4        # target_a_xy(2) + target_b_xy(2)
        self.n_bodies = 2
        self.state_dim = 58      # ant_a(29) + ant_b(29), no targets
        self.goal_indices = jp.array([58, 59])  # target_a position in full obs

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # Fix target positions (swap destinations)
        L = self._corridor_half_length
        # target_a at right end, target_b at left end
        q = q.at[-4:].set(jp.array([L, 0.0, -L, 0.0]))
        qd = qd.at[-4:].set(0)

        # Reset ant positions: A at left, B at right
        q = q.at[0].set(-L)   # ant_a x
        q = q.at[1].set(0.0)  # ant_a y
        q = q.at[15].set(L)   # ant_b x
        q = q.at[16].set(0.0) # ant_b y

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "dist_a": zero,
            "dist_b": zero,
            "success": zero,
            "success_a": zero,
            "success_b": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        obs = self._get_obs(pipeline_state)

        # Health check for both ants
        min_z, max_z = self._healthy_z_range
        z_a = pipeline_state.x.pos[0, 2]
        z_b = pipeline_state.x.pos[self._torso_b_idx, 2]
        is_healthy_a = jp.where(z_a < min_z, 0.0, 1.0)
        is_healthy_a = jp.where(z_a > max_z, 0.0, is_healthy_a)
        is_healthy_b = jp.where(z_b < min_z, 0.0, 1.0)
        is_healthy_b = jp.where(z_b > max_z, 0.0, is_healthy_b)
        is_healthy = is_healthy_a * is_healthy_b

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        # Distance: each ant to its own target
        pos_a = pipeline_state.x.pos[0, :2]
        pos_b = pipeline_state.x.pos[self._torso_b_idx, :2]
        target_a = obs[-4:-2]  # target for ant A
        target_b = obs[-2:]    # target for ant B

        dist_a = jp.linalg.norm(pos_a - target_a)
        dist_b = jp.linalg.norm(pos_b - target_b)
        mean_dist = (dist_a + dist_b) / 2.0

        reward = -mean_dist + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        success_a = jp.array(dist_a < 0.5, dtype=float)
        success_b = jp.array(dist_b < 0.5, dtype=float)
        success = success_a * success_b  # both must reach their targets

        state.metrics.update(
            dist_a=dist_a,
            dist_b=dist_b,
            success=success,
            success_a=success_a,
            success_b=success_b,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Full observation: [ant_a_body(29), ant_b_body(29), target_a_xy(2), target_b_xy(2)]."""
        # q layout: ant_a(15) + ant_b(15) + target_a(2) + target_b(2) = 34
        # qd layout: ant_a(14) + ant_b(14) + target(4) = 32
        qpos_a = pipeline_state.q[:15]
        qvel_a = pipeline_state.qd[:14]
        qpos_b = pipeline_state.q[15:30]
        qvel_b = pipeline_state.qd[14:28]

        target_a_pos = pipeline_state.q[-4:-2]
        target_b_pos = pipeline_state.q[-2:]

        return jp.concatenate([qpos_a, qvel_a, qpos_b, qvel_b,
                               target_a_pos, target_b_pos])

    @staticmethod
    def get_obs_agent(full_obs, agent_id):
        """Extract partial observation for one agent.
        Agent 0: own_body(29) + target_a(2) = 31
        Agent 1: own_body(29) + target_b(2) = 31
        """
        if agent_id == 0:
            return jp.concatenate([full_obs[:29], full_obs[-4:-2]])
        else:
            return jp.concatenate([full_obs[29:58], full_obs[-2:]])
