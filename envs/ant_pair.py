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


class AntPair(PipelineEnv):
    """Two independent ants navigating to a shared goal.

    q layout:  [ant_a_free(7) + ant_a_joints(8) + ant_b_free(7) + ant_b_joints(8) + target_xy(2)] = 32
    qd layout: [ant_a_free(6) + ant_a_joints(8) + ant_b_free(6) + ant_b_joints(8) + target_xy(2)] = 30

    Full obs (60): [ant_a_qpos(15), ant_a_qvel(14), ant_b_qpos(15), ant_b_qvel(14), target_xy(2)]
    Agent A partial obs (31): [ant_a_qpos(15), ant_a_qvel(14), target_xy(2)]
    Agent B partial obs (31): [ant_b_qpos(15), ant_b_qvel(14), target_xy(2)]

    Success: both ants within 0.5m of target.
    Done: either ant falls (unhealthy).
    Reward: -mean_dist + healthy_reward.
    Action: 16-dim (8 per ant). Training script splits actions per agent.
    """

    def __init__(
        self,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        reset_noise_scale=0.1,
        target_dist: float = 10.0,
        backend="spring",
        **kwargs,
    ):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets", "ant_pair.xml"
        )
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._target_dist = target_dist

        # Find torso_b body index in the link list
        self._torso_b_idx = list(sys.link_names).index("torso_b")

        # Observation / goal dimensions
        self.obs_dim_full = 60   # 15+14 + 15+14 + 2
        self.obs_dim_agent = 31  # 15+14 + 2
        self.goal_dim = 2
        self.n_bodies = 2

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_target, rng_offset = jax.random.split(rng, 5)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # Both ants near origin with small random offsets
        # Ant A: q[0:2] = xy offset, Ant B: q[15:17] = xy offset
        offsets = jax.random.uniform(rng_offset, (4,), minval=-0.5, maxval=0.5)
        q = q.at[0].set(offsets[0])
        q = q.at[1].set(offsets[1])
        q = q.at[15].set(offsets[2])
        q = q.at[16].set(offsets[3])

        # Set the target position
        _, target = self._random_target(rng_target)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "dist_a": zero,
            "dist_b": zero,
            "mean_dist": zero,
            "success": zero,
            "success_easy": zero,
            "reward_survive": zero,
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        # Health check for both ants
        min_z, max_z = self._healthy_z_range

        z_a = pipeline_state.x.pos[0, 2]
        is_healthy_a = jp.where(z_a < min_z, 0.0, 1.0)
        is_healthy_a = jp.where(z_a > max_z, 0.0, is_healthy_a)

        z_b = pipeline_state.x.pos[self._torso_b_idx, 2]
        is_healthy_b = jp.where(z_b < min_z, 0.0, 1.0)
        is_healthy_b = jp.where(z_b > max_z, 0.0, is_healthy_b)

        # Done if EITHER ant falls
        is_healthy = is_healthy_a * is_healthy_b

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        obs = self._get_obs(pipeline_state)

        # Target position
        target_xy = pipeline_state.x.pos[-1][:2]

        # Distance of each ant to target
        pos_a_xy = pipeline_state.x.pos[0][:2]
        pos_b_xy = pipeline_state.x.pos[self._torso_b_idx][:2]
        dist_a = jp.linalg.norm(pos_a_xy - target_xy)
        dist_b = jp.linalg.norm(pos_b_xy - target_xy)
        mean_dist = (dist_a + dist_b) / 2.0

        reward = -mean_dist + healthy_reward
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Success: BOTH ants within threshold
        success = jp.array((dist_a < 0.5) & (dist_b < 0.5), dtype=float)
        success_easy = jp.array((dist_a < 2.0) & (dist_b < 2.0), dtype=float)

        state.metrics.update(
            dist_a=dist_a,
            dist_b=dist_b,
            mean_dist=mean_dist,
            success=success,
            success_easy=success_easy,
            reward_survive=healthy_reward,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Full observation: [ant_a_qpos(15), ant_a_qvel(14), ant_b_qpos(15), ant_b_qvel(14), target_xy(2)]."""
        # q layout: ant_a(15) + ant_b(15) + target(2) = 32
        qpos_a = pipeline_state.q[:15]     # 3 pos + 4 quat + 8 joints
        qpos_b = pipeline_state.q[15:30]   # 3 pos + 4 quat + 8 joints

        # qd layout: ant_a(14) + ant_b(14) + target(2) = 30
        qvel_a = pipeline_state.qd[:14]    # 6 free + 8 joints
        qvel_b = pipeline_state.qd[14:28]  # 6 free + 8 joints

        target_xy = pipeline_state.x.pos[-1][:2]

        return jp.concatenate([qpos_a, qvel_a, qpos_b, qvel_b, target_xy])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle on the xy plane."""
        rng, rng1 = jax.random.split(rng)
        ang = jp.pi * 2.0 * jax.random.uniform(rng1)
        target_x = self._target_dist * jp.cos(ang)
        target_y = self._target_dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])
