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


class AntPairBall(PipelineEnv):
    """Two separate ants cooperatively pushing a ball to a target.

    Each ant has 8 actuators (16 total). The ball is spawned between the ants
    and the target is spawned at a configurable distance away.

    Observation structure:
        Full obs (62): [ant_a_qpos(15), ant_a_qvel(14),
                        ant_b_qpos(15), ant_b_qvel(14),
                        ball_xy(2), target_xy(2)]
        Agent A partial obs (33): [ant_a_qpos(15), ant_a_qvel(14),
                                   ball_xy(2), target_xy(2)]
        Agent B partial obs (33): [ant_b_qpos(15), ant_b_qvel(14),
                                   ball_xy(2), target_xy(2)]
    """

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        reset_noise_scale=0.1,
        target_dist: float = 5.0,
        backend="spring",
        **kwargs,
    ):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'assets', 'ant_pair_ball.xml',
        )
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({'opt.timestep': 0.005})
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

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._target_dist = target_dist

        # Body indices in the link array
        self._torso_a_idx = self.sys.link_names.index('torso_a')
        self._torso_b_idx = self.sys.link_names.index('torso_b')
        self._object_idx = self.sys.link_names.index('object')

        # q layout: ant_a_qpos(15) + ant_b_qpos(15) + ball(2) + target(2) = 34
        # qd layout: ant_a_qvel(14) + ant_b_qvel(14) + ball(2) + target(2) = 32
        self._ant_a_q_end = 15
        self._ant_b_q_start = 15
        self._ant_b_q_end = 30
        self._ant_a_qd_end = 14
        self._ant_b_qd_start = 14
        self._ant_b_qd_end = 28

        # Dimensions exposed for the training script
        self.obs_dim_full = 62
        self.obs_dim_agent = 33
        self.goal_dim = 2
        self.n_bodies = 2

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # Random target and ball placement
        _, target, obj = self._random_target(rng3)

        # Last 4 q entries: ball_xy(2) + target_xy(2)
        q = q.at[-4:].set(jp.concatenate([obj, target]))
        qd = qd.at[-4:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_survive": zero,
            "reward_ctrl": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
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
        z_a = pipeline_state.x.pos[self._torso_a_idx, 2]
        z_b = pipeline_state.x.pos[self._torso_b_idx, 2]

        is_healthy_a = jp.where(z_a < min_z, 0.0, 1.0)
        is_healthy_a = jp.where(z_a > max_z, 0.0, is_healthy_a)
        is_healthy_b = jp.where(z_b < min_z, 0.0, 1.0)
        is_healthy_b = jp.where(z_b > max_z, 0.0, is_healthy_b)
        is_healthy = is_healthy_a * is_healthy_b  # done if either falls

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)

        # Distance between ball and target (last 4 entries of obs)
        ball_xy = obs[-4:-2]
        target_xy = obs[-2:]
        dist = jp.linalg.norm(ball_xy - target_xy)

        reward = -dist + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)

        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Full observation: both ants' states + ball + target.

        Returns array of shape (62,):
            [ant_a_qpos(15), ant_a_qvel(14),
             ant_b_qpos(15), ant_b_qvel(14),
             ball_xy(2), target_xy(2)]
        """
        # Strip ball(2) + target(2) from the end of q and qd
        ant_a_qpos = pipeline_state.q[:self._ant_a_q_end]
        ant_a_qvel = pipeline_state.qd[:self._ant_a_qd_end]

        ant_b_qpos = pipeline_state.q[self._ant_b_q_start:self._ant_b_q_end]
        ant_b_qvel = pipeline_state.qd[self._ant_b_qd_start:self._ant_b_qd_end]

        ball_xy = pipeline_state.x.pos[self._object_idx][:2]
        target_xy = pipeline_state.x.pos[-1][:2]

        return jp.concatenate([
            ant_a_qpos, ant_a_qvel,
            ant_b_qpos, ant_b_qvel,
            ball_xy, target_xy,
        ])

    def get_obs_agent(self, full_obs: jax.Array, agent_id: int) -> jax.Array:
        """Extract partial observation for a single agent.

        Args:
            full_obs: Full observation array of shape (62,).
            agent_id: 0 for ant A, 1 for ant B.

        Returns:
            Partial obs of shape (33,): [qpos(15), qvel(14), ball_xy(2), target_xy(2)]
        """
        shared = full_obs[-4:]  # ball_xy + target_xy
        if agent_id == 0:
            own = full_obs[:29]  # ant_a_qpos(15) + ant_a_qvel(14)
        else:
            own = full_obs[29:58]  # ant_b_qpos(15) + ant_b_qvel(14)
        return jp.concatenate([own, shared])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Returns a target and ball location.

        Target is placed at ``target_dist`` from the midpoint between the two
        ants in a random direction. Ball is placed between the ants with a
        small random offset.
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = self._target_dist
        ang = jp.pi * 2.0 * jax.random.uniform(rng1)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)

        ang_obj = jp.pi * 2.0 * jax.random.uniform(rng2)
        obj_x_offset = jp.cos(ang_obj)
        obj_y_offset = jp.sin(ang_obj)

        target_pos = jp.array([target_x, target_y])
        # Ball near the midpoint between the two ants (x~1.5) with small offset
        obj_pos = target_pos * 0.2 + jp.array([obj_x_offset, obj_y_offset])
        return rng, target_pos, obj_pos
