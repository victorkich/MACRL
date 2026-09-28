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
import xml.etree.ElementTree as ET

# Two-ant maze environment.
# Both ants spawn in the 'R' cell and must both reach the goal.
# Maze generation adapted from ant_maze.py

RESET = R = 'r'
GOAL = G = 'g'

U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5


def find_robot(structure, size_scaling):
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                return i * size_scaling, j * size_scaling


def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])
    return jp.array(goals)


def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_pair_maze.xml")

    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                      j * maze_size_scaling,
                                      MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling,
                                       0.5 * maze_size_scaling,
                                       MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    # Update init_qpos: set both ants' xy to robot spawn position with small offsets
    # init_qpos layout: ant_a(15) + ant_b(15) + target(2) = 32
    # ant_a at indices 0,1 (x,y); ant_b at indices 15,16 (x,y)
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data").split()
    # Ant A: spawn at (robot_x - 0.5, robot_y)
    data[0] = str(robot_x - 0.5)
    data[1] = str(robot_y)
    # Ant B: spawn at (robot_x + 0.5, robot_y)
    data[15] = str(robot_x + 0.5)
    data[16] = str(robot_y)
    torso.set("data", " ".join(data))

    # Add static top-down camera centered on maze
    rows, cols = len(maze_layout), len(maze_layout[0])
    cx = (rows - 1) * maze_size_scaling / 2
    cy = (cols - 1) * maze_size_scaling / 2
    cam_z = max(rows, cols) * maze_size_scaling * 0.9
    ET.SubElement(
        worldbody, "camera",
        name="maze_top",
        mode="fixed",
        pos=f"{cx} {cy} {cam_z}",
        xyaxes="1 0 0 0 1 0",
    )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_goals


class AntPairMaze(PipelineEnv):
    """Two ants in a maze. Success requires both ants within 0.5m of the goal.

    Full obs: [ant_a_body(29), ant_b_body(29), target_xy(2)] = 60 dims
    Partial obs per agent: [own_body(29), target_xy(2)] = 31 dims
    """

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 2.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        **kwargs,
    ):
        xml_string, possible_goals = make_maze(maze_layout_name, maze_size_scaling)

        sys = mjcf.loads(xml_string)
        self.possible_goals = possible_goals

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
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Per-ant body obs: qpos(15) + qvel(14) = 29 dims (with xy included)
        # Full obs: ant_a(29) + ant_b(29) + target(2) = 60
        # State (no goal): ant_a(29) + ant_b(29) = 58
        self.state_dim = 58
        self.obs_dim_full = 60
        self.obs_dim_agent = 31
        self.goal_dim = 2
        self.n_bodies = 2
        self.goal_indices = jp.array([58, 59])
        self._torso_b_idx = list(self.sys.link_names).index('torso_b')

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # Set the target q, qd (last 2 elements)
        _, target = self._random_target(rng3)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_survive": zero,
            "reward_ctrl": zero,
            "dist_a": zero,
            "dist_b": zero,
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

        # Health check on ant A (body index 0)
        min_z, max_z = self._healthy_z_range
        is_healthy_a = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy_a = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy_a)
        # Health check on ant B (body index 1 — second torso)
        is_healthy_b = jp.where(pipeline_state.x.pos[self._torso_b_idx, 2] < min_z, 0.0, 1.0)
        is_healthy_b = jp.where(pipeline_state.x.pos[self._torso_b_idx, 2] > max_z, 0.0, is_healthy_b)

        is_healthy = is_healthy_a * is_healthy_b

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        target_xy = obs[-2:]
        # Ant A xy position
        ant_a_xy = obs[:2]
        # Ant B xy position (starts at index 29)
        ant_b_xy = obs[29:31]

        dist_a = jp.linalg.norm(ant_a_xy - target_xy)
        dist_b = jp.linalg.norm(ant_b_xy - target_xy)
        dist = jp.maximum(dist_a, dist_b)

        # Both ants must be within threshold
        success = jp.array((dist_a < 0.5) & (dist_b < 0.5), dtype=float)
        success_easy = jp.array((dist_a < 2.0) & (dist_b < 2.0), dtype=float)

        reward = -dist + healthy_reward - ctrl_cost

        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            dist_a=dist_a,
            dist_b=dist_b,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe both ants and target.

        Full obs: [ant_a_body(29), ant_b_body(29), target_xy(2)] = 60 dims
        ant body = qpos(15) + qvel(14) = 29 (xy included)

        qpos layout: ant_a(15) | ant_b(15) | target(2)
        qvel layout: ant_a(14) | ant_b(14) | target(2)
        """
        # Ant A: qpos indices 0:15, qvel indices 0:14
        qpos_a = pipeline_state.q[:15]
        qvel_a = pipeline_state.qd[:14]

        # Ant B: qpos indices 15:30, qvel indices 14:28
        qpos_b = pipeline_state.q[15:30]
        qvel_b = pipeline_state.qd[14:28]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos_a = qpos_a[2:]
            qpos_b = qpos_b[2:]

        return jp.concatenate([qpos_a, qvel_a, qpos_b, qvel_b, target_pos])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a random target location chosen from possible goal cells."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]
