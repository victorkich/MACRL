# Derived from the environments of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

import os
from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import xml.etree.ElementTree as ET

# Two ants + T-shaped load through slit maze.
# T crossbar (5 units) is wider than slit (4 units) → must rotate to pass.
# Directly inspired by "Cooperative geometric puzzle solving in ants vs humans" (PNAS 2025).

RESET = R = 'r'
GOAL = G = 'g'

# Two chambers separated by a wall with a single slit
T_SLIT_SIMPLE = [[1, 1, 1, 1, 1, 1, 1],
                  [1, R, G, 1, G, G, 1],
                  [1, G, G, G, G, G, 1],
                  [1, G, G, 1, G, G, 1],
                  [1, 1, 1, 1, 1, 1, 1]]

# Three chambers, two slits at different vertical positions
T_SLIT_DOUBLE = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, R, G, 1, G, 1, G, G, 1],
                  [1, G, G, G, G, 1, G, G, 1],
                  [1, G, G, 1, G, G, G, G, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]

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
    if maze_layout_name == "t_slit":
        maze_layout = T_SLIT_SIMPLE
    elif maze_layout_name == "t_slit_double":
        maze_layout = T_SLIT_DOUBLE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_pair_tload.xml")
    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    # Filter goals: only right-chamber goals are kept so every episode requires slit
    # traversal. Same idea as chicane's adjacent-goal filter — left-chamber goals are
    # trivially reachable by the spawned T without solving the cooperative puzzle.
    # T_SLIT_SIMPLE: wall at col 3 (world y=12); right chamber starts at col 4 (y>=16).
    if maze_layout_name == "t_slit":
        possible_goals = jp.array([g for g in possible_goals.tolist() if g[1] >= 16.0])

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    rows, cols = len(maze_layout), len(maze_layout[0])
    for i in range(rows):
        for j in range(cols):
            if maze_layout[i][j] == 1:
                # Boundary walls: full thickness. Internal walls: thin slabs.
                is_boundary = (i == 0 or i == rows - 1 or j == 0 or j == cols - 1)
                if is_boundary:
                    sx = 0.5 * maze_size_scaling
                    sy = 0.5 * maze_size_scaling
                else:
                    # Thin internal walls (0.3 units half-size in y = 0.6 total)
                    # Keeps full x-size so wall blocks still form continuous barriers
                    sx = 0.5 * maze_size_scaling
                    sy = 0.3
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                      j * maze_size_scaling,
                                      MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (sx, sy,
                                       MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box", material="",
                    contype="1", conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    # Enable ant leg collision with object: set conaffinity=2 on ant capsule geoms only
    for geom in tree.findall(".//geom"):
        name = geom.get("name", "")
        if geom.get("type") == "capsule" and not name.startswith("tc") and not name.startswith("ts"):
            geom.set("conaffinity", "2")
    # Object capsules: contype=3 to match both walls (bit 0) and ant legs (bit 1)
    for geom in tree.findall(".//geom"):
        name = geom.get("name", "")
        if name.startswith("tc") or name.startswith("ts"):
            geom.set("contype", "3")
            geom.set("conaffinity", "1")

    # init_qpos: ant_a(15) + ant_b(15) + tload(3: x,y,θ) + target(2) = 35
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data").split()
    # T at θ=π/2 (crossbar along y) to fit corridor. Shift +2 in y to avoid wall clip.
    tload_y = robot_y + 2.0
    data[0] = str(robot_x - 1.0)
    data[1] = str(tload_y)
    data[15] = str(robot_x + 1.0)
    data[16] = str(tload_y)
    data[30] = str(robot_x)
    data[31] = str(tload_y)
    data[32] = str(1.5708)
    torso.set("data", " ".join(data))

    # Top-down camera
    rows, cols = len(maze_layout), len(maze_layout[0])
    cx = (rows - 1) * maze_size_scaling / 2
    cy = (cols - 1) * maze_size_scaling / 2
    cam_z = max(rows, cols) * maze_size_scaling * 0.9
    ET.SubElement(worldbody, "camera", name="maze_top", mode="fixed",
                  pos=f"{cx} {cy} {cam_z}", xyaxes="1 0 0 0 1 0")

    tree = tree.getroot()
    return ET.tostring(tree), possible_goals


class AntPairTLoad(PipelineEnv):
    """Two ants + T-shaped load through slit maze. T must rotate to pass through slits.

    Full obs: [ant_a(29), ant_b(29), tload_θ(1), tload_xy(2), 0, target_xy(2)] = 64
    Per-agent obs: [own_body(29), tload_θxy(3), goal(3), agent_id(2)] = 37 (matches chicane)
    HER goal slice = obs[61:64] = (tload_θ, tload_x, tload_y) — the actual object pose.
    """

    def __init__(self, ctrl_cost_weight=0.5, healthy_reward=1.0,
                 terminate_when_unhealthy=True, healthy_z_range=(0.2, 2.0),
                 reset_noise_scale=0.1, exclude_current_positions_from_observation=False,
                 backend="generalized", maze_layout_name="t_slit",
                 maze_size_scaling=4.0, **kwargs):
        xml_string, possible_goals = make_maze(maze_layout_name, maze_size_scaling)
        sys = mjcf.loads(xml_string)
        self.possible_goals = possible_goals

        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10
        if backend == "positional":
            sys = sys.replace(actuator=sys.actuator.replace(
                gear=200 * jp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._object_idx = self.sys.link_names.index('object')
        self._torso_b_idx = list(self.sys.link_names).index('torso_b')

        # 3D object state: (θ, x, y) — same layout as chicane.
        # HER slices the last goal_dim dims of state → gets the real (θ, x, y) pose,
        # which the reward measures (xy) consistently.
        # State: ant_a(29) + ant_b(29) + (θ, x, y)(3) = 61; full obs = 64 (state + 3D goal)
        self.state_dim = 61
        self.obs_dim_full = 64
        self.obs_dim_agent = 35  # body(29) + tload_θxy(3) + goal(3); agent_id added by training
        self.goal_dim = 3
        self.n_bodies = 2
        self.goal_indices = jp.array([61, 62, 63])

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # FIXED T spawn (matches chicane's low-variance approach for trainability).
        # Position: x=8 (centered on slit x-range [6,10]), y=6 (deep in left chamber).
        # Orientation: θ=π — stem points toward slit (+y direction); crossbar is the
        # base. Ants spawn behind the crossbar (-y side), where they can push the T
        # forward through the slit. Single random variable per episode = target only.
        new_x  = jp.array(8.0)
        new_y  = jp.array(6.0)
        new_th = jp.array(jp.pi)
        q = q.at[30].set(new_x).at[31].set(new_y).at[32].set(new_th)

        # Ants spawn at local "above crossbar" (+y); rotated by θ=π → world -y side.
        cos_t, sin_t = jp.cos(new_th), jp.sin(new_th)
        def local_to_world(dx, dy):
            return (new_x + dx * cos_t - dy * sin_t,
                    new_y + dx * sin_t + dy * cos_t)
        ant_a_x, ant_a_y = local_to_world(-0.8, 1.8)
        ant_b_x, ant_b_y = local_to_world( 0.8, 1.8)
        q = q.at[0].set(ant_a_x).at[1].set(ant_a_y)
        q = q.at[15].set(ant_b_x).at[16].set(ant_b_y)

        _, target = self._random_target(rng3)
        q = q.at[-2:].set(target)
        qd = qd.at[-5:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {"reward_survive": zero, "reward_ctrl": zero,
                   "dist": zero, "success": zero, "success_easy": zero}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update({"seed": 0})
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # NO MORE PYTHON HACKS. Physics is handled by Brax natively via vel_damping=-1
        # in the XML <custom> block (Brax PR #509 fix for the pusher env).

        if "steps" in state.info:
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]

        min_z, max_z = self._healthy_z_range
        is_healthy_a = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy_a = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy_a)
        is_healthy_b = jp.where(pipeline_state.x.pos[self._torso_b_idx, 2] < min_z, 0.0, 1.0)
        is_healthy_b = jp.where(pipeline_state.x.pos[self._torso_b_idx, 2] > max_z, 0.0, is_healthy_b)
        is_healthy = is_healthy_a * is_healthy_b

        healthy_reward = self._healthy_reward if self._terminate_when_unhealthy else self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Distance: tload center xy to target xy (use pipeline state directly)
        tload_xy = pipeline_state.x.pos[self._object_idx][:2]
        target_xy = pipeline_state.x.pos[-1][:2]
        dist = jp.linalg.norm(tload_xy - target_xy)

        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        reward = -dist + healthy_reward - ctrl_cost

        state.metrics.update(reward_survive=healthy_reward, reward_ctrl=-ctrl_cost,
                             dist=dist, success=success, success_easy=success_easy)
        state.info.update({"seed": seed})
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Full obs: [ant_a(29), ant_b(29), tload_θ(1), tload_xy(2), 0, target_xy(2)] = 64 dims.
        Layout matches chicane. HER slices [61:64] = (θ, x, y) — the actual object pose."""
        qpos_a = pipeline_state.q[:15]
        qvel_a = pipeline_state.qd[:14]
        qpos_b = pipeline_state.q[15:30]
        qvel_b = pipeline_state.qd[14:28]

        tload_theta = pipeline_state.q[32:33]
        tload_xy = pipeline_state.x.pos[self._object_idx][:2]
        target_xy = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos_a = qpos_a[2:]
            qpos_b = qpos_b[2:]

        return jp.concatenate([qpos_a, qvel_a, qpos_b, qvel_b,
                                tload_theta, tload_xy, jp.zeros(1), target_xy])

    def _random_target(self, rng):
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]
