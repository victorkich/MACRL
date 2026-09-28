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

# Two ants + rigid bar in a chicane (S-shaped passage).
# Bar is longer than corridor width → must rotate at each corner.
# Cooperation required: single ant can't control both rotation and translation.

RESET = R = 'r'
GOAL = G = 'g'

U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

CHICANE_S = [[1, 1, 1, 1, 1, 1],
             [1, R, G, G, G, 1],
             [1, 1, 1, 1, G, 1],
             [1, G, G, G, G, 1],
             [1, G, 1, 1, 1, 1],
             [1, G, G, G, G, 1],
             [1, 1, 1, 1, 1, 1]]

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
    elif maze_layout_name == "chicane_s":
        maze_layout = CHICANE_S
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_pair_chicane.xml")
    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    # Filter goals so every episode requires real chicane navigation.
    # Two filters combined:
    #   (a) drop goals on the same row as the spawn — the spawn row is an open
    #       corridor, so a same-row goal is solvable by pushing the bar straight
    #       without going around any corner;
    #   (b) drop goals within 6m of the bar spawn — defensive safety net for any
    #       adjacent goal that survived (a).
    import math
    bar_spawn_x, bar_spawn_y = robot_x, robot_y + 2.0
    possible_goals = jp.array([
        g for g in possible_goals.tolist()
        if abs(g[0] - robot_x) > 0.5
        and math.hypot(g[0] - bar_spawn_x, g[1] - bar_spawn_y) >= 6.0
    ])

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                      j * maze_size_scaling,
                                      MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling,
                                       0.5 * maze_size_scaling,
                                       MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box", material="",
                    contype="1", conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    # Enable ant leg collision with object: set conaffinity=2 on ant capsule geoms
    # Object capsule uses contype=3 so it matches both bit 0 (walls/feet) and bit 1 (legs)
    for geom in tree.findall(".//geom"):
        name = geom.get("name", "")
        if geom.get("type") == "capsule" and not name.startswith("bar_"):
            geom.set("conaffinity", "2")
    for geom in tree.findall(".//geom"):
        name = geom.get("name", "")
        if name.startswith("bar_c"):
            geom.set("contype", "3")
            geom.set("conaffinity", "1")

    # init_qpos: ant_a(15) + ant_b(15) + bar(3: x,y,θ) + target(2) = 35
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data").split()
    # Bar at θ=π/2 (along y), half-length=2.5. Shift +2 in y to avoid wall clip.
    bar_y = robot_y + 2.0
    data[0] = str(robot_x - 1.0)
    data[1] = str(bar_y)
    data[15] = str(robot_x + 1.0)
    data[16] = str(bar_y)
    data[30] = str(robot_x)
    data[31] = str(bar_y)
    data[32] = str(1.5708)
    torso.set("data", " ".join(data))

    # Top-down camera
    rows, cols = len(maze_layout), len(maze_layout[0])
    cx = (rows - 1) * maze_size_scaling / 2
    cy = (cols - 1) * maze_size_scaling / 2
    cam_z = max(rows, cols) * maze_size_scaling * 0.9
    ET.SubElement(worldbody, "camera", name="maze_top", mode="fixed",
                  pos=f"{cx} {cy} {cam_z}", xyaxes="1 0 0 0 1 0")

    # Third-person oblique camera — tracks the bar (object) automatically.
    # mode="targetbody" makes the camera always point at the target body regardless
    # of where it moves, giving a natural third-person view of the action.
    ET.SubElement(worldbody, "camera", name="third_person", mode="targetbody",
                  target="object", pos=f"{-4} {-4} 12")

    tree = tree.getroot()
    return ET.tostring(tree), possible_goals


class AntPairChicane(PipelineEnv):
    """Two ants + rigid bar in a chicane. Bar must rotate at corners.

    Full obs: [ant_a(29), ant_b(29), bar_θ(1), bar_xy(2), target_xy(2)] = 63
    Per-agent obs: [own_body(29), bar_θxy(3), target(2), agent_id(2)] = 36
    """

    def __init__(self, ctrl_cost_weight=0.5, healthy_reward=1.0,
                 terminate_when_unhealthy=True, healthy_z_range=(0.2, 2.0),
                 reset_noise_scale=0.1, exclude_current_positions_from_observation=False,
                 backend="generalized", maze_layout_name="u_maze",
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

        # Obs: ant_a(29) + ant_b(29) + bar_θ(1) + bar_xy(2) + target(3: θ=0,x,y) = 64
        # 3D goal [θ, x, y] for CRL contrastive loss orientation signal (matches state order)
        self.state_dim = 61    # without target
        self.obs_dim_full = 64
        self.obs_dim_agent = 35  # body(29) + bar_θxy(3) + goal(3); agent_id added by training
        self.goal_dim = 3
        self.n_bodies = 2
        self.goal_indices = jp.array([61, 62, 63])

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        _, target = self._random_target(rng3)
        q = q.at[-2:].set(target)
        # Zero object and target velocities
        qd = qd.at[-5:].set(0)  # bar(3) + target(2)

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

        # NO MORE PYTHON HACKS. Physics handled natively by Brax spring backend
        # with vel_damping=-1 in the XML <custom> block (Brax PR #509 fix).

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

        # Distance: bar center xy to target xy (pipeline state — independent of obs layout)
        bar_xy = pipeline_state.x.pos[self._object_idx][:2]
        target_xy = pipeline_state.x.pos[-1][:2]
        dist = jp.linalg.norm(bar_xy - target_xy)

        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        reward = -dist + healthy_reward - ctrl_cost

        state.metrics.update(reward_survive=healthy_reward, reward_ctrl=-ctrl_cost,
                             dist=dist, success=success, success_easy=success_easy)
        state.info.update({"seed": seed})
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Full obs: [ant_a(29), ant_b(29), bar_θ(1), bar_xy(2), target(3: θ=0,x,y)] = 64 dims.
        Target order matches state order [θ, x, y] for CRL goal consistency."""
        qpos_a = pipeline_state.q[:15]
        qvel_a = pipeline_state.qd[:14]
        qpos_b = pipeline_state.q[15:30]
        qvel_b = pipeline_state.qd[14:28]

        bar_theta = pipeline_state.q[32:33]
        bar_xy = pipeline_state.x.pos[self._object_idx][:2]
        target_xy = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos_a = qpos_a[2:]
            qpos_b = qpos_b[2:]

        return jp.concatenate([qpos_a, qvel_a, qpos_b, qvel_b, bar_theta, bar_xy, jp.zeros(1), target_xy])

    def _random_target(self, rng):
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]
