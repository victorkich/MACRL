# Derived from the environments of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

"""Chicane env with N physical ants (N ∈ {2, 3, 4}).

Extends AntPairChicane by dynamically generating an XML with N ant bodies
(instead of the fixed 2 in the original XML). All ants cooperate on pushing
a rigid bar through an S-shaped maze.
"""
import os
from typing import Tuple
import xml.etree.ElementTree as ET

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp


RESET = R = 'r'
GOAL = G = 'g'

U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5


def _ant_body_xml(prefix: str, x_pos: float, y_pos: float) -> str:
    """One ant body XML template with given prefix ('a', 'b', 'c', 'd')."""
    p = prefix
    # The camera only lives on the first ant (a_torso).
    cam = ('<camera name="track" mode="trackcom" pos="0 0 14" '
           'xyaxes="1 0 0 0 1 0"/>') if p == "a" else ""
    return f"""
    <body name="torso_{p}" pos="{x_pos} {y_pos} 0.75">
      {cam}
      <geom name="{p}_torso_geom" pos="0 0 0" size="0.25" type="sphere" conaffinity="1"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="{p}_root" pos="0 0 0" type="free"/>
      <body name="{p}_front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="{p}_aux_1_geom" size="0.08" type="capsule"/>
        <body name="{p}_aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="{p}_hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="{p}_left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="{p}_ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="{p}_left_ankle_geom" size="0.08" type="capsule"/>
            <geom name="{p}_left_foot_geom" contype="1" pos="0.4 0.4 0" size="0.08" type="sphere" mass="0" conaffinity="1"/>
          </body>
        </body>
      </body>
      <body name="{p}_front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="{p}_aux_2_geom" size="0.08" type="capsule"/>
        <body name="{p}_aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="{p}_hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="{p}_right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="{p}_ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="{p}_right_ankle_geom" size="0.08" type="capsule"/>
            <geom name="{p}_right_foot_geom" contype="1" pos="-0.4 0.4 0" size="0.08" type="sphere" mass="0" conaffinity="1"/>
          </body>
        </body>
      </body>
      <body name="{p}_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="{p}_aux_3_geom" size="0.08" type="capsule"/>
        <body name="{p}_aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="{p}_hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="{p}_back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="{p}_ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="{p}_third_ankle_geom" size="0.08" type="capsule"/>
            <geom name="{p}_third_foot_geom" contype="1" pos="-0.4 -0.4 0" size="0.08" type="sphere" mass="0" conaffinity="1"/>
          </body>
        </body>
      </body>
      <body name="{p}_right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="{p}_aux_4_geom" size="0.08" type="capsule"/>
        <body name="{p}_aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="{p}_hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="{p}_rightback_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="{p}_ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="{p}_fourth_ankle_geom" size="0.08" type="capsule"/>
            <geom name="{p}_fourth_foot_geom" contype="1" pos="0.4 -0.4 0" size="0.08" type="sphere" mass="0" conaffinity="1"/>
          </body>
        </body>
      </body>
    </body>
"""


def _ant_actuators_xml(prefix: str) -> str:
    """Actuator section for one ant (8 motors)."""
    p = prefix
    return f"""
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{p}_ankle_3" gear="150"/>
"""


def build_multi_chicane_xml(n_ants: int) -> str:
    """Generate the full chicane XML for n_ants physical ants (2..4)."""
    assert n_ants in (2, 3, 4), "n_ants must be 2, 3, or 4"
    prefixes = ["a", "b", "c", "d"][:n_ants]
    # Initial placeholder positions (overwritten by make_maze's init_qpos edit)
    init_positions = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)][:n_ants]

    # init_qpos: (ant_body=15) × n + bar(3) + target(2) = 15n + 5
    # Default per-ant qpos: [x, y, 0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    per_ant = "{x} {y} 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0"
    init_qpos_parts = []
    for prefix, (x, y) in zip(prefixes, init_positions):
        init_qpos_parts.append(per_ant.format(x=x, y=y))
    init_qpos_parts.append("0.0 0.0 0.0")  # bar x, y, theta
    init_qpos_parts.append("0.0 0.0")       # target x, y
    init_qpos = " ".join(init_qpos_parts)

    ant_bodies = "".join(
        _ant_body_xml(p, x, y) for p, (x, y) in zip(prefixes, init_positions)
    )
    actuators = "".join(_ant_actuators_xml(p) for p in prefixes)

    return f"""<mujoco model="ant_multi_chicane_{n_ants}a">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.01" iterations="4" />
  <custom>
    <numeric data="{init_qpos}" name="init_qpos"/>
    <numeric data="2000" name="constraint_limit_stiffness"/>
    <numeric data="10000" name="constraint_stiffness"/>
    <numeric data="10" name="constraint_ang_damping"/>
    <numeric data="20" name="constraint_vel_damping"/>
    <numeric data="0.5" name="joint_scale_pos"/>
    <numeric data="0.2" name="joint_scale_ang"/>
    <numeric data="-1" name="ang_damping"/>
    <numeric data="-1" name="vel_damping"/>
    <numeric data="1" name="spring_mass_scale"/>
    <numeric data="1" name="spring_inertia_scale"/>
    <numeric data="15" name="solver_maxls"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>

    {ant_bodies}

    <body name="object" pos="0.0 0.0 0.4">
      <geom name="bar_c0" type="capsule" fromto="-1.75 0 0 1.75 0 0" size="0.175"
            rgba="0.8 0.2 0.2 1" density="3.0" contype="1" conaffinity="1"/>
      <joint name="obj_slidex" type="slide" pos="0 0 0" axis="1 0 0" range="-50 50"/>
      <joint name="obj_slidey" type="slide" pos="0 0 0" axis="0 1 0" range="-50 50"/>
      <joint name="obj_hingez" type="hinge" pos="0 0 0" axis="0 0 1" limited="false"/>
    </body>

    <body name="target" pos="0 0 0.01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-50 50" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-50 50" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" size=".9" type="sphere" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


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


def make_multi_maze(n_ants, maze_layout_name, maze_size_scaling):
    import math
    assert maze_layout_name == "u_maze"
    maze_layout = U_MAZE
    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    # Same goal filter as 2-ant chicane: exclude same-row + <6m goals.
    bar_spawn_x, bar_spawn_y = robot_x, robot_y + 2.0
    possible_goals = jp.array([
        g for g in possible_goals.tolist()
        if abs(g[0] - robot_x) > 0.5
        and math.hypot(g[0] - bar_spawn_x, g[1] - bar_spawn_y) >= 6.0
    ])

    xml_string = build_multi_chicane_xml(n_ants)
    root = ET.fromstring(xml_string)
    worldbody = root.find(".//worldbody")

    # Add maze walls
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

    # Ant leg <-> bar collision config (matches 2-ant chicane)
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if geom.get("type") == "capsule" and not name.startswith("bar_"):
            geom.set("conaffinity", "2")
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.startswith("bar_c"):
            geom.set("contype", "3")
            geom.set("conaffinity", "1")

    # init_qpos layout: (ant_i = 15) × n_ants + bar(3) + target(2)
    torso = root.find(".//numeric[@name='init_qpos']")
    data = torso.get("data").split()

    bar_y = robot_y + 2.0
    # Ant spawn positions around the bar (bar is along y at θ=π/2, length 3.5).
    # 2 ants: on either side in x.
    # 3 ants: two on sides, one trailing (-y).
    # 4 ants: two on sides at bar_y, two trailing (-y).
    ant_offsets = {
        2: [(-1.0, 0.0), (1.0, 0.0)],
        3: [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.5)],
        4: [(-1.0, 0.5), (1.0, 0.5), (-1.0, -1.5), (1.0, -1.5)],
    }[n_ants]

    for i, (dx, dy) in enumerate(ant_offsets):
        base = i * 15
        data[base + 0] = str(robot_x + dx)
        data[base + 1] = str(bar_y + dy)

    # Bar at (robot_x, bar_y, θ=π/2)
    bar_base = n_ants * 15
    data[bar_base + 0] = str(robot_x)
    data[bar_base + 1] = str(bar_y)
    data[bar_base + 2] = str(1.5708)
    # Target xy stays at [bar_base+3, bar_base+4] — overwritten in reset()
    torso.set("data", " ".join(data))

    # Top-down camera
    rows, cols = len(maze_layout), len(maze_layout[0])
    cx = (rows - 1) * maze_size_scaling / 2
    cy = (cols - 1) * maze_size_scaling / 2
    cam_z = max(rows, cols) * maze_size_scaling * 0.9
    ET.SubElement(worldbody, "camera", name="maze_top", mode="fixed",
                  pos=f"{cx} {cy} {cam_z}", xyaxes="1 0 0 0 1 0")

    return ET.tostring(root), possible_goals


class AntMultiChicane(PipelineEnv):
    """Chicane with N ∈ {2, 3, 4} physical ants cooperatively pushing a bar."""

    def __init__(self, n_ants=2, ctrl_cost_weight=0.5, healthy_reward=1.0,
                 terminate_when_unhealthy=True, healthy_z_range=(0.2, 2.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False,
                 backend="generalized", maze_layout_name="u_maze",
                 maze_size_scaling=4.0, **kwargs):
        assert n_ants in (2, 3, 4)
        self.n_ants = n_ants
        xml_string, possible_goals = make_multi_maze(n_ants, maze_layout_name, maze_size_scaling)
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
        # Torso indices in link_names for all ants
        prefixes = ["a", "b", "c", "d"][:n_ants]
        self._torso_indices = [self.sys.link_names.index(f"torso_{p}") for p in prefixes]

        # Obs layout: ant_i(29)*n + bar_θ(1) + bar_xy(2) + zeros(1) + target_xy(2)
        # = 29n + 6
        body_dim = 29
        self.body_dim = body_dim
        self.state_dim = body_dim * n_ants + 3   # without target placeholder + target_xy
        self.obs_dim_full = body_dim * n_ants + 6
        self.obs_dim_agent = body_dim + 3 + 3     # body + bar_θxy + goal
        self.goal_dim = 3
        self.n_bodies = n_ants
        self.goal_indices = jp.array([self.state_dim, self.state_dim + 1, self.state_dim + 2])

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        _, target = self._random_target(rng3)
        q = q.at[-2:].set(target)
        qd = qd.at[-5:].set(0)   # bar(3) + target(2) velocities zeroed

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

        if "steps" in state.info:
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]

        min_z, max_z = self._healthy_z_range
        # All ants must be healthy
        is_healthy = jp.array(1.0)
        for ti in self._torso_indices:
            h = jp.where(pipeline_state.x.pos[ti, 2] < min_z, 0.0, 1.0)
            h = jp.where(pipeline_state.x.pos[ti, 2] > max_z, 0.0, h)
            is_healthy = is_healthy * h

        healthy_reward = (self._healthy_reward if self._terminate_when_unhealthy
                         else self._healthy_reward * is_healthy)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

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
        # Per-ant: qpos(15) + qvel(14) = 29
        ant_obs_parts = []
        for i in range(self.n_ants):
            qpos_i = pipeline_state.q[i * 15:(i + 1) * 15]
            qvel_i = pipeline_state.qd[i * 14:(i + 1) * 14]
            if self._exclude_current_positions_from_observation:
                qpos_i = qpos_i[2:]
            ant_obs_parts.append(qpos_i)
            ant_obs_parts.append(qvel_i)

        # Bar state: theta at q[n_ants*15 + 2], xy from x.pos[object_idx]
        bar_theta = pipeline_state.q[self.n_ants * 15 + 2:self.n_ants * 15 + 3]
        bar_xy = pipeline_state.x.pos[self._object_idx][:2]
        target_xy = pipeline_state.x.pos[-1][:2]

        return jp.concatenate(ant_obs_parts + [bar_theta, bar_xy, jp.zeros(1), target_xy])

    def _random_target(self, rng):
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]
