# Parts adapted from the training script of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

# ============================================================
# MACRL — Multi-Agent Contrastive RL
# CTDE: Centralized Training, Decentralized Execution
#
#   - Centralized SA_encoder over the joint (state, action)
#   - Decentralized actor, parameters shared across agents,
#     each receiving (state_i, goal)
#   - Replay buffer stores joint observations and actions
#   - Goal relabeling uses agent-0's trajectory for episode
#     boundaries and future-goal sampling
#   - Team reward is stored but unused in the gradient: the
#     contrastive critic is reward-free
#   - Depth-muP residual scaling for deep networks
# ============================================================

import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import functools
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

import enlighten
from brax import envs
from etils import epath
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue


# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    exp_name: str = "train_marl"
    gpu: str = ""
    seed: int = 1000
    track: bool = False
    wandb_project_name: str = "macrl"
    wandb_entity: str = None
    wandb_mode: str = "online"
    wandb_dir: str = "."
    wandb_group: str = "."
    capture_vis: bool = False
    checkpoint: bool = True

    # Environment
    env_id: str = "ant"
    episode_length: int = 1000
    # filled at runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0
    multi_body_goal: int = 0          # 1 = achieved pos is mean of both ants' xy (ant_pair envs)
    use_ball_achieved_goal: bool = False
    per_agent_goal: int = 0           # 1 = each agent has its own target (corridor env)
    min_pool_critic: int = 0          # 1 = use min instead of mean for factored critic aggregation
    # Critic aggregation on decoupled-body envs:
    #   "mean"        factored, per-agent embeddings mean-pooled (default)
    #   "min"         factored, min-pooled
    #   "concat"      monolithic, one encoder over the joint state and action
    #   "independent" no pooling, one contrastive term per agent
    critic_agg: str = "mean"
    # Only used with critic_agg="independent". 0 = contrast against the goal in
    # the observation, 1 = against the agent's own future position.
    # Requires goal_dim == 2.
    indep_own_goal: int = 0
    object_obs_dim: int = 0           # dims of object state in obs (2=ball xy, 3=bar/T-load xyθ)

    # Multi-agent
    n_agents: int = 2                  # number of cooperative agents
    ma_reward: str = "mean"            # "mean" | "sum" | "min" — team reward aggregation
    cooperative: int = 0               # 1 = split-joint mode (one shared body)
    partial_obs: int = 0               # 0 = full obs, 1 = agent sees own body only
    shared_params: int = 1             # CRL always uses shared params (kept for CLI compat)

    # Algorithm
    total_env_steps: int = 100_000_000
    num_epochs: int = 100
    num_envs: int = 512                # envs per agent (total = n_agents * num_envs)
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10_000
    min_replay_size: int = 1_000
    unroll_length: int = 62

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    g_encoder_depth: int = 0  # 0 = same as critic_depth
    actor_skip_connections: int = 0
    critic_skip_connections: int = 0

    num_episodes_per_env: int = 1
    training_steps_multiplier: int = 1
    use_all_batches: int = 0
    num_sgd_batches_per_training_step: int = 800

    eval_actor: int = 0                # 0 = deterministic, 1 = stochastic
    expl_actor: int = 1                # 1 = stochastic exploration

    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0

    # Depth-µP
    mup_residual_scaling: int = 1
    mup_scale_actor: int = 0
    zero_init_residual_last: int = 1

    # filled at runtime
    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0


# ----------------------------
# Initializers
# ----------------------------
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def _get_res_scale(network_depth: int, enable_mup: bool) -> float:
    n_blocks = max(1, int(network_depth) // 4)
    if not enable_mup:
        return 1.0
    return float(1.0 / np.sqrt(n_blocks))


# ----------------------------
# Residual block
# ----------------------------
def residual_block(x, width, normalize, activation, res_scale, zero_init_last):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    last_init = nn.initializers.zeros if zero_init_last else lecun_uniform
    x = nn.Dense(width, kernel_init=last_init, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    return identity + res_scale * x


# ----------------------------
# Networks
# ----------------------------
def use_factored_critic(args):
    """True when the critic encodes agents separately instead of the joint input.

    Decoupled-body envs use the factored critic. critic_agg="concat" selects the
    monolithic encoder instead.
    """
    decoupled = bool(getattr(args, 'multi_body_goal', 0) or
                     getattr(args, 'per_agent_goal', 0))
    return decoupled and getattr(args, 'critic_agg', 'mean') != 'concat'


def use_min_pool(args):
    return (getattr(args, 'critic_agg', 'mean') == 'min'
            or bool(getattr(args, 'min_pool_critic', 0)))


def use_action_factored_independent(args):
    """Independent critic on a shared-body env.

    Shared-body tasks split one body's joints across agents, so there is no
    per-agent body state to slice. The independent variant factors over actions
    instead: agent i scores (full state, its own action slice) against the goal,
    one InfoNCE term per agent, no pooling.
    """
    decoupled = bool(getattr(args, 'multi_body_goal', 0) or
                     getattr(args, 'per_agent_goal', 0))
    return (not decoupled) and getattr(args, 'critic_agg', 'mean') == 'independent'


def use_independent(args):
    """Independent per-agent contrastive critics, no pooling.

    Each agent scores its own state-action against the goal with its own InfoNCE
    matrix and the per-agent losses are averaged, so there is no cross-agent
    coupling in the critic loss.
    """
    return getattr(args, 'critic_agg', 'mean') == 'independent'


class SA_encoder(nn.Module):
    """
    Centralized state-action encoder.
    Inputs: joint_state (obs_dim * n_agents), joint_action (action_size * n_agents)
    Output: embedding vector of size 64
    """
    norm_type: str = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 1
    zero_init_residual_last: int = 1

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)
        activation = nn.relu if self.use_relu else nn.swish

        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_init_last = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_init_last)

        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    """
    Goal encoder.
    Input: goal vector (goal_dim,)
    Output: embedding vector of size 64
    """
    norm_type: str = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 1
    zero_init_residual_last: int = 1

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)
        activation = nn.relu if self.use_relu else nn.swish

        x = g
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_init_last = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_init_last)

        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    """
    Decentralized actor — parameter-shared across agents.
    Input: per-agent observation (obs_dim + goal_dim = obs_size_single)
    Output: (mean, log_std) for tanh-squashed Gaussian policy
    """
    action_size: int
    norm_type: str = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 0
    zero_init_residual_last: int = 1
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)
        activation = nn.relu if self.use_relu else nn.swish

        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_init_last = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_init_last)

        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


# ----------------------------
# Training state
# ----------------------------
@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray   # joint obs: (n_agents * obs_size_single,)
    action: jnp.ndarray        # joint action: (n_agents * action_size,)
    reward: jnp.ndarray        # team reward scalar
    discount: jnp.ndarray      # from agent-0's done flag
    extras: jnp.ndarray = ()


def load_params(path):
    with epath.Path(path).open("rb") as f:
        return pickle.loads(f.read())


def save_params(path, params):
    with epath.Path(path).open("wb") as f:
        f.write(pickle.dumps(params))


# ----------------------------
# CRL relabeling for multi-agent
# ----------------------------
@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_crl_fn_ma(buffer_config, transition, sample_key):
    """
    Goal relabeling for decoupled-body envs, one obs slice per agent.

    transition.observation has shape (episode_len, n_agents * obs_size_single)
    Layout per timestep:
      [obs_0 | obs_1 | ... | obs_{n-1}]
      Each obs_i is obs_size_single wide (raw env obs = state_i + env_goal_i).

    After relabeling:
      new_obs = [state_0 | state_1 | ... | state_{n-1} | hindsight_goal]
      where state_i = obs_i[:obs_dim]  (kinematic state, no env goal)
      and hindsight_goal = future_obs_0[goal_start_idx:goal_end_idx]
      (sampled from agent-0's future trajectory — shared team goal)

    transition.action has shape (episode_len, n_agents * action_size)
    """
    gamma, obs_dim, goal_start_idx, goal_end_idx, n_agents, obs_size_single = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)

    # Geometric discount probability matrix (upper-triangular)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount_mat = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount_mat

    # Restrict to same trajectory via agent-0's episode seed
    seeds = transition.extras["state_extras"]["seed"]  # (seq_len,)
    same_ep = jnp.equal(seeds[:, None], seeds[None, :]).astype(jnp.float32)
    probs = probs * same_ep + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))  # (seq_len,)

    # Future joint observations (using sampled future index)
    future_joint_obs = jnp.take(transition.observation, goal_index[:-1], axis=0)
    # (seq_len-1, n_agents * obs_size_single)

    # Hindsight goal from agent-0's future kinematic state
    # Agent-0's raw obs occupies the first obs_size_single dims
    goal = future_joint_obs[:, goal_start_idx:goal_end_idx]
    # (seq_len-1, goal_dim)

    # Extract per-agent state portions and concatenate → joint state
    # Agent i raw obs: [i*obs_size_single : (i+1)*obs_size_single]
    # Agent i state:   [i*obs_size_single : i*obs_size_single + obs_dim]
    joint_state = jnp.concatenate(
        [transition.observation[:-1, i * obs_size_single: i * obs_size_single + obs_dim]
         for i in range(n_agents)],
        axis=-1,
    )  # (seq_len-1, n_agents * obs_dim)

    future_joint_state = jnp.concatenate(
        [future_joint_obs[:, i * obs_size_single: i * obs_size_single + obs_dim]
         for i in range(n_agents)],
        axis=-1,
    )  # (seq_len-1, n_agents * obs_dim)

    # New observation for training: [joint_state | hindsight_goal]
    new_obs = jnp.concatenate([joint_state, goal], axis=-1)
    # (seq_len-1, n_agents * obs_dim + goal_dim)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "seed":       jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
        },
        "state":        joint_state,
        "future_state": future_joint_state,
        "future_action": jnp.take(transition.action, goal_index[:-1], axis=0),
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=    jnp.squeeze(transition.action[:-1]),
        reward=    jnp.squeeze(transition.reward[:-1]),
        discount=  jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


# ----------------------------
# CRL relabeling — cooperative (single shared body)
# ----------------------------
@functools.partial(jax.jit, static_argnames=("buffer_config",))
def flatten_crl_fn_coop(buffer_config, transition, sample_key):
    """
    Goal relabeling for shared-body envs, one obs for the whole team.

    transition.observation: (episode_len, obs_size_single)   — single body obs
    transition.action:      (episode_len, full_action_size)  — full joint action

    After relabeling:
      new_obs = [state (obs_dim) | hindsight_goal (goal_dim)]
    The actor receives (new_obs + one-hot agent_id) at training time.
    """
    gamma, obs_dim, goal_start_idx, goal_end_idx, multi_body_goal, use_ball_goal, per_agent_goal, goal_dim = buffer_config

    seq_len     = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)

    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount_mat   = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs          = is_future_mask * discount_mat

    seeds  = transition.extras["state_extras"]["seed"]
    same_ep = jnp.equal(seeds[:, None], seeds[None, :]).astype(jnp.float32)
    probs   = probs * same_ep + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))  # (seq_len,)

    future_obs   = jnp.take(transition.observation, goal_index[:-1], axis=0)
    if per_agent_goal:
        # Corridor: goal = concat of both ants' achieved xy (each aims for the other's start)
        goal = jnp.concatenate([future_obs[:, 0:2], future_obs[:, 29:31]], axis=-1)  # (T-1, 4)
    elif use_ball_goal:
        # Object envs: goal = last goal_dim dims of state (ball_xy or bar_θxy)
        goal = future_obs[:, obs_dim - goal_dim:obs_dim]              # (T-1, goal_dim)
    elif multi_body_goal:
        # Two-body envs with factored critic: goal = mean of both ants' achieved xy.
        goal = (future_obs[:, 0:2] + future_obs[:, 29:31]) / 2.0    # (T-1, 2)
    else:
        goal = future_obs[:, goal_start_idx:goal_end_idx]            # (T-1, goal_dim)
    state        = transition.observation[:-1, :obs_dim]               # (T-1, obs_dim)
    future_state = future_obs[:, :obs_dim]                             # (T-1, obs_dim)

    new_obs = jnp.concatenate([state, goal], axis=-1)                  # (T-1, obs_dim+goal_dim)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "seed":       jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
        },
        "state":         state,
        "future_state":  future_state,
        "future_action": jnp.take(transition.action, goal_index[:-1], axis=0),
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=    jnp.squeeze(transition.action[:-1]),
        reward=    jnp.squeeze(transition.reward[:-1]),
        discount=  jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("=" * 60, flush=True)
    print("MA-CRL  —  Multi-Agent Contrastive RL (CTDE)", flush=True)
    print("=" * 60, flush=True)
    # any _coop env_id auto-enables cooperative split-joint mode
    if "_coop" in args.env_id:
        args.cooperative = 1
    if args.n_agents == 1:
        args.cooperative = 1  # trivially cooperative; avoids reshape mismatch in eval
    print(f"  env_id      : {args.env_id}", flush=True)
    print(f"  n_agents    : {args.n_agents}", flush=True)
    print(f"  cooperative : {bool(args.cooperative)}", flush=True)
    print(f"  num_envs    : {args.num_envs}", flush=True)
    print(f"  critic_depth: {args.critic_depth}  actor_depth: {args.actor_depth}", flush=True)
    print(f"  depth-µP    : mup={args.mup_residual_scaling}  mup_actor={args.mup_scale_actor}", flush=True)
    print("=" * 60, flush=True)

    # Derived counts (num_training_steps_per_epoch recomputed after cooperative mode is resolved)
    total_envs = args.n_agents * args.num_envs
    args.env_steps_per_actor_step = total_envs * args.unroll_length  # may be overridden for coop
    args.num_prefill_env_steps    = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps  = int(np.ceil(args.min_replay_size / args.unroll_length))

    run_name = (
        f"MARL_{args.env_id}_nagents:{args.n_agents}_{args.batch_size}_{args.total_env_steps}"
        f"_nenvs:{args.num_envs}_cw:{args.critic_network_width}_aw:{args.actor_network_width}"
        f"_cd:{args.critic_depth}_ad:{args.actor_depth}"
        f"_mup:{args.mup_residual_scaling}_seed:{args.seed}"
    )

    if args.track:
        if args.wandb_group == ".":
            args.wandb_group = None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    if args.checkpoint:
        from pathlib import Path
        from datetime import datetime
        short_run_name = f"runs/marl_{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.makedirs(save_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    # ----------------------------
    # Environment setup
    # ----------------------------
    def make_env(env_id=None):
        env_id = env_id or args.env_id
        print(f"  creating env: {env_id}", flush=True)
        if env_id == "reacher":
            from envs.reacher import Reacher
            env = Reacher(backend="spring")
            args.obs_dim = 10; args.goal_start_idx = 4; args.goal_end_idx = 7
        elif env_id == "pusher":
            from envs.pusher import Pusher
            env = Pusher(backend="spring")
            args.obs_dim = 20; args.goal_start_idx = 10; args.goal_end_idx = 13
        elif env_id in ("ant", "ant_coop", "ant_easy"):
            from envs.ant import Ant
            env = Ant(backend="spring",
                      exclude_current_positions_from_observation=False,
                      terminate_when_unhealthy=True,
                      target_dist=3.0 if env_id == "ant_easy" else 10.0)
            args.obs_dim = 29; args.goal_start_idx = 0; args.goal_end_idx = 2
            if env_id == "ant_coop":
                args.cooperative = 1  # auto-enable split-joint cooperative mode
        elif "ant" in env_id and "maze" in env_id and "pair" not in env_id:
            if "gen" not in env_id:
                from envs.ant_maze import AntMaze
                _tmp = env_id.removesuffix("_coop").removeprefix("ant_")
                if _tmp == "maze":            maze_name = "u_maze"
                elif _tmp.startswith("maze_"): maze_name = _tmp[5:]
                else:                          maze_name = _tmp
                env = AntMaze(backend="spring",
                              exclude_current_positions_from_observation=False,
                              terminate_when_unhealthy=True,
                              maze_layout_name=maze_name)
            else:
                from envs.ant_maze_generalization import AntMazeGeneralization
                gen_idx = env_id.find("gen")
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=env_id[4:gen_idx - 1],
                    generalization_config=env_id[gen_idx + 4:],
                )
            args.obs_dim = 29; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif env_id in ("ant_ball", "ball_easy"):
            from envs.ant_ball import AntBall
            env = AntBall(backend="spring",
                          exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=True,
                          target_dist=2.0 if env_id == "ball_easy" else 5.0)
            args.obs_dim = 31; args.goal_start_idx = 28; args.goal_end_idx = 30
        elif env_id == "ant_push":
            from envs.ant_push import AntPush
            env = AntPush(backend="mjx")
            args.obs_dim = 31; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif env_id == "humanoid":
            from envs.humanoid import Humanoid
            env = Humanoid(backend="spring",
                           exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=True)
            args.obs_dim = 268; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif "humanoid" in env_id and "maze" in env_id:
            from envs.humanoid_maze import HumanoidMaze
            env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
            args.obs_dim = 268; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif env_id == "arm_reach":
            from envs.manipulation.arm_reach import ArmReach
            env = ArmReach(backend="mjx")
            args.obs_dim = 13; args.goal_start_idx = 7; args.goal_end_idx = 10
        elif env_id == "arm_binpick_easy":
            from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
            env = ArmBinpickEasy(backend="mjx")
            args.obs_dim = 17; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif env_id == "arm_binpick_hard":
            from envs.manipulation.arm_binpick_hard import ArmBinpickHard
            env = ArmBinpickHard(backend="mjx")
            args.obs_dim = 17; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif env_id == "arm_binpick_easy_EEF":
            from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
            env = ArmBinpickEasyEEF(backend="mjx")
            args.obs_dim = 11; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif "arm_grasp" in env_id:
            from envs.manipulation.arm_grasp import ArmGrasp
            cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
            env = ArmGrasp(cube_noise_scale=cube_noise_scale, backend="mjx")
            args.obs_dim = 23; args.goal_start_idx = 16; args.goal_end_idx = 23
        elif env_id == "arm_push_easy":
            from envs.manipulation.arm_push_easy import ArmPushEasy
            env = ArmPushEasy(backend="mjx")
            args.obs_dim = 17; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif env_id == "arm_push_hard":
            from envs.manipulation.arm_push_hard import ArmPushHard
            env = ArmPushHard(backend="mjx")
            args.obs_dim = 17; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif env_id == "point2d":
            from envs.point2d import Point2D
            env = Point2D(target_dist=5.0)
            args.obs_dim = 6; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif env_id == "ant_pair":
            from envs.ant_pair import AntPair
            env = AntPair(backend="spring", exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=True, target_dist=10.0)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = 1
        elif env_id == "ant_pair_ball":
            from envs.ant_pair_ball import AntPairBall
            env = AntPairBall(backend="spring", exclude_current_positions_from_observation=False,
                              terminate_when_unhealthy=True, target_dist=5.0)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
        elif env_id == "ant_pair_corridor":
            from envs.ant_pair_corridor import AntPairCorridor
            env = AntPairCorridor(backend="spring", terminate_when_unhealthy=True)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.per_agent_goal = 1
        elif env_id in ("ant_pair_chicane_3a", "ant_pair_chicane_4a"):
            # Multi-ant chicane: 3 or 4 physical ants cooperatively pushing the bar.
            from envs.ant_multi_chicane import AntMultiChicane
            n_ants = int(env_id[-2])  # "3a" or "4a"
            env = AntMultiChicane(n_ants=n_ants, backend="spring",
                                  exclude_current_positions_from_observation=False,
                                  terminate_when_unhealthy=True, maze_layout_name="u_maze")
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.object_obs_dim = 3
            args.multi_body_goal = 1
        elif env_id.startswith("ant_pair_chicane"):
            from envs.ant_pair_chicane import AntPairChicane
            _name = env_id.replace("ant_pair_chicane_", "").replace("ant_pair_chicane", "u_maze")
            env = AntPairChicane(backend="spring", exclude_current_positions_from_observation=False,
                                 terminate_when_unhealthy=True, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.object_obs_dim = 3  # bar pose (θ, x, y)
            args.multi_body_goal = 1
        elif env_id.startswith("ant_pair_tload"):
            from envs.ant_pair_tload import AntPairTLoad
            _name = env_id.replace("ant_pair_tload_", "").replace("ant_pair_tload", "t_slit")
            env = AntPairTLoad(backend="spring", exclude_current_positions_from_observation=False,
                               terminate_when_unhealthy=True, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.object_obs_dim = 3  # (θ, x, y) — same layout as chicane, HER slices the real pose
            args.multi_body_goal = 1
        elif env_id.startswith("ant_pair_maze_ball_coop"):
            from envs.ant_pair_maze_ball import AntPairMazeBall
            _name = env_id.replace("ant_pair_maze_ball_coop_", "").replace("ant_pair_maze_ball_coop", "u_maze")
            env = AntPairMazeBall(backend="spring", exclude_current_positions_from_observation=False,
                                  terminate_when_unhealthy=True, maze_layout_name=_name,
                                  xml_name="ant_pair_maze_ball_coop.xml",
                                  simultaneous_touch_radius=2.0)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.multi_body_goal = 1
        elif env_id.startswith("ant_pair_maze_ball"):
            from envs.ant_pair_maze_ball import AntPairMazeBall
            _name = env_id.replace("ant_pair_maze_ball_", "").replace("ant_pair_maze_ball", "u_maze")
            env = AntPairMazeBall(backend="spring", exclude_current_positions_from_observation=False,
                                  terminate_when_unhealthy=True, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.multi_body_goal = 1
        elif env_id.startswith("ant_pair_maze"):
            from envs.ant_pair_maze import AntPairMaze
            _name = env_id.replace("ant_pair_maze_", "").replace("ant_pair_maze", "u_maze")
            env = AntPairMaze(backend="spring", exclude_current_positions_from_observation=False,
                              terminate_when_unhealthy=True, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = 1
        else:
            raise NotImplementedError(f"Unknown env_id: {env_id}")
        return env

    env = make_env()
    env = envs.training.wrap(env, episode_length=args.episode_length)

    obs_size_single    = env.observation_size   # per-agent obs dim (includes env goal)
    action_size_single = env.action_size        # full env action dim
    # Pair envs set args.goal_dim explicitly. Non-pair branches only set
    # goal_start_idx/goal_end_idx — derive goal_dim from those if missing.
    if not hasattr(args, "goal_dim"):
        args.goal_dim = args.goal_end_idx - args.goal_start_idx
    goal_dim           = args.goal_dim

    # ------------------------------------------------------------------
    # Cooperative mode: n_agents split the joints of one shared body.
    # Each agent controls (action_size_single // n_agents) joints.
    # All agents observe the full state; the actor input is augmented
    # with a one-hot agent ID so parameter-shared agents specialize.
    #
    # Non-cooperative mode: n_agents run in independent environments.
    # ------------------------------------------------------------------
    if args.cooperative:
        assert action_size_single % args.n_agents == 0, (
            f"action_size ({action_size_single}) must divide evenly by "
            f"n_agents ({args.n_agents})"
        )
        action_size_per_agent = action_size_single // args.n_agents
        full_action_size      = action_size_single          # what env.step() expects
        if args.partial_obs and hasattr(args, "obs_dim_agent"):
            actor_input_size = args.obs_dim_agent + args.n_agents
        else:
            actor_input_size = obs_size_single + args.n_agents   # obs + one-hot agent ID
        sim_envs              = args.num_envs               # single sim per parallel slot
        args.env_steps_per_actor_step = args.num_envs * args.unroll_length
        if use_factored_critic(args):
            # Factored critic: SA_encoder called per-agent with (body_i + shared_obj, action_i)
            body_dim = 29
            shared_obj_dim = args.obs_dim - args.n_agents * body_dim  # 0=maze, 2=ball, 3=bar/T
            sa_state_dim  = body_dim + shared_obj_dim
            sa_action_dim = action_size_per_agent
        elif use_action_factored_independent(args):
            # Shared-body independent critic: full state, but only one agent's
            # action slice. The smaller action input is inherent to independent
            # learning, since an agent does not observe its partner's action.
            sa_state_dim  = args.obs_dim
            sa_action_dim = action_size_per_agent
        else:
            sa_state_dim  = args.obs_dim
            sa_action_dim = full_action_size
    else:
        action_size_per_agent = action_size_single
        full_action_size      = action_size_single * args.n_agents
        actor_input_size      = obs_size_single
        sim_envs              = total_envs
        sa_state_dim          = args.obs_dim * args.n_agents
        sa_action_dim         = action_size_single * args.n_agents

    # Recompute now that env_steps_per_actor_step is finalised (coop may have lowered it)
    args.num_training_steps_per_epoch = max(1, (
        (args.total_env_steps - args.num_prefill_env_steps)
        // (args.num_epochs * args.env_steps_per_actor_step)
    ))
    total_training_steps = (
        args.num_epochs
        * args.num_training_steps_per_epoch
        * args.training_steps_multiplier
    )
    total_env_steps_actual = (
        args.num_prefill_env_steps
        + total_training_steps * args.env_steps_per_actor_step
    )

    env_keys  = jax.random.split(env_key, sim_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step  = jax.jit(env.step)

    eval_num_envs = args.num_eval_envs if args.cooperative else args.n_agents * args.num_eval_envs

    print(f"  obs_size_single  : {obs_size_single}", flush=True)
    print(f"  action_size      : {action_size_single}  (per-agent: {action_size_per_agent})", flush=True)
    print(f"  goal_dim         : {goal_dim}", flush=True)
    if args.cooperative:
        print(f"  mode             : cooperative split-joint  ({args.n_agents} agents × {action_size_per_agent} joints)", flush=True)
        print(f"  critic input     : state={sa_state_dim}  action={sa_action_dim}", flush=True)
    else:
        print(f"  joint_obs_dim    : {args.n_agents * obs_size_single}  (buffer storage)", flush=True)
        print(f"  joint_state_dim  : {sa_state_dim}  (critic input)", flush=True)
    print(f"  --- training schedule ---", flush=True)
    print(f"  env_steps/actor_step : {args.env_steps_per_actor_step:,}", flush=True)
    print(f"  prefill_actor_steps  : {args.num_prefill_actor_steps}", flush=True)
    print(f"  train_steps/epoch    : {args.num_training_steps_per_epoch}", flush=True)
    print(f"  total epochs         : {args.num_epochs}", flush=True)
    print(f"  total env steps      : ~{total_env_steps_actual:,.0f}  (target: {args.total_env_steps:,})", flush=True)
    print(flush=True)

    eval_env = make_env(args.env_id)
    eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)
    eval_env_keys = jax.random.split(eval_env_key, eval_num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)

    # ----------------------------
    # Networks
    # ----------------------------
    # Actor: per-agent, parameter-shared.
    # Cooperative:     input = obs_size_single + n_agents (+ one-hot agent ID)
    # Non-cooperative: input = obs_size_single
    # Output: action_size_per_agent (cooperative) or action_size_single (non-coop)
    actor = Actor(
        action_size=action_size_per_agent,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        use_relu=args.use_relu,
        mup_residual_scaling=int(bool(args.mup_scale_actor) and bool(args.mup_residual_scaling)),
        zero_init_residual_last=args.zero_init_residual_last,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, actor_input_size])),
        tx=optax.adam(learning_rate=args.actor_lr),
    )
    print(f"  Actor params: {sum(x.size for x in jax.tree_util.tree_leaves(actor_state.params)):,}", flush=True)

    # SA_encoder (CENTRALIZED):
    # Cooperative:     input = (single body state, full joint action)
    # Non-cooperative: input = (joint state all agents, joint action all agents)
    sa_encoder = SA_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        use_relu=args.use_relu,
        mup_residual_scaling=args.mup_residual_scaling,
        zero_init_residual_last=args.zero_init_residual_last,
    )
    sa_encoder_params = sa_encoder.init(
        sa_key,
        np.ones([1, sa_state_dim]),
        np.ones([1, sa_action_dim]),
    )

    # G_encoder: input = goal_dim (shared goal from agent-0's future trajectory)
    g_depth = args.g_encoder_depth if args.g_encoder_depth > 0 else args.critic_depth
    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=g_depth,
        use_relu=args.use_relu,
        mup_residual_scaling=args.mup_residual_scaling,
        zero_init_residual_last=args.zero_init_residual_last,
    )
    g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_dim]))

    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=args.critic_lr),
    )
    print(f"  SA_encoder params: {sum(x.size for x in jax.tree_util.tree_leaves(sa_encoder_params)):,}", flush=True)
    print(f"  G_encoder  params: {sum(x.size for x in jax.tree_util.tree_leaves(g_encoder_params)):,}", flush=True)

    # Entropy coefficient: sum of per-agent entropies
    target_entropy = -args.entropy_param * args.n_agents * action_size_per_agent
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    # ----------------------------
    # Replay buffer
    # ----------------------------
    if args.cooperative:
        # Single body: store raw obs + full joint action
        dummy_obs    = jnp.zeros((obs_size_single,))
        dummy_action = jnp.zeros((full_action_size,))
    else:
        # Independent bodies: store joint obs + joint action across agents
        dummy_obs    = jnp.zeros((args.n_agents * obs_size_single,))
        dummy_action = jnp.zeros((args.n_agents * action_size_single,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}},
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal  = jax.jit(buffer.sample_internal)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,          # buffer tracks num_envs groups, not total_envs
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # ----------------------------
    # Actor steps (MA versions)
    # ----------------------------
    def _build_joint_transition(env_state, nstate, actions_all, extra_fields):
        """
        From a stepped env_state (total_envs,) and actions (n_agents, num_envs, action_size),
        build a Transition with joint obs/action for the buffer.
        Agent-0 slice provides reward, discount, seed, truncation for the group.
        """
        n_agents   = args.n_agents
        num_envs   = args.num_envs

        # Raw obs: (total_envs, obs_size_single) → (n_agents, num_envs, obs_size_single)
        obs_per_agent = env_state.obs.reshape(n_agents, num_envs, obs_size_single)
        # Joint obs: (num_envs, n_agents * obs_size_single)
        joint_obs = obs_per_agent.transpose(1, 0, 2).reshape(num_envs, n_agents * obs_size_single)
        # Joint action: (num_envs, n_agents * action_size_single)
        joint_action = actions_all.transpose(1, 0, 2).reshape(num_envs, n_agents * action_size_single)

        # Team reward
        rewards_per_agent = nstate.reward.reshape(n_agents, num_envs)
        if args.ma_reward == "sum":
            team_reward = rewards_per_agent.sum(0)
        elif args.ma_reward == "min":
            team_reward = rewards_per_agent.min(0)
        else:  # mean
            team_reward = rewards_per_agent.mean(0)

        # Discount & extras from agent-0 (indices 0:num_envs)
        agent0_done     = nstate.done[:num_envs]
        agent0_discount = 1.0 - agent0_done
        state_extras    = {x: nstate.info[x][:num_envs] for x in extra_fields}

        return Transition(
            observation=joint_obs,
            action=joint_action,
            reward=team_reward,
            discount=agent0_discount,
            extras={"state_extras": state_extras},
        )

    # ----------------------------
    # Cooperative actor steps
    # (single simulation, n_agents each contribute action_size_per_agent joints)
    # Each agent receives its own one-hot ID appended to the shared observation
    # so the parameter-shared network can specialize per agent.
    # ----------------------------
    def _coop_obs_i(obs, i):
        """Build per-agent observation. Full or partial depending on args.partial_obs."""
        if args.partial_obs and hasattr(args, "obs_dim_agent"):
            body_dim = 29
            own_body = obs[..., i * body_dim:(i + 1) * body_dim]
            if getattr(args, 'per_agent_goal', 0):
                # Corridor: goals = [target_a(2), target_b(2)], each agent gets its own
                goal_per = args.goal_dim // args.n_agents  # 2
                goal = obs[..., -(args.goal_dim - i * goal_per):] if i == 0 else obs[..., -goal_per:]
                # agent 0: obs[-4:-2] = target_a; agent 1: obs[-2:] = target_b
                goal = obs[..., -(args.goal_dim - i * goal_per):-(args.goal_dim - (i + 1) * goal_per)] if i < args.n_agents - 1 else obs[..., -goal_per:]
                agent_obs = jnp.concatenate([own_body, goal], axis=-1)
            elif getattr(args, 'object_obs_dim', 0):
                # Bar/T-load: extract object state (θ, x, y) before goal
                n_obj = args.object_obs_dim
                obj_state = obs[..., -args.goal_dim - n_obj:-args.goal_dim]
                goal = obs[..., -args.goal_dim:]
                agent_obs = jnp.concatenate([own_body, obj_state, goal], axis=-1)
            elif hasattr(args, 'use_ball_achieved_goal') and args.use_ball_achieved_goal:
                ball_xy = obs[..., -args.goal_dim - 2:-args.goal_dim]
                goal = obs[..., -args.goal_dim:]
                agent_obs = jnp.concatenate([own_body, ball_xy, goal], axis=-1)
            else:
                goal = obs[..., -args.goal_dim:]
                agent_obs = jnp.concatenate([own_body, goal], axis=-1)
        else:
            agent_obs = obs
        agent_id = jnp.zeros((*agent_obs.shape[:-1], args.n_agents), dtype=obs.dtype).at[..., i].set(1.0)
        return jnp.concatenate([agent_obs, agent_id], axis=-1)

    def cooperative_deterministic_actor_step(training_state, env, env_state, extra_fields):
        sub_actions = []
        for i in range(args.n_agents):
            obs_i     = _coop_obs_i(env_state.obs, i)
            mean_i, _ = actor.apply(training_state.actor_state.params, obs_i)
            sub_actions.append(nn.tanh(mean_i))
        full_action = jnp.concatenate(sub_actions, axis=-1)  # (num_envs, full_action_size)
        nstate = env.step(env_state, full_action)
        return nstate, Transition(
            observation=env_state.obs,
            action=full_action,
            reward=nstate.reward,
            discount=1.0 - nstate.done,
            extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
        )

    def cooperative_stochastic_actor_step(training_state, env, env_state, key, extra_fields):
        agent_keys  = jax.random.split(key, args.n_agents)
        sub_actions = []
        for i in range(args.n_agents):
            obs_i          = _coop_obs_i(env_state.obs, i)
            means_i, lsd_i = actor.apply(training_state.actor_state.params, obs_i)
            stds_i         = jnp.exp(lsd_i)
            noise_i        = jax.random.normal(agent_keys[i], shape=means_i.shape, dtype=means_i.dtype)
            sub_actions.append(nn.tanh(means_i + stds_i * noise_i))
        full_action = jnp.concatenate(sub_actions, axis=-1)
        nstate = env.step(env_state, full_action)
        return nstate, Transition(
            observation=env_state.obs,
            action=full_action,
            reward=nstate.reward,
            discount=1.0 - nstate.done,
            extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
        )

    def deterministic_actor_step_ma(training_state, env, env_state, extra_fields):
        obs_all = env_state.obs.reshape(args.n_agents, args.num_envs, obs_size_single)
        # vmap actor over agent axis (parameter sharing)
        means_all, _ = jax.vmap(
            lambda o: actor.apply(training_state.actor_state.params, o)
        )(obs_all)
        actions_all = nn.tanh(means_all)
        nstate = env.step(env_state, actions_all.reshape(total_envs, action_size_single))
        return nstate, _build_joint_transition(env_state, nstate, actions_all, extra_fields)

    def stochastic_actor_step_ma(training_state, env, env_state, key, extra_fields):
        obs_all     = env_state.obs.reshape(args.n_agents, args.num_envs, obs_size_single)
        agent_keys  = jax.random.split(key, args.n_agents)

        means_all, log_stds_all = jax.vmap(
            lambda o: actor.apply(training_state.actor_state.params, o)
        )(obs_all)
        stds_all    = jnp.exp(log_stds_all)
        noise_all   = jax.vmap(
            lambda m, k: jax.random.normal(k, shape=m.shape, dtype=m.dtype)
        )(means_all, agent_keys)
        actions_all = nn.tanh(means_all + stds_all * noise_all)

        nstate = env.step(env_state, actions_all.reshape(total_envs, action_size_single))
        return nstate, _build_joint_transition(env_state, nstate, actions_all, extra_fields)

    # ----------------------------
    # Experience collection
    # ----------------------------
    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        def f(carry, _):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.cooperative:
                if args.expl_actor == 1:
                    nstate, transition = cooperative_stochastic_actor_step(
                        training_state, env, env_state, current_key,
                        extra_fields=("truncation", "seed"),
                    )
                else:
                    nstate, transition = cooperative_deterministic_actor_step(
                        training_state, env, env_state,
                        extra_fields=("truncation", "seed"),
                    )
            elif args.expl_actor == 1:
                nstate, transition = stochastic_actor_step_ma(
                    training_state, env, env_state, current_key,
                    extra_fields=("truncation", "seed"),
                )
            else:
                nstate, transition = deterministic_actor_step_ma(
                    training_state, env, env_state,
                    extra_fields=("truncation", "seed"),
                )
            return (nstate, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, _):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, key)
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=args.num_prefill_actor_steps,
        )[0]

    # ----------------------------
    # Gradient updates
    # ----------------------------
    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        transitions = jax.tree_util.tree_map(lambda x: x[:args.batch_size], transitions)

        def actor_loss_fn(actor_params, critic_params, log_alpha, transitions, key):
            agent_keys     = jax.random.split(key, args.n_agents)
            actions_list   = []
            log_probs_list = []

            if args.cooperative:
                # obs layout: [state (obs_dim) | hindsight_goal (goal_dim)]
                obs   = transitions.observation
                state = obs[:, :args.obs_dim]                    # (B, obs_dim)
                goal  = obs[:, args.obs_dim:]                    # (B, goal_dim)
                body_dim = 29

                for i in range(args.n_agents):
                    obs_i = _coop_obs_i(obs, i)
                    means_i, log_stds_i = actor.apply(actor_params, obs_i)
                    stds_i   = jnp.exp(log_stds_i)
                    x_t_i    = means_i + stds_i * jax.random.normal(agent_keys[i], means_i.shape, dtype=means_i.dtype)
                    action_i = nn.tanh(x_t_i)
                    lp_i     = jax.scipy.stats.norm.logpdf(x_t_i, loc=means_i, scale=stds_i)
                    lp_i    -= jnp.log((1 - jnp.square(action_i)) + 1e-6)
                    actions_list.append(action_i)
                    log_probs_list.append(lp_i.sum(-1))

                full_action = jnp.concatenate(actions_list, axis=-1)  # (B, full_action_size)

                if use_factored_critic(args):
                    # Factored critic: call SA_encoder per-agent with shared object state
                    shared_state = state[:, args.n_agents * body_dim:]  # object state (0/2/3 dims)
                    sa_reprs = []
                    for i in range(args.n_agents):
                        body_i   = state[:, i * body_dim:(i + 1) * body_dim]
                        body_i   = jnp.concatenate([body_i, shared_state], axis=-1)
                        action_i = actions_list[i]
                        sa_reprs.append(sa_encoder.apply(critic_params["sa_encoder"], body_i, action_i))

                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)

                    if use_min_pool(args):
                        # Min-pool: Q = min over agents of -||SA_i - G||
                        # Prevents free-riding: worst agent is the bottleneck
                        per_agent_q = []
                        for sa_i in sa_reprs:
                            per_agent_q.append(-jnp.sqrt(jnp.sum((sa_i - g_repr) ** 2, axis=-1)))
                        qf_pi = jnp.stack(per_agent_q, axis=0).min(axis=0)  # (B,)
                    elif use_independent(args):
                        # Independent (IPPO-style): each agent scores against the
                        # goal on its own; the actor objective averages per-agent
                        # values with no pooled team representation.
                        per_agent_q = [
                            -jnp.sqrt(jnp.sum((sa_i - g_repr) ** 2, axis=-1))
                            for sa_i in sa_reprs]
                        qf_pi = jnp.stack(per_agent_q, axis=0).mean(axis=0)  # (B,)
                    else:
                        sa_repr = jnp.stack(sa_reprs, axis=0).mean(axis=0)
                        qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
                elif use_action_factored_independent(args):
                    # Shared body, independent: agent i scores (full state, own
                    # action slice) against the goal, averaged with no pooling.
                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                    per_agent_q = []
                    for i in range(args.n_agents):
                        sa_i = sa_encoder.apply(critic_params["sa_encoder"],
                                                state, actions_list[i])
                        per_agent_q.append(
                            -jnp.sqrt(jnp.sum((sa_i - g_repr) ** 2, axis=-1)))
                    qf_pi = jnp.stack(per_agent_q, axis=0).mean(axis=0)
                else:
                    sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, full_action)
                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                    qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            else:
                # obs layout: [joint_states (n*obs_dim) | hindsight_goal (goal_dim)]
                obs  = transitions.observation
                goal = obs[:, args.n_agents * args.obs_dim:]

                for i in range(args.n_agents):
                    state_i  = obs[:, i * args.obs_dim: (i + 1) * args.obs_dim]
                    obs_i    = jnp.concatenate([state_i, goal], axis=-1)
                    means_i, log_stds_i = actor.apply(actor_params, obs_i)
                    stds_i   = jnp.exp(log_stds_i)
                    x_t_i    = means_i + stds_i * jax.random.normal(agent_keys[i], means_i.shape, dtype=means_i.dtype)
                    action_i = nn.tanh(x_t_i)
                    lp_i     = jax.scipy.stats.norm.logpdf(x_t_i, loc=means_i, scale=stds_i)
                    lp_i    -= jnp.log((1 - jnp.square(action_i)) + 1e-6)
                    actions_list.append(action_i)
                    log_probs_list.append(lp_i.sum(-1))

                joint_state  = obs[:, :args.n_agents * args.obs_dim]
                full_action  = jnp.concatenate(actions_list, axis=-1)
                sa_repr = sa_encoder.apply(critic_params["sa_encoder"], joint_state, full_action)
                g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            log_prob = jnp.stack(log_probs_list, axis=-1).sum(-1)

            if args.disable_entropy:
                loss = -jnp.mean(qf_pi)
            else:
                loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

            return loss, log_prob

        def alpha_loss_fn(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - target_entropy))

        (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.alpha_state.params["log_alpha"],
            transitions,
            key,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
            training_state.alpha_state.params, log_prob
        )
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)
        training_state  = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss":      actor_loss,
            "alpha_loss":      alpha_loss,
            "log_alpha":       training_state.alpha_state.params["log_alpha"],
        }
        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        transitions = jax.tree_util.tree_map(lambda x: x[:args.batch_size], transitions)

        def critic_loss_fn(critic_params, transitions, key):
            obs = transitions.observation
            logits_list = None   # set only by the independent-critic path
            if args.cooperative:
                # obs layout: [state (obs_dim) | goal (goal_dim)]
                state       = obs[:, :args.obs_dim]
                goal        = obs[:, args.obs_dim:]
                full_action = transitions.action
                body_dim = 29

                if use_factored_critic(args):
                    # Factored critic: call SA_encoder per-agent with shared object state
                    shared_state = state[:, args.n_agents * body_dim:]
                    sa_reprs = []
                    for i in range(args.n_agents):
                        body_i   = state[:, i * body_dim:(i + 1) * body_dim]
                        body_i   = jnp.concatenate([body_i, shared_state], axis=-1)
                        action_i = full_action[:, i * action_size_per_agent:(i + 1) * action_size_per_agent]
                        sa_reprs.append(sa_encoder.apply(critic_params["sa_encoder"], body_i, action_i))

                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)

                    per_agent_logits = [
                        -jnp.sqrt(jnp.sum(
                            (sa_i[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                        for sa_i in sa_reprs]
                    if use_min_pool(args):
                        # Min-pool contrastive loss: per-agent logit matrices, take min
                        logits = jnp.stack(per_agent_logits, axis=0).min(axis=0)  # (B, B)
                    elif use_independent(args):
                        if getattr(args, 'indep_own_goal', 0):
                            # Positive is agent i's own future position. Its xy
                            # sits at the agent's body slot in future_state.
                            fut = transitions.extras["future_state"]
                            logits_list = []
                            for i, sa_i in enumerate(sa_reprs):
                                g_i = fut[:, i * body_dim:i * body_dim + 2]
                                gr_i = g_encoder.apply(critic_params["g_encoder"], g_i)
                                logits_list.append(-jnp.sqrt(jnp.sum(
                                    (sa_i[:, None, :] - gr_i[None, :, :]) ** 2, axis=-1)))
                        else:
                            # Independent: keep the per-agent matrices; the NCE loss
                            # is averaged over agents with no pooling (see below).
                            logits_list = per_agent_logits
                    else:
                        sa_repr = jnp.stack(sa_reprs, axis=0).mean(axis=0)
                        logits = -jnp.sqrt(jnp.sum(
                            (sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                elif use_action_factored_independent(args):
                    # Shared body, independent: one InfoNCE term per agent over
                    # (full state, own action slice), averaged with no pooling.
                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                    logits_list = []
                    for i in range(args.n_agents):
                        action_i = full_action[:, i * action_size_per_agent:
                                               (i + 1) * action_size_per_agent]
                        sa_i = sa_encoder.apply(critic_params["sa_encoder"],
                                                state, action_i)
                        logits_list.append(-jnp.sqrt(jnp.sum(
                            (sa_i[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)))
                else:
                    sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, full_action)
                    g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                    logits = -jnp.sqrt(jnp.sum(
                        (sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            else:
                # obs layout: [joint_state (n*obs_dim) | goal (goal_dim)]
                joint_state  = obs[:, :args.n_agents * args.obs_dim]
                goal         = obs[:, args.n_agents * args.obs_dim:]
                joint_action = transitions.action
                sa_repr = sa_encoder.apply(critic_params["sa_encoder"], joint_state, joint_action)
                g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
                logits = -jnp.sqrt(jnp.sum(
                    (sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))

            # NCE contrastive loss: diagonal = (s,a)→g matches; off-diagonal = negatives.
            # Independent mode averages the loss over per-agent matrices; every
            # other mode has a single team matrix.
            if logits_list is None:
                logits_list = [logits]

            loss = 0.0
            logsumexp_terms = []
            for lg in logits_list:
                loss = loss - jnp.mean(jnp.diag(lg) - jax.nn.logsumexp(lg, axis=1))
                lse = jax.nn.logsumexp(lg + 1e-6, axis=1)
                logsumexp_terms.append(lse)
                loss = loss + args.logsumexp_penalty_coeff * jnp.mean(lse ** 2)
            loss = loss / len(logits_list)
            logsumexp = jnp.stack(logsumexp_terms, axis=0).mean(axis=0)

            return loss, (logsumexp, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))

        (loss, (logsumexp, *_)), grad = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            training_state.critic_state.params, transitions, key
        )
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state   = training_state.replace(critic_state=new_critic_state)

        return training_state, {
            "critic_loss": loss,
            "logsumexp":   logsumexp.mean(),
        }

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
        metrics = {**actor_metrics, **critic_metrics}
        return (training_state, key), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        exp_key1, exp_key2, sample_key, train_key, sgd_key = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(training_state, env_state, buffer_state, exp_key1)
        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step
        )

        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *transitions_list
        )

        # CRL relabeling
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        if args.cooperative:
            buffer_cfg = (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx, int(getattr(args, 'multi_body_goal', 0)), int(getattr(args, 'use_ball_achieved_goal', 0)), int(getattr(args, 'per_agent_goal', 0)), args.goal_dim)
            transitions = jax.vmap(flatten_crl_fn_coop, in_axes=(None, 0, 0))(
                buffer_cfg, transitions, batch_keys
            )
        else:
            buffer_cfg = (
                args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx,
                args.n_agents, obs_size_single,
            )
            transitions = jax.vmap(flatten_crl_fn_ma, in_axes=(None, 0, 0))(
                buffer_cfg, transitions, batch_keys
            )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
        )

        permutation = jax.random.permutation(exp_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        num_full = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_full * args.batch_size], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]), transitions
        )

        if args.use_all_batches == 0:
            n_total  = transitions.observation.shape[0]
            selected = jax.random.permutation(sgd_key, n_total)[:args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(lambda x: x[selected], transitions)

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, train_key), transitions)
        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier),
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    # ----------------------------
    # Prefill
    # ----------------------------
    print("Prefilling replay buffer ...", flush=True)
    key, prefill_key = jax.random.split(key)
    t_prefill = time.time()
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )
    print(f"  Prefill done in {time.time() - t_prefill:.1f}s  "
          f"(buffer size: {replay_buffer.size(buffer_state)})", flush=True)

    # ----------------------------
    # Evaluator
    # ----------------------------
    if args.cooperative:
        def _eval_actor_step(ts, env, es, extra_fields):
            nstate, transition = cooperative_deterministic_actor_step(ts, env, es, extra_fields)
            return nstate, transition
    elif args.eval_actor == 0:
        def _eval_actor_step(ts, env, es, extra_fields):
            nstate, transition = deterministic_actor_step_ma(ts, env, es, extra_fields)
            return nstate, transition
    else:
        key, eval_actor_key = jax.random.split(key)
        def _eval_actor_step(ts, env, es, extra_fields):
            nstate, transition = stochastic_actor_step_ma(ts, env, es, eval_actor_key, extra_fields)
            return nstate, transition

    evaluator = CrlEvaluator(
        _eval_actor_step,
        eval_env,
        num_eval_envs=eval_num_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    training_walltime = 0.0
    start_time        = time.time()
    print("\nStarting training ...", flush=True)

    enl_manager = enlighten.get_manager()

    epoch_bar = enl_manager.counter(
        total=args.num_epochs,
        desc="Training",
        unit="epoch",
        color="cyan",
        bar_format=(
            "{desc}{desc_pad}{percentage:3.0f}%|{bar}| "
            "{count}/{total} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
        ),
    )
    status_bar = enl_manager.status_bar(
        status_format=(
            "Epoch {ep}/{total} │ steps={steps} │ "
            "A_loss={actor_loss} │ C_loss={critic_loss} │ "
            "α={log_alpha} │ reward={reward} │ suc={success} │ dist={dist}"
        ),
        color="bold_white_on_black",
        ep=0, total=args.num_epochs,
        steps="n/a", actor_loss="n/a", critic_loss="n/a",
        log_alpha="n/a", reward="n/a", success="n/a", dist="n/a",
    )

    for ne in range(args.num_epochs):
        t0 = time.time()
        key, epoch_key = jax.random.split(key)

        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_time        = time.time() - t0
        training_walltime += epoch_time
        hours_passed      = (time.time() - start_time) / 3600

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_time

        metrics = {
            "training/sps":           sps,
            "training/walltime":      training_walltime,
            "training/envsteps":      training_state.env_steps.item(),
            "training/gradient_steps": training_state.gradient_steps.item(),
            **{f"training/{k}": float(v) for k, v in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        # ---- terminal summary ----
        actor_loss  = metrics.get("training/actor_loss",  float("nan"))
        critic_loss = metrics.get("training/critic_loss", float("nan"))
        log_alpha   = metrics.get("training/log_alpha",   float("nan"))
        logsumexp   = metrics.get("training/logsumexp",   float("nan"))
        success     = metrics.get("eval/episode_success", float("nan"))
        dist        = metrics.get("eval/episode_dist",    float("nan"))
        reward      = metrics.get("eval/episode_reward",  float("nan"))
        env_steps   = metrics["training/envsteps"]

        suc_str    = f"{success:.4f}" if not np.isnan(success) else "n/a"
        dist_str   = f"{dist:.2f}"    if not np.isnan(dist)    else "n/a"
        reward_str = f"{reward:.2f}"  if not np.isnan(reward)  else "n/a"

        # persistent status bar (bottom of terminal)
        status_bar.update(
            ep=ne + 1, total=args.num_epochs,
            steps=f"{env_steps:.2e}",
            actor_loss=f"{actor_loss:.4f}",
            critic_loss=f"{critic_loss:.4f}",
            log_alpha=f"{log_alpha:.3f}",
            reward=reward_str,
            success=suc_str,
            dist=dist_str,
        )

        # scrolling log line printed above the bars
        enl_manager.write(
            f"[Epoch {ne + 1:>4}/{args.num_epochs}] "
            f"steps={env_steps:.2e}  sps={sps:,.0f}  "
            f"actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}  "
            f"log_alpha={log_alpha:.3f}  logsumexp={logsumexp:.3f}  "
            f"reward={reward_str}  success={suc_str}  dist={dist_str}  "
            f"time={epoch_time:.1f}s  elapsed={hours_passed:.2f}h"
        )
        print(f"PROG steps={env_steps:.2e} suc={suc_str}", flush=True)

        epoch_bar.update()

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                save_params(f"{save_path}/step_{int(training_state.env_steps)}.pkl", params)

        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline":
                trigger_sync()

    # ----------------------------
    # Final checkpoint
    # ----------------------------
    if args.checkpoint:
        params = (
            training_state.alpha_state.params,
            training_state.actor_state.params,
            training_state.critic_state.params,
        )
        save_params(f"{save_path}/final.pkl", params)
        print(f"\nCheckpoint saved to {save_path}/final.pkl", flush=True)

    epoch_bar.close()
    status_bar.close()
    enl_manager.stop()

    total_hours = (time.time() - start_time) / 3600
    print(f"\nTraining complete.  Total time: {total_hours:.2f}h", flush=True)
