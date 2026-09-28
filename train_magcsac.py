# Parts adapted from the training script of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

# ============================================================
# MA-GCSAC  —  Multi-Agent Goal-Conditioned Soft Actor-Critic
#
# Goal-conditioned baseline for MA-CRL comparison.
# Identical architecture to MA-SAC but with:
#   1. Goal-reaching reward  (-dist_to_goal dense, or sparse 0/1)
#   2. Q-network conditioned on (state + goal), not just state
#   3. Optional Hindsight Experience Replay (HER, "future" strategy)
#
# The ant env already appends the target position to obs:
#   obs = [qpos(15) | qvel(14) | target_xy(2)]  =  31 dims
# so obs[:2] = ant x,y  and  obs[-2:] = goal x,y.
#
# Cooperative:  n_agents share one body (split joints)
# ============================================================

import os, jax, flax, tyro, time, optax, wandb, pickle, random, functools
import wandb_osh, numpy as np, flax.linen as nn, jax.numpy as jnp
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
    exp_name: str = "train_magcsac"
    gpu: str = ""
    seed: int = 1000
    track: bool = False
    wandb_project_name: str = "macrl"
    wandb_entity: str = None
    wandb_mode: str = "online"
    wandb_dir: str = "."
    wandb_group: str = "."
    checkpoint: bool = True

    env_id: str = "ant"
    episode_length: int = 1000
    obs_dim: int = 0          # filled at runtime (body state dims, excl. goal)
    goal_dim: int = 2         # target x,y
    obs_size: int = 0         # filled at runtime (obs_dim + goal_dim)
    goal_start_idx: int = 0
    goal_end_idx: int = 0
    use_ball_achieved_goal: bool = False  # ant_ball: achieved = ball pos, not ant pos
    multi_body_goal: int = 0             # 1 = achieved pos is mean of both ants' xy (ant_pair envs)
    per_agent_goal: int = 0              # 1 = corridor: each agent has its own 2D goal (goal_dim=4)
    per_agent_her: int = 0               # 1 = relabel and reward on own position, not the team mean

    n_agents: int = 2
    ma_reward: str = "mean"
    cooperative: int = 0
    partial_obs: int = 0        # 0 = full obs (current), 1 = agent sees own body only
    shared_params: int = 0      # 0 = separate params per agent (current), 1 = shared with one-hot ID

    # Goal-reaching reward
    reward_type: str = "sparse"  # "dense" = -dist_to_goal | "sparse" = float(dist<0.5)
    healthy_reward_bonus: float = 0.0  # per-step survival bonus (helps ant stay upright)
    terminate_when_unhealthy: int = 1  # 0 = never terminate (full 1000-step episodes for HER)

    # HER
    use_her: int = 1             # 1 = enable Hindsight Experience Replay
    her_ratio: float = 0.8       # fraction of each batch relabelled with HER goals

    total_env_steps: int = 100_000_000
    num_epochs: int = 100
    num_envs: int = 512
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005

    max_replay_size: int = 10_000
    min_replay_size: int = 1_000
    unroll_length: int = 62

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4

    num_episodes_per_env: int = 1
    num_sgd_batches: int = 1        # gradient updates per trajectory sample (CRL uses 800)
    training_steps_multiplier: int = 1
    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0
    use_layer_norm: int = 1   # 0 = plain MLP (no LayerNorm), matches JaxGCRL baselines
    use_residual: int = 1     # 0 = no skip connections (plain MLP)

    mup_residual_scaling: int = 1
    mup_scale_actor: int = 0
    zero_init_residual_last: int = 1

    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0


# ----------------------------
# Initializers & utils
# ----------------------------
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def _get_res_scale(depth: int, enable: bool) -> float:
    n = max(1, int(depth) // 4)
    return float(1.0 / np.sqrt(n)) if enable else 1.0


def residual_block(x, width, normalize, activation, res_scale, zero_init_last, use_residual=True):
    identity = x
    for _ in range(3):
        x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
    last_init = nn.initializers.zeros if zero_init_last else lecun_uniform
    x = nn.Dense(width, kernel_init=last_init, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    return (identity + res_scale * x) if use_residual else x


# ----------------------------
# Networks
# ----------------------------
class QNetwork(nn.Module):
    """Centralized Q-network: Q(state+goal, joint_action) -> scalar."""
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    use_layer_norm: int = 1
    use_residual: int = 1
    mup_residual_scaling: int = 1
    zero_init_residual_last: int = 1

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        normalize  = (lambda y: nn.LayerNorm()(y)) if self.use_layer_norm else (lambda y: y)
        activation = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_last  = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_last,
                               use_residual=bool(self.use_residual))
        x = nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x.squeeze(-1)


class Actor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    use_layer_norm: int = 1
    use_residual: int = 1
    mup_residual_scaling: int = 0
    zero_init_residual_last: int = 1
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x):
        normalize  = (lambda y: nn.LayerNorm()(y)) if self.use_layer_norm else (lambda y: y)
        activation = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_last  = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_last,
                               use_residual=bool(self.use_residual))
        mean    = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


# ----------------------------
# State containers
# ----------------------------
@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_states: Any
    critic_state: TrainState
    target_critic_params: Any
    alpha_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    next_observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


def save_params(path, p):
    with epath.Path(path).open("wb") as f: f.write(pickle.dumps(p))


# ----------------------------
# HER relabeling
# ----------------------------
def make_her_relabel(reward_type, her_ratio, use_ball_achieved_goal=False, multi_body_goal=False, per_agent_goal=False, object_obs_dim=0, goal_dim=2):
    """Returns a JIT-compatible HER relabeling function (future strategy, episode-aware).

    Uses the seed field in extras to detect episode boundaries so that future
    goals are always sampled from the *same* ant life — not from a later life
    after an auto-reset.  Without this, short-episode envs (terminate_when_unhealthy)
    produce goals near the reset origin, collapsing the learned policy to
    "return to origin" and giving 0 % evaluation success.
    """
    def her_relabel(trs, key):
        B, T, _ = trs.observation.shape
        key, mask_key, idx_key = jax.random.split(key, 3)

        # ── Episode boundary detection via seed ──────────────────────────────
        seeds = trs.extras["state_extras"]["seed"]   # (B, T)
        seed_change = jnp.concatenate([
            jnp.zeros((B, 1), dtype=jnp.int32),
            (seeds[:, 1:] != seeds[:, :-1]).astype(jnp.int32)
        ], axis=1)                                    # 1 where a new episode starts
        # seed_change_next[b,t] = 1  iff  step t+1 belongs to a new episode
        seed_change_next = jnp.concatenate(
            [seed_change[:, 1:], jnp.ones((B, 1), dtype=jnp.int32)], axis=1)

        # Right-to-left scan: ep_remaining[b,t] = steps still in the same
        # episode *after* step t  (0 = last step of the episode).
        def _scan_ep(carry, sc_next_t):
            new = jnp.where(sc_next_t > 0.5, jnp.zeros_like(carry), carry + 1)
            return new, new
        _, ep_rem_rev = jax.lax.scan(
            _scan_ep, jnp.zeros(B, dtype=jnp.int32),
            jnp.flip(seed_change_next.T, axis=0))    # scan from t=T-1 down to t=0
        ep_remaining = jnp.flip(ep_rem_rev, axis=0).T  # (B, T)

        # ── Sample future index within the same episode ───────────────────────
        t_arange  = jnp.arange(T)
        rand_frac = jax.random.uniform(idx_key, (B, T))
        ep_safe   = jnp.maximum(ep_remaining, 1)             # avoid /0 when clipping
        future_off = (rand_frac * ep_safe).astype(jnp.int32) + 1   # in [1, ep_remaining]
        future_idx = jnp.clip(t_arange[None, :] + future_off, 0, T - 1)  # (B, T)

        # Per-step HER mask: skip if no valid future same-episode step exists
        has_future = ep_remaining > 0                        # (B, T)
        her_mask   = (jax.random.uniform(mask_key, (B, T)) < her_ratio) & has_future

        if object_obs_dim > 0:
            # Object envs (chicane/tload): future goal = object (theta, x, y) from future state
            n_obj = object_obs_dim
            future_pos = jax.vmap(lambda ob, idx: ob[idx, -goal_dim - n_obj:-goal_dim])(trs.observation, future_idx)
        elif use_ball_achieved_goal:
            future_pos = jax.vmap(lambda ob, idx: ob[idx, -4:-2])(trs.observation, future_idx)
        elif per_agent_goal:
            future_pos = jax.vmap(lambda ob, idx: jnp.concatenate([ob[idx, :2], ob[idx, 29:31]], axis=-1))(trs.observation, future_idx)
        elif multi_body_goal:
            future_pos = jax.vmap(lambda ob, idx: (ob[idx, :2] + ob[idx, 29:31]) / 2.0)(trs.observation, future_idx)
        else:
            future_pos = jax.vmap(lambda ob, idx: ob[idx, :2])(trs.observation, future_idx)

        goal_d = goal_dim if object_obs_dim > 0 else (4 if per_agent_goal else 2)
        new_goal     = jnp.where(her_mask[:, :, None], future_pos,
                                 trs.observation[:, :, -goal_d:])
        new_obs      = trs.observation.at[:, :, -goal_d:].set(new_goal)
        new_next_obs = trs.next_observation.at[:, :, -goal_d:].set(new_goal)

        if object_obs_dim > 0:
            # Object envs: achieved = object xy, goal = goal xy (both 2D for distance)
            achieved_next = trs.next_observation[:, :, -goal_dim - object_obs_dim + 1:-goal_dim]  # obj x,y
            goal_xy = new_goal[:, :, -2:]  # skip theta placeholder, take x,y
            dist = jnp.linalg.norm(achieved_next - goal_xy, axis=-1)
            if reward_type == "sparse":
                new_rew_relabeled = (dist < 0.5).astype(jnp.float32)
            else:
                new_rew_relabeled = -dist
            new_reward = jnp.where(her_mask, new_rew_relabeled, trs.reward)
            return trs._replace(observation=new_obs, next_observation=new_next_obs,
                                reward=new_reward)
        elif use_ball_achieved_goal:
            achieved_next = trs.next_observation[:, :, -4:-2]
        elif per_agent_goal:
            dist_a = jnp.linalg.norm(trs.next_observation[:, :, :2] - new_goal[:, :, :2], axis=-1)
            dist_b = jnp.linalg.norm(trs.next_observation[:, :, 29:31] - new_goal[:, :, 2:4], axis=-1)
            dist = jnp.maximum(dist_a, dist_b)
            if reward_type == "sparse":
                new_rew_relabeled = (dist < 0.5).astype(jnp.float32)
            else:
                new_rew_relabeled = -dist
            new_reward = jnp.where(her_mask, new_rew_relabeled, trs.reward)
            return trs._replace(observation=new_obs, next_observation=new_next_obs,
                                reward=new_reward)
        elif multi_body_goal:
            # Both ants must reach: use max distance for sparse reward
            dist_a = jnp.linalg.norm(trs.next_observation[:, :, :2] - new_goal, axis=-1)
            dist_b = jnp.linalg.norm(trs.next_observation[:, :, 29:31] - new_goal, axis=-1)
            dist = jnp.maximum(dist_a, dist_b)
            if reward_type == "sparse":
                new_rew_relabeled = (dist < 0.5).astype(jnp.float32)
            else:
                new_rew_relabeled = -dist
            new_reward = jnp.where(her_mask, new_rew_relabeled, trs.reward)
            return trs._replace(observation=new_obs, next_observation=new_next_obs,
                                reward=new_reward)
        else:
            achieved_next = trs.next_observation[:, :, :2]
        dist = jnp.linalg.norm(achieved_next - new_goal, axis=-1)  # (B, T)
        if reward_type == "sparse":
            new_rew_relabeled = (dist < 0.5).astype(jnp.float32)
        else:
            new_rew_relabeled = -dist
        new_reward = jnp.where(her_mask, new_rew_relabeled, trs.reward)

        return trs._replace(observation=new_obs, next_observation=new_next_obs,
                            reward=new_reward)
    return her_relabel


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.env_id.endswith("_coop"):
        args.cooperative = 1
    if args.n_agents == 1:
        args.cooperative = 1  # trivially cooperative; avoids reshape mismatch in actor_step

    print("=" * 60, flush=True)
    print("MA-GCSAC  —  Multi-Agent Goal-Conditioned SAC", flush=True)
    print("=" * 60, flush=True)
    print(f"  env_id      : {args.env_id}", flush=True)
    print(f"  n_agents    : {args.n_agents}", flush=True)
    print(f"  cooperative : {bool(args.cooperative)}", flush=True)
    print(f"  num_envs    : {args.num_envs}", flush=True)
    print(f"  reward_type : {args.reward_type}", flush=True)
    print(f"  use_her     : {bool(args.use_her)}  her_ratio={args.her_ratio}", flush=True)
    print("=" * 60, flush=True)

    total_envs = args.n_agents * args.num_envs
    args.env_steps_per_actor_step = total_envs * args.unroll_length
    args.num_prefill_env_steps    = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps  = int(np.ceil(args.min_replay_size / args.unroll_length))

    run_name = (f"MAGCSAC_{args.env_id}_na:{args.n_agents}_{args.total_env_steps}"
                f"_ne:{args.num_envs}_cd:{args.critic_depth}_ad:{args.actor_depth}"
                f"_her:{bool(args.use_her)}_rt:{args.reward_type}_s:{args.seed}")

    if args.track:
        if args.wandb_group == ".": args.wandb_group = None
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group, dir=args.wandb_dir,
                   config=vars(args), name=run_name, save_code=True)
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    if args.checkpoint:
        from pathlib import Path
        from datetime import datetime
        save_path = Path(args.wandb_dir) / Path(
            f"runs/magcsac_{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(save_path, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, eval_key = jax.random.split(key, 5)

    # ----------------------------
    # Environment
    # ----------------------------
    def make_env(env_id=None):
        eid = env_id or args.env_id
        print(f"  creating env: {eid}", flush=True)
        twu = bool(args.terminate_when_unhealthy)
        if eid in ("ant", "ant_coop", "ant_easy"):
            from envs.ant import Ant
            env = Ant(backend="spring", exclude_current_positions_from_observation=False,
                      terminate_when_unhealthy=twu,
                      target_dist=3.0 if eid == "ant_easy" else 10.0)
            args.obs_dim = 29; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
        elif eid in ("ant_maze_coop",) or ("ant" in eid and "maze" in eid and "pair" not in eid):
            from envs.ant_maze import AntMaze
            _tmp = eid.removesuffix("_coop").removeprefix("ant_"); name = "u_maze" if _tmp == "maze" else (_tmp[5:] if _tmp.startswith("maze_") else _tmp)
            env = AntMaze(backend="spring", exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=twu,
                          maze_layout_name=name if name else "U_MAZE")
            args.obs_dim = 29; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
        elif eid in ("ant_ball", "ball_easy"):
            from envs.ant_ball import AntBall
            env = AntBall(backend="spring", exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=twu,
                          target_dist=2.0 if eid == "ball_easy" else 5.0)
            args.obs_dim = 29; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.use_ball_achieved_goal = True
        elif eid in ("humanoid", "humanoid_coop"):
            from envs.humanoid import Humanoid
            env = Humanoid(backend="spring", exclude_current_positions_from_observation=False,
                           terminate_when_unhealthy=twu)
            args.obs_dim = 268; args.goal_dim = 3
            args.goal_start_idx = 0; args.goal_end_idx = 3
        elif eid in ("humanoid_maze_coop",) or ("humanoid" in eid and "maze" in eid):
            from envs.humanoid_maze import HumanoidMaze
            name = eid.replace("_coop","").replace("humanoid_maze_","")
            env = HumanoidMaze(backend="spring", maze_layout_name=name if name else "U_MAZE")
            args.obs_dim = 268; args.goal_dim = 3
            args.goal_start_idx = 0; args.goal_end_idx = 3
        elif eid == "point2d":
            from envs.point2d import Point2D
            env = Point2D(target_dist=5.0)
            args.obs_dim = 6; args.goal_dim = 2; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif eid == "ant_pair":
            from envs.ant_pair import AntPair
            env = AntPair(backend="spring", exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=twu, target_dist=10.0)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = True
        elif eid == "ant_pair_ball":
            from envs.ant_pair_ball import AntPairBall
            env = AntPairBall(backend="spring", exclude_current_positions_from_observation=False,
                              terminate_when_unhealthy=twu, target_dist=5.0)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = True
            args.use_ball_achieved_goal = True
        elif eid == "ant_pair_corridor":
            from envs.ant_pair_corridor import AntPairCorridor
            env = AntPairCorridor(backend="spring", terminate_when_unhealthy=twu)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.per_agent_goal = 1
        elif eid.startswith("ant_pair_chicane"):
            from envs.ant_pair_chicane import AntPairChicane
            _name = eid.replace("ant_pair_chicane_", "").replace("ant_pair_chicane", "u_maze")
            env = AntPairChicane(backend="spring", exclude_current_positions_from_observation=False,
                                 terminate_when_unhealthy=twu, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.object_obs_dim = 3
            args.multi_body_goal = True
        elif eid.startswith("ant_pair_tload"):
            from envs.ant_pair_tload import AntPairTLoad
            _name = eid.replace("ant_pair_tload_", "").replace("ant_pair_tload", "t_slit")
            env = AntPairTLoad(backend="spring", exclude_current_positions_from_observation=False,
                               terminate_when_unhealthy=twu, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = env.goal_dim
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.use_ball_achieved_goal = True
            args.object_obs_dim = 3
            args.multi_body_goal = True
        elif eid.startswith("ant_pair_maze_ball_coop"):
            from envs.ant_pair_maze_ball import AntPairMazeBall
            _name = eid.replace("ant_pair_maze_ball_coop_", "").replace("ant_pair_maze_ball_coop", "u_maze")
            env = AntPairMazeBall(backend="spring", exclude_current_positions_from_observation=False,
                                  terminate_when_unhealthy=twu, maze_layout_name=_name,
                                  xml_name="ant_pair_maze_ball_coop.xml",
                                  simultaneous_touch_radius=2.0)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = True
            args.use_ball_achieved_goal = True
        elif eid.startswith("ant_pair_maze_ball"):
            from envs.ant_pair_maze_ball import AntPairMazeBall
            _name = eid.replace("ant_pair_maze_ball_", "").replace("ant_pair_maze_ball", "u_maze")
            env = AntPairMazeBall(backend="spring", exclude_current_positions_from_observation=False,
                                  terminate_when_unhealthy=twu, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = True
            args.use_ball_achieved_goal = True
        elif eid.startswith("ant_pair_maze"):
            from envs.ant_pair_maze import AntPairMaze
            _name = eid.replace("ant_pair_maze_", "").replace("ant_pair_maze", "u_maze")
            env = AntPairMaze(backend="spring", exclude_current_positions_from_observation=False,
                              terminate_when_unhealthy=twu, maze_layout_name=_name)
            args.obs_dim = env.obs_dim_full - env.goal_dim; args.goal_dim = 2
            args.goal_start_idx = 0; args.goal_end_idx = 2
            args.obs_dim_agent = env.obs_dim_agent
            args.multi_body_goal = True
        else:
            raise NotImplementedError(f"Unknown env_id: {eid}")
        return env

    env      = make_env()

    # per_agent_her relabels to an agent's own future position instead of the
    # team mean, giving a denser signal on the pair envs. Set after make_env,
    # which forces multi_body_goal on. Evaluation is unaffected.
    if getattr(args, 'per_agent_her', 0):
        args.multi_body_goal = False
        print("  per_agent_her=1: multi_body_goal disabled, HER relabels to own position",
              flush=True)

    env      = envs.training.wrap(env, episode_length=args.episode_length)
    obs_size = env.observation_size   # e.g. 31 for ant (body 29 + goal 2)
    act_size = env.action_size
    args.obs_size = obs_size

    if args.cooperative:
        base = act_size // args.n_agents
        rem  = act_size % args.n_agents
        joints_per_agent    = [base + (1 if i < rem else 0) for i in range(args.n_agents)]
        action_slice_starts = [sum(joints_per_agent[:i]) for i in range(args.n_agents)]
        if args.partial_obs and hasattr(args, "obs_dim_agent"):
            actor_input_size = args.obs_dim_agent + args.n_agents
        else:
            actor_input_size = obs_size + args.n_agents   # obs + one-hot agent ID
        sim_envs            = args.num_envs
        args.env_steps_per_actor_step = args.num_envs * args.unroll_length
        # Q conditioned on (state + goal): use full obs_size
        sa_state_dim        = obs_size
        sa_action_dim       = act_size
    else:
        joints_per_agent    = [act_size] * args.n_agents
        action_slice_starts = [0] * args.n_agents
        actor_input_size    = obs_size
        sim_envs            = total_envs
        sa_state_dim        = obs_size * args.n_agents
        sa_action_dim       = act_size * args.n_agents

    args.num_training_steps_per_epoch = max(1, (
        (args.total_env_steps - args.num_prefill_env_steps)
        // (args.num_epochs * args.env_steps_per_actor_step)
    ))
    total_env_steps_actual = (args.num_prefill_env_steps +
        args.num_epochs * args.num_training_steps_per_epoch
        * args.training_steps_multiplier * args.env_steps_per_actor_step)

    eval_num_envs = args.num_eval_envs if args.cooperative else args.n_agents * args.num_eval_envs
    eval_env = make_env(); eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)

    print(f"  obs_size: {obs_size}  act_size: {act_size}  joints_per_agent: {joints_per_agent}", flush=True)
    print(f"  Q-network state input: {sa_state_dim}  (includes goal)", flush=True)
    print(f"  total env steps: ~{total_env_steps_actual:,.0f}", flush=True)

    env_keys  = jax.random.split(env_key, sim_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step  = jax.jit(env.step)

    # ----------------------------
    # Networks
    # ----------------------------
    actor_keys = jax.random.split(key, args.n_agents + 1); key = actor_keys[0]
    if args.shared_params:
        a = Actor(action_size=joints_per_agent[0], network_width=args.actor_network_width,
                  network_depth=args.actor_depth, use_relu=args.use_relu,
                  use_layer_norm=args.use_layer_norm, use_residual=args.use_residual,
                  mup_residual_scaling=int(bool(args.mup_scale_actor) and bool(args.mup_residual_scaling)),
                  zero_init_residual_last=args.zero_init_residual_last)
        inp = np.ones([1, actor_input_size])
        ast = TrainState.create(apply_fn=a.apply,
                                params=a.init(actor_keys[1], inp),
                                tx=optax.adam(args.actor_lr))
        actors = [a] * args.n_agents
        actor_states = [ast]  # single shared state
    else:
        actors, actor_states = [], []
        for i in range(args.n_agents):
            a = Actor(action_size=joints_per_agent[i], network_width=args.actor_network_width,
                      network_depth=args.actor_depth, use_relu=args.use_relu,
                      use_layer_norm=args.use_layer_norm, use_residual=args.use_residual,
                      mup_residual_scaling=int(bool(args.mup_scale_actor) and bool(args.mup_residual_scaling)),
                      zero_init_residual_last=args.zero_init_residual_last)
            inp = np.ones([1, actor_input_size])
            ast = TrainState.create(apply_fn=a.apply,
                                    params=a.init(actor_keys[i + 1], inp),
                                    tx=optax.adam(args.actor_lr))
            actors.append(a); actor_states.append(ast)

    key, q_key1, q_key2 = jax.random.split(key, 3)
    qnet = QNetwork(network_width=args.critic_network_width, network_depth=args.critic_depth,
                    use_relu=args.use_relu, use_layer_norm=args.use_layer_norm,
                    use_residual=args.use_residual, mup_residual_scaling=args.mup_residual_scaling,
                    zero_init_residual_last=args.zero_init_residual_last)
    q1_params = qnet.init(q_key1, np.ones([1, sa_state_dim]), np.ones([1, sa_action_dim]))
    q2_params = qnet.init(q_key2, np.ones([1, sa_state_dim]), np.ones([1, sa_action_dim]))
    critic_state = TrainState.create(apply_fn=None,
                                     params={"q1": q1_params, "q2": q2_params},
                                     tx=optax.adam(args.critic_lr))
    target_critic_params = {"q1": q1_params, "q2": q2_params}

    key, alpha_key = jax.random.split(key)
    target_entropy = -args.entropy_param * sum(joints_per_agent)
    alpha_state = TrainState.create(apply_fn=None,
                                    params={"log_alpha": jnp.asarray(0.0)},
                                    tx=optax.adam(args.alpha_lr))

    training_state = TrainingState(
        env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()),
        actor_states=actor_states, critic_state=critic_state,
        target_critic_params=target_critic_params, alpha_state=alpha_state)

    print(f"  Actor[0] params: {sum(x.size for x in jax.tree_util.tree_leaves(actor_states[0].params)):,}", flush=True)
    print(f"  Q-network params: {sum(x.size for x in jax.tree_util.tree_leaves(q1_params)):,}", flush=True)

    # ----------------------------
    # Replay buffer
    # ----------------------------
    if args.cooperative:
        dummy_obs = jnp.zeros((obs_size,)); dummy_act = jnp.zeros((act_size,))
    else:
        dummy_obs = jnp.zeros((args.n_agents * obs_size,)); dummy_act = jnp.zeros((args.n_agents * act_size,))

    dummy_transition = Transition(
        observation=dummy_obs, next_observation=dummy_obs,
        action=dummy_act, reward=0.0, discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}})

    def jit_wrap(buf):
        buf.insert_internal = jax.jit(buf.insert_internal)
        buf.sample_internal  = jax.jit(buf.sample_internal)
        return buf

    replay_buffer = jit_wrap(TrajectoryUniformSamplingQueue(
        max_replay_size=args.max_replay_size, dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size, num_envs=args.num_envs,
        episode_length=args.episode_length))
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _coop_obs_i(obs, i):
        """Build per-agent observation. Full or partial depending on args.partial_obs."""
        if args.partial_obs and hasattr(args, "obs_dim_agent"):
            # ant_pair envs: obs = [ant_a(29), ant_b(29), ...extras..., goal]
            body_dim = 29  # qpos(15) + qvel(14) per ant
            own_body = obs[..., i * body_dim:(i + 1) * body_dim]
            if getattr(args, 'per_agent_goal', 0):
                goal_per = args.goal_dim // args.n_agents  # 2
                if i < args.n_agents - 1:
                    goal = obs[..., -(args.goal_dim - i * goal_per):-(args.goal_dim - (i + 1) * goal_per)]
                else:
                    goal = obs[..., -goal_per:]
                agent_obs = jnp.concatenate([own_body, goal], axis=-1)
            elif getattr(args, 'object_obs_dim', 0):
                # Object envs (chicane/tload): extract full object state before goal
                n_obj = args.object_obs_dim
                obj_state = obs[..., -args.goal_dim - n_obj:-args.goal_dim]
                goal = obs[..., -args.goal_dim:]
                agent_obs = jnp.concatenate([own_body, obj_state, goal], axis=-1)
            elif args.use_ball_achieved_goal:
                # Ball envs: include ball_xy (2 dims) before goal
                goal = obs[..., -args.goal_dim:]
                ball_xy = obs[..., -args.goal_dim - 2:-args.goal_dim]
                agent_obs = jnp.concatenate([own_body, ball_xy, goal], axis=-1)
            else:
                goal = obs[..., -args.goal_dim:]
                agent_obs = jnp.concatenate([own_body, goal], axis=-1)
        else:
            # Original behavior: full obs for all agents
            agent_obs = obs
        aid = jnp.zeros((*agent_obs.shape[:-1], args.n_agents), dtype=obs.dtype).at[..., i].set(1.0)
        return jnp.concatenate([agent_obs, aid], axis=-1)

    def _achieved_pos(obs):
        """Extract achieved position from observation."""
        if getattr(args, 'object_obs_dim', 0):
            # Object envs (chicane/tload): achieved = last goal_dim dims of state
            # For 3D goal (theta,x,y): achieved pos for HER distance = object xy
            n_obj = args.object_obs_dim
            return obs[..., -args.goal_dim - n_obj + 1:-args.goal_dim]  # object x,y (skip theta)
        elif args.use_ball_achieved_goal:
            return obs[..., -4:-2]                            # ball xy
        elif args.multi_body_goal:
            return (obs[..., :2] + obs[..., 29:31]) / 2.0    # mean of both ants' xy
        else:
            return obs[..., :2]                               # ant xy

    def _goal_reward(next_obs):
        """Compute goal-reaching reward from next observation."""
        achieved = _achieved_pos(next_obs)
        goal = next_obs[..., -args.goal_dim:]
        if getattr(args, 'object_obs_dim', 0):
            # Object envs (chicane/tload): reward = dist(object_xy, target_xy)
            # Object xy = last 2 of the 3D object state; target xy = last 2 of goal
            obj_xy = achieved                         # already extracted as object x,y
            goal_xy = goal[..., -2:]                  # skip theta placeholder, take x,y
            dist = jnp.linalg.norm(obj_xy - goal_xy, axis=-1)
        elif getattr(args, 'per_agent_goal', 0):
            goal_a = next_obs[..., -4:-2]
            goal_b = next_obs[..., -2:]
            dist_a = jnp.linalg.norm(next_obs[..., :2] - goal_a, axis=-1)
            dist_b = jnp.linalg.norm(next_obs[..., 29:31] - goal_b, axis=-1)
            dist = jnp.maximum(dist_a, dist_b)
        elif args.multi_body_goal:
            # Both ants must reach the goal: use max distance
            dist_a = jnp.linalg.norm(next_obs[..., :2] - goal, axis=-1)
            dist_b = jnp.linalg.norm(next_obs[..., 29:31] - goal, axis=-1)
            dist = jnp.maximum(dist_a, dist_b)
        else:
            dist = jnp.linalg.norm(achieved - goal, axis=-1)
        if args.reward_type == "sparse":
            return (dist < 0.5).astype(jnp.float32)
        else:
            return -dist

    # HER relabeling function (closed over reward_type, her_ratio, and env type)
    her_relabel = make_her_relabel(args.reward_type, args.her_ratio,
                                   use_ball_achieved_goal=args.use_ball_achieved_goal,
                                   multi_body_goal=bool(args.multi_body_goal),
                                   per_agent_goal=bool(args.per_agent_goal),
                                   object_obs_dim=getattr(args, 'object_obs_dim', 0),
                                   goal_dim=args.goal_dim)

    def actor_step(ts, env, env_state, key, stochastic, extra_fields):
        agent_keys = jax.random.split(key, args.n_agents)
        sub_actions, log_probs = [], []
        for i, a in enumerate(actors):
            ap = ts.actor_states[0 if args.shared_params else i].params
            inp = _coop_obs_i(env_state.obs, i) if args.cooperative else \
                  env_state.obs[:, i*obs_size:(i+1)*obs_size]
            m, ls = a.apply(ap, inp); s = jnp.exp(ls)
            xt = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype) if stochastic else m
            act = nn.tanh(xt)
            lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act**2 + 1e-6)).sum(-1)
            sub_actions.append(act); log_probs.append(lp)
        full_act = jnp.concatenate(sub_actions, axis=-1)
        env_act  = full_act if args.cooperative else full_act.reshape(sim_envs, act_size)
        nstate   = env.step(env_state, env_act)

        # Health signal: (1-done) penalises falling when twu=True.
        # When twu=False, done=0 always; use env's is_healthy from reward_survive.
        if bool(args.terminate_when_unhealthy):
            _health = 1.0 - nstate.done
        else:
            _health = nstate.metrics["reward_survive"]

        if args.cooperative:
            cur_obs = env_state.obs; nxt_obs = nstate.obs
            disc = 1.0 - nstate.done
            rew  = _goal_reward(nstate.obs) + args.healthy_reward_bonus * _health
            extras = {"state_extras": {x: nstate.info[x] for x in extra_fields}}
        else:
            n_ag, nenv = args.n_agents, args.num_envs
            obs_j  = env_state.obs.reshape(n_ag, nenv, obs_size).transpose(1,0,2).reshape(nenv, n_ag*obs_size)
            nobs_j = nstate.obs.reshape(n_ag, nenv, obs_size).transpose(1,0,2).reshape(nenv, n_ag*obs_size)
            cur_obs = obs_j; nxt_obs = nobs_j
            disc = 1.0 - nstate.done[:nenv]
            rew  = _goal_reward(nstate.obs[:nenv]) + args.healthy_reward_bonus * _health[:nenv]
            extras = {"state_extras": {x: nstate.info[x][:nenv] for x in extra_fields}}

        return nstate, Transition(observation=cur_obs, next_observation=nxt_obs,
                                   action=full_act, reward=rew, discount=disc, extras=extras)

    @jax.jit
    def get_experience(ts, env_state, buffer_state, key):
        def f(carry, _):
            es, k = carry; k, nk = jax.random.split(k)
            nes, tr = actor_step(ts, env, es, k, True, ("truncation", "seed"))
            return (nes, nk), tr
        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill(ts, env_state, buffer_state, key):
        def f(carry, _):
            ts, es, bs, k = carry; k, nk = jax.random.split(k)
            es, bs = get_experience(ts, es, bs, k)
            ts = ts.replace(env_steps=ts.env_steps + args.env_steps_per_actor_step)
            return (ts, es, bs, nk), ()
        return jax.lax.scan(f, (ts, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    # ----------------------------
    # SAC Updates
    # ----------------------------
    def soft_update(params, target_params, tau):
        return jax.tree_util.tree_map(lambda p, tp: tau*p + (1-tau)*tp, params, target_params)

    @jax.jit
    def update_critic(transitions, ts, key):
        transitions = jax.tree_util.tree_map(lambda x: x[:args.batch_size], transitions)
        alpha = jnp.exp(ts.alpha_state.params["log_alpha"])

        next_obs  = transitions.next_observation
        agent_keys = jax.random.split(key, args.n_agents)
        next_sub_acts, next_log_probs = [], []
        for i, a in enumerate(actors):
            ap = ts.actor_states[0 if args.shared_params else i].params
            inp = _coop_obs_i(next_obs, i) if args.cooperative else \
                  next_obs[:, i*obs_size:(i+1)*obs_size]
            m, ls = a.apply(ap, inp); s = jnp.exp(ls)
            xt = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
            act = nn.tanh(xt)
            lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act**2 + 1e-6)).sum(-1)
            next_sub_acts.append(act); next_log_probs.append(lp)
        next_action   = jnp.concatenate(next_sub_acts, axis=-1)
        next_log_prob = sum(next_log_probs)

        if args.cooperative:
            # Q conditioned on full obs (includes goal)
            next_state = next_obs
            cur_state  = transitions.observation
        else:
            next_state = next_obs
            cur_state  = transitions.observation

        next_q1 = qnet.apply(ts.target_critic_params["q1"], next_state, next_action)
        next_q2 = qnet.apply(ts.target_critic_params["q2"], next_state, next_action)
        next_q  = jnp.minimum(next_q1, next_q2) - alpha * next_log_prob
        target_q = transitions.reward + args.gamma * transitions.discount * next_q

        def loss_fn(critic_params):
            q1 = qnet.apply(critic_params["q1"], cur_state, transitions.action)
            q2 = qnet.apply(critic_params["q2"], cur_state, transitions.action)
            return jnp.mean((q1 - target_q)**2) + jnp.mean((q2 - target_q)**2)

        loss, grad = jax.value_and_grad(loss_fn)(ts.critic_state.params)
        new_critic = ts.critic_state.apply_gradients(grads=grad)
        new_target = soft_update(new_critic.params, ts.target_critic_params, args.tau)
        ts = ts.replace(critic_state=new_critic, target_critic_params=new_target)
        return ts, {"critic_loss": loss}

    @jax.jit
    def update_actors_and_alpha(transitions, ts, key):
        transitions = jax.tree_util.tree_map(lambda x: x[:args.batch_size], transitions)
        obs   = transitions.observation
        alpha = jnp.exp(ts.alpha_state.params["log_alpha"])
        agent_keys = jax.random.split(key, args.n_agents)

        # Q conditioned on full obs (includes goal)
        cur_state = obs

        # Compute joint action and joint log-prob (for alpha update)
        all_acts, all_log_probs = [], []
        for i, a in enumerate(actors):
            ast = ts.actor_states[0 if args.shared_params else i]
            inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*obs_size:(i+1)*obs_size]
            m, ls = a.apply(ast.params, inp); s = jnp.exp(ls)
            xt  = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
            act = nn.tanh(xt)
            lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act**2 + 1e-6)).sum(-1)
            all_acts.append(act); all_log_probs.append(lp)
        full_act  = jnp.concatenate(all_acts, axis=-1)
        log_prob  = sum(all_log_probs)

        # Per-agent actor gradient
        total_actor_loss = 0.0
        if args.shared_params:
            # Shared params: accumulate gradients across agents, apply once
            shared_ast = ts.actor_states[0]
            acc_grad = None
            for i, a in enumerate(actors):
                def loss_fn_i(ap, i=i):
                    inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*obs_size:(i+1)*obs_size]
                    m, ls = a.apply(ap, inp); s = jnp.exp(ls)
                    xt  = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
                    act = nn.tanh(xt)
                    lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act**2 + 1e-6)).sum(-1)
                    sl = action_slice_starts[i]
                    full_act_i = full_act.at[:, sl:sl+joints_per_agent[i]].set(act)
                    q   = qnet.apply(ts.critic_state.params["q1"], cur_state, full_act_i)
                    if args.disable_entropy:
                        return -jnp.mean(q)
                    return jnp.mean(alpha * lp - q)
                loss_i, grad_i = jax.value_and_grad(loss_fn_i)(shared_ast.params)
                total_actor_loss = total_actor_loss + loss_i
                acc_grad = grad_i if acc_grad is None else jax.tree_util.tree_map(lambda a, b: a + b, acc_grad, grad_i)
            avg_grad = jax.tree_util.tree_map(lambda g: g / args.n_agents, acc_grad)
            new_actor_states = [shared_ast.apply_gradients(grads=avg_grad)]
        else:
            new_actor_states = []
            for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
                def loss_fn_i(ap, i=i):
                    inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*obs_size:(i+1)*obs_size]
                    m, ls = a.apply(ap, inp); s = jnp.exp(ls)
                    xt  = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
                    act = nn.tanh(xt)
                    lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act**2 + 1e-6)).sum(-1)
                    # Replace agent i's slice with differentiable `act` so Q-gradient flows
                    sl = action_slice_starts[i]
                    full_act_i = full_act.at[:, sl:sl+joints_per_agent[i]].set(act)
                    q   = qnet.apply(ts.critic_state.params["q1"], cur_state, full_act_i)
                    if args.disable_entropy:
                        return -jnp.mean(q)
                    return jnp.mean(alpha * lp - q)
                loss_i, grad_i = jax.value_and_grad(loss_fn_i)(ast.params)
                total_actor_loss = total_actor_loss + loss_i
                new_actor_states.append(ast.apply_gradients(grads=grad_i))

        # Alpha update
        def alpha_loss_fn(ap):
            return jnp.mean(jnp.exp(ap["log_alpha"]) * (-log_prob - target_entropy))
        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(ts.alpha_state.params)
        new_alpha = ts.alpha_state.apply_gradients(grads=alpha_grad)

        ts = ts.replace(actor_states=new_actor_states, alpha_state=new_alpha)
        return ts, {"actor_loss": total_actor_loss / args.n_agents,
                    "log_alpha": new_alpha.params["log_alpha"]}

    @jax.jit
    def sgd_step(carry, transitions):
        # transitions.observation: (num_envs, episode_length, obs_size)
        ts, key = carry; key, hk, ck, ak = jax.random.split(key, 4)

        # Optional HER relabeling before flattening
        if args.use_her:
            transitions = her_relabel(transitions, hk)

        flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), transitions)
        goal_reward = jnp.mean(flat.reward)
        N = flat.observation.shape[0]
        n_batches = min(N // args.batch_size, max(1, args.num_sgd_batches))
        perm = jax.random.permutation(key, N); key, _ = jax.random.split(key)
        flat_p = jax.tree_util.tree_map(lambda x: x[perm[:n_batches * args.batch_size]], flat)
        batched = jax.tree_util.tree_map(
            lambda x: x.reshape((n_batches, args.batch_size) + x.shape[1:]), flat_p)

        def _inner(carry, batch):
            ts_i, k_i = carry; k_i, ck, ak = jax.random.split(k_i, 3)
            ts_i, c_m = update_critic(batch, ts_i, ck)
            ts_i, a_m = update_actors_and_alpha(batch, ts_i, ak)
            ts_i = ts_i.replace(gradient_steps=ts_i.gradient_steps + 1)
            return (ts_i, k_i), {**c_m, **a_m}

        (ts, key), inner_m = jax.lax.scan(_inner, (ts, key), batched)
        avg_m = jax.tree_util.tree_map(jnp.mean, inner_m)
        return (ts, key), {"goal_reward": goal_reward, **avg_m}

    @jax.jit
    def training_epoch(ts, env_state, buffer_state, key):
        def f(carry, t):
            ts, es, bs, k = carry; k, sk = jax.random.split(k)
            es, bs = get_experience(ts, es, bs, sk)
            ts = ts.replace(env_steps=ts.env_steps + args.env_steps_per_actor_step)
            tl_list = []
            for _ in range(args.num_episodes_per_env):
                bs, new_tr = replay_buffer.sample(bs)
                tl_list.append(new_tr)
            trs = jax.tree_util.tree_map(lambda *a: jnp.stack(a, axis=0), *tl_list)
            (ts, k), metrics = jax.lax.scan(sgd_step, (ts, k), trs)
            return (ts, es, bs, k), metrics
        (ts, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (ts, env_state, buffer_state, key),
            jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))
        return ts, env_state, buffer_state, metrics

    # ----------------------------
    # Evaluator
    # ----------------------------
    def eval_actor_step(ts, ev, es, extra_fields):
        nstate, tr = actor_step(ts, ev, es, jax.random.PRNGKey(0), False, extra_fields)
        return nstate, tr

    evaluator = CrlEvaluator(eval_actor_step, eval_env, num_eval_envs=eval_num_envs,
                              episode_length=args.episode_length, key=eval_key)

    # ----------------------------
    # Prefill
    # ----------------------------
    print("Prefilling replay buffer ...", flush=True)
    t0 = time.time()
    training_state, env_state, buffer_state, _ = prefill(training_state, env_state, buffer_state, key)
    print(f"  Prefill done in {time.time()-t0:.1f}s  (buffer size: {replay_buffer.size(buffer_state)})", flush=True)

    # ----------------------------
    # Training loop
    # ----------------------------
    training_walltime = 0.0; start_time = time.time()
    print("\nStarting training ...", flush=True)

    enl = enlighten.get_manager()
    epoch_bar = enl.counter(total=args.num_epochs, desc="MA-GCSAC", unit="epoch", color="cyan",
                             bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count}/{total} [{elapsed}<{eta}]")
    status_bar = enl.status_bar(
        status_format="Ep {ep}/{total} │ steps={steps} │ A={actor_loss} │ C={critic_loss} │ α={log_alpha} │ gr={goal_reward} │ suc={success}",
        color="bold_white_on_black",
        ep=0, total=args.num_epochs, steps="n/a", actor_loss="n/a",
        critic_loss="n/a", log_alpha="n/a", goal_reward="n/a", success="n/a")

    for ne in range(args.num_epochs):
        t0 = time.time(); key, ek = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, ek)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_time = time.time() - t0; training_walltime += epoch_time
        sps = args.env_steps_per_actor_step * args.num_training_steps_per_epoch / epoch_time

        metrics = {"training/sps": sps, "training/walltime": training_walltime,
                   "training/envsteps": training_state.env_steps.item(),
                   **{f"training/{k}": float(v) for k, v in metrics.items()}}
        metrics = evaluator.run_evaluation(training_state, metrics)

        al  = metrics.get("training/actor_loss",   float("nan"))
        cl  = metrics.get("training/critic_loss",  float("nan"))
        la  = metrics.get("training/log_alpha",    float("nan"))
        gr  = metrics.get("training/goal_reward",  float("nan"))
        suc = metrics.get("eval/episode_success",  float("nan"))
        st  = metrics["training/envsteps"]

        status_bar.update(ep=ne+1, total=args.num_epochs, steps=f"{st:.2e}",
                          actor_loss=f"{al:.4f}", critic_loss=f"{cl:.4f}",
                          log_alpha=f"{la:.3f}",
                          goal_reward=f"{gr:.4f}" if not np.isnan(gr) else "n/a",
                          success=f"{suc:.4f}" if not np.isnan(suc) else "n/a")
        enl.write(f"[Ep {ne+1:>4}/{args.num_epochs}] steps={st:.2e}  sps={sps:,.0f}"
                  f"  A={al:.4f}  C={cl:.4f}  α={la:.3f}"
                  f"  gr={gr:.4f}  suc={suc:.4f}"
                  f"  t={epoch_time:.1f}s  elapsed={(time.time()-start_time)/3600:.2f}h")
        print(f"PROG steps={st:.2e} suc={suc:.4f} gr={gr:.4f}", flush=True)
        epoch_bar.update()

        if args.checkpoint and (ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0):
            actor_params = (training_state.actor_states[0].params if args.shared_params
                            else [s.params for s in training_state.actor_states])
            save_params(f"{save_path}/step_{int(training_state.env_steps)}.pkl",
                        (training_state.alpha_state.params,
                         actor_params,
                         training_state.critic_state.params))
        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline": trigger_sync()

    epoch_bar.close(); status_bar.close(); enl.stop()

    if args.checkpoint:
        actor_params = (training_state.actor_states[0].params if args.shared_params
                        else [s.params for s in training_state.actor_states])
        save_params(f"{save_path}/final.pkl",
                    (training_state.alpha_state.params,
                     actor_params,
                     training_state.critic_state.params))
    print(f"\nTraining complete.  Total time: {(time.time()-start_time)/3600:.2f}h", flush=True)
