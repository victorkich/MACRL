# Parts adapted from the training script of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

# ============================================================
# MAPPO  —  Multi-Agent Proximal Policy Optimisation (CTDE)
#
# Yu et al. 2022 "The Surprising Effectiveness of PPO in Cooperative MARL".
# On-policy rollout collection, centralized value function (critic),
# decentralized actors.  GAE advantage estimation.
#
# Cooperative:     n_agents share one body (split joints)
# Non-cooperative: n_agents run in independent environments
# ============================================================

import os, jax, flax, tyro, time, optax, wandb, pickle, random
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


@dataclass
class Args:
    exp_name: str = "train_mappo"
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
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    n_agents: int = 2
    ma_reward: str = "mean"
    cooperative: int = 0

    total_env_steps: int = 100_000_000
    num_epochs: int = 100
    num_envs: int = 512
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    # PPO hyperparameters
    rollout_length: int = 128       # steps per env before update
    ppo_epochs: int = 4             # gradient epochs per rollout
    num_minibatches: int = 4        # minibatches per ppo epoch
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4

    use_relu: int = 0
    mup_residual_scaling: int = 1
    mup_scale_actor: int = 0
    zero_init_residual_last: int = 1

    env_steps_per_rollout: int = 0
    num_rollouts_per_epoch: int = 0
    num_training_steps_per_epoch: int = 0


lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def _get_res_scale(depth, enable):
    n = max(1, int(depth) // 4)
    return float(1.0 / np.sqrt(n)) if enable else 1.0


def residual_block(x, width, normalize, activation, res_scale, zero_init_last):
    identity = x
    for _ in range(3):
        x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
    li = nn.initializers.zeros if zero_init_last else lecun_uniform
    x = nn.Dense(width, kernel_init=li, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    return identity + res_scale * x


class Actor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 0
    zero_init_residual_last: int = 1
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x):
        norm = lambda y: nn.LayerNorm()(y)
        act  = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = norm(x); x = act(x)
        rs = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zl = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, norm, act, rs, zl)
        mean    = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


class ValueNetwork(nn.Module):
    """Centralized value function: joint_state -> scalar V(s)."""
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 1
    zero_init_residual_last: int = 1

    @nn.compact
    def __call__(self, x):
        norm = lambda y: nn.LayerNorm()(y)
        act  = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = norm(x); x = act(x)
        rs = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zl = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, norm, act, rs, zl)
        return nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init)(x).squeeze(-1)


@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_states: Any    # per-agent TrainState
    critic_state: TrainState


class RolloutTransition(NamedTuple):
    observation: jnp.ndarray       # joint state input to value function
    actor_obs: jnp.ndarray         # per-agent obs list [n_agents, num_envs, obs_i]
    action: jnp.ndarray            # full joint action [num_envs, full_act]
    log_prob: jnp.ndarray          # sum of per-agent log_probs [num_envs]
    value: jnp.ndarray             # V(s) [num_envs]
    reward: jnp.ndarray            # team reward [num_envs]
    discount: jnp.ndarray          # 1 - done [num_envs]


def save_params(path, p):
    with epath.Path(path).open("wb") as f: f.write(pickle.dumps(p))


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.env_id.endswith("_coop"):
        args.cooperative = 1
    if args.n_agents == 1:
        args.cooperative = 1  # trivially cooperative; avoids reshape mismatch

    print("=" * 60, flush=True)
    print("MAPPO  —  Multi-Agent PPO (CTDE)", flush=True)
    print("=" * 60, flush=True)
    print(f"  env_id      : {args.env_id}", flush=True)
    print(f"  n_agents    : {args.n_agents}", flush=True)
    print(f"  cooperative : {bool(args.cooperative)}", flush=True)
    print(f"  num_envs    : {args.num_envs}", flush=True)
    print(f"  rollout_len : {args.rollout_length}  ppo_epochs: {args.ppo_epochs}", flush=True)
    print("=" * 60, flush=True)

    random.seed(args.seed); np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, env_key, eval_key = jax.random.split(key, 3)

    run_name = (f"MAPPO_{args.env_id}_na:{args.n_agents}_{args.total_env_steps}"
                f"_ne:{args.num_envs}_cd:{args.critic_depth}_ad:{args.actor_depth}_s:{args.seed}")

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
        save_path = Path(args.wandb_dir) / Path(f"runs/mappo_{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(save_path, exist_ok=True)

    # ----------------------------
    # Environment
    # ----------------------------
    def make_env(env_id=None):
        eid = env_id or args.env_id
        print(f"  creating env: {eid}", flush=True)
        if eid in ("ant", "ant_coop"):
            from envs.ant import Ant
            env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 29; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif eid in ("humanoid", "humanoid_coop"):
            from envs.humanoid import Humanoid
            env = Humanoid(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 268; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif "ant" in eid and "maze" in eid:
            from envs.ant_maze import AntMaze
            _tmp = eid.removesuffix("_coop").removeprefix("ant_")
            name = "u_maze" if _tmp == "maze" else (_tmp[5:] if _tmp.startswith("maze_") else _tmp)
            env = AntMaze(backend="spring", exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=True, maze_layout_name=name)
            args.obs_dim = 29; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif "humanoid" in eid and "maze" in eid:
            from envs.humanoid_maze import HumanoidMaze
            name = eid.replace("_coop","").replace("humanoid_maze_","") or "U_MAZE"
            env = HumanoidMaze(backend="spring", maze_layout_name=name)
            args.obs_dim = 268; args.goal_start_idx = 0; args.goal_end_idx = 3
        else:
            raise NotImplementedError(f"Unknown env_id: {eid}")
        return env

    env  = make_env(); env = envs.training.wrap(env, episode_length=args.episode_length)
    OBS  = env.observation_size; ACT = env.action_size

    if args.cooperative:
        base = ACT // args.n_agents; rem = ACT % args.n_agents
        jpa  = [base + (1 if i < rem else 0) for i in range(args.n_agents)]
        sim_envs     = args.num_envs
        actor_input  = OBS + args.n_agents    # + one-hot agent ID
        value_input  = args.obs_dim           # centralized value sees body state
    else:
        jpa          = [ACT] * args.n_agents
        sim_envs     = args.n_agents * args.num_envs
        actor_input  = OBS
        value_input  = args.obs_dim * args.n_agents

    args.env_steps_per_rollout = sim_envs * args.rollout_length
    args.num_rollouts_per_epoch = max(1, args.total_env_steps // (args.num_epochs * args.env_steps_per_rollout))
    args.num_training_steps_per_epoch = args.num_rollouts_per_epoch  # one rollout = one "step"
    total_env_steps_actual = args.num_epochs * args.num_rollouts_per_epoch * args.env_steps_per_rollout

    eval_num_envs = args.num_eval_envs if args.cooperative else args.n_agents * args.num_eval_envs
    eval_env = make_env(); eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)

    print(f"  obs: {OBS}  act: {ACT}  joints_per_agent: {jpa}", flush=True)
    print(f"  rollouts/epoch: {args.num_rollouts_per_epoch}  steps/rollout: {args.env_steps_per_rollout:,}", flush=True)
    print(f"  total env steps: ~{total_env_steps_actual:,.0f}", flush=True)

    env_keys  = jax.random.split(env_key, sim_envs)
    env_state = jax.jit(env.reset)(env_keys); env.step = jax.jit(env.step)

    # ----------------------------
    # Networks
    # ----------------------------
    akeys = jax.random.split(key, args.n_agents + 2); key = akeys[0]
    actors, actor_states = [], []
    for i in range(args.n_agents):
        a  = Actor(action_size=jpa[i], network_width=args.actor_network_width,
                   network_depth=args.actor_depth, use_relu=args.use_relu,
                   mup_residual_scaling=int(bool(args.mup_scale_actor) and bool(args.mup_residual_scaling)),
                   zero_init_residual_last=args.zero_init_residual_last)
        tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.actor_lr))
        st = TrainState.create(apply_fn=a.apply,
                               params=a.init(akeys[i+1], np.ones([1, actor_input])),
                               tx=tx)
        actors.append(a); actor_states.append(st)

    key, vk = jax.random.split(key)
    vnet = ValueNetwork(network_width=args.critic_network_width, network_depth=args.critic_depth,
                        use_relu=args.use_relu, mup_residual_scaling=args.mup_residual_scaling,
                        zero_init_residual_last=args.zero_init_residual_last)
    vtx  = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.critic_lr))
    critic_state = TrainState.create(apply_fn=vnet.apply,
                                     params=vnet.init(vk, np.ones([1, value_input])),
                                     tx=vtx)

    training_state = TrainingState(env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()),
                                   actor_states=actor_states, critic_state=critic_state)

    print(f"  Actor[0] params: {sum(x.size for x in jax.tree_util.tree_leaves(actor_states[0].params)):,}", flush=True)
    print(f"  Value params   : {sum(x.size for x in jax.tree_util.tree_leaves(critic_state.params)):,}", flush=True)

    # ----------------------------
    # Rollout collection
    # ----------------------------
    def _coop_obs_i(obs, i):
        aid = jnp.zeros((*obs.shape[:-1], args.n_agents), dtype=obs.dtype).at[..., i].set(1.0)
        return jnp.concatenate([obs, aid], axis=-1)

    def get_actor_obs(env_obs):
        """Returns per-agent obs in shape (n_agents, num_envs, obs_i)."""
        if args.cooperative:
            return [_coop_obs_i(env_obs, i) for i in range(args.n_agents)]
        else:
            return [env_obs[i*args.num_envs:(i+1)*args.num_envs] for i in range(args.n_agents)]

    def get_value_obs(env_obs):
        """Returns centralized state input to value network."""
        if args.cooperative:
            return env_obs[:, :args.obs_dim]
        else:
            return jnp.concatenate(
                [env_obs[i*args.num_envs:(i+1)*args.num_envs, :args.obs_dim]
                 for i in range(args.n_agents)], axis=-1)

    @jax.jit
    def collect_rollout(ts, env_state, key):
        """Collect rollout_length steps; returns (env_state, RolloutTransition)."""
        def step(carry, _):
            es, k = carry; k, sk = jax.random.split(k)
            agent_keys = jax.random.split(sk, args.n_agents)

            actor_obs_list = get_actor_obs(es.obs)
            value_obs      = get_value_obs(es.obs)
            val            = vnet.apply(ts.critic_state.params, value_obs)

            sub_acts, log_probs = [], []
            for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
                m, ls = a.apply(ast.params, actor_obs_list[i])
                s     = jnp.exp(ls)
                xt    = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
                act   = nn.tanh(xt)
                lp    = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1-act**2+1e-6)).sum(-1)
                sub_acts.append(act); log_probs.append(lp)

            full_act  = jnp.concatenate(sub_acts, axis=-1)
            log_prob  = sum(log_probs)
            env_act   = full_act if args.cooperative else full_act.reshape(sim_envs, ACT)
            nstate    = env.step(es, env_act)

            if args.cooperative:
                rew  = nstate.reward; disc = 1.0 - nstate.done
            else:
                rews = nstate.reward.reshape(args.n_agents, args.num_envs)
                rew  = rews.mean(0) if args.ma_reward == "mean" else (rews.sum(0) if args.ma_reward == "sum" else rews.min(0))
                disc = 1.0 - nstate.done[:args.num_envs]

            # Stack actor_obs for storage
            actor_obs_stacked = jnp.stack(actor_obs_list, axis=0)  # (n_agents, num_envs, obs_i)

            tr = RolloutTransition(
                observation=value_obs, actor_obs=actor_obs_stacked,
                action=full_act, log_prob=log_prob, value=val,
                reward=rew, discount=disc)
            return (nstate, k), tr

        (env_state, _), rollout = jax.lax.scan(step, (env_state, key), (), length=args.rollout_length)
        return env_state, rollout

    # ----------------------------
    # GAE Advantage Estimation
    # ----------------------------
    @jax.jit
    def compute_gae(rollout, last_value):
        """
        rollout: RolloutTransition with shapes (T, num_envs, ...)
        last_value: V(s_T) shape (num_envs,)
        Returns: advantages (T, num_envs), returns (T, num_envs)
        """
        T = rollout.reward.shape[0]
        rewards   = rollout.reward    # (T, num_envs)
        values    = rollout.value     # (T, num_envs)
        discounts = rollout.discount  # (T, num_envs)

        def body(carry, t):
            gae_next, val_next = carry
            r   = rewards[T - 1 - t]
            v   = values[T - 1 - t]
            d   = discounts[T - 1 - t]
            delta = r + args.gamma * d * val_next - v
            gae   = delta + args.gamma * args.gae_lambda * d * gae_next
            return (gae, v), (gae, gae + v)

        _, (advantages_rev, returns_rev) = jax.lax.scan(
            body, (jnp.zeros_like(last_value), last_value),
            jnp.arange(T))
        advantages = jnp.flip(advantages_rev, axis=0)  # (T, num_envs)
        returns    = jnp.flip(returns_rev,    axis=0)
        return advantages, returns

    # ----------------------------
    # PPO Update
    # ----------------------------
    @jax.jit
    def ppo_update(ts, rollout, advantages, returns, key):
        """Multiple epochs of minibatch PPO updates."""
        T, N = rollout.reward.shape   # T=rollout_length, N=num_envs

        # Flatten (T, N, ...) -> (T*N, ...)
        def flat(x): return x.reshape(T * N, *x.shape[2:])
        obs_flat   = flat(rollout.observation)      # (TN, value_input)
        act_flat   = flat(rollout.action)
        lp_flat    = flat(rollout.log_prob)
        adv_flat   = flat(advantages)
        ret_flat   = flat(returns)

        # Normalise advantages — clamp std to avoid NaN when value fn is accurate
        adv_std = adv_flat.std()
        adv_flat = (adv_flat - adv_flat.mean()) / jnp.maximum(adv_std, 1e-6)

        minibatch_size = (T * N) // args.num_minibatches
        metrics_acc = {"actor_loss": jnp.zeros(()), "value_loss": jnp.zeros(()), "entropy": jnp.zeros(())}

        for _ in range(args.ppo_epochs):
            key, sk = jax.random.split(key)
            perm     = jax.random.permutation(sk, T * N)

            for mb in range(args.num_minibatches):
                idx    = perm[mb * minibatch_size:(mb + 1) * minibatch_size]
                mb_obs = obs_flat[idx]; mb_act = act_flat[idx]
                mb_lp  = lp_flat[idx];  mb_adv = adv_flat[idx]
                mb_ret = ret_flat[idx]

                # Per-agent actor obs for this minibatch
                # actor_obs: (T, n_agents, N, obs_i) -> (T, N, n_agents, obs_i) -> (T*N, n_agents, obs_i)
                # Must use T-major order to align with obs_flat/adv_flat/ret_flat
                mb_aobs = rollout.actor_obs.transpose(0, 2, 1, 3).reshape(T*N, args.n_agents, actor_input)[idx]

                def actor_losses(actor_params_list):
                    total_lp  = jnp.zeros(minibatch_size)
                    total_ent = jnp.zeros(())
                    for i, a in enumerate(actors):
                        sl_start = sum(jpa[:i])
                        obs_i  = mb_aobs[:, i, :]
                        # Evaluate log prob of the STORED action (correct PPO ratio)
                        act_i  = mb_act[:, sl_start:sl_start + jpa[i]]
                        xt     = jnp.arctanh(jnp.clip(act_i, -1 + 1e-6, 1 - 1e-6))
                        m, ls  = a.apply(actor_params_list[i], obs_i)
                        s      = jnp.exp(ls)
                        lp     = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act_i**2 + 1e-6)).sum(-1)
                        total_lp  = total_lp + lp
                        # Analytical entropy of N(m, exp(ls)) per dim: 0.5*(1+log(2π)) + ls
                        # Bounded in [-8, 6] nats (ls clamped to [-5,2]), avoids tanh-correction instability
                        total_ent = total_ent + (0.5 * (1.0 + jnp.log(2.0 * jnp.pi)) + ls).sum(-1).mean()
                    ratio      = jnp.exp(total_lp - mb_lp)
                    clip1      = ratio * mb_adv
                    clip2      = jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
                    actor_loss = -jnp.mean(jnp.minimum(clip1, clip2))
                    entropy    = total_ent / args.n_agents
                    return actor_loss - args.ent_coef * entropy, (actor_loss, entropy)

                def value_loss(vp):
                    v = vnet.apply(vp, mb_obs)
                    return args.vf_coef * jnp.mean((v - mb_ret) ** 2)

                (al, (al_raw, ent)), al_grad = jax.value_and_grad(
                    lambda aps: actor_losses(aps), has_aux=True)(
                        [ast.params for ast in ts.actor_states])
                vl, vl_grad = jax.value_and_grad(value_loss)(ts.critic_state.params)

                # Apply gradients per-agent
                new_actor_states = [
                    ast.apply_gradients(grads=al_grad[i])
                    for i, ast in enumerate(ts.actor_states)]
                new_critic = ts.critic_state.apply_gradients(grads=vl_grad)
                ts = ts.replace(actor_states=new_actor_states, critic_state=new_critic,
                                gradient_steps=ts.gradient_steps + 1)

                metrics_acc["actor_loss"] = metrics_acc["actor_loss"] + al_raw
                metrics_acc["value_loss"] = metrics_acc["value_loss"] + vl
                metrics_acc["entropy"]    = metrics_acc["entropy"]    + ent

        n_updates = args.ppo_epochs * args.num_minibatches
        return ts, {k: v / n_updates for k, v in metrics_acc.items()}

    @jax.jit
    def training_epoch(ts, env_state, key):
        def rollout_and_update(carry, _):
            ts, es, k = carry; k, rk, uk = jax.random.split(k, 3)
            es, rollout = collect_rollout(ts, es, rk)
            # Last value for GAE bootstrap
            last_val_obs = get_value_obs(es.obs)
            last_val = vnet.apply(ts.critic_state.params, last_val_obs)
            advantages, returns = compute_gae(rollout, last_val)
            ts, metrics = ppo_update(ts, rollout, advantages, returns, uk)
            ts = ts.replace(env_steps=ts.env_steps + args.env_steps_per_rollout)
            return (ts, es, k), metrics

        (ts, env_state, key), metrics = jax.lax.scan(
            rollout_and_update,
            (ts, env_state, key),
            jnp.arange(args.num_rollouts_per_epoch))
        return ts, env_state, metrics

    # ----------------------------
    # Evaluator  (uses deterministic mean actions)
    # ----------------------------
    def eval_step(ts, ev, es, extra_fields):
        actor_obs_list = get_actor_obs(es.obs)
        sub_acts = []
        for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
            m, _ = a.apply(ast.params, actor_obs_list[i])
            sub_acts.append(nn.tanh(m))
        full_act  = jnp.concatenate(sub_acts, axis=-1)
        env_act   = full_act if args.cooperative else full_act.reshape(sim_envs, ACT)
        nstate    = ev.step(es, env_act)

        if args.cooperative:
            obs_j = es.obs; rew = nstate.reward; disc = 1.0 - nstate.done
        else:
            n_ag = args.n_agents
            obs_j = es.obs.reshape(n_ag, args.num_eval_envs, OBS).transpose(1,0,2).reshape(eval_num_envs, n_ag*OBS)
            rew   = nstate.reward[:args.num_eval_envs]; disc = 1.0 - nstate.done[:args.num_eval_envs]

        tr = {"observation": obs_j, "action": full_act, "reward": rew, "discount": disc,
              "extras": {"state_extras": {x: nstate.info[x] for x in extra_fields}}}
        return nstate, tr

    evaluator = CrlEvaluator(eval_step, eval_env, num_eval_envs=eval_num_envs,
                              episode_length=args.episode_length, key=eval_key)

    # ----------------------------
    # Training loop
    # ----------------------------
    start_time = time.time(); training_walltime = 0.0
    print("\nStarting training ...", flush=True)

    enl = enlighten.get_manager()
    epoch_bar  = enl.counter(total=args.num_epochs, desc="MAPPO", unit="epoch", color="magenta",
                              bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count}/{total} [{elapsed}<{eta}]")
    status_bar = enl.status_bar(
        status_format="Ep {ep}/{total} │ steps={steps} │ A={actor_loss} │ V={value_loss} │ ent={entropy} │ rew={reward} │ suc={success}",
        color="bold_white_on_black",
        ep=0, total=args.num_epochs, steps="n/a",
        actor_loss="n/a", value_loss="n/a", entropy="n/a", reward="n/a", success="n/a")

    for ne in range(args.num_epochs):
        t0 = time.time(); key, ek = jax.random.split(key)
        training_state, env_state, metrics = training_epoch(training_state, env_state, ek)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_time = time.time() - t0; training_walltime += epoch_time
        sps = args.env_steps_per_rollout * args.num_rollouts_per_epoch / epoch_time

        metrics = {"training/sps": sps, "training/walltime": training_walltime,
                   "training/envsteps": training_state.env_steps.item(),
                   **{f"training/{k}": float(v) for k, v in metrics.items()}}
        metrics = evaluator.run_evaluation(training_state, metrics)

        al  = metrics.get("training/actor_loss",  float("nan"))
        vl  = metrics.get("training/value_loss",  float("nan"))
        ent = metrics.get("training/entropy",      float("nan"))
        suc = metrics.get("eval/episode_success",  float("nan"))
        rew = metrics.get("eval/episode_reward",   float("nan"))
        st  = metrics["training/envsteps"]

        status_bar.update(ep=ne+1, total=args.num_epochs, steps=f"{st:.2e}",
                          actor_loss=f"{al:.4f}", value_loss=f"{vl:.4f}", entropy=f"{ent:.3f}",
                          reward=f"{rew:.2f}" if not np.isnan(rew) else "n/a",
                          success=f"{suc:.4f}" if not np.isnan(suc) else "n/a")
        enl.write(f"[Ep {ne+1:>4}/{args.num_epochs}] steps={st:.2e}  sps={sps:,.0f}"
                  f"  A={al:.4f}  V={vl:.4f}  ent={ent:.3f}"
                  f"  rew={rew:.2f}  suc={suc:.4f}"
                  f"  t={epoch_time:.1f}s  elapsed={(time.time()-start_time)/3600:.2f}h")
        epoch_bar.update()

        if args.checkpoint and (ne < 5 or ne >= args.num_epochs-5 or ne % 10 == 0):
            save_params(f"{save_path}/step_{int(training_state.env_steps)}.pkl",
                        ([s.params for s in training_state.actor_states], training_state.critic_state.params))
        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline": trigger_sync()

    epoch_bar.close(); status_bar.close(); enl.stop()
    if args.checkpoint:
        save_params(f"{save_path}/final.pkl",
                    ([s.params for s in training_state.actor_states], training_state.critic_state.params))
    print(f"\nTraining complete.  Total time: {(time.time()-start_time)/3600:.2f}h", flush=True)
