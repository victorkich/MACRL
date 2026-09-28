# Parts adapted from the training script of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

# ============================================================
# MADDPG  —  Multi-Agent Deep Deterministic Policy Gradient (CTDE)
#
# Lowe et al. 2017.  Reward-maximizing baseline for MA-CRL comparison.
# Deterministic actors + centralized Q-network + Polyak target networks.
# Exploration via Gaussian action noise during rollout collection.
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
from buffer import TrajectoryUniformSamplingQueue


@dataclass
class Args:
    exp_name: str = "train_maddpg"
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
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    expl_noise: float = 0.3    # Gaussian exploration std

    max_replay_size: int = 10_000
    min_replay_size: int = 1_000
    unroll_length: int = 62

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4

    num_episodes_per_env: int = 1
    training_steps_multiplier: int = 1
    use_relu: int = 0
    mup_residual_scaling: int = 1
    mup_scale_actor: int = 0
    zero_init_residual_last: int = 1
    max_grad_norm: float = 0.5

    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
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


class QNetwork(nn.Module):
    """Centralized Q-network: Q(state, action) -> scalar."""
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 1
    zero_init_residual_last: int = 1

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        norm = lambda y: nn.LayerNorm()(y)
        act  = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = norm(x); x = act(x)
        rs = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zl = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, norm, act, rs, zl)
        return nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init)(x).squeeze(-1)


class DeterministicActor(nn.Module):
    """Deterministic actor: obs -> tanh-scaled action."""
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 0
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
        return nn.tanh(nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x))


@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_states: Any          # per-agent TrainState
    critic_state: TrainState   # centralized Q
    target_actor_params: Any   # Polyak targets
    target_critic_params: Any


class Transition(NamedTuple):
    observation: jnp.ndarray
    next_observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


def save_params(path, p):
    with epath.Path(path).open("wb") as f: f.write(pickle.dumps(p))


def soft_update(params, target, tau):
    return jax.tree_util.tree_map(lambda p, t: tau*p + (1-tau)*t, params, target)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.env_id.endswith("_coop"):
        args.cooperative = 1
    if args.n_agents == 1:
        args.cooperative = 1  # trivially cooperative; avoids reshape mismatch

    print("=" * 60, flush=True)
    print("MADDPG  —  Multi-Agent DDPG (CTDE)", flush=True)
    print("=" * 60, flush=True)
    print(f"  env_id      : {args.env_id}", flush=True)
    print(f"  n_agents    : {args.n_agents}", flush=True)
    print(f"  cooperative : {bool(args.cooperative)}", flush=True)
    print(f"  num_envs    : {args.num_envs}", flush=True)
    print("=" * 60, flush=True)

    total_envs = args.n_agents * args.num_envs
    args.env_steps_per_actor_step = total_envs * args.unroll_length
    args.num_prefill_env_steps    = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps  = int(np.ceil(args.min_replay_size / args.unroll_length))

    run_name = (f"MADDPG_{args.env_id}_na:{args.n_agents}_{args.total_env_steps}"
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
        save_path = Path(args.wandb_dir) / Path(f"runs/maddpg_{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(save_path, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buf_key, env_key, eval_key = jax.random.split(key, 4)

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
        sim_envs = args.num_envs
        args.env_steps_per_actor_step = args.num_envs * args.unroll_length
        actor_input = OBS + args.n_agents
        sa_sdim = args.obs_dim; sa_adim = ACT
    else:
        jpa  = [ACT] * args.n_agents
        sim_envs = total_envs
        actor_input = OBS
        sa_sdim = args.obs_dim * args.n_agents; sa_adim = ACT * args.n_agents

    args.num_training_steps_per_epoch = max(1, (
        (args.total_env_steps - args.num_prefill_env_steps)
        // (args.num_epochs * args.env_steps_per_actor_step)))
    eval_num_envs = args.num_eval_envs if args.cooperative else args.n_agents * args.num_eval_envs
    eval_env = make_env(); eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)

    print(f"  obs: {OBS}  act: {ACT}  joints_per_agent: {jpa}", flush=True)
    print(f"  train_steps/epoch: {args.num_training_steps_per_epoch}  total_steps: ~{args.num_epochs * args.num_training_steps_per_epoch * args.env_steps_per_actor_step:,.0f}", flush=True)

    env_keys  = jax.random.split(env_key, sim_envs)
    env_state = jax.jit(env.reset)(env_keys); env.step = jax.jit(env.step)

    # ----------------------------
    # Networks
    # ----------------------------
    akeys = jax.random.split(key, args.n_agents + 2); key = akeys[0]
    actors, actor_states = [], []
    for i in range(args.n_agents):
        a  = DeterministicActor(action_size=jpa[i], network_width=args.actor_network_width,
                                 network_depth=args.actor_depth, use_relu=args.use_relu,
                                 mup_residual_scaling=int(bool(args.mup_scale_actor) and bool(args.mup_residual_scaling)),
                                 zero_init_residual_last=args.zero_init_residual_last)
        st = TrainState.create(apply_fn=a.apply,
                               params=a.init(akeys[i+1], np.ones([1, actor_input])),
                               tx=optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.actor_lr)))
        actors.append(a); actor_states.append(st)

    key, qk = jax.random.split(key)
    qnet = QNetwork(network_width=args.critic_network_width, network_depth=args.critic_depth,
                    use_relu=args.use_relu, mup_residual_scaling=args.mup_residual_scaling,
                    zero_init_residual_last=args.zero_init_residual_last)
    q_params = qnet.init(qk, np.ones([1, sa_sdim]), np.ones([1, sa_adim]))
    critic_state = TrainState.create(apply_fn=None, params=q_params, tx=optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.critic_lr)))

    target_actor_params = [s.params for s in actor_states]
    target_critic_params = q_params

    training_state = TrainingState(
        env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()),
        actor_states=actor_states, critic_state=critic_state,
        target_actor_params=target_actor_params, target_critic_params=target_critic_params)

    # ----------------------------
    # Buffer
    # ----------------------------
    if args.cooperative:
        d_obs = jnp.zeros((OBS,)); d_act = jnp.zeros((ACT,))
    else:
        d_obs = jnp.zeros((args.n_agents * OBS,)); d_act = jnp.zeros((args.n_agents * ACT,))

    dummy_tr = Transition(observation=d_obs, next_observation=d_obs, action=d_act,
                          reward=0.0, discount=0.0,
                          extras={"state_extras": {"truncation": 0.0, "seed": 0.0}})

    def jit_wrap(b):
        b.insert_internal = jax.jit(b.insert_internal)
        b.sample_internal  = jax.jit(b.sample_internal)
        return b

    buf = jit_wrap(TrajectoryUniformSamplingQueue(
        max_replay_size=args.max_replay_size, dummy_data_sample=dummy_tr,
        sample_batch_size=args.batch_size, num_envs=args.num_envs,
        episode_length=args.episode_length))
    buffer_state = jax.jit(buf.init)(buf_key)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _coop_obs_i(obs, i):
        aid = jnp.zeros((*obs.shape[:-1], args.n_agents), dtype=obs.dtype).at[..., i].set(1.0)
        return jnp.concatenate([obs, aid], axis=-1)

    def get_joint(ts, obs, target=False):
        """Build full action from deterministic actors."""
        params_list = ts.target_actor_params if target else [s.params for s in ts.actor_states]
        sub = []
        for i, (a, p) in enumerate(zip(actors, params_list)):
            inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*OBS:(i+1)*OBS]
            sub.append(a.apply(p, inp))
        return jnp.concatenate(sub, axis=-1)

    def get_state(obs, next_obs=None):
        if args.cooperative:
            s = obs[:, :args.obs_dim]
            return (s, next_obs[:, :args.obs_dim]) if next_obs is not None else s
        s = jnp.concatenate([obs[:, i*OBS:i*OBS+args.obs_dim] for i in range(args.n_agents)], axis=-1)
        if next_obs is not None:
            ns = jnp.concatenate([next_obs[:, i*OBS:i*OBS+args.obs_dim] for i in range(args.n_agents)], axis=-1)
            return s, ns
        return s

    def build_transition(env_state, nstate, full_act, extra_fields):
        if args.cooperative:
            obs_j  = env_state.obs; nobs_j = nstate.obs
            rew    = nstate.reward; disc   = 1.0 - nstate.done
            extras = {"state_extras": {x: nstate.info[x] for x in extra_fields}}
        else:
            n_ag, nenv = args.n_agents, args.num_envs
            obs_j  = env_state.obs.reshape(n_ag, nenv, OBS).transpose(1,0,2).reshape(nenv, n_ag*OBS)
            nobs_j = nstate.obs.reshape(n_ag, nenv, OBS).transpose(1,0,2).reshape(nenv, n_ag*OBS)
            rews   = nstate.reward.reshape(n_ag, nenv)
            rew    = rews.mean(0) if args.ma_reward == "mean" else (rews.sum(0) if args.ma_reward == "sum" else rews.min(0))
            disc   = 1.0 - nstate.done[:nenv]
            extras = {"state_extras": {x: nstate.info[x][:nenv] for x in extra_fields}}
        return Transition(observation=obs_j, next_observation=nobs_j,
                          action=full_act, reward=rew, discount=disc, extras=extras)

    def actor_step(ts, env, env_state, key, explore, extra_fields):
        ap    = [s.params for s in ts.actor_states]
        obs_i = env_state.obs if args.cooperative else \
                env_state.obs.reshape(args.n_agents, args.num_envs, OBS).transpose(1,0,2).reshape(args.num_envs, args.n_agents*OBS)
        sub   = []
        for i, (a, p) in enumerate(zip(actors, ap)):
            inp = _coop_obs_i(env_state.obs, i) if args.cooperative else obs_i[:, i*OBS:(i+1)*OBS]
            act = a.apply(p, inp)
            if explore:
                key, nk = jax.random.split(key)
                act = jnp.clip(act + args.expl_noise * jax.random.normal(nk, act.shape, dtype=act.dtype), -1.0, 1.0)
            sub.append(act)
        full_act = jnp.concatenate(sub, axis=-1)
        env_act  = full_act if args.cooperative else full_act.reshape(sim_envs, ACT)
        nstate   = env.step(env_state, env_act)
        return nstate, build_transition(env_state, nstate, full_act, extra_fields)

    @jax.jit
    def get_experience(ts, env_state, buf_state, key):
        def f(carry, _):
            es, k = carry; k, nk = jax.random.split(k)
            nes, tr = actor_step(ts, env, es, k, True, ("truncation","seed"))
            return (nes, nk), tr
        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        return env_state, buf.insert(buf_state, data)

    def prefill(ts, env_state, buf_state, key):
        def f(carry, _):
            ts, es, bs, k = carry; k, nk = jax.random.split(k)
            es, bs = get_experience(ts, es, bs, k)
            ts = ts.replace(env_steps=ts.env_steps + args.env_steps_per_actor_step)
            return (ts, es, bs, nk), ()
        return jax.lax.scan(f, (ts, env_state, buf_state, key), (), length=args.num_prefill_actor_steps)[0]

    # ----------------------------
    # MADDPG Updates
    # ----------------------------
    @jax.jit
    def update_step(ts, transitions):
        transitions = jax.tree_util.tree_map(lambda x: x[:args.batch_size], transitions)
        obs  = transitions.observation; nobs = transitions.next_observation

        if args.cooperative:
            s = obs[:, :args.obs_dim]; ns = nobs[:, :args.obs_dim]
        else:
            s  = jnp.concatenate([obs[:, i*OBS:i*OBS+args.obs_dim] for i in range(args.n_agents)], axis=-1)
            ns = jnp.concatenate([nobs[:, i*OBS:i*OBS+args.obs_dim] for i in range(args.n_agents)], axis=-1)

        # Target Q: r + γ * Q_target(s', μ_target(s'))
        next_acts_sub = []
        for i, (a, tp) in enumerate(zip(actors, ts.target_actor_params)):
            inp = _coop_obs_i(nobs, i) if args.cooperative else nobs[:, i*OBS:(i+1)*OBS]
            next_acts_sub.append(a.apply(tp, inp))
        next_act  = jnp.concatenate(next_acts_sub, axis=-1)
        target_q  = (transitions.reward + args.gamma * transitions.discount *
                     qnet.apply(ts.target_critic_params, ns, next_act))

        # Critic update
        def critic_loss(qp):
            return jnp.mean((qnet.apply(qp, s, transitions.action) - target_q) ** 2)
        c_loss, c_grad = jax.value_and_grad(critic_loss)(ts.critic_state.params)
        new_critic = ts.critic_state.apply_gradients(grads=c_grad)

        # Actor updates (each agent independently maximizes Q)
        new_actor_states = []; total_actor_loss = 0.0
        cur_acts_sub = []  # build current full action once
        for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
            inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*OBS:(i+1)*OBS]
            cur_acts_sub.append(a.apply(ast.params, inp))
        cur_acts = jnp.concatenate(cur_acts_sub, axis=-1)

        for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
            def actor_loss_i(ap, i=i):
                inp  = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*OBS:(i+1)*OBS]
                act_i = a.apply(ap, inp)
                # Replace agent i's slice in the joint action
                sl_start = sum(jpa[:i])
                full = cur_acts.at[:, sl_start:sl_start+jpa[i]].set(act_i)
                return -jnp.mean(qnet.apply(ts.critic_state.params, s, full))
            a_loss, a_grad = jax.value_and_grad(actor_loss_i)(ast.params)
            total_actor_loss += a_loss
            new_actor_states.append(ast.apply_gradients(grads=a_grad))

        # Soft update targets
        new_target_actor  = [soft_update(na.params, tp, args.tau)
                              for na, tp in zip(new_actor_states, ts.target_actor_params)]
        new_target_critic = soft_update(new_critic.params, ts.target_critic_params, args.tau)

        new_ts = ts.replace(
            actor_states=new_actor_states, critic_state=new_critic,
            target_actor_params=new_target_actor, target_critic_params=new_target_critic,
            gradient_steps=ts.gradient_steps + 1)

        return new_ts, {"critic_loss": c_loss, "actor_loss": total_actor_loss / args.n_agents}

    @jax.jit
    def extract_transitions(raw):
        """Flatten trajectory buffer samples to (s,a,r,s') tuples."""
        obs  = raw.observation; nobs = raw.next_observation
        return raw._replace(
            observation=jnp.reshape(obs[..., :obs.shape[-1]], (-1, obs.shape[-1]), order="F"),
            next_observation=jnp.reshape(nobs, (-1, nobs.shape[-1]), order="F"),
            action=jnp.reshape(raw.action, (-1, raw.action.shape[-1]), order="F"),
            reward=jnp.reshape(raw.reward, (-1,), order="F"),
            discount=jnp.reshape(raw.discount, (-1,), order="F"),
        )

    @jax.jit
    def training_epoch(ts, env_state, buf_state, key):
        def f(carry, _):
            ts, es, bs, k = carry; k, ek, sk = jax.random.split(k, 3)
            es, bs = get_experience(ts, es, bs, ek)
            ts = ts.replace(env_steps=ts.env_steps + args.env_steps_per_actor_step)
            tl = []
            for _ in range(args.num_episodes_per_env):
                bs, new_tr = buf.sample(bs); tl.append(new_tr)
            trs = jax.tree_util.tree_map(lambda *a: jnp.concatenate(a, 0), *tl)
            # Extract consecutive (s,a,r,s') from each trajectory
            obs_cur  = trs.observation[:, :-1, :]
            obs_next = trs.observation[:, 1:, :]
            act_cur  = trs.action[:, :-1, :]
            rew_cur  = trs.reward[:, :-1]
            disc_cur = trs.discount[:, :-1]
            B, T = obs_cur.shape[:2]
            tr_flat = Transition(
                observation=obs_cur.reshape(B*T, -1),
                next_observation=obs_next.reshape(B*T, -1),
                action=act_cur.reshape(B*T, -1),
                reward=rew_cur.reshape(B*T),
                discount=disc_cur.reshape(B*T),
                extras={})
            ts, metrics = update_step(ts, tr_flat)
            return (ts, es, bs, k), metrics
        (ts, env_state, buf_state, key), metrics = jax.lax.scan(
            f, (ts, env_state, buf_state, key),
            jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))
        return ts, env_state, buf_state, metrics

    # ----------------------------
    # Evaluator
    # ----------------------------
    def eval_step(ts, ev, es, extra_fields):
        nstate, tr = actor_step(ts, ev, es, jax.random.PRNGKey(0), False, extra_fields)
        return nstate, tr

    evaluator = CrlEvaluator(eval_step, eval_env, num_eval_envs=eval_num_envs,
                              episode_length=args.episode_length, key=eval_key)

    # ----------------------------
    # Prefill + training loop
    # ----------------------------
    print("Prefilling replay buffer ...", flush=True)
    t0 = time.time()
    training_state, env_state, buffer_state, _ = prefill(training_state, env_state, buffer_state, key)
    print(f"  Prefill done in {time.time()-t0:.1f}s  (buffer size: {buf.size(buffer_state)})", flush=True)

    start_time = time.time(); training_walltime = 0.0
    print("\nStarting training ...", flush=True)

    enl = enlighten.get_manager()
    epoch_bar  = enl.counter(total=args.num_epochs, desc="MADDPG", unit="epoch", color="yellow",
                              bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count}/{total} [{elapsed}<{eta}]")
    status_bar = enl.status_bar(
        status_format="Ep {ep}/{total} │ steps={steps} │ A={actor_loss} │ C={critic_loss} │ rew={reward} │ suc={success}",
        color="bold_white_on_black",
        ep=0, total=args.num_epochs, steps="n/a",
        actor_loss="n/a", critic_loss="n/a", reward="n/a", success="n/a")

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

        al  = metrics.get("training/actor_loss",  float("nan"))
        cl  = metrics.get("training/critic_loss", float("nan"))
        suc = metrics.get("eval/episode_success", float("nan"))
        rew = metrics.get("eval/episode_reward",  float("nan"))
        st  = metrics["training/envsteps"]

        status_bar.update(ep=ne+1, total=args.num_epochs, steps=f"{st:.2e}",
                          actor_loss=f"{al:.4f}", critic_loss=f"{cl:.4f}",
                          reward=f"{rew:.2f}" if not np.isnan(rew) else "n/a",
                          success=f"{suc:.4f}" if not np.isnan(suc) else "n/a")
        enl.write(f"[Ep {ne+1:>4}/{args.num_epochs}] steps={st:.2e}  sps={sps:,.0f}"
                  f"  A={al:.4f}  C={cl:.4f}"
                  f"  rew={rew:.2f}  suc={suc:.4f}"
                  f"  t={epoch_time:.1f}s  elapsed={(time.time()-start_time)/3600:.2f}h")
        epoch_bar.update()

        if args.checkpoint and (ne < 5 or ne >= args.num_epochs-5 or ne % 10 == 0):
            save_params(f"{save_path}/step_{int(training_state.env_steps)}.pkl",
                        ([s.params for s in training_state.actor_states],
                         training_state.critic_state.params))
        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline": trigger_sync()

    epoch_bar.close(); status_bar.close(); enl.stop()
    if args.checkpoint:
        save_params(f"{save_path}/final.pkl",
                    ([s.params for s in training_state.actor_states], training_state.critic_state.params))
    print(f"\nTraining complete.  Total time: {(time.time()-start_time)/3600:.2f}h", flush=True)
