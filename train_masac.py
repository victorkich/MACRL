# Parts adapted from the training script of scaling-crl (https://github.com/wang-kevin3290/scaling-crl).
# Copyright 2023 FLAIR. Licensed under the Apache License 2.0 (see LICENSE-APACHE).
# Modified by the MACRL authors.

# ============================================================
# MA-SAC  —  Multi-Agent Soft Actor-Critic (CTDE)
#
# Reward-maximizing baseline for MA-CRL comparison.
# Uses twin soft Q-networks (Bellman TD) instead of contrastive critic.
# Supports cooperative split-joint mode and independent-agent mode.
#
# Cooperative:     n_agents share one body (split joints)
# Non-cooperative: n_agents run in independent environments
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
    exp_name: str = "train_masac"
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
    obs_dim: int = 0          # filled at runtime
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
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005          # Polyak target update

    max_replay_size: int = 10_000
    min_replay_size: int = 1_000
    unroll_length: int = 62

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4

    num_episodes_per_env: int = 1
    training_steps_multiplier: int = 1
    use_all_batches: int = 0
    num_sgd_batches_per_training_step: int = 800

    eval_actor: int = 0
    expl_actor: int = 1
    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0

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


def residual_block(x, width, normalize, activation, res_scale, zero_init_last):
    identity = x
    for _ in range(3):
        x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
    last_init = nn.initializers.zeros if zero_init_last else lecun_uniform
    x = nn.Dense(width, kernel_init=last_init, bias_init=bias_init)(x)
    x = normalize(x); x = activation(x)
    return identity + res_scale * x


# ----------------------------
# Networks
# ----------------------------
class QNetwork(nn.Module):
    """Centralized soft Q-network: Q(state, action) -> scalar."""
    network_width: int = 256
    network_depth: int = 4
    use_relu: int = 0
    mup_residual_scaling: int = 1
    zero_init_residual_last: int = 1

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        normalize  = lambda y: nn.LayerNorm()(y)
        activation = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_last  = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_last)
        x = nn.Dense(1, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x.squeeze(-1)


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
        normalize  = lambda y: nn.LayerNorm()(y)
        activation = nn.relu if self.use_relu else nn.swish
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x); x = activation(x)
        res_scale = _get_res_scale(self.network_depth, bool(self.mup_residual_scaling))
        zero_last  = bool(self.zero_init_residual_last)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation, res_scale, zero_last)
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
    actor_states: Any          # list/tuple of per-agent TrainState
    critic_state: TrainState   # shared twin Q (Q1, Q2 in params)
    target_critic_params: Any  # Polyak target copy
    alpha_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    next_observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


def load_params(path):
    with epath.Path(path).open("rb") as f: return pickle.loads(f.read())

def save_params(path, params):
    with epath.Path(path).open("wb") as f: f.write(pickle.dumps(params))



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
        args.cooperative = 1  # trivially cooperative; avoids reshape mismatch

    print("=" * 60, flush=True)
    print("MA-SAC  —  Multi-Agent Soft Actor-Critic (CTDE)", flush=True)
    print("=" * 60, flush=True)
    print(f"  env_id      : {args.env_id}", flush=True)
    print(f"  n_agents    : {args.n_agents}", flush=True)
    print(f"  cooperative : {bool(args.cooperative)}", flush=True)
    print(f"  num_envs    : {args.num_envs}", flush=True)
    print(f"  critic_depth: {args.critic_depth}  actor_depth: {args.actor_depth}", flush=True)
    print("=" * 60, flush=True)

    total_envs = args.n_agents * args.num_envs
    args.env_steps_per_actor_step = total_envs * args.unroll_length
    args.num_prefill_env_steps    = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps  = int(np.ceil(args.min_replay_size / args.unroll_length))

    run_name = (f"MASAC_{args.env_id}_na:{args.n_agents}_{args.total_env_steps}"
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
        save_path = Path(args.wandb_dir) / Path(f"runs/masac_{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
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
        if eid in ("ant", "ant_coop"):
            from envs.ant import Ant
            env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 29; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif "ant" in eid and "maze" in eid:
            from envs.ant_maze import AntMaze
            _tmp = eid.removesuffix("_coop").removeprefix("ant_")
            name = "u_maze" if _tmp == "maze" else (_tmp[5:] if _tmp.startswith("maze_") else _tmp)
            env = AntMaze(backend="spring", exclude_current_positions_from_observation=False,
                          terminate_when_unhealthy=True, maze_layout_name=name)
            args.obs_dim = 29; args.goal_start_idx = 0; args.goal_end_idx = 2
        elif eid in ("humanoid", "humanoid_coop"):
            from envs.humanoid import Humanoid
            env = Humanoid(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
            args.obs_dim = 268; args.goal_start_idx = 0; args.goal_end_idx = 3
        elif eid in ("humanoid_maze_coop",) or ("humanoid" in eid and "maze" in eid):
            from envs.humanoid_maze import HumanoidMaze
            name = eid.replace("_coop","").replace("humanoid_maze_","")
            env = HumanoidMaze(backend="spring", maze_layout_name=name if name else "U_MAZE")
            args.obs_dim = 268; args.goal_start_idx = 0; args.goal_end_idx = 3
        else:
            raise NotImplementedError(f"Unknown env_id: {eid}")
        return env

    env      = make_env()
    env      = envs.training.wrap(env, episode_length=args.episode_length)
    obs_size = env.observation_size
    act_size = env.action_size

    # Cooperative joint split (supports uneven: e.g. humanoid 17 joints, 2 agents -> [9,8])
    if args.cooperative:
        base = act_size // args.n_agents
        rem  = act_size % args.n_agents
        joints_per_agent  = [base + (1 if i < rem else 0) for i in range(args.n_agents)]
        action_slice_starts = [sum(joints_per_agent[:i]) for i in range(args.n_agents)]
        actor_input_size  = obs_size + args.n_agents   # + one-hot agent ID
        sim_envs          = args.num_envs
        args.env_steps_per_actor_step = args.num_envs * args.unroll_length
        sa_state_dim      = args.obs_dim
        sa_action_dim     = act_size
    else:
        joints_per_agent  = [act_size] * args.n_agents
        action_slice_starts = [0] * args.n_agents
        actor_input_size  = obs_size
        sim_envs          = total_envs
        sa_state_dim      = args.obs_dim * args.n_agents
        sa_action_dim     = act_size * args.n_agents

    args.num_training_steps_per_epoch = max(1, (
        (args.total_env_steps - args.num_prefill_env_steps)
        // (args.num_epochs * args.env_steps_per_actor_step)
    ))
    total_env_steps_actual = (args.num_prefill_env_steps +
        args.num_epochs * args.num_training_steps_per_epoch * args.training_steps_multiplier * args.env_steps_per_actor_step)

    eval_num_envs = args.num_eval_envs if args.cooperative else args.n_agents * args.num_eval_envs
    eval_env = make_env(); eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)

    print(f"  obs_size: {obs_size}  act_size: {act_size}  joints_per_agent: {joints_per_agent}", flush=True)
    print(f"  --- training schedule ---", flush=True)
    print(f"  env_steps/actor_step: {args.env_steps_per_actor_step:,}", flush=True)
    print(f"  train_steps/epoch   : {args.num_training_steps_per_epoch}", flush=True)
    print(f"  total env steps     : ~{total_env_steps_actual:,.0f}  (target: {args.total_env_steps:,})", flush=True)
    print(flush=True)

    env_keys  = jax.random.split(env_key, sim_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step  = jax.jit(env.step)

    # ----------------------------
    # Networks  (per-agent actors, shared twin Q)
    # ----------------------------
    actor_keys = jax.random.split(key, args.n_agents + 1); key = actor_keys[0]
    actors, actor_states = [], []
    for i in range(args.n_agents):
        a = Actor(action_size=joints_per_agent[i], network_width=args.actor_network_width,
                  network_depth=args.actor_depth, use_relu=args.use_relu,
                  mup_residual_scaling=int(bool(args.mup_scale_actor) and bool(args.mup_residual_scaling)),
                  zero_init_residual_last=args.zero_init_residual_last)
        inp = np.ones([1, actor_input_size])
        ast = TrainState.create(apply_fn=a.apply,
                                params=a.init(actor_keys[i + 1], inp),
                                tx=optax.adam(args.actor_lr))
        actors.append(a); actor_states.append(ast)

    key, q_key1, q_key2 = jax.random.split(key, 3)
    qnet = QNetwork(network_width=args.critic_network_width, network_depth=args.critic_depth,
                    use_relu=args.use_relu, mup_residual_scaling=args.mup_residual_scaling,
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
    # Replay buffer  (stores Transition with next_observation in extras)
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
    # Actor steps
    # ----------------------------
    def _coop_obs_i(obs, i):
        aid = jnp.zeros((*obs.shape[:-1], args.n_agents), dtype=obs.dtype).at[..., i].set(1.0)
        return jnp.concatenate([obs, aid], axis=-1)

    def build_full_action(actor_params_list, obs, key, stochastic):
        sub_actions, log_probs = [], []
        agent_keys = jax.random.split(key, args.n_agents)
        for i, (a, ap) in enumerate(zip(actors, actor_params_list)):
            inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*obs_size:(i+1)*obs_size]
            m, ls = a.apply(ap, inp); s = jnp.exp(ls)
            if stochastic:
                xt = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
            else:
                xt = m
            act = nn.tanh(xt)
            lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1 - act**2 + 1e-6)).sum(-1)
            sub_actions.append(act); log_probs.append(lp)
        return jnp.concatenate(sub_actions, axis=-1), sum(log_probs)

    def actor_step(ts, env, env_state, key, stochastic, extra_fields):
        ap_list = [s.params for s in ts.actor_states]
        obs = env_state.obs
        if not args.cooperative:
            obs = jnp.tile(obs.reshape(args.n_agents, args.num_envs, obs_size)
                           .transpose(1,0,2).reshape(sim_envs, obs_size), (1,1))
            # Actually obs is already (sim_envs, obs_size)
        full_act, _ = build_full_action(ap_list, env_state.obs
                                         if args.cooperative else
                                         env_state.obs.reshape(args.n_agents, args.num_envs, obs_size)
                                         .transpose(1,0,2).reshape(args.num_envs, args.n_agents*obs_size)
                                         if not args.cooperative else env_state.obs,
                                         key, stochastic)
        nstate = env.step(env_state, full_act if args.cooperative else
                          full_act.reshape(sim_envs, act_size))

        if args.cooperative:
            cur_obs = env_state.obs; nxt_obs = nstate.obs
            rew = nstate.reward; disc = 1.0 - nstate.done
            extras = {"state_extras": {x: nstate.info[x] for x in extra_fields}}
        else:
            n_ag, nenv = args.n_agents, args.num_envs
            obs_j  = env_state.obs.reshape(n_ag, nenv, obs_size).transpose(1,0,2).reshape(nenv, n_ag*obs_size)
            nobs_j = nstate.obs.reshape(n_ag, nenv, obs_size).transpose(1,0,2).reshape(nenv, n_ag*obs_size)
            cur_obs = obs_j; nxt_obs = nobs_j
            rews = nstate.reward.reshape(n_ag, nenv)
            rew  = rews.mean(0) if args.ma_reward == "mean" else (rews.sum(0) if args.ma_reward == "sum" else rews.min(0))
            disc = 1.0 - nstate.done[:nenv]
            extras = {"state_extras": {x: nstate.info[x][:nenv] for x in extra_fields}}

        return nstate, Transition(observation=cur_obs, next_observation=nxt_obs,
                                   action=full_act, reward=rew, discount=disc, extras=extras)

    @jax.jit
    def get_experience(ts, env_state, buffer_state, key):
        def f(carry, _):
            es, k = carry; k, nk = jax.random.split(k)
            nes, tr = actor_step(ts, env, es, k, args.expl_actor == 1, ("truncation","seed"))
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

        # Compute next actions and log_probs from each agent's actor
        next_obs = transitions.next_observation
        agent_keys = jax.random.split(key, args.n_agents)
        next_sub_acts, next_log_probs = [], []
        for i, (a, ap) in enumerate(zip(actors, [s.params for s in ts.actor_states])):
            inp = _coop_obs_i(next_obs, i) if args.cooperative else next_obs[:, i*args.obs_dim:(i+1)*args.obs_dim]
            m, ls = a.apply(ap, inp); s = jnp.exp(ls)
            xt = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
            act = nn.tanh(xt)
            lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1-act**2+1e-6)).sum(-1)
            next_sub_acts.append(act); next_log_probs.append(lp)
        next_action   = jnp.concatenate(next_sub_acts, axis=-1)
        next_log_prob = sum(next_log_probs)

        if args.cooperative:
            next_state = next_obs[:, :args.obs_dim]
            cur_state  = transitions.observation[:, :args.obs_dim]
        else:
            next_state = jnp.concatenate([next_obs[:, i*obs_size: i*obs_size+args.obs_dim]
                                           for i in range(args.n_agents)], axis=-1)
            cur_state  = jnp.concatenate([transitions.observation[:, i*obs_size: i*obs_size+args.obs_dim]
                                           for i in range(args.n_agents)], axis=-1)

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
        obs = transitions.observation
        alpha = jnp.exp(ts.alpha_state.params["log_alpha"])
        agent_keys = jax.random.split(key, args.n_agents)

        if args.cooperative:
            cur_state = obs[:, :args.obs_dim]
        else:
            cur_state = jnp.concatenate([obs[:, i*obs_size: i*obs_size+args.obs_dim]
                                          for i in range(args.n_agents)], axis=-1)

        total_actor_loss = 0.0

        # Full actor update: compute all actions jointly
        all_log_probs = []
        all_acts      = []
        for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
            inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*args.obs_dim:(i+1)*args.obs_dim]
            m, ls = a.apply(ast.params, inp); s = jnp.exp(ls)
            xt   = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
            act  = nn.tanh(xt)
            lp   = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1-act**2+1e-6)).sum(-1)
            all_acts.append(act); all_log_probs.append(lp)
        full_act  = jnp.concatenate(all_acts, axis=-1)
        log_prob  = sum(all_log_probs)

        # Per-agent actor gradient
        new_actor_states = []
        for i, (a, ast) in enumerate(zip(actors, ts.actor_states)):
            def loss_fn_i(ap, i=i):
                inp = _coop_obs_i(obs, i) if args.cooperative else obs[:, i*args.obs_dim:(i+1)*args.obs_dim]
                m, ls = a.apply(ap, inp); s = jnp.exp(ls)
                xt  = m + s * jax.random.normal(agent_keys[i], m.shape, dtype=m.dtype)
                act = nn.tanh(xt)
                lp  = (jax.scipy.stats.norm.logpdf(xt, m, s) - jnp.log(1-act**2+1e-6)).sum(-1)
                q   = qnet.apply(ts.critic_state.params["q1"], cur_state, full_act)
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
        return ts, {"actor_loss": total_actor_loss / args.n_agents, "log_alpha": new_alpha.params["log_alpha"]}

    @jax.jit
    def sgd_step(carry, transitions):
        # transitions.observation shape: (num_envs, episode_length, obs_size)
        ts, key = carry; key, ck, ak = jax.random.split(key, 3)
        # Flatten (num_envs, episode_length, ...) -> (num_envs*episode_length, ...)
        flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), transitions)
        perm = jax.random.permutation(ck, flat.observation.shape[0])
        sampled = jax.tree_util.tree_map(lambda x: x[perm[:args.batch_size]], flat)

        ts, critic_metrics = update_critic(sampled, ts, ck)
        ts, actor_metrics  = update_actors_and_alpha(sampled, ts, ak)
        ts = ts.replace(gradient_steps=ts.gradient_steps + 1)
        return (ts, key), {**critic_metrics, **actor_metrics}

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
            # Stack along a new leading axis: (num_episodes_per_env, num_envs, episode_length, ...)
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
                              episode_length=args.episode_length, key=eval_env_key)

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
    epoch_bar = enl.counter(total=args.num_epochs, desc="MA-SAC", unit="epoch", color="green",
                             bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count}/{total} [{elapsed}<{eta}]")
    status_bar = enl.status_bar(
        status_format="Ep {ep}/{total} │ steps={steps} │ A={actor_loss} │ C={critic_loss} │ α={log_alpha} │ rew={reward} │ suc={success}",
        color="bold_white_on_black",
        ep=0, total=args.num_epochs, steps="n/a", actor_loss="n/a",
        critic_loss="n/a", log_alpha="n/a", reward="n/a", success="n/a")

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
        la  = metrics.get("training/log_alpha",   float("nan"))
        suc = metrics.get("eval/episode_success", float("nan"))
        rew = metrics.get("eval/episode_reward",  float("nan"))
        st  = metrics["training/envsteps"]

        status_bar.update(ep=ne+1, total=args.num_epochs, steps=f"{st:.2e}",
                          actor_loss=f"{al:.4f}", critic_loss=f"{cl:.4f}",
                          log_alpha=f"{la:.3f}", reward=f"{rew:.2f}" if not np.isnan(rew) else "n/a",
                          success=f"{suc:.4f}" if not np.isnan(suc) else "n/a")
        enl.write(f"[Ep {ne+1:>4}/{args.num_epochs}] steps={st:.2e}  sps={sps:,.0f}"
                  f"  A={al:.4f}  C={cl:.4f}  α={la:.3f}"
                  f"  rew={rew:.2f}  suc={suc:.4f}"
                  f"  t={epoch_time:.1f}s  elapsed={(time.time()-start_time)/3600:.2f}h")
        epoch_bar.update()

        if args.checkpoint and (ne < 5 or ne >= args.num_epochs-5 or ne % 10 == 0):
            save_params(f"{save_path}/step_{int(training_state.env_steps)}.pkl",
                        (training_state.alpha_state.params,
                         [s.params for s in training_state.actor_states],
                         training_state.critic_state.params))
        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline": trigger_sync()

    epoch_bar.close(); status_bar.close(); enl.stop()

    if args.checkpoint:
        save_params(f"{save_path}/final.pkl",
                    (training_state.alpha_state.params,
                     [s.params for s in training_state.actor_states],
                     training_state.critic_state.params))
    print(f"\nTraining complete.  Total time: {(time.time()-start_time)/3600:.2f}h", flush=True)
