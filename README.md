# Scalable Multi-Agent Contrastive Reinforcement Learning

**Webpage:** https://victorkich.github.io/MACRL/ · **Paper:** https://openreview.net/pdf?id=cCWvLZocDZ

Reference implementation of MACRL and the baselines it is compared against.

MACRL trains a **centralized contrastive critic** with parameter-shared
decentralized actors. On decoupled-body tasks the critic is **factored**: each
agent is encoded separately from its own body state, the shared object state and
its own action, and the per-agent embeddings are pooled into a team
representation that is scored against a goal embedding.

This repository contains only what is needed to rerun the experiments. Analysis,
plotting and the paper's result data are not included.

## Install

Python 3.12 and a CUDA 12 GPU.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for every reported experiment.
Weights and Biases logging is off by default, so nothing needs configuring to
run. Pass `--track` to enable it.

Quick check that the install works, roughly two minutes on one GPU:

```bash
python train_marl_crl.py --env_id ant_pair_maze --n_agents 2 \
    --cooperative 1 --partial_obs 1 --total_env_steps 300000 --num_envs 32
```

Progress prints as `PROG steps=<env steps> suc=<success rate>`.

## Running MACRL

```bash
python train_marl_crl.py \
    --env_id ant_pair_maze --n_agents 2 \
    --cooperative 1 --partial_obs 1 --multi_body_goal 1 \
    --actor_depth 64 --critic_depth 64 --mup_residual_scaling 1 \
    --total_env_steps 200000000 --seed 1000
```

Depth is set with `--actor_depth` and `--critic_depth` together. The paper's
`D=4` and `D=64` results differ only in these two flags. `--mup_residual_scaling 1`
applies the `1/sqrt(L)` residual scaling and should stay on at depth.

### Environments

| `--env_id` | Task | Required flags |
| --- | --- | --- |
| `ant_easy`, `ant_coop` | Shared-body locomotion | `--cooperative 1` |
| `ant_maze_coop` | Maze U | `--cooperative 1` |
| `ant_maze_big_maze_coop` | Maze Big | `--cooperative 1` |
| `ant_maze_hardest_maze_coop` | Maze Hard | `--cooperative 1` |
| `ant_pair_maze` | Pair Maze U | `--cooperative 1 --partial_obs 1 --multi_body_goal 1` |
| `ant_pair_maze_big_maze` | Pair Maze Big | same as above |
| `ant_pair_maze_hardest_maze` | Pair Maze Hard | same as above |
| `ant_pair_maze_ball_coop` | Coop Ball U | `--cooperative 1 --partial_obs 1` |
| `ant_pair_maze_ball_coop_big_maze` | Coop Ball Big | `--cooperative 1 --partial_obs 1` |
| `ant_pair_corridor` | Pair Corridor | `--cooperative 1 --partial_obs 1` |
| `ant_pair_chicane` | Chicane | `--cooperative 1 --partial_obs 1` |

Agent count is `--n_agents` (2 by default, and 1, 2, 4, 8 for the shared-body
scaling study). Budgets used in the paper are 100M env steps for Ant Easy, Ant
Coop, Maze U, Maze Big and Pair Corridor; 200M for Maze Hard and Pair Maze U;
300M for Pair Maze Big, Pair Maze Hard, Coop Ball U, Coop Ball Big and Chicane;
and 400M for Coop Ball Hard (`ant_pair_maze_ball_coop_hardest_maze`). Each
reported number is final performance: the mean of the last five evaluations of a
seed, averaged over seeds. In the main results table MACRL uses seeds 1000 to
1004, and up to 1007 where the centralized-versus-independent comparison added
seeds; the baselines use 1000 to 1002. The paper gives the seed count of every
ablation and depth sweep.

### Critic aggregation

`--critic_agg` selects how the centralized critic combines agents on
decoupled-body tasks. Goal semantics are identical across all four, so the flag
isolates aggregation.

| Value | Critic |
| --- | --- |
| `mean` | Factored, mean-pooled per-agent embeddings. Paper default. |
| `min` | Factored, min-pooled. The weakest agent bottlenecks the team score. |
| `concat` | Monolithic. One encoder over the joint state and joint action. |
| `independent` | No pooling. One contrastive term per agent. |

`--critic_agg independent` with `--indep_own_goal 1` gives each agent its own
future position as the positive instead of the team goal, which is well posed
under partial observation. This pair of flags reproduces the centralized against
independent comparison.

## Running the baselines

All baselines take the same environment flags as above.

| Script | Method |
| --- | --- |
| `train_magcsac.py` | MA-SAC + HER, goal-conditioned |
| `train_maddpg_her.py` | MADDPG + HER |
| `train_magciql.py` | MA-GCIQL |
| `train_magcbc.py` | MA-GCBC |
| `train_masac.py` | MA-SAC, reward only |
| `train_maddpg.py` | MADDPG, reward only |
| `train_mappo.py` | MAPPO |

```bash
python train_magcsac.py --env_id ant_pair_maze --n_agents 2 \
    --cooperative 1 --partial_obs 1 --total_env_steps 200000000 --seed 1000
```

The strengthened baselines in the paper use a residual backbone matched to
MACRL, LayerNorm and multiple gradient updates per environment step. Those are
the defaults here. `train_magcsac.py --per_agent_her 1` relabels to an agent's
own future position rather than the team mean.

## Layout

```
train_marl_crl.py     MACRL
train_ma*.py          baselines
buffer.py             replay buffer and future-goal relabeling
evaluator.py          evaluation rollouts and success computation
envs/                 Brax environments and MuJoCo XML assets
```

Useful shared flags: `--seed`, `--num_envs`, `--total_env_steps`, `--gpu`,
`--track`, `--checkpoint`. Run any script with `--help` for the full list.

## Notes

Environments are built on Brax with the spring backend. Success is measured
against the goal the episode was commanded to reach, which is sampled uniformly
per episode from a fixed set of valid targets, using the same sampler during
training and evaluation.

Single-agent manipulation environments from scaling-crl are not
included, since no experiment in the paper uses them.

Very deep runs are limited by host RAM during XLA compilation rather than by
GPU memory. `D=64` needs roughly 7GB, `D=512` roughly 27GB and `D=1024` up to
73GB.

## Acknowledgements and license

This code builds on [scaling-crl](https://github.com/wang-kevin3290/scaling-crl)
(Wang et al., 2025), which builds on [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL).
Files copied or adapted from scaling-crl remain under the Apache License 2.0
([LICENSE-APACHE](LICENSE-APACHE)); they are listed in [NOTICE](NOTICE) and marked
in their headers. All other code is released under the MIT License
([LICENSE](LICENSE)).
