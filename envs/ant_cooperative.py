"""
Cooperative Ant environment for split-joint multi-agent control.

The Ant has 8 actuated joints (2 per leg × 4 legs).
With n_agents agents, agent i controls joints [i*k : (i+1)*k] where k = 8 // n_agents.

  n_agents=2 → agent-0 controls front legs (4 joints), agent-1 controls rear legs
  n_agents=4 → each agent controls one leg (2 joints)
  n_agents=8 → each agent controls one joint

All agents share the same physics simulation and observe the full body state.
No single agent can locomote the ant independently — cooperation is necessary.

The training script (train_marl_crl.py --cooperative 1 --env_id ant_coop) handles
the action split; this class is a documented alias for the standard Ant.
"""
from envs.ant import Ant


class AntCooperative(Ant):
    """
    Ant with split-joint cooperative multi-agent control.
    Functionally identical to Ant — the joint split is managed by the trainer.
    """

    @staticmethod
    def joints_per_agent(n_agents: int) -> int:
        assert 8 % n_agents == 0, f"n_agents ({n_agents}) must divide 8 evenly"
        return 8 // n_agents
