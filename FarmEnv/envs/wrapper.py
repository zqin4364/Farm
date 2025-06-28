from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import numpy as np

from FarmEnv.envs.core import World
from FarmEnv.envs.config import B_total, DEFAULT_HISTORY_LENGTH

class FarmEnvWrapper(ParallelEnv):
    def __init__(self, num_farmers, total_res=B_total, max_step=50):
        super().__init__()

        #create env
        self.world = World(
            num_farmers=num_farmers,
            total_res=total_res,
            max_step=max_step
        )

        self.farmer_names = list(self.world.farmers.keys())
        self.ai_name = "AI"
        self.agents = self.farmer_names + [self.ai_name]
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            agent: self._get_obs_space(agent) for agent in self.agents
        }

        self.action_spaces = {
            agent: self._get_action_space(agent) for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        obs = self.world.reset(seed=seed, options=options)
        obs["AI"]["history"] = self.normalize_history()
        self.agents = self.possible_agents[:]
        return obs

    def step(self, actions):
        # 确保所有智能体都有动作
        full_actions = {}
        for agent in self.agents:
            if agent in actions:
                full_actions[agent] = actions[agent]
            else:
                # 为缺失的智能体提供默认动作
                if agent.startswith("Farmer"):
                    full_actions[agent] = {"request": 0.0}
                else:
                    full_actions[agent] = {"allocations": {f"Farmer_{i}": 0.0} for i in range(len(self.farmer_names))}

            obs, rewards, infos, dones = self.world.step(full_actions)

            if "AI" in obs:
                obs["AI"]["history"] = self.normalize_history()

            # 确保所有智能体都有完成状态
            full_dones = {agent: dones.get(agent, True) for agent in self.agents}
            self.agents = [agent for agent in self.agents if not full_dones[agent]]

            return obs, rewards, full_dones, infos

    def _get_obs_space(self, agent):
        if agent.startswith("Farmer"):
            return spaces.Dict({
                "grow_stage": spaces.Discrete(5),
                "weather_s": spaces.Discrete(3)
            })
        elif agent == "AI":
            return spaces.Dict({
                "weather_s": spaces.Discrete(3),
                "current_res": spaces.Box(low=0.0, high=self.world.total_res, shape=()),
                "credit": spaces.Box(low=0.0, high=1.0, shape=(len(self.farmer_names),)),
                "history": spaces.Box(low=0.0, high=1.0, shape=(len(self.farmer_names), 3))
            })

    def _get_action_space(self, agent):
        if agent.startswith("Farmer"):
            return spaces.Dict({
                "request": spaces.Box(
                    low=0.0, high=self.world.total_res, shape=(), dtype=np.float32
                )
            })
        elif agent == "AI":
            return spaces.Dict({
                "allocations": spaces.Box(
                    low=0.0, high=self.world.total_res, shape=(len(self.farmer_names),), dtype=np.float32
                )
            })
        else:
            raise ValueError(f"Unknown agent{agent}")

    def normalize_history(self):
        weather_map = {"Drought": -1, "Normal": 0, "Rainy": 1}
        result = np.zeros((len(self.farmer_names), 3), dtype=np.float32)

        for i, name in enumerate(self.farmer_names):
            history = list(self.world.ai_agent.history[name])
            if history:
                record = history[-1]
                result[i, 0] = record["grow_stage"]/4.0
                w_val = weather_map.get(record["weather_s"], 0)
                result[i, 1] = (w_val + 1) / 2.0
                result[i, 2] = record["allocated_res"] / max(self.world.total_res, 1e-6)

        return result

