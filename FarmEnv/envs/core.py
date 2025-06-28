import numpy as np
from collections import deque
from pettingzoo import ParallelEnv

from .config import B_total
from FarmEnv.envs.utils.true_model import True_Demand
from FarmEnv.envs.utils.bayesian_estimator import BayesianEstimator
from FarmEnv.envs.config import B_total, DEFAULT_HISTORY_LENGTH

class Entity(object):
    def __init__(self):
        pass

class Agent(Entity):
    def __init__(self, name):
        self.name = name


#weather
class Weather():
    """
    Weather Entity
    """
    def __init__(self):
        self.states = ["Drought", "Normal", "Rainy"]
        self.probs = [0.3, 0.5, 0.2]
        self.values = {
            "Drought": -1.0,
            "Normal": 0.0,
            "Rainy": 1.0
        }

        self.current_state = None
        self.current_val = None
        # init weather
        self.update()

    def update(self):
        self.current_state = np.random.choice(self.states, p=self.probs)
        self.current_val = self.values[self.current_state]

    def get_weather(self):
        return self.current_state

    def get_value(self):
        return self.current_val


#crop
class Crop():
    """
    Crop Entity
    """
    def __init__(self):
        #growth stage = {sow, sprout, grow, blossom, ripe} => {0, 1, 2, 3, 4}
        self.grow_stage = 0

        #The reward for each stage
        self.base_reward = 1.0
        self.stage_weight = [0.2, 0.4, 0.6, 0.8, 1.0]

    def grow(self,  allocated_res, actual_demand, weather_val, mu1=1, mu2=1, mu3=1):
        theta = allocated_res / actual_demand
        z = mu1 * theta + mu2 * weather_val + mu3
        grow_prob = 1 / (1 + np.exp(-z))

        rand_val = np.random.rand()
        if grow_prob > rand_val:
            self.grow_stage = min(4, self.grow_stage + 1)



    def get_stage_reward(self):
        weight = self.stage_weight[self.grow_stage]
        return self.base_reward * weight


class FarmerAgent(Agent):
    def __init__(self, name, weather, policy_fn = None):
        super(Agent, self).__init__()
        self.name = name
        #farmer's request
        self.request = 0.0
        self.weather = weather
        self.crop = Crop()
        self.policy_fn = policy_fn if policy_fn is not None else self.default_policy

    #defalut use estimate request
    def default_policy(self, obs):
        grow_stage = obs["grow_stage"]
        weather_val = obs["weather_val"]

        if grow_stage == 4:
            return 0.0
        else:
            return 1.0  #temporary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def generate_request(self, obs):
        """based on weather and grow stage"""
        self.request = self.policy_fn(obs)
        return self.request

    #actual control by env, nor not farmer
    def apply_res(self, allocated_res, actual_demand, weather_val):
        self.crop.grow(allocated_res, actual_demand, weather_val)
        new_stage = self.crop.grow_stage
        return new_stage


class AIAgent(Agent):
    def __init__(self, name, agents, weather, current_res, policy_fn=None):
        super(Agent, self).__init__()
        self.name = name
        self.agents = agents
        self.current_res = current_res
        self.weather = weather
        self.policy_fn = policy_fn if policy_fn is not None else self.default_fn

        #Guess the actual request by history and bayesian estimator.
        self.history = {
            agent_name : deque(maxlen=DEFAULT_HISTORY_LENGTH) for agent_name in self.agents.keys()
        }
        #default credit is 1.0
        self.credit = {
            agent_name : 1.0 for agent_name in self.agents.keys()
        }

        self.estimator = {
            agent_name : BayesianEstimator() for agent_name in self.agents.keys()
        }

    def default_fn(self):
        pass

    def update_history(self, agent_name, obs, allocated_res):
        self.history[agent_name].append({
            "grow_stage": obs["grow_stage"],
            "weather_s": obs["weather_s"],
            "allocated_res": allocated_res
        })

    def update_estimator(self, agent_name, obs):
        history = self.history[agent_name]
        if not history:
            return

        last_record = history[-1]
        last_stage = last_record["grow_stage"]
        current_stage = obs["grow_stage"]

        grown = current_stage > last_stage

        grow_stage = last_stage
        weather_s = last_record["weather_s"]
        allocated_res = last_record["allocated_res"]

        self.estimator[agent_name].update(
            grow_stage = grow_stage,
            weather_s = weather_s,
            allocated_res=allocated_res,
            grown=grown
        )

    def estimate_demand(self, agent_name, grow_stage, weather_s, gamma1=0.5, gamma2=0.5):
        base_estimate = self.estimator[agent_name].estimate(grow_stage, weather_s)

        credit = self.credit.get(agent_name, 1.0)
        credit_estimate = base_estimate * (gamma1 + gamma2 * credit)

        adjust_estimate = max(0.1, min(credit_estimate, 10.0))

        return adjust_estimate

    def update_credit(self, agent_name, estimated_demand, request_demand, lambda1 = 0.1):
        old_credit = self.credit.get(agent_name, 1.0)
        error = request_demand - estimated_demand
        score = np.exp(-error ** 2)
        new_credit = (1 - lambda1) * old_credit + lambda1 * score

        self.credit[agent_name] = new_credit

    def allocated_resoureces(self, obs_dict, gamma=0.5):
        demands = {}
        weight = {}
        allocations = {}
        grow_stage = {}

        #policy allocations
        for agent_name, agent in self.agents.items():
            obs = obs_dict[agent_name]
            grow_stage[agent_name] = obs["grow_stage"]
            weather_s = self.weather.get_weather()

            policy_input = {
                "grow_stage": grow_stage[agent_name],
                "weather_s": weather_s,
                "credit": self.credit[agent_name]
            }

            policy_demand = self.policy_fn(policy_input)
            demands[agent_name] = policy_demand

        #credit allocations
        exp_credit = {agent_name : np.exp(self.credit[agent_name]) for agent_name in self.agents.keys()}
        sum_exp = sum(exp_credit.values())

        for name in demands:
            weight[name] = exp_credit[name] / sum_exp if sum_exp > 0 else len(self.agents)

        #combine
        for agent_name, agent in self.agents.items():
            hybrid_demand = (1 - gamma) * demands[agent_name] + gamma * self.current_res
            allocation = max(0.0, min(hybrid_demand, self.current_res))
            allocations[agent_name] = allocation

        total_allocations = sum(allocations.values())
        if total_allocations > self.current_res:
            scale = self.current_res / total_allocations
            for agent_name in self.agents.keys():
                allocations[agent_name] *= scale
            total_allocations = self.current_res

        self.current_res -= total_allocations

        return allocations #dict:{agents: allocation}


class World(ParallelEnv):
    metadata = {"render_modes": [], "name": "farm_v0"}

    def __init__(self, num_farmers, total_res=B_total, max_step=50):
        self.num_farmers = num_farmers
        self.total_res = total_res
        self.max_step = max_step

        #Init weather
        self.weather = Weather()
        #init farmer for dict
        self.farmers = {f"Farmer_{i}": FarmerAgent(name=f"Farmer_{i}", weather=self.weather) for i in range(num_farmers)}
        self.ai_agent = AIAgent(name="AI", agents=self.farmers, weather=self.weather, current_res=self.total_res)

        self.timestep = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.timestep = 0
        self.weather.update()

        for farmer in self.farmers.values():
            farmer.crop = Crop()
            #farmer.request = 0

        self.ai_agent.current_res = self.total_res

        observations = {agent_name: self._get_obs(agent_name=agent_name) for agent_name in self.farmers}
        observations["AI"] = self._get_obs("AI")

        return observations

    def step(self, actions: dict[str, dict]):
        """
        One step of an episode
        :param actions: actions: {"AI": {"allocations": {"Farmer_0": x}}, "farmer": {"farmer_0": {"request": y}}}
            AI: {"allocations": {"Farmer_0": x}}
            farmer: {"farmer_0": {"request": y}}
        """
        if self.timestep >= self.max_step:
            raise RuntimeError("Max timestep reached")

        self.timestep += 1
        self.weather.update()

        #Farmer's action: request \hat(d_i^t)
        requests = {}
        for agent_name, farmer in self.farmers.items():
            action = actions.get(agent_name, {})
            request = action.get("request", 0.0)
            requests[agent_name] = request

        #update estimator, creidt and estimate actual request
        estimated_demands = {}
        for agent_name, farmer in self.farmers.items():
            obs = self._get_obs(agent_name=agent_name)

            estimated_demand = self.ai_agent.estimate_demand(agent_name=agent_name,
                                                             grow_stage=obs["grow_stage"],
                                                             weather_s=self.weather.get_weather())
            estimated_demands[agent_name] = estimated_demand

            self.ai_agent.update_credit(agent_name=agent_name,
                                        estimated_demand=estimated_demand,
                                        request_demand=requests[agent_name])

        ai_action = actions.get("AI", {})
        allocations = ai_action.get("allocations", {})

        #farmer apply resource to update grow_stage and compute reward
        observations = {}
        rewards = {}
        infos = {}
        dones = {}
        farmer_rewards = {}

        for agent_name, farmer in self.farmers.items():
            allocation = allocations.get(agent_name, 0.0)
            obs = self._get_obs(agent_name)
            true_demand = True_Demand(weather_s=self.weather.get_weather(), grow_stage=obs["grow_stage"])
            farmer.crop.grow(allocated_res=allocation,
                             actual_demand=true_demand,
                             weather_val=self.weather.get_value())

            reward = self._compute_farmer_reward(agent_name=agent_name,
                                                 request=requests[agent_name],
                                                 allocation=allocation,
                                                 obs=obs,
                                                 credit=self.ai_agent.credit[agent_name])

            observations[agent_name] = obs
            rewards[agent_name] = reward
            farmer_rewards[agent_name] = reward
            infos[agent_name] = {f"True Demand": true_demand}
            dones[agent_name] = False

        #AI' view
        observations["AI"] = self._get_obs(agent_name="AI")
        rewards["AI"] = self._compute_ai_reward(farmer_rewards=farmer_rewards)
        infos["AI"] = {"Estimate Demands:": estimated_demands}
        dones["AI"] = False

        done_flag = self.timestep >= self.max_step
        for k in dones:
            dones[k] = done_flag

        return observations, rewards, infos, dones

    def _get_obs(self, agent_name):
        if agent_name.startswith("Farmer"):
            farmer = self.farmers[agent_name]
            return {
                "grow_stage": farmer.crop.grow_stage,
                "weather_s": self.weather.get_weather()
            }

        elif agent_name == "AI":
            return {
                "weather_s": self.weather.get_weather(),
                "current_res": self.ai_agent.current_res,
                "credit": self.ai_agent.credit,
                "history": self.ai_agent.history
            }

        else:
            raise ValueError(f"Unknown agent_name: {agent_name}")

    def _compute_farmer_reward(self, agent_name, request, allocation, obs, credit, alpha1=1.0, alpha2=0.1):

        farmer = self.farmers[agent_name]
        crop = farmer.crop
        grow_stage = obs["grow_stage"]
        weather_s = obs["weather_s"]

        true_demand = True_Demand( weather_s=weather_s, grow_stage=grow_stage)
        true_demand = max(true_demand, 1e-6)

        stage_reward = crop.get_stage_reward()

        error = (request - true_demand) ** 2
        penalty = alpha2 * error

        reward_weight = allocation / true_demand
        reward = alpha1 * stage_reward * reward_weight - alpha2 * (2 - credit) * penalty

        return reward

    def _compute_ai_reward(self, farmer_rewards: dict):
        """
        The reward is sum of farmers rewards of AI
        :param farmer_rewards: {"farmer.name": reward}
        :return: rewards of AI
        """
        return sum(farmer_rewards.values())