import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from tqdm import tqdm
from FarmEnv.envs.wrapper import FarmEnvWrapper  # 导入您的环境

import matplotlib
matplotlib.use('TkAgg')  # 使用 Tkinter 交互式后端
import matplotlib.pyplot as plt


# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# 配置参数
class Config:
    # 环境参数
    NUM_FARMERS = 2
    TOTAL_RES = 100
    MAX_STEP = 50

    # 训练参数
    NUM_EPISODES = 10000  # 增加训练回合数到100,000
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    LEARNING_RATE = 5e-5  # 降低学习率

    # 网络参数
    HIDDEN_SIZE = 128  # 增加隐藏层大小

    # 经验缓冲区
    BUFFER_SIZE = 1000
    BATCH_SIZE = 128  # 增加批量大小
    NUM_EPOCHS = 4

    # 日志和保存
    LOG_INTERVAL = 100  # 增加日志间隔
    PLOT_INTERVAL = 1000  # 绘图间隔
    SAVE_INTERVAL = 1000  # 模型保存间隔
    SAVE_DIR = "ippo_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 归一化参数
    REWARD_SCALE = 0.01  # 奖励缩放因子


# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 更深的神经网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE),
            nn.ReLU()
        )

        # Actor 头 - 输出动作均值
        self.actor = nn.Linear(Config.HIDDEN_SIZE, output_size)

        # Critic 头 - 输出状态值
        self.critic = nn.Linear(Config.HIDDEN_SIZE, 1)

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        """正交初始化权重"""
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value


# IPPO 训练器
class IPPOTrainer:
    def __init__(self, env):
        self.env = env
        self.agents = {}

        # 为每个智能体创建策略网络和优化器
        for agent_name in env.possible_agents:
            # 确定输入和输出大小
            if agent_name.startswith("Farmer_"):
                input_size = 2  # grow_stage + weather_s (编码后)
                output_size = 1  # request值
            else:  # AI 智能体
                # weather_s + current_res + credit + history
                input_size = 1 + 1 + len(env.farmer_names) + len(env.farmer_names) * 3
                output_size = len(env.farmer_names)  # 每个农民的分配量

            policy = PolicyNetwork(input_size, output_size)
            optimizer = optim.Adam(policy.parameters(), lr=Config.LEARNING_RATE)

            # 添加学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=100
            )

            self.agents[agent_name] = {
                "policy": policy,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "buffer": deque(maxlen=Config.BUFFER_SIZE),
                "input_size": input_size,
                "output_size": output_size,
                "reward_stats": {"mean": 0, "std": 1}  # 奖励统计
            }

    def normalize_obs(self, obs, agent_name):
        """预处理观察为张量并归一化"""
        if agent_name.startswith("Farmer_"):
            # 农民观察: {'grow_stage': int, 'weather_s': str}
            # 将天气字符串编码为数字: Drought=0, Normal=1, Rainy=2
            weather_mapping = {"Drought": 0, "Normal": 1, "Rainy": 2}
            weather_code = weather_mapping.get(obs["weather_s"], 1)
            obs_tensor = torch.tensor([obs["grow_stage"], weather_code], dtype=torch.float32)
            # 归一化
            obs_tensor[0] /= 4.0  # 生长阶段归一化到 [0, 1]
            return obs_tensor

        elif agent_name == "AI":
            # AI观察: {'weather_s': str, 'current_res': float, 'credit': dict, 'history': np.array}
            weather_mapping = {"Drought": 0, "Normal": 1, "Rainy": 2}
            weather_code = weather_mapping.get(obs["weather_s"], 1)

            # 信用值
            credit_values = [obs["credit"][name] for name in self.env.farmer_names]

            # 历史记录
            history = obs["history"].flatten()

            # 组合所有特征
            ai_obs = np.concatenate([
                [weather_code, obs["current_res"]],
                credit_values,
                history
            ])

            # 归一化
            ai_obs[0] /= 2.0  # 天气归一化
            ai_obs[1] /= Config.TOTAL_RES  # 资源归一化
            credit_values = np.array(credit_values)
            if credit_values.max() > credit_values.min():
                credit_values = (credit_values - credit_values.min()) / (credit_values.max() - credit_values.min())
            ai_obs[2:2 + len(credit_values)] = credit_values
            ai_obs[2 + len(credit_values):] = history  # 历史已经在wrapper中归一化

            return torch.tensor(ai_obs, dtype=torch.float32)

    def act(self, obs, agent_name, deterministic=False):
        """为智能体选择动作"""
        obs_tensor = self.normalize_obs(obs, agent_name)
        agent = self.agents[agent_name]

        # 获取动作均值和状态值
        action_mean, value = agent["policy"](obs_tensor.unsqueeze(0))

        # 创建高斯分布 (固定标准差为1)
        dist = torch.distributions.Normal(action_mean, 1.0)

        if deterministic:
            action = action_mean  # 测试时使用均值
        else:
            action = dist.sample()  # 训练时采样

        log_prob = dist.log_prob(action).sum()

        # 转换为环境期望的动作格式
        if agent_name.startswith("Farmer_"):
            action_dict = {"request": action.item()}
            action_value = action  # 存储动作值
        else:  # AI 智能体
            # 注意：action的形状是(1, num_farmers)
            allocations = {f"Farmer_{i}": max(0, action[0, i].item()) for i in range(len(self.env.farmer_names))}
            action_dict = {"allocations": allocations}
            action_value = action.squeeze(0)  # 存储动作值

        return action_dict, log_prob, value.item(), obs_tensor, action_value

    def normalize_reward(self, agent_name, reward):
        """奖励归一化"""
        agent = self.agents[agent_name]
        stats = agent["reward_stats"]

        # 更新统计
        stats["mean"] = 0.99 * stats["mean"] + 0.01 * reward
        stats["std"] = 0.99 * stats["std"] + 0.01 * (reward - stats["mean"]) ** 2

        # 归一化
        normalized_reward = (reward - stats["mean"]) / (np.sqrt(stats["std"]) + 1e-8)
        return normalized_reward * Config.REWARD_SCALE

    def store_experience(self, agent_name, state, action_value, log_prob, reward, next_state, done, value):
        """存储经验到缓冲区"""
        # 归一化奖励
        normalized_reward = self.normalize_reward(agent_name, reward)

        # 确保所有张量都被分离，避免保留计算图
        state = state.detach().clone()
        action_value = action_value.detach().clone()
        log_prob = log_prob.detach().clone()
        next_state = next_state.detach().clone()

        self.agents[agent_name]["buffer"].append({
            "state": state,
            "action": action_value,
            "log_prob": log_prob,
            "reward": normalized_reward,
            "next_state": next_state,
            "done": done,
            "value": value
        })

    def update_agent(self, agent_name):
        """更新智能体策略"""
        agent = self.agents[agent_name]
        buffer = agent["buffer"]

        if len(buffer) < Config.BATCH_SIZE:
            return None  # 没有足够的数据更新

        total_loss = 0

        for _ in range(Config.NUM_EPOCHS):
            # 随机采样一批经验
            batch = random.sample(buffer, Config.BATCH_SIZE)

            # 解包批次数据
            states = torch.stack([exp["state"] for exp in batch])
            actions = torch.stack([exp["action"] for exp in batch])
            old_log_probs = torch.stack([exp["log_prob"] for exp in batch])
            rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32)
            next_states = torch.stack([exp["next_state"] for exp in batch])
            dones = torch.tensor([exp["done"] for exp in batch], dtype=torch.float32)
            old_values = torch.tensor([exp["value"] for exp in batch], dtype=torch.float32)

            # 计算优势估计
            with torch.no_grad():
                _, next_values = agent["policy"](next_states)
                next_values = next_values.squeeze()

                # 计算GAE
                advantages = torch.zeros_like(rewards)
                last_advantage = 0
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_non_terminal = 1.0 - dones[t]
                        next_value = next_values[t]
                    else:
                        next_non_terminal = 1.0 - dones[t]
                        next_value = old_values[t + 1]

                    delta = rewards[t] + Config.GAMMA * next_value * next_non_terminal - old_values[t]
                    advantages[t] = delta + Config.GAMMA * Config.GAE_LAMBDA * next_non_terminal * last_advantage
                    last_advantage = advantages[t]

                # 归一化优势
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns = advantages + old_values

            # 计算新策略的概率
            action_means, values = agent["policy"](states)
            dist = torch.distributions.Normal(action_means, 1.0)
            new_log_probs = dist.log_prob(actions).sum(dim=1)

            # 计算比率
            ratio = (new_log_probs - old_log_probs).exp()

            # 策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - Config.CLIP_EPSILON,
                                1.0 + Config.CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            # 熵奖励
            entropy = dist.entropy().mean()

            # 总损失
            loss = policy_loss + Config.VALUE_COEF * value_loss - Config.ENTROPY_COEF * entropy
            total_loss += loss.item()

            # 优化步骤
            agent["optimizer"].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent["policy"].parameters(), Config.MAX_GRAD_NORM)
            agent["optimizer"].step()

        # 更新学习率
        agent["scheduler"].step(total_loss / Config.NUM_EPOCHS)

        return total_loss / Config.NUM_EPOCHS

    def save_models(self, episode):
        """保存模型"""
        for agent_name, agent_data in self.agents.items():
            torch.save(
                agent_data["policy"].state_dict(),
                os.path.join(Config.SAVE_DIR, f"{agent_name}_ep_{episode}.pth")
            )

    def plot_rewards(self, ai_rewards, episode):
        """绘制并保存奖励曲线"""
        plt.figure(figsize=(12, 8))

        # 绘制原始奖励
        plt.subplot(2, 1, 1)
        plt.plot(ai_rewards)
        plt.title(f"AI Agent Reward (Episode {episode})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        # 绘制滑动平均奖励
        plt.subplot(2, 1, 2)
        window_size = max(1, len(ai_rewards) // 100)
        moving_avg = np.convolve(ai_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(ai_rewards)), moving_avg)
        plt.title(f"AI Agent Moving Average Reward (Window Size={window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(Config.SAVE_DIR, f"ai_rewards_ep_{episode}.png"))
        plt.close()

        # 保存奖励数据
        np.save(os.path.join(Config.SAVE_DIR, f"ai_rewards_ep_{episode}.npy"), np.array(ai_rewards))


# 训练函数
def train_ippo():
    config = Config()

    # 创建环境
    env = FarmEnvWrapper(
        num_farmers=config.NUM_FARMERS,
        total_res=config.TOTAL_RES,
        max_step=config.MAX_STEP
    )

    # 创建训练器
    trainer = IPPOTrainer(env)

    # 记录奖励
    episode_rewards = {agent: [] for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    ai_rewards = []  # 专门记录AI奖励

    # 早停机制
    best_ai_reward = -float('inf')
    no_improve_count = 0
    early_stop_threshold = 10000  # 10000回合无改进则停止

    # 训练循环
    for episode in tqdm(range(config.NUM_EPISODES), desc="Training", unit="episode"):
        obs = env.reset()
        done = {agent: False for agent in env.agents}
        episode_reward = {agent: 0 for agent in env.possible_agents}

        step_count = 0
        while not all(done.values()) and step_count < config.MAX_STEP:
            actions = {}
            log_probs = {}
            values = {}
            states = {}
            action_values = {}  # 存储动作值

            # 每个智能体选择动作
            for agent_name in env.agents:
                agent_obs = obs[agent_name]
                action_dict, log_prob, value, state_tensor, action_value = trainer.act(agent_obs, agent_name)

                actions[agent_name] = action_dict
                log_probs[agent_name] = log_prob
                values[agent_name] = value
                states[agent_name] = state_tensor
                action_values[agent_name] = action_value

            # 执行动作
            next_obs, rewards, dones, infos = env.step(actions)

            # 更新奖励
            for agent_name in env.agents:
                episode_reward[agent_name] += rewards[agent_name]
                total_rewards[agent_name] += rewards[agent_name]

            # 存储经验
            for agent_name in env.agents:
                # 预处理下一个观察
                next_state_tensor = trainer.normalize_obs(next_obs[agent_name], agent_name)

                # 存储经验 (使用detach避免保留计算图)
                trainer.store_experience(
                    agent_name,
                    states[agent_name],
                    action_values[agent_name],
                    log_probs[agent_name],
                    rewards[agent_name],
                    next_state_tensor,
                    dones[agent_name],
                    values[agent_name]
                )

            # 更新观察
            obs = next_obs
            step_count += 1

        # 记录回合奖励
        for agent in env.possible_agents:
            episode_rewards[agent].append(episode_reward.get(agent, 0))

        # 特别记录AI奖励
        ai_reward = episode_reward.get("AI", 0)
        ai_rewards.append(ai_reward)

        # 更新所有智能体
        update_losses = {}
        for agent_name in env.possible_agents:
            loss = trainer.update_agent(agent_name)
            if loss is not None:
                update_losses[agent_name] = loss

        # 早停机制
        if ai_reward > best_ai_reward:
            best_ai_reward = ai_reward
            no_improve_count = 0
            trainer.save_models("best")
        else:
            no_improve_count += 1
            if no_improve_count > early_stop_threshold:
                print(f"\nEarly stopping at episode {episode} due to no improvement")
                break

        # 定期打印统计信息
        if episode % config.LOG_INTERVAL == 0:
            avg_rewards = {}
            for agent in env.possible_agents:
                # 计算最近100回合的平均奖励
                recent_rewards = episode_rewards[agent][-100:] if len(episode_rewards[agent]) >= 100 else \
                episode_rewards[agent]
                avg_rewards[agent] = np.mean(recent_rewards) if recent_rewards else 0

            print(f"\nEpisode {episode}/{config.NUM_EPISODES}")
            for agent in env.possible_agents:
                print(f"  {agent}: Avg Reward = {avg_rewards[agent]:.2f}, Total = {total_rewards[agent]:.2f}")

            if update_losses:
                print("  Update Losses:")
                for agent, loss in update_losses.items():
                    print(f"    {agent}: {loss:.4f}")

            # 打印当前学习率
            lr = trainer.agents["AI"]["optimizer"].param_groups[0]['lr']
            print(f"  Learning Rate: {lr:.2e}")

        # 定期保存模型和绘制奖励曲线
        if episode % config.SAVE_INTERVAL == 0 and episode > 0:
            trainer.save_models(episode)

        if episode % config.PLOT_INTERVAL == 0 and episode > 0:
            trainer.plot_rewards(ai_rewards, episode)

    # 最终保存模型和奖励曲线
    trainer.save_models("final")
    trainer.plot_rewards(ai_rewards, "final")

    print("\nTraining completed!")
    return trainer, ai_rewards


# 测试函数
def test_ippo(trainer):
    config = Config()

    # 创建环境
    env = FarmEnvWrapper(
        num_farmers=config.NUM_FARMERS,
        total_res=config.TOTAL_RES,
        max_step=config.MAX_STEP
    )

    # 加载模型
    for agent_name in env.possible_agents:
        model_path = os.path.join(Config.SAVE_DIR, f"{agent_name}_ep_best.pth")
        if os.path.exists(model_path):
            trainer.agents[agent_name]["policy"].load_state_dict(torch.load(model_path))
            print(f"Loaded best model for {agent_name}")
        else:
            model_path = os.path.join(Config.SAVE_DIR, f"{agent_name}_ep_final.pth")
            if os.path.exists(model_path):
                trainer.agents[agent_name]["policy"].load_state_dict(torch.load(model_path))
                print(f"Loaded final model for {agent_name}")
            else:
                print(f"Warning: Model for {agent_name} not found")

    # 运行测试
    num_episodes = 10
    total_rewards = {agent: 0 for agent in env.possible_agents}

    for episode in range(num_episodes):
        obs = env.reset()
        done = {agent: False for agent in env.agents}
        episode_rewards = {agent: 0 for agent in env.agents}

        step_count = 0
        while not all(done.values()) and step_count < config.MAX_STEP:
            actions = {}

            # 每个智能体选择动作 (测试时使用确定性策略)
            for agent_name in env.agents:
                agent_obs = obs[agent_name]
                action_dict, _, _, _, _ = trainer.act(agent_obs, agent_name, deterministic=True)
                actions[agent_name] = action_dict

            # 执行动作
            next_obs, rewards, dones, infos = env.step(actions)

            # 更新奖励
            for agent_name in env.agents:
                episode_rewards[agent_name] += rewards[agent_name]

            obs = next_obs
            step_count += 1

        # 打印回合结果
        print(f"\nTest Episode {episode + 1} Results:")
        for agent_id, reward in episode_rewards.items():
            print(f"  {agent_id}: Total Reward = {reward:.2f}")
            total_rewards[agent_id] += reward

    # 打印平均奖励
    print("\nAverage Rewards:")
    for agent_id in env.possible_agents:
        avg_reward = total_rewards[agent_id] / num_episodes
        print(f"  {agent_id}: {avg_reward:.2f}")


if __name__ == "__main__":
    # 训练模型
    trainer, ai_rewards = train_ippo()

    # 测试模型
    test_ippo(trainer)

    # 绘制最终奖励曲线
    plt.figure(figsize=(12, 6))

    # 原始奖励
    plt.subplot(1, 2, 1)
    plt.plot(ai_rewards)
    plt.title("AI Agent Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    # 滑动平均奖励
    plt.subplot(1, 2, 2)
    window_size = max(1, len(ai_rewards) // 100)
    moving_avg = np.convolve(ai_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, len(ai_rewards)), moving_avg)
    plt.title(f"Moving Average Reward (Window Size={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, "final_ai_rewards.png"))
    plt.show()