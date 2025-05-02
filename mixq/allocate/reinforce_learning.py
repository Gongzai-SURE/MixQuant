import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 超参数
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
CLIP_EPSILON = 0.2
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
UPDATE_ITERS = 10

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BitAllocationEnv:
    def __init__(self, layer_sizes, bits, F, alpha, R):
        # 将numpy数组转换为GPU张量
        self.layer_sizes = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.bits = torch.tensor(bits, dtype=torch.float32, device=device)
        self.F = torch.tensor(F, dtype=torch.float32, device=device)
        self.alpha = alpha
        self.R = R
        self.original_size = torch.sum(self.layer_sizes) * 32  # GPU计算
        self.max_budget = self.original_size * R
        self.n_layers = len(layer_sizes)
        self.reset()

    def reset(self):
        self.current_layer = 0
        self.allocated_bits = []
        self.used_budget = torch.tensor(0.0, device=device)  # GPU张量
        return self._get_state()

    def _get_state(self):
        
        state = torch.zeros(3 + self.n_layers * 2, device=device)
        
        # 当前进度
        state[0] = self.current_layer / self.n_layers
        
        # 预算使用率
        state[1] = self.used_budget / self.max_budget

        # Fisher信息
        state[2:2+self.n_layers] = self.F / torch.max(self.F)

        # 层大小归一化
        state[3+self.n_layers:] = self.layer_sizes / torch.max(self.layer_sizes)
        
        return state

    def step(self, action):
        """全部使用GPU张量计算"""
        bit_value = self.bits[action]
        layer_size = self.layer_sizes[self.current_layer]
        
        new_usage = self.used_budget + layer_size * bit_value
        
        if new_usage > self.max_budget:
            reward = torch.tensor(-1000.0, device=device)
            done = True
        else:
            self.allocated_bits.append(bit_value.item())  
            self.used_budget = new_usage
            self.current_layer += 1
            reward = torch.tensor(0.1 if self.current_layer < self.n_layers else 0.0, device=device)
            done = self.current_layer >= self.n_layers
            
            if done:
                reward = self._calculate_final_reward()
        
        next_state = self._get_state()
        return next_state, reward, done, {}

    def _calculate_final_reward(self):
        """GPU加速的奖励计算"""
        allocated_bits_tensor = torch.tensor(self.allocated_bits, device=device)
        loss = torch.sum(self.F * torch.exp(-self.alpha * allocated_bits_tensor))
        return -loss

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.to(device)  # 确保网络在GPU上
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.to(device)  # 确保网络在GPU上
    
    def forward(self, state):
        return self.net(state)

class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = deque(maxlen=10000)
        
    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        # 批量数据直接从GPU获取
        batch = random.sample(self.buffer, BATCH_SIZE)
        states = torch.stack([t[0] for t in batch]).to(device)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=device)
        next_states = torch.stack([t[3] for t in batch]).to(device)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=device)
        
        # 价值计算
        with torch.no_grad():
            target_v = rewards + GAMMA * (1 - dones) * self.critic(next_states).squeeze()
        
        # 优势计算
        V = self.critic(states).squeeze()
        advantage = (target_v - V).detach()
        
        critic_loss = nn.MSELoss()(V, target_v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        
        old_probs = self.actor(states).detach()
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        for _ in range(UPDATE_ITERS):
            new_probs = self.actor(states)
            new_dist = torch.distributions.Categorical(new_probs)
            log_probs = new_dist.log_prob(actions)
            
            ratio = torch.exp(log_probs - torch.log(old_probs))
            clipped_ratio = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON)
            
            actor_loss = -torch.min(ratio*advantage, clipped_ratio*advantage).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()

 
def train(bits = [2, 3, 4, 8], F = None, layer_sizes = None, alpha = 1, R = 0.25):
    # 初始化环境和Agent
    env = BitAllocationEnv(layer_sizes, bits, F, alpha, R)
    state_dim = env._get_state().shape[0]
    action_dim = len(bits)
    agent = PPO(state_dim, action_dim)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.buffer.append((
                state.cpu().detach(),  
                action,
                reward.cpu().item(),
                next_state.cpu().detach(),
                done
            ))
            
            state = next_state
            total_reward += reward.item()
        
        agent.update()
        
        # 打印进度
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{EPISODES} | Total Reward: {total_reward:.2f}")

    # 测试阶段
    with torch.no_grad():
        test_env = BitAllocationEnv(layer_sizes, bits, F, alpha, R)
        state = test_env.reset()
        allocations = []
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = test_env.step(action)
            allocations.append(bits[action])
            if done:
                break
            state = next_state

    print("\nFinal Bit Allocation:")
    print(f"Layers: {len(layer_sizes)}")
    print(f"Allocated Bits: {allocations}")
    print(f"Total Usage: {sum([l*b for l,b in zip(layer_sizes,allocations)])} bits")
    print(f"Original Usage: {test_env.original_size.item()} bits")
    print(f"Compression Rate: {np.sum(layer_sizes * (np.array(allocations) / 16))/np.sum(layer_sizes):.2%}")
    return allocations


if __name__ == "__main__":
    # 环境参数配置
    import random
    import json

    def load_json(file):
        # 加载json文件数据
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data = []
        # 遍历字典中的值并平铺
        for id, block in enumerate(json_data):
            for key, value in json_data[block].items():
                data.append(value)
        return np.array(data)


    # 参数设置
    bits = [2, 3, 4, 8]  # 可选位宽
    F = load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/fisher_data.json')
    layer_sizes = load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/LayersParams.json')
    N = len(F)  # 层数
    R = 0.25  # 压缩率
    alpha = 1.5  # 目标函数中的衰减系数
    _ = train(bits,F,layer_sizes,alpha,R)