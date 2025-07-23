import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
# from .allocate_utils import *
from allocate_utils import *

# 超参数
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
CLIP_EPSILON = 0.2
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
UPDATE_ITERS = 10

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BitAllocationEnv:
    def __init__(self, layer_sizes, bits, F, alpha, R, origin_bit):
        # 初始化参数
        self.layer_sizes = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.bits = torch.tensor(bits, dtype=torch.float32, device=device)
        self.F = torch.tensor(F, dtype=torch.float32, device=device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        self.R = R
        self.original_size = torch.sum(self.layer_sizes) * 16
        self.origin_bit = torch.tensor(origin_bit, dtype=torch.float32, device=device)
        self.max_budget = self.original_size * R
        self.n_layers = len(F)
        
        # 最优动作记录（初始化为均匀分配方案）
        self.best_allocations = [int(R * origin_bit)] * self.n_layers
        self.best_reward = self._calculate_reward_for_allocation(self.best_allocations)
        
        self.reset()

    def reset(self):
        self.current_layer = 0
        self.allocated_bits = []
        self.used_budget = torch.tensor(0.0, device=device)
        return self._get_state()

    def _get_state(self):
        state = torch.zeros(3 + self.n_layers * 2, device=device)
        state[0] = self.current_layer / self.n_layers
        state[1] = self.used_budget / self.max_budget
        state[2:2+self.n_layers] = self.F / torch.max(self.F)
        state[3+self.n_layers:] = self.layer_sizes / torch.max(self.layer_sizes)
        return state

    def step(self, action):
        bit_value = self.bits[action]
        layer_size = self.layer_sizes[self.current_layer]
        new_usage = self.used_budget + layer_size * bit_value

        # 预算检查
        if abs(new_usage - self.max_budget) < 30000:
            reward = torch.tensor(100, device=device)
            done = True
        else:
            self.allocated_bits.append(bit_value.item())
            self.used_budget = new_usage
            self.current_layer += 1
            reward = torch.tensor(0.1 if self.current_layer < self.n_layers else 0.0, device=device)
            done = self.current_layer >= self.n_layers
            
            if done:
                reward = self._calculate_final_reward()
                # 更新最优分配方案
                current_reward = reward.item()
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.best_allocations = self.allocated_bits.copy()
        
        next_state = self._get_state()
        return next_state, reward, done, {}

    def _calculate_reward_for_allocation(self, allocations):

        allocated_bits_tensor = torch.tensor(allocations, device=device)

        # 精度损失项
        accuracy_terms = [
            F_i * (torch.exp(-self.alpha * (bit_i/self.origin_bit)))
            for F_i, bit_i in zip(self.F, allocated_bits_tensor)
        ]
        accuracy_loss = torch.sum(torch.stack(accuracy_terms))
        
        # 预算惩罚项
        total_usage = torch.sum(self.layer_sizes * allocated_bits_tensor)
        budget_penalty = torch.abs(total_usage - self.max_budget) / self.max_budget * 10
        
        return (-accuracy_loss - budget_penalty).item()
    
    def _calculate_final_reward(self):
        return torch.tensor(
            self._calculate_reward_for_allocation(self.allocated_bits),
            device=device
        )

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
        self.to(device) 
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.to(device)  
    
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
        
        # 更新critic
        critic_loss = nn.MSELoss()(V, target_v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        
        # 更新actor
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


def train(bits=[3, 4, 5], F=None, layer_sizes=None, alpha=1, R=0.25, sameLayerReset=False, origin_bit=16):
    if sameLayerReset:
        layer_group_fisher = same_layer_reset(F)
        return train_layer_group(
            bits=bits, 
            layer_sizes=layer_sizes, 
            layer_groups=layer_group_fisher,
            alpha=alpha, 
            R=R, 
            origin_bit=origin_bit
        )
    else:
        return train_all_layer(
            bits=bits, 
            F=F, 
            layer_sizes=layer_sizes, 
            alpha=alpha, 
            R=R, 
            origin_bit=origin_bit
        )

def train_layer_group(bits=[3, 4, 5], layer_sizes=None, layer_groups=None, alpha=1, R=0.25, 
                     origin_bit=16):
    
    # 分组处理
    layer_sizes_groups = same_layer_reset(layer_sizes)
    allocations = {key: [] for key in layer_groups.keys()}
    best_overall_allocations = None

    for group_name, fishers in layer_groups.items():
        group_F = np.array(fishers)
        group_sizes = layer_sizes_groups[group_name]
        
        # 训练当前组
        env = BitAllocationEnv(group_sizes, bits, group_F, alpha, R, origin_bit)
        agent = PPO(env._get_state().shape[0], len(bits))
        
        # 训练循环
        for episode in range(EPISODES):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, _, done, _ = env.step(action)
                state = next_state
            agent.update()
        
        # 记录当前组的最优分配
        allocations[group_name] = env.best_allocations
        
    # 更新全局最优
    best_overall_allocations = pack_list(allocations)
    print("Best allocations for each group:", allocations)
    print("Best overall allocations:", best_overall_allocations)

    return best_overall_allocations

def train_all_layer(bits=[3, 4, 5], F=None, layer_sizes=None, alpha=1, R=0.25, origin_bit=16):
    env = BitAllocationEnv(layer_sizes, bits, F, alpha, R, origin_bit)
    agent = PPO(env._get_state().shape[0], len(bits))

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
        agent.update()
    
    print("Best allocations found:", env.best_allocations)

    return env.best_allocations

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
    bits = [3, 4, 5]  # 可选位宽
    F = load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/fisher_data.json')
    layer_sizes = load_json('/root/autodl-tmp/methods/mix_quantize/model_info/llama2-7b/LayersParams.json')
    N = len(F)  # 层数
    sameLayerReset = False  # 是否使用同名层分配模式
    R = 0.28  # 压缩率
    alpha = 20  # 目标函数中的衰减系数
    _ = train(bits,F,layer_sizes,alpha,R,sameLayerReset)