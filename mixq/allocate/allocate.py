'''
Basic quantitative strategy allocation function
Receive the quantization bit, target bit and the number of model layers, and return the bit width allocation strategy 
(in dictionary form, the keyword is the bit width value, and the value is the number to be quantized)

parameters:
bits: int, the number of bits to be quantized in each layer eg.[2,4,8]
target_bits: float, the target bit of total quantization eg.4.1
layers_num: int, the number of model layers
return: dict, the bit width allocation strategy eg. {2: 10, 4: 15, 8: 15}

constraints and requests:
1. The average number of bits allocated to each layer is less than the target bit
\frac{1}{n} {\textstyle \sum_{1}^{n}} b_{l_i}\le b_{tar} 
2. The quantization bit width assigned to each layer is as large as possible
3. The total number of bits assigned to each layer is an integer,and bit width is in bits
'''
import pulp
import json
import math
import optuna
from scipy.optimize import dual_annealing
import numpy as np
from loguru import logger
pulp.LpSolverDefault.msg = False


class Allocation:
    def __init__(self, bits=None, 
                 layer_sizes=None, fisher = None, ppl = None, 
                 target_bit=None, alpha=0.1, top_r = 0.1,
                 strategy=None, allocation=None):
        self.bits = bits                                    # bit width list
        self.original_bit = 16                              # original bit width   
        self.layer_sizes = self.modify_size(layer_sizes)    # layer size list
        self.layer_num = len(self.layer_sizes)              # number of layers
        self.fisher = self.set_Fisher(fisher)               # fisher information list
        self.ppl = self.set_ppl(ppl)                        # perplexity list                                         
        self.target_bit = target_bit                        # target bits
        self.R = self.target_bit / self.original_bit        # compression rate
        self.alpha = alpha                                  # hyperparameter, used to control the trade-off between compression and accuracy
        self.top_r = top_r                                  # top r percentage for ppl allocation
        self.strategy = strategy                            # allocation strategy
        self.allocation_result = allocation                 # store allocation results

    def set_bits(self, bits):
        self.bits = bits

    def set_layer_sizes(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def set_R(self, R):
        self.R = R
    
    def set_Fisher(self, fisher):
        if fisher is not None:
            if isinstance(fisher,dict):
                fisher = list(fisher.values())
                all_values = []
                for index, block in enumerate(fisher):
                    for key, value in block.items():
                        for k, v in value.items():
                            all_values.append(v)
                self.fisher = all_values
            else:
                self.fisher = fisher
        else:
            self.fisher = None

    def set_ppl(self, ppl):
        if ppl is not None:
            if isinstance(ppl,dict):
                ppl = list(ppl.values())
                all_values = []
                for index, block in enumerate(ppl):
                    for key, value in block.items():
                        for k, v in value.items():
                            all_values.append(v)
                self.ppl = all_values
            else:
                self.ppl = ppl
        else:
            self.ppl = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_top_r(self, top_r):
        self.top_r = top_r

    def set_strategy(self, strategy):
        self.strategy = strategy

    def get_allocation_result(self):
        return self.allocation_result
    
    def check_allocation_result(self, allocation_result = None):
        actutal_compression = sum(layer_size * (bit / self.original_bit) 
                                  for layer_size, bit in zip(self.layer_sizes, allocation_result)
                                  ) / sum(self.layer_sizes)
        # 计算实际压缩率，过压缩返回1，欠压缩返回2，符合压缩率返回0
        if actutal_compression == self.R:
            return 0
        elif actutal_compression < self.R:
            return 1
        elif actutal_compression > self.R:
            return 2       
        

    def modify_size(self,layer_sizes):
        # 将fisher格式变为转为list
        if isinstance(layer_sizes,dict):
            layer_sizes = list(layer_sizes.values())
            all_values = []
            for index, block in enumerate(layer_sizes):
                for key, value in block.items():
                    all_values.append(value)
            return all_values
        else:
            return layer_sizes
        
    def allocate(self):
        if self.fisher is not None:
            if self.strategy == "greedy":
                self.allocation_result = self._greedy_allocation()
            elif self.strategy == "genetic":
                self.allocation_result = self._genetic_allocation()
            elif self.strategy == "rl":
                self.allocation_result = self._reinforcement_learning_allocation()
            elif self.strategy == "annealing":
                self.allocation_result = self._annealing_allocation()
            elif self.strategy == "bayesian":
                self.allocation_result = self._bayesian_allocation()
            elif self.strategy == "random":
                self.allocation_result = self._random_allocation()
            else:
                raise ValueError("Unknown allocation strategy. Supported strategies: 'greedy', 'genetic', 'reinforcement_learning', 'annealing', 'bayesian' and 'random'")
            

        if self.ppl is not None:
            self._ppl_allocation()

        return self.allocation_result
        
        
        
    def _greedy_allocation(self):
        # 初始化
        bit_allocation = [min(self.bits)] * self.layer_num
        P_total = sum(self.layer_sizes)
        P_current = sum(p_i * (bit_i / self.original_bit) for p_i, bit_i in zip(self.layer_sizes, bit_allocation))
        
        if self.check_allocation_result(bit_allocation) == 1:
            while True:
                # 计算当前目标函数值
                current_objective = sum(F_i * math.exp(-self.alpha * (self.original_bit / bit_i)) for F_i, bit_i in zip(self.fisher, bit_allocation))
                
                # 寻找最优的层进行位宽增加
                best_delta = -float('inf')
                best_layer = -1
                best_new_bit = -1
                
                for i in range(self.layer_num):
                    current_bit = bit_allocation[i]
                    if current_bit < max(self.bits):
                        # 找到下一个更高的位宽
                        next_bit = min([b for b in self.bits if b > current_bit])
                        # 计算目标函数的变化量
                        delta = self.fisher[i] * (math.exp(-self.alpha * (self.original_bit / next_bit)) - math.exp(-self.alpha * (self.original_bit / current_bit)))
                        if delta > best_delta:
                            best_delta = delta
                            best_layer = i
                            best_new_bit = next_bit
                
                # 如果没有可以增加的层，终止
                if best_layer == -1:
                    break
                
                # 尝试增加位宽
                new_P_current = P_current + self.layer_sizes[best_layer] * ((best_new_bit - bit_allocation[best_layer]) / self.original_bit)
                
                # 检查是否满足压缩率约束
                if new_P_current <= P_total * self.R:
                    # 更新位宽和总参数规模
                    bit_allocation[best_layer] = best_new_bit
                    P_current = new_P_current
                else:
                    # 无法增加，终止
                    break
        elif self.check_allocation_result(bit_allocation) == 2:
            # 找到精度损失最小的层进行位宽减少
            while True:
                # 计算当前目标函数值
                current_objective = sum(F_i * math.exp(-self.alpha * (self.original_bit / bit_i)) for F_i, bit_i in zip(self.fisher, bit_allocation))
                # 寻找最优的层进行位宽减少
                best_delta = float('inf')
                best_layer = -1
                best_new_bit = -1
                for i in range(self.layer_num):
                    current_bit = bit_allocation[i]
                    if current_bit > min(self.bits):
                        # 找到下一个更低的位宽
                        next_bit = max([b for b in self.bits if b < current_bit])
                        # 计算目标函数的变化量
                        delta = self.fisher[i] * (math.exp(-self.alpha * (self.original_bit / next_bit)) - math.exp(-self.alpha * (self.original_bit / current_bit)))
                        if delta < best_delta:
                            best_delta = delta
                            best_layer = i
                            best_new_bit = next_bit
                # 如果没有可以减少的层，终止
                if best_layer == -1:
                    break
                # 尝试减少位宽 
                new_P_current = P_current - self.layer_sizes[best_layer] * ((bit_allocation[best_layer] - best_new_bit) / self.original_bit)
                # 检查是否满足压缩率约束
                if new_P_current >= P_total * self.R:
                    # 更新位宽和总参数规模
                    bit_allocation[best_layer] = best_new_bit
                    P_current = new_P_current
                else:
                    # 无法减少，终止
                    break
                
        return bit_allocation
    
    def _annealing_allocation(self):
        P_total = sum(self.layer_sizes)

        # 控制位宽到可选位宽上
        def map_to_discrete(continuous_value):
            return min(self.bits, key=lambda x: abs(x - continuous_value))

        # 目标函数
        def objective_function(bit_allocation):
            bit_allocation = np.array([map_to_discrete(bit) for bit in bit_allocation])
            objective = np.sum(self.fisher * np.exp(-self.alpha * (self.original_bit / bit_allocation)))
            
            compressed_size = np.sum(self.layer_sizes * (bit_allocation / self.original_bit))
            constraint = compressed_size - P_total * self.R
            
            penalty = 1e6 * max(0, abs(constraint))
            return objective + penalty

        bounds = [(min(self.bits), max(self.bits)) for _ in range(self.layer_num)]

        result = dual_annealing(objective_function, bounds, maxiter=1000, seed=42)

        bit_allocation = np.array([map_to_discrete(bit) for bit in result.x])
        
        return bit_allocation

    def _genetic_allocation(self):
        from .genetic import GeneticAlgorithm
        # 初始化
        genetic = GeneticAlgorithm(self.bits, self.layer_sizes, self.fisher, self.original_bit, self.R, self.alpha)
        genetic.run()
        return genetic.get_best_individual()
    
    def _bayesian_allocation(self):
        from .bayesian import BayesianOptimization
        # 初始化
        bayesian = BayesianOptimization(self.bits, self.layer_sizes, self.fisher, self.original_bit, self.R, self.alpha)
        bayesian.run()
        return bayesian.get_best_trial()

    def _reinforcement_learning_allocation(self):
        from .reinforce_learning import train
        return train(self.bits, self.fisher, self.layer_sizes, self.R, self.alpha)

    def _ppl_allocation(self):
        if self.ppl is None:
            raise ValueError("Please provide perplexity values for PPL allocation.")
        allocation_res = [int(self.target_bit)] * self.layer_num
        logger.info(f"top_r: {self.top_r}, layer_num: {self.layer_num}, target_bit: {self.target_bit}")

        top_k = int(self.top_r * self.layer_num)
        if top_k == 0:
            self.allocation_result = allocation_res
            return 0
        data = np.array(self.ppl)
        top_indices = np.argsort(data)[-top_k:][::-1]  
        bottom_indices = np.argsort(data)[:top_k]

        # 调整困惑度越大，说明该层量化后对结果的影响越大，因此分配的位宽越大
        for i in top_indices:
            allocation_res[i] = max(self.bits)
        for i in bottom_indices:
            allocation_res[i] = min(self.bits)
        
        self.allocation_result = allocation_res

    def finetuning_allocation(self):
        if self.allocation_result is None:
            raise ValueError("Please run the allocation method first.")
        else:
            check = self.check_allocation_result(self.allocation_result)
            if check:
                logger.info("The allocation result does not meet the compression rate constraint. Starting fine-tuning...")
                # 进行微调,采用贪心算法将位宽分配结果调整至符合压缩率约束
                original_allocation = self.allocation_result.copy()

                # 按照贪心算法进行微调
                # self.allocation_result = self._greedy_allocation(original_allocation)

                # 按照层顺序进行微调
                if check == 1:
                    # 将最小位宽调高,使其符合压缩率约束
                    P_current = sum(p_i * (bit_i / self.original_bit) for p_i, bit_i in zip(self.layer_sizes, self.allocation_result))
                    while P_current < sum(self.layer_sizes) * self.R:
                        # 找到最小的位宽候选层
                        min_bit = min(self.allocation_result)
                        target_layer = []
                        for i in range(len(self.allocation_result)):
                            if self.allocation_result[i] == min_bit:
                                target_layer.append(i)
                        # 调高最小位宽候选层的位宽
                        for i in target_layer:
                            if self.allocation_result[i] < max(self.bits):
                                self.allocation_result[i] = min([b for b in self.bits if b > self.allocation_result[i]])
                                P_current += self.layer_sizes[i] * ((self.allocation_result[i] - min_bit) / self.original_bit)
                                if P_current >= sum(self.layer_sizes) * self.R:
                                    print("Fine-tuning completed. New allocation result:", self.allocation_result)
                                    break
                elif check == 2:
                    P_current = sum(p_i * (bit_i / self.original_bit) for p_i, bit_i in zip(self.layer_sizes, self.allocation_result))
                    while P_current > sum(self.layer_sizes) * self.R:
                        # 找到最大的位宽候选层
                        max_bit = max(self.allocation_result)
                        target_layer = []
                        for i in range(len(self.allocation_result)):
                            if self.allocation_result[i] == max_bit:
                                target_layer.append(i)
                        # 调低最大位宽候选层的位宽
                        for i in target_layer:
                            if self.allocation_result[i] > min(self.bits):
                                self.allocation_result[i] = max([b for b in self.bits if b < self.allocation_result[i]])
                                P_current -= self.layer_sizes[i] * ((max_bit - self.allocation_result[i]) / self.original_bit)
                                if P_current <= sum(self.layer_sizes) * self.R:
                                    break
                            
                



    # def _random_allocation(self):
    #     randoms = RandomAllocation(self.bits, self.layer_sizes, self.fisher, self.original_bit, self.R, self.alpha)
    #     genetic.run()
    #     return genetic.get_best_individual()





""" def Greedy_allocation_list(bits, target_bits, layers_num):
    allocation_strategy = {}
    bits_num = len(bits)
    bits.sort(reverse=True)
    total_bits = target_bits * layers_num
    # Formulate a linear pr  ogramming problem
    prob = pulp.LpProblem("Maximize_Z", pulp.LpMaximize)
    # Define variables
    vars = []
    for i in range(bits_num):
        vars.append(pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger))
    # Objective function
    prob += sum([bits[i] * vars[i] for i in range(bits_num)])
    # Constraints
    prob += sum([vars[i] for i in range(bits_num)]) == layers_num
    prob += sum([vars[i]*bits[i] for i in range(bits_num)]) <= total_bits
    # Solve the problem
    prob.solve()
    # Output results
    for v in prob.variables():
        allocation_strategy[bits[int(v.name[1:])]] = int(v.varValue)
    allocation_strategy = dict(sorted(allocation_strategy.items(), key=lambda x: x[0]))
    return allocation_strategy

def allocation_is_legal(allocation_strategy, bits, target_bits, layers_num):
    bits_num = len(bits)
    total_bits = target_bits * layers_num
    # The average number of bits allocated to each layer is less than the target bit
    for i in range(bits_num):
        if bits[i] not in allocation_strategy:
            raise ValueError('The bit width is not in the allocation strategy')
        total_bits -= allocation_strategy[bits[i]] * bits[i]
    if  total_bits < 0:
        raise ValueError('The total number of bits assigned to each layer is greater than the target bit')
    
def load_json(file):
    # 加载json文件数据
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sort_FI_layer(data):
    from collections import defaultdict
    values_dict = defaultdict(list)
    for block in data:
        for key, value in data[block].items():
            values_dict[key].append(value)

    rank_dict = {}
    for key, values in values_dict.items():
        sorted_values = sorted(values, reverse=True)
        ranks = {v: i + 1 for i, v in enumerate(sorted_values)}
        rank_dict[key] = [ranks[v] for v in values]
    return rank_dict

def sort_FI_in_all(data):
    from collections import defaultdict

    all_values = []
    for index, block in enumerate(data):
        for key, value in data[index].items():
            all_values.append((f"{index}_{key}", value)) 

    sorted_values = sorted(all_values, key=lambda x: x[1], reverse=False)
    rank_dict = {}
    for rank, (combined_key, _) in enumerate(sorted_values, start=1):
        index, key = combined_key.split('_', 1)  
        rank_dict[(int(index), key)] = rank  

    result = defaultdict(dict)
    for (index, key), rank in rank_dict.items():
        result[index][key] = rank

    return dict(result)

def judge_bits(rank,strategy):
    total_nums = 0
    for bit,nums in strategy.items():
        total_nums += nums
        if rank < total_nums or rank == total_nums:
            return bit

def get_bits_list(layers_score, strategy):
    sorted_score = sort_FI_in_all(layers_score)
    res = []
    for index,layer_score_rank in sorted_score.items():
        layers_bit = {}
        for layer_name,score_rank in layer_score_rank.items():
            bit = judge_bits(score_rank,strategy)
            layers_bit[layer_name] = bit
        res.append(layers_bit)
    return res
 """