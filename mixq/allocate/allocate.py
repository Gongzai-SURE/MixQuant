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

import numpy as np
from loguru import logger
from .allocate_utils import *

class Allocation:
    def __init__(self, bits=None, 
                 layer_sizes=None, fisher = None, ppl = None, 
                 target_bit=None, alpha=0.1, top_r = 0.1,
                 strategy=None, allocation=None, sameLayerReset=False):
        self.bits = bits                                    # bit width list
        self.original_bit = 16                              # original bit width   
        self.layer_sizes = modify_size(layer_sizes)    # layer size list
        self.layer_num = len(self.layer_sizes)              # number of layers
        self.fisher = self.set_Fisher(fisher)               # fisher information list
        self.ppl = self.set_ppl(ppl)                        # perplexity list                                         
        self.target_bit = target_bit                        # target bits
        self.R = self.target_bit / self.original_bit        # compression rate
        self.alpha = alpha                                  # hyperparameter, used to control the trade-off between compression and accuracy
        self.top_r = top_r                                  # top r percentage for ppl allocation
        self.strategy = strategy                            # allocation strategy
        self.allocation_result = allocation                 # store allocation results
        self.sameLayerReset = sameLayerReset                # whether to reset bits between same name layer

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

    def objective_function(self, bit_allocation):
        if self.fisher is None:
            raise ValueError("Fisher information is not set. Please provide Fisher information for the objective function.")
        # 计算精度损失
        accuracy_loss = sum(F_i * (np.exp(-self.alpha * (bit_i/self.original_bit))- np.exp(-self.alpha)) / (np.exp(-self.alpha * (1.5/self.original_bit))- np.exp(-self.alpha)) \
                           for F_i, bit_i in zip(self.fisher, bit_allocation))

        return accuracy_loss

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
        from .greeady import GreedyBitAllocation
        # 初始化贪心算法
        greedy = GreedyBitAllocation(self.bits, self.layer_sizes, self.fisher,self.original_bit, self.R, self.alpha, self.sameLayerReset)
        greedy.allocate()
        return greedy.get_allocation_result()
        
    
    def _annealing_allocation(self):
        from scipy.optimize import dual_annealing
        P_total = sum(self.layer_sizes)

        # 控制位宽到可选位宽上
        def map_to_discrete(continuous_value):
            return min(self.bits, key=lambda x: abs(x - continuous_value))

        # 目标函数
        def objective_function_all(bit_allocation):
            bit_allocation = np.array([map_to_discrete(bit) for bit in bit_allocation])
            objective = self.objective_function(bit_allocation)
            
            compressed_size = np.sum(self.layer_sizes * (bit_allocation / self.original_bit))
            constraint = compressed_size - P_total * self.R
            
            penalty = 1e6 * max(0, abs(constraint))
            return objective + penalty

        bounds = [(min(self.bits), max(self.bits)) for _ in range(self.layer_num)]

        result = dual_annealing(objective_function_all, bounds, maxiter=1000, seed=42)

        bit_allocation = np.array([map_to_discrete(bit) for bit in result.x])
        
        return bit_allocation

    def _genetic_allocation(self):
        from .genetic import GeneticAlgorithm
        # 初始化
        genetic = GeneticAlgorithm(self.bits, self.layer_sizes, self.fisher, self.original_bit, self.R, self.alpha)
        result,log= genetic.run()
        logger.info(f"Genetic Algorithm Log: {log}")
        return genetic.get_best_individual()
    
    def _bayesian_allocation(self):
        from .bayesian import BayesianOptimization
        # 初始化
        bayesian = BayesianOptimization(self.bits, self.layer_sizes, self.fisher, self.original_bit, self.R, self.alpha)
        bayesian.run()
        return bayesian.get_best_trial()

    def _reinforcement_learning_allocation(self):
        from .reinforce_learning import train
        return train(self.bits, self.fisher, self.layer_sizes, self.alpha, self.R, self.sameLayerReset)

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
                
                original_allocation = self.allocation_result.copy()

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
                                           
