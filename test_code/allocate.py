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
pulp.LpSolverDefault.msg = False

def Greedy_allocation_list(bits, target_bits, layers_num):
    allocation_strategy = {}
    bits_num = len(bits)
    bits.sort(reverse=True)
    total_bits = target_bits * layers_num
    # Formulate a linear programming problem
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
        for key, value in block.items():
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
        for key, value in block.items():
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
