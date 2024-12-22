import sys
sys.path.append('methods/mix_quantize')
from mixq.allocate import *

bits = [2,4,8]
target_bits = 4.3
layers_num = 40

allocation_strategy = Greedy_allocation_list(bits, target_bits, layers_num)

print(allocation_strategy)

allocate_is_legal(allocation_strategy, bits, target_bits, layers_num)

