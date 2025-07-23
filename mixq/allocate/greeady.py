import math
import numpy as np
from loguru import logger
from .allocate_utils import *

class GreedyBitAllocation:
    def __init__(self, bits, p, F, B, R, alpha, sameLayerReset=False):
        """
        Initialize the greedy bit allocation algorithm
        
        Parameters:
        bits (list): Available bit-width options (sorted)
        p (list): Layer sizes (parameter counts)
        F (list): Fisher information values for each layer
        B (int): Original bit-width (before quantization)
        R (float): Target compression ratio
        alpha (float): Scaling factor for the objective function
        """
        self.bits = bits
        self.layer_sizes = p
        self.fisher = F
        self.original_bit = B
        self.R = R
        self.alpha = alpha
        self.layer_num = len(p)
        self.P_total = np.sum(p)
        self.sameLayerReset = sameLayerReset
        self.allocate_result = None

        if self.sameLayerReset:
            self.layer_fisher = same_layer_reset(self.fisher)
            self.layer_num = len(p)// len(self.layer_fisher)

    def get_allocation_result(self):
        if self.allocate_result is None:
            raise ValueError("Allocation has not been performed yet.")
        return self.allocate_result

        
    def objective_function(self, bit_allocation):
        """Calculate the objective function value for a given bit allocation"""
        # 判断bit_allocation与fisher长度是否一致
        if len(bit_allocation) != len(self.fisher):
            raise ValueError("Bit allocation length must match Fisher information length.")
        return sum(F_i * (np.exp(-self.alpha * (bit_i/self.original_bit))- np.exp(-self.alpha)) / (math.exp(-self.alpha) - np.exp(-self.alpha * (1.5/self.original_bit))) \
                           for F_i, bit_i in zip(self.fisher, bit_allocation))
    
    def _compute_delta(self, bit, new_bit):
        """Compute the normalized delta for a bit change"""
        return (np.exp(-self.alpha * (new_bit/self.original_bit)) - np.exp(-self.alpha * (bit/self.original_bit)))

    def _compute_normalized_delta(self, F_i, bit, new_bit):
        """Compute the Fisher-weighted normalized delta"""
        delta = self._compute_delta(bit, new_bit)
        return F_i * delta / (math.exp(-self.alpha) - np.exp(-self.alpha * (1.5/self.original_bit)))


    def _check_allocation_result(self, allocation_result):
        """Check if the allocation meets the compression target"""
        actutal_compression = sum(
            layer_size * (bit / self.original_bit)
            for layer_size, bit in zip(self.layer_sizes, allocation_result)
        ) / self.P_total
        
        if actutal_compression == self.R:
            return 0
        elif actutal_compression < self.R:
            return 1  # Under-compressed
        else:
            return 2  # Over-compressed
    
    def allocate(self):
        if self.sameLayerReset:
            layer_result = {key: [] for key in self.layer_fisher.keys()}
            for key, values in self.layer_fisher.items():
                self.fisher = values
                layer_result[key] = self.samelayer_allocation()
                logger.info(f"Layer {key} allocation: {layer_result[key]}")
            self.allocate_result = pack_list(layer_result)
        else:
            self.total_allocation()

    def total_allocation(self):
        """Perform the greedy bit allocation"""
        # Initialize with minimum bit-width
        bit_allocation = [min(self.bits)] * self.layer_num
        P_current = sum(
            p_i * (bit_i / self.original_bit) 
            for p_i, bit_i in zip(self.layer_sizes, bit_allocation)
        )

        allocation_status = self._check_allocation_result(bit_allocation)        
        if allocation_status == 1:  # Under-compressed
            while True:
                # Find the best layer to increase bit-width
                best_delta = -float('inf')
                best_layer = -1
                best_new_bit = -1
                
                for i in range(self.layer_num):
                    current_bit = bit_allocation[i]
                    if current_bit < max(self.bits):
                        # Find next higher bit-width
                        next_bit = min([b for b in self.bits if b > current_bit])
                        # Calculate delta in objective function
                        delta = self._compute_normalized_delta(self.fisher[i], current_bit, next_bit)
                        if delta > best_delta:
                            best_delta = delta
                            best_layer = i
                            best_new_bit = next_bit
                
                # If no layer can be increased, stop
                if best_layer == -1:
                    break
                
                # Try to increase bit-width
                new_P_current = P_current + self.layer_sizes[best_layer] * (
                    (best_new_bit - bit_allocation[best_layer]) / self.original_bit
                )
                
                # Check compression constraint
                if new_P_current <= self.P_total * self.R:
                    bit_allocation[best_layer] = best_new_bit
                    P_current = new_P_current
                else:
                    break
                    
        elif allocation_status == 2:  # Over-compressed
            while True:
                # Find the best layer to decrease bit-width
                best_delta = float('inf')
                best_layer = -1
                best_new_bit = -1
                
                for i in range(self.layer_num):
                    current_bit = bit_allocation[i]
                    if current_bit > min(self.bits):
                        # Find next lower bit-width
                        next_bit = max([b for b in self.bits if b < current_bit])
                        # Calculate delta in objective function
                        delta = self._compute_normalized_delta(self.fisher[i], current_bit, next_bit)
                        if delta < best_delta:
                            best_delta = delta
                            best_layer = i
                            best_new_bit = next_bit
                
                # If no layer can be decreased, stop
                if best_layer == -1:
                    break
                
                # Try to decrease bit-width
                new_P_current = P_current - self.layer_sizes[best_layer] * (
                    (bit_allocation[best_layer] - best_new_bit) / self.original_bit
                )
                
                # Check compression constraint
                if new_P_current >= self.P_total * self.R:
                    bit_allocation[best_layer] = best_new_bit
                    P_current = new_P_current
                else:
                    break
                    
        self.allocate_result = bit_allocation
        
    def samelayer_allocation(self, max_iter=100):
        """
        Perform optimized same-layer bit allocation with precomputed deltas
        
        Parameters:
        max_iter (int): Maximum number of iterations to prevent infinite loops
        
        Returns:
        list: Optimized bit allocation
        """
        # Initialize all layers to closest available bit to target average
        closest_bit = min(self.bits, key=lambda x: abs(x - self.original_bit*self.R))
        bit_allocation = np.array([closest_bit] * self.layer_num)
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            # Precompute current objective
            current_obj = self.objective_function(bit_allocation)
            best_new_obj = current_obj
            best_swap = None
            
            # Find all possible beneficial swaps
            for i in range(self.layer_num):
                current_i_bit = bit_allocation[i]
                
                # Try increase i-th layer's bit
                higher_bits = [b for b in self.bits if b > current_i_bit]
                if not higher_bits:
                    continue
                    
                next_i_bit = min(higher_bits)
                
                for j in range(self.layer_num):
                    if i == j:
                        continue
                        
                    current_j_bit = bit_allocation[j]
                    
                    # Try decrease j-th layer's bit
                    lower_bits = [b for b in self.bits if b < current_j_bit]
                    if not lower_bits:
                        continue
                        
                    next_j_bit = max(lower_bits)
                    
                    # Check bit-width balance
                    if (next_i_bit - current_i_bit) != (current_j_bit - next_j_bit):
                        continue
                    
                    # Create temp allocation
                    temp_allocation = bit_allocation.copy()
                    temp_allocation[i] = next_i_bit
                    temp_allocation[j] = next_j_bit
                    
                    # Calculate new objective
                    new_obj = self.objective_function(temp_allocation)
                    
                    # Check for improvement
                    if new_obj > best_new_obj:
                        best_new_obj = new_obj
                        best_swap = (i, j, next_i_bit, next_j_bit)
            
            # Execute the best swap if found
            if best_swap is not None:
                i, j, next_i_bit, next_j_bit = best_swap
                bit_allocation[i] = next_i_bit
                bit_allocation[j] = next_j_bit
                improved = True
        
        return bit_allocation