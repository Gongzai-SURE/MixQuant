import torch
import inspect
import logging
import functools
import torch.nn as nn
import transformers
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict


class AwqLayerQuantizer:
    def __init__(
            self,
            layer,
            w_bit: int,
            group_size = 128,
            zero_point = True,
            apply_clip = True,
            quantizer: Optional[object] = None
    )-> None:
            self.layer = layer
            self.bits = w_bit
            self.group_size = group_size
            self.zero_point = zero_point
            self.apply_clip = apply_clip    
            self.dev = self.layer.weight.device
            W = layer.weight.data.clone()

            if isinstance(self.layer, nn.Conv2d): 
                W = W.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                W = W.t()
            
            self.rows = W.shape[0]
            self.columns = W.shape[1]  
            self.nsamples = 0 
            
            self.quantizer = None

