import math
import time
from loguru import logger
import torch
import torch.nn as nn
import transformers

from .quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class Mixq_Linear:
    def __init__(self, layer, wbits):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d): 
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1]  
        self.nsamples = 0 
        self.bits = wbits
        self.quantizer = None
        self.quant_loss_threshold = None
        
    
    #定义量化损失阈值计算函数,根据目标层的参数量(正比)，层最大值和最小值，所处块的id（id号越大说明层越深），计算量化损失阈值
    def get_layer_loss_threshold(self,layer,pentalty_index,bit_differ):
        a = pentalty_index[0]
        b = pentalty_index[1]
        # 目标比特惩罚项
        c = pentalty_index[2]  #与当前层对齐，越往后的层对最终的量化精度的影响越小
        layer_Params_mun = layer.weight.data.numel()
        layer_max,layer_min = layer.weight.data.max(),layer.weight.data.min()
        quant_loss_threshold = a * layer_Params_mun * b* (layer_max - layer_min) * 1.25 + c * bit_differ
        self.quant_loss_threshold = quant_loss_threshold


    # 对层按照group的粒度量化
    def fasterquant(
        self,quantbit, blocksize=128, groupsize=-1
    ):
        self.quantizer.set_bits(quantbit)
        # check 当前层的量化策略
        if self.quantizer.quantize:
            self.quantizer.find_params(self.layer.weight.data, weight=True, num=16)

        # 量化参数搜索次数，量化位数越高，搜索次数约少
        # search_echop = round(16/quantbit-1)*4
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        tick = time.time()

        Losses = torch.zeros_like(W)
        Error = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Error1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)

            for i in range(count):
                w = W1[:, i]
                # group包处理
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize),(self.columns))], weight=True, num=8)
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 
                Error1[:,i] = w - q

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 
            Error[:,i1:i2] = Error1

        torch.cuda.synchronize()
        # 保留两位小数
        # logger.info(f"Quantilization_bits:{quantbit} time :{(time.time() - tick):.2f} layer loss: {torch.sum(Losses).item():.4f}.")
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        return Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype),torch.sum(Losses).item()


    def free(self):
        self.Losses = None
        torch.cuda.empty_cache()