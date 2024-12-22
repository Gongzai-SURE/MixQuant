import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers.models.falcon.modeling_falcon import FalconLinear

try:
    import owq_cuda
except:
    logger.info('OWQ CUDA kernel extension is not installed.')

def quantize(x, scale, zero, minq, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)

def quantize_efficient(x_round, scale, zero, minq, maxq):
    q = torch.clamp(x_round + zero, minq, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):
    def __init__(
            self,
            bits, perchannel=False, sym=False, 
            mse=False, norm=2.4, 
        ):
        super(Quantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        
        self.bits = bits
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.perchannel = perchannel
        self.n_levels = 2 ** bits
        
        if self.sym:
            self.minq, self.maxq = -((self.n_levels - 1) // 2 + 1), (self.n_levels - 1) // 2
        else:
            self.minq, self.maxq = 0, self.n_levels - 1
        
        self.num = 100
        self.eps = torch.tensor(1e-8)
    
    def set_bits(self, bits):
        self.bits = bits
        self.n_levels = 2 ** bits
        if self.sym:
            self.minq, self.maxq = -((self.n_levels - 1) // 2 + 1), (self.n_levels - 1) // 2
        else:
            self.minq, self.maxq = 0, self.n_levels - 1

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.perchannel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)
    
    def find_params(self, x, weight=False, num=100):
        self.num = num
        dev = x.device
        minq, maxq = self.minq, self.maxq
        
        shape = x.shape
        if self.perchannel: # row-wise
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.mse:
            if self.perchannel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            if self.sym:
                xrange = torch.max(xmin.abs(), xmax)
                zero = torch.zeros_like(xmin)
                if self.perchannel:
                    zero = zero.reshape(new_shape)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max(tmp_max / -minq, self.eps)
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                    score = self.lp_loss(x, x_q, 2.4)
                    best_max = torch.where(score < best_score, tmp_max, best_max)
                    best_score = torch.min(score, best_score)
                
                max_val = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max(max_val / -minq, self.eps)
                self.zero = torch.zeros_like(self.scale)
            else:
                xrange = xmax - xmin
                tmp_min = torch.zeros_like(xmin)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max((tmp_max - tmp_min) / (maxq - minq), self.eps)
                    delta = scale.clone()
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    for zp in range(0, self.n_levels):
                        new_min = tmp_min - zp * delta
                        new_max = tmp_max - zp * delta
                        zero = torch.clamp(minq - torch.round(new_min / delta), minq, maxq)
                        if self.perchannel:
                            zero = zero.reshape(new_shape)
                        x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                        score = self.lp_loss(x, x_q, 2.4)
                        best_min = torch.where(score < best_score, new_min, best_min)
                        best_max = torch.where(score < best_score, new_max, best_max)
                        best_score = torch.min(best_score, score)
            
                min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
                max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max((max_val_pos - min_val_neg) / (maxq - minq), self.eps)
                self.zero = torch.clamp(minq - torch.round(min_val_neg / self.scale), minq, maxq)
        else:
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                if torch.any(tmp):
                    xmin[tmp] = -xmax[tmp]

            tmp = (xmin == 0) & (xmax == 0) 
            xmin[tmp] = -1
            xmax[tmp] = +1

            if self.sym:
                self.scale = xmax / -minq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / maxq
                self.zero = torch.round(-xmin / self.scale)
        
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.minq, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

def make_quant(module, quant_infos, wbits, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quant_infos:
            setattr(
                module, attr, 
                QuantLinear(wbits, 
                            tmp.in_features, 
                            tmp.out_features,  
                            tmp.bias is not None, 
                            tmp.weight.dtype,
                            name1).to(tmp.weight.device)
            )
    for name1, child in module.named_children():
        make_quant(child, quant_infos, wbits, name + '.' + name1 if name != '' else name1)

def lm_pack(model, quantinfos, wbits, linears=[nn.Linear, FalconLinear]):
    from mixq.utils.misc import find_layers
    layers = find_layers(model, linears)
    layers = {n: layers[n] for n in quantinfos}
    make_quant(model, quantinfos, wbits)
    qlayers = find_layers(model, [QuantLinear])
    logger.info('Packing ...')
    for name in qlayers:
        quantinfos[name] = quantinfos[name].cpu()
                           # quantinfos[name].bits,
        qlayers[name].pack(layers[name],
                           quantinfos[name].scale, 
                           quantinfos[name].zero, 
                           )
    logger.info('Done.')
    return model


class QuantLinear(nn.Module):
    def __init__(self, bits, infeatures, outfeatures, bias, dtype, name):
        super().__init__()
        assert bits in [3,4], "Only 3,4 bits are supported."
        
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer('scales', torch.zeros((outfeatures, 1), dtype=dtype))
        self.register_buffer('zeros', torch.zeros((outfeatures // 2, 1), dtype=torch.uint8))
        self.register_buffer('bias', torch.zeros(outfeatures, dtype=dtype))
        
        self.faster = True
        self.dtype = dtype
        self.name = name
        
    def pack(self, linear, scales, zeros, sym:bool=False):
        dtype = linear.weight.dtype
        
        if sym:
            zeros += 2**(self.bits - 1)
            
        if linear.bias is not None:
            self.bias = linear.bias.to(dtype)
            
        intweight = torch.round((linear.weight.data + zeros * scales) / scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (self.infeatures // 32 * self.bits, self.outfeatures), dtype=np.uint32
        )
        self.scales = scales.to(dtype)
        zeros = zeros.to(torch.uint8)
        zeros_int = torch.zeros((zeros.shape[0] // 2, zeros.shape[1]), dtype=torch.uint8)
        for i in range(zeros_int.shape[0]):
            zeros_int[i] = (zeros[2*i] | zeros[2*i + 1] << 4)
        self.zeros = zeros_int
        
        i = 0
        row = 0
        if self.bits == 3:
            while row < qweight.shape[0]:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):    
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):    
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
        elif self.bits == 4:
            while row < qweight.shape[0]:
                for j in range(i, i + 8):
                    qweight[row] |= intweight[j] << (4 * (j - i))
                i += 8
                row += 1
        else:
            raise NotImplementedError

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)
    
    def set_kernel(self, faster):
        # set operation kernel
        if self.bits == 3:
            if faster:
                self.matvec = owq_cuda.vecquant3matmul_faster
                self.outmatvec = owq_cuda.vecquant3outliermatmul_faster
                self.dequant = owq_cuda.matquant3dequant_faster
            else:
                self.matvec = owq_cuda.vecquant3matmul
                self.outmatvec = owq_cuda.vecquant3outliermatmul
                self.dequant = owq_cuda.matquant3dequant
        elif self.bits == 4:
            if faster:
                self.matvec = owq_cuda.vecquant4matmul_faster
                self.outmatvec = owq_cuda.vecquant4outliermatmul_faster
                self.dequant = owq_cuda.matquant4dequant_faster
            else:
                self.matvec = owq_cuda.vecquant4matmul
                self.outmatvec = owq_cuda.vecquant4outliermatmul
                self.dequant = owq_cuda.matquant4dequant
        else: # support only 3, 4 bits
            raise NotImplementedError
        
        if self.faster:
            self.forward = self.forward_faster
        else:
            self.forward = self.forward_normal
    
    def forward_faster(self, x):
        if x.shape[-1] == x.numel():
            y = self.bias.clone()
            self.matvec(
                x, self.qweight, y,
                self.scales, self.zeros
                )
            y = y.to(x.dtype)
        else:
            out = torch.empty((self.infeatures, self.outfeatures), dtype=x.dtype, device=x.device)
            self.dequant(self.qweight, out, self.scales, self.zeros)
            y = torch.nn.functional.linear(x, out.t(), self.bias)
        return y
    
    def forward_normal(self, x):
        if x.shape[-1] == x.numel():
            dtype = x.dtype
            y = self.bias.float()
            x = x.float()
            self.matvec(
                x, self.qweight, y,
                self.scales, self.zeros
                )
            y = y.to(dtype)
        else:
            out = torch.empty((self.infeatures, self.outfeatures), dtype=torch.float, device=x.device)
            self.dequant(self.qweight, out, self.scales, self.zeros)
            y = torch.nn.functional.linear(x, out.t().to(x.dtype), self.bias)
        return y