'''
This file contains the function to split a model across multiple GPUs.
'''
import torch.nn as nn
from .misc import parsing_layers


def split_model_across_gpus(model, meta, num_gpus):
    layers, pre_layers, post_layers = parsing_layers(model, meta)
    
    # 将 pre_layers 分配到第一个 GPU 上
    for i, pre_layer in enumerate(pre_layers):
        pre_layers[i] = pre_layer.to('cuda:0')
    
    # 将 transformer 层等比例地分配到多个 GPU 上
    num_transformer_layers = len(layers)
    layers_per_gpu = num_transformer_layers // num_gpus
    model_parts = pre_layers

    for i in range(num_gpus):
        start_idx = i * layers_per_gpu
        end_idx = (i + 1) * layers_per_gpu if i != num_gpus - 1 else num_transformer_layers
        model_part = nn.Sequential(*layers[start_idx:end_idx]).to(f'cuda:{i}')
        model_parts.append(model_part)

    # 将 post_layers 分配到最后一个 GPU 上
    for i, post_layer in enumerate(post_layers):
        post_layers[i] = post_layer.to(f'cuda:{num_gpus-1}')
    model_parts.extend(post_layers)

    return model_parts