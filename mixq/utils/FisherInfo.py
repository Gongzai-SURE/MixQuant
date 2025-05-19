import torch
# from torchviz import make_dot
import copy
from .misc import find_layers
import gc
from loguru import logger
import warnings
from .multi_gpus import *
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ..linear import Mixq_Linear
from ..quant import Quantizer

warnings.filterwarnings("ignore", category=UserWarning, message=".*Warning.*")

# 随机生成扰动数据
def random_data(param, percentage = 0.2):
    torch.manual_seed(0)
    # 符号随机矩阵
    sign = 2 * torch.randint(0, 2, param.shape) - 1
    res = sign.to(param.device) * (param * percentage) 
    return res.requires_grad_()

# 真实量化扰动
def Quantization_perturbation(bit, layer):
    mixq_linear = Mixq_Linear(layer, bit)
    mixq_linear.quantizer = Quantizer(bit, perchannel=True)
    original_weight = layer.weight.data.clone().to(torch.float)
    W_quant,_ = mixq_linear.fasterquant(quantbit = bit, groupsize=128)
    perturbation = W_quant - original_weight
    del original_weight, W_quant
    torch.cuda.empty_cache()
    return perturbation

def create_position_ids(length: int, device: str = "cpu") -> torch.Tensor:
    """
    Args:
        length (int): 序列长度。
    Returns:
        torch.Tensor: 一个形状为 (1, length) 的 position_ids 张量。
    """
    position_ids = torch.arange(0, length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)  # shape: (1, length)
    return position_ids

def create_attention_mask(length: int, device: str = "cpu") -> torch.Tensor:
    attention_mask = torch.ones(length, dtype=torch.long, device=device)
    attention_mask[-1] = 0  
    attention_mask = attention_mask.unsqueeze(0)  # shape: (1, length)
    
    return attention_mask

# 计算fisher信息矩阵7 (梯度传递采用全模型传递方式,添加扰动方式为“组件层”添加，扰动数据生成方式为真实量化扰动)
def evaluate_fisher_information_with_quant_perturb_sub_block(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    
    # model.to(args.device)
    layer_batch = 1
    # 初始化各个目标参数
    est_fisher_info = {}
    original_fisher = {}
    modified_fisher = {}
    original_perplexitys = []
    modified_perplexitys = {}
    batch_num = dataset[0][0].size()[0] 

    epoch_iterator = tqdm(dataset, desc="Iteration")
    for model_part in model_parts:
        model_part.train()

    # 计算原始困惑度以及每一层的fihser information
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to('cuda:0') for t in batch)
        
        batch_num = batch[0].size()[0]

        for i in range(batch_num):
            inputs = batch[0][i].unsqueeze(0)
            inp_kwargs = {
                "attention_mask": None,
                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
            }
            # 前向传播
            outputs = inputs
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Embedding layer
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                elif j == len(model_parts) - 1:  # Post layer
                    device = f'cuda:{num_gpus - 1}'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{(j-1) % num_gpus}'
                    for item in inp_kwargs:
                        if inp_kwargs[item] is not None:
                            inp_kwargs[item] = inp_kwargs[item].to(device)
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            model.lm_head.to(f'cuda:{num_gpus - 1}')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # 计算原始困惑度
            loss_fn = nn.CrossEntropyLoss()
            perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            original_perplexitys.append(perplexity)
            
            # 反向传播
            for model_part in model_parts:
                model_part.zero_grad()
            perplexity.backward()

            # 获取初始Fisher info
            block_number = 0
            for i, model_part in enumerate(model_parts):
                if i == 0 or i == len(model_parts) - 1:
                    continue
                for seq_layer in model_part:
                    original_fisher[block_number] = {}
                    target_layers = find_layers(seq_layer)
                    for name, param in seq_layer.named_parameters():
                        layer_name = name.rsplit('.', 1)[0]
                        if layer_name not in target_layers.keys():
                            continue
                        if param.requires_grad:
                            if param.grad is not None:
                                if layer_name not in original_fisher[block_number]:
                                    original_fisher[block_number][layer_name] = torch.sum(param.grad.detach() ** 2).to("cpu")
                                else:
                                    original_fisher[block_number][layer_name] += torch.sum(param.grad.detach() ** 2).to("cpu")
                    block_number += 1

            del outputs, logits, shift_logits, shift_labels, perplexity, inputs
        torch.cuda.empty_cache()
        batch = tuple(t.to('cpu') for t in batch)

    # Nomalize original fisher information
    for block_id,_ in enumerate(original_fisher):
        for name in original_fisher[block_id].keys():
            original_fisher[block_id][name] = original_fisher[block_id][name]/(batch_num*len(dataset))
    
    # 计算原始困惑度平均值
    original_perplexity = sum(original_perplexitys)/len(original_perplexitys)
    logger.info(f"Original_perplexity : {original_perplexity}")

    # 计算每一层添加扰动后的困惑度变化以及对应fisher information
    test_bits = args.test_bit
    for bit in test_bits:
        est_fisher_info[bit] = {}
        modified_perplexitys[bit] = {}
        modified_fisher[bit] = {}
        block_number = 0
        for i,model_part in enumerate(model_parts):
            if i == 0 or i == len(model_parts) - 1:
                continue
            block_iterator = tqdm(model_part, desc=f"Cuda:{(i-1) % num_gpus} ModelBlock")
            for _, seq_layer in enumerate(block_iterator):
            # for seq_layer in model_part:
                modified_fisher[bit][block_number] = {}
                modified_perplexitys[bit][block_number] = {}
                target_layers = find_layers(seq_layer)
                count_layer = 0
                delta_thetas = {}
                param_list = {}
                # 采用层组件添加扰动的方式，设置层组件规模为layer_batch，即每 layer_batch 层为一组 进行随机扰动的添加
                for name, param in seq_layer.named_parameters():
                    layer_name = name.rsplit('.', 1)[0]
                    if layer_name not in target_layers.keys():
                        continue
                    
                    count_layer += 1

                    # 扰动噪声 δθ，作为模拟量化带来扰动的波动
                    with torch.no_grad():
                        delta_theta = Quantization_perturbation(bit, target_layers[layer_name])
                        # delta_theta = random_data(param,args.perturb_percentage)
                        param.add_(delta_theta)
                        delta_thetas[layer_name] = delta_theta
                        param_list[layer_name] = param
                    
                    param.requires_grad = True

                    if count_layer % layer_batch == 0 or count_layer == len(target_layers):
                        # 计算fisher信息矩阵
                        for step, batch in enumerate(epoch_iterator):
                            batch = tuple(t.to('cuda:0') for t in batch)

                            for i in range(batch_num):
                                inputs = batch[0][i].unsqueeze(0)
                                inp_kwargs = {
                                    "attention_mask": None,
                                    "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
                                }
                                # 前向传播
                                outputs = inputs
                                for j, model_part in enumerate(model_parts):
                                    if j == 0:  # Embedding layer
                                        device = 'cuda:0'
                                        outputs = model_part(outputs.to(device))
                                    elif j == len(model_parts) - 1:  # Post layer
                                        device = f'cuda:{num_gpus - 1}'
                                        outputs = model_part(outputs.to(device))
                                    else:  # Transformer layers
                                        device = f'cuda:{(j-1) % num_gpus}'
                                        for item in inp_kwargs:
                                            if inp_kwargs[item] is not None:
                                                inp_kwargs[item] = inp_kwargs[item].to(device)
                                        for layer in model_part:
                                            outputs = layer(outputs.to(device), **inp_kwargs)[0]

                                model.lm_head.to(f'cuda:{num_gpus - 1}')
                                logits = model.lm_head(outputs)
                                inputs = inputs.to(f'cuda:{num_gpus - 1}')
                                shift_logits = logits[:, :-1, :].contiguous()
                                shift_labels = inputs[:, 1:]

                                # 计算添加扰动后的困惑度大小
                                loss_fn = nn.CrossEntropyLoss()
                                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                                for layer_name in param_list.keys():
                                    if layer_name not in modified_perplexitys[bit][block_number]:
                                        modified_perplexitys[bit][block_number][layer_name] = loss.to("cpu")
                                    else:
                                        modified_perplexitys[bit][block_number][layer_name] += loss.to("cpu")
                                # loss = perplexity - original_perplexitys[step]

                                # 反向传播
                                for model_part in model_parts:
                                    model_part.zero_grad()

                                loss.backward()

                                # 计算 Fisher 信息矩阵
                                for layer_name in param_list.keys():
                                    if param_list[layer_name].grad is not None:
                                        if layer_name not in modified_fisher[bit][block_number]:
                                            modified_fisher[bit][block_number][layer_name] = torch.sum(param_list[layer_name].grad.detach() ** 2).to("cpu")
                                        else:
                                            modified_fisher[bit][block_number][layer_name] += torch.sum(param_list[layer_name].grad.detach() ** 2).to("cpu")

                                del inputs, outputs, logits, shift_logits, shift_labels
                                torch.cuda.empty_cache()
                        # 恢复原始参数
                        with torch.no_grad():
                            for key in delta_thetas.keys():
                                param_list[key].sub_(delta_thetas[key])
                            # param.sub_(delta_theta)
                        del delta_theta, delta_thetas, param_list
                        delta_thetas = {}
                        param_list = {}
                        torch.cuda.empty_cache()
                del target_layers
                block_number += 1

        # Nomalize modeified fisher information and modified perplexitys
        
        for block_id,_ in enumerate(modified_fisher[bit]):
            for name in modified_fisher[bit][block_id].keys():
                modified_fisher[bit][block_id][name] = modified_fisher[bit][block_id][name]/(batch_num*len(dataset))
                modified_perplexitys[bit][block_id][name] = modified_perplexitys[bit][block_id][name]/(batch_num*len(dataset))

        # calculate estimate fisher infomation
        for block_id, _ in enumerate(modified_fisher[bit]):
            est_fisher_info[bit][block_id] = {}
            for name in modified_fisher[bit][block_id].keys():
                # est_fisher_info[bit][block_id][name] = (modified_fisher[bit][block_id][name] - original_fisher[block_id][name])/ original_fisher[block_id][name]
                est_fisher_info[bit][block_id][name] = modified_fisher[bit][block_id][name] - original_fisher[block_id][name]


    for model_part in model_parts:
        model_part.to('cpu')

    torch.cuda.empty_cache()
    
    return est_fisher_info, modified_perplexitys, original_perplexity


# 计算fisher信息矩阵6 (梯度传递采用全模型传递方式,添加扰动方式为逐层添加，增加前后梯度变化，扰动数据生成方式为真实量化扰动)
def evaluate_fisher_information_with_quant_perturb_sub(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    # model.to(args.device)
    # 初始化各个目标参数
    est_fisher_info = {}
    original_fisher = {}
    modified_fisher = {}
    original_perplexitys = []
    modified_perplexitys = {}
    batch_num = dataset[0][0].size()[0] 

    epoch_iterator = tqdm(dataset, desc="Iteration")
    for model_part in model_parts:
        model_part.train()

    # 计算原始困惑度以及每一层的fihser information
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to('cuda:0') for t in batch)
        
        batch_num = batch[0].size()[0]

        for i in range(batch_num):
            inputs = batch[0][i].unsqueeze(0)
            inp_kwargs = {
                "attention_mask": None,
                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
            }
            # 前向传播
            outputs = inputs
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Embedding layer
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                elif j == len(model_parts) - 1:  # Post layer
                    device = f'cuda:{num_gpus - 1}'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{(j-1) % num_gpus}'
                    for item in inp_kwargs:
                        if inp_kwargs[item] is not None:
                            inp_kwargs[item] = inp_kwargs[item].to(device)
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            model.lm_head.to(f'cuda:{num_gpus - 1}')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # 计算原始困惑度
            loss_fn = nn.CrossEntropyLoss()
            perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            original_perplexitys.append(perplexity)
            
            # 反向传播
            for model_part in model_parts:
                model_part.zero_grad()
            perplexity.backward()

            # 获取初始Fisher info
            block_number = 0
            for i, model_part in enumerate(model_parts):
                if i == 0 or i == len(model_parts) - 1:
                    continue
                for seq_layer in model_part:
                    original_fisher[block_number] = {}
                    target_layers = find_layers(seq_layer)
                    for name, param in seq_layer.named_parameters():
                        layer_name = name.rsplit('.', 1)[0]
                        if layer_name not in target_layers.keys():
                            continue
                        if param.requires_grad:
                            if param.grad is not None:
                                if layer_name not in original_fisher[block_number]:
                                    original_fisher[block_number][layer_name] = torch.sum(param.grad.detach() ** 2).to("cpu")
                                else:
                                    original_fisher[block_number][layer_name] += torch.sum(param.grad.detach() ** 2).to("cpu")
                    block_number += 1

            del outputs, logits, shift_logits, shift_labels, perplexity, inputs
        torch.cuda.empty_cache()
        batch = tuple(t.to('cpu') for t in batch)

    # Nomalize original fisher information
    for block_id,_ in enumerate(original_fisher):
        for name in original_fisher[block_id].keys():
            original_fisher[block_id][name] = original_fisher[block_id][name]/(batch_num*len(dataset))
    
    # 计算原始困惑度平均值
    original_perplexity = sum(original_perplexitys)/len(original_perplexitys)
    print(f"Original_perplexity : {original_perplexity}")

    # 计算每一层添加扰动后的困惑度变化以及对应fisher information
    test_bits = [8, 4, 3, 2]
    for bit in test_bits:
        est_fisher_info[bit] = {}
        modified_perplexitys[bit] = {}
        modified_fisher[bit] = {}
        block_number = 0
        for i,model_part in enumerate(model_parts):
            if i == 0 or i == len(model_parts) - 1:
                continue
            block_iterator = tqdm(model_part, desc=f"Cuda:{(i-1) % num_gpus} ModelBlock")
            for step, seq_layer in enumerate(block_iterator):
            # for seq_layer in model_part:
                modified_fisher[bit][block_number] = {}
                modified_perplexitys[bit][block_number] = {}
                target_layers = find_layers(seq_layer)
                # 逐层添加扰动并进行fisher info的计算
                for name, param in seq_layer.named_parameters():
                    layer_name = name.rsplit('.', 1)[0]
                    if layer_name not in target_layers.keys():
                        continue

                    # 扰动噪声 δθ，作为模拟量化带来扰动的波动
                    with torch.no_grad():
                        delta_theta = Quantization_perturbation(bit, target_layers[layer_name])
                        param.add_(delta_theta)
                    
                    param.requires_grad = True

                    # 计算fisher信息矩阵
                    for step, batch in enumerate(epoch_iterator):
                        batch = tuple(t.to('cuda:0') for t in batch)

                        for i in range(batch_num):
                            inputs = batch[0][i].unsqueeze(0)
                            inp_kwargs = {
                                "attention_mask": None,
                                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
                            }
                            # 前向传播
                            outputs = inputs
                            for j, model_part in enumerate(model_parts):
                                if j == 0:  # Embedding layer
                                    device = 'cuda:0'
                                    outputs = model_part(outputs.to(device))
                                elif j == len(model_parts) - 1:  # Post layer
                                    device = f'cuda:{num_gpus - 1}'
                                    outputs = model_part(outputs.to(device))
                                else:  # Transformer layers
                                    device = f'cuda:{(j-1) % num_gpus}'
                                    for item in inp_kwargs:
                                        if inp_kwargs[item] is not None:
                                            inp_kwargs[item] = inp_kwargs[item].to(device)
                                    for layer in model_part:
                                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

                            model.lm_head.to(f'cuda:{num_gpus - 1}')
                            logits = model.lm_head(outputs)
                            inputs = inputs.to(f'cuda:{num_gpus - 1}')
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = inputs[:, 1:]

                            # 计算添加扰动后的困惑度大小
                            loss_fn = nn.CrossEntropyLoss()
                            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            if layer_name not in modified_perplexitys[bit][block_number]:
                                modified_perplexitys[bit][block_number][layer_name] = loss.to("cpu")
                            else:
                                modified_perplexitys[bit][block_number][layer_name] += loss.to("cpu")
                            # loss = perplexity - original_perplexitys[step]

                            # 反向传播
                            for model_part in model_parts:
                                model_part.zero_grad()

                            loss.backward()

                            # 计算 Fisher 信息矩阵
                            if param.grad is not None:
                                if layer_name not in modified_fisher[bit][block_number]:
                                    modified_fisher[bit][block_number][layer_name] = torch.sum(param.grad.detach() ** 2).to("cpu")
                                else:
                                    modified_fisher[bit][block_number][layer_name] += torch.sum(param.grad.detach() ** 2).to("cpu")
                    # 恢复原始参数
                    with torch.no_grad():
                        param.sub_(delta_theta)
                    del outputs, logits, shift_logits, shift_labels, loss, inputs, delta_theta
                    torch.cuda.empty_cache()
                del target_layers
                block_number += 1

        # Nomalize modeified fisher information and modified perplexitys
        
        for block_id,_ in enumerate(modified_fisher[bit]):
            for name in modified_fisher[bit][block_id].keys():
                modified_fisher[bit][block_id][name] = modified_fisher[bit][block_id][name]/(batch_num*len(dataset))
                modified_perplexitys[bit][block_id][name] = modified_perplexitys[bit][block_id][name]/(batch_num*len(dataset))

        # calculate estimate fisher infomation
        for block_id, _ in enumerate(modified_fisher[bit]):
            est_fisher_info[bit][block_id] = {}
            for name in modified_fisher[bit][block_id].keys():
                # est_fisher_info[bit][block_id][name] = (modified_fisher[bit][block_id][name] - original_fisher[block_id][name]) / original_fisher[block_id][name]
                est_fisher_info[bit][block_id][name] = modified_fisher[bit][block_id][name] - original_fisher[block_id][name]


    for model_part in model_parts:
        model_part.to('cpu')
    
    return est_fisher_info, modified_perplexitys, original_perplexity


# 计算fisher信息矩阵5 (梯度传递采用全模型传递方式,添加扰动方式为逐层添加，增加前后梯度变化，扰动数据生成方式为扰动幅度)
def evaluate_fisher_information_with_random_perturb_sub(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    # model.to(args.device)
    # 初始化各个目标参数
    est_fisher_info = {}
    original_fisher = {}
    modified_fisher = {}
    original_perplexitys = []
    modified_perplexitys = {}
    batch_num = dataset[0][0].size()[0] 

    epoch_iterator = tqdm(dataset, desc="Iteration")
    for model_part in model_parts:
        model_part.train()

    # 计算原始困惑度以及每一层的fihser information
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to('cuda:0') for t in batch)
        
        batch_num = batch[0].size()[0]

        for i in range(batch_num):
            inputs = batch[0][i].unsqueeze(0)
            inp_kwargs = {
                "attention_mask": None,
                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
            }
            # 前向传播
            outputs = inputs
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Embedding layer
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                elif j == len(model_parts) - 1:  # Post layer
                    device = f'cuda:{num_gpus - 1}'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{(j-1) % num_gpus}'
                    for item in inp_kwargs:
                        if inp_kwargs[item] is not None:
                            inp_kwargs[item] = inp_kwargs[item].to(device)
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            model.lm_head.to(f'cuda:{num_gpus - 1}')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # 计算原始困惑度
            loss_fn = nn.CrossEntropyLoss()
            perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            original_perplexitys.append(perplexity)
            
            # 反向传播
            for model_part in model_parts:
                model_part.zero_grad()
            perplexity.backward()

            # 获取初始Fisher info
            block_number = 0
            for i, model_part in enumerate(model_parts):
                if i == 0 or i == len(model_parts) - 1:
                    continue
                for seq_layer in model_part:
                    original_fisher[block_number] = {}
                    target_layers = find_layers(seq_layer)
                    for name, param in seq_layer.named_parameters():
                        layer_name = name.rsplit('.', 1)[0]
                        if layer_name not in target_layers.keys():
                            continue
                        if param.requires_grad:
                            if param.grad is not None:
                                if name not in original_fisher[block_number]:
                                    original_fisher[block_number][name] = torch.sum(param.grad.detach() ** 2).to("cpu")
                                else:
                                    original_fisher[block_number][name] += torch.sum(param.grad.detach() ** 2).to("cpu")
                    block_number += 1

            del outputs, logits, shift_logits, shift_labels, perplexity, inputs
        torch.cuda.empty_cache()
        batch = tuple(t.to('cpu') for t in batch)

    # Nomalize original fisher information
    for block_id,_ in enumerate(original_fisher):
        for name in original_fisher[block_id].keys():
            original_fisher[block_id][name] = original_fisher[block_id][name]/(batch_num*len(dataset))
    
    # 计算原始困惑度平均值
    original_perplexity = sum(original_perplexitys)/len(original_perplexitys)
    print(f"Original_perplexity : {original_perplexity}")

    # 计算每一层添加扰动后的困惑度变化以及对应fisher information
    block_number = 0
    for i,model_part in enumerate(model_parts):
        if i == 0 or i == len(model_parts) - 1:
            continue
        block_iterator = tqdm(model_part, desc=f"Cuda:{(i-1) % num_gpus} ModelBlock")
        for step, seq_layer in enumerate(block_iterator):
        # for seq_layer in model_part:
            modified_fisher[block_number] = {}
            modified_perplexitys[block_number] = {}
            target_layers = find_layers(seq_layer)
            # 逐层添加扰动并进行fisher info的计算
            for name, param in seq_layer.named_parameters():
                layer_name = name.rsplit('.', 1)[0]
                if layer_name not in target_layers.keys():
                    continue

                # 扰动噪声 δθ，作为模拟量化带来扰动的波动
                with torch.no_grad():
                    delta_theta = random_data(param,args.perturb_percentage)
                    param.add_(delta_theta)
                
                param.requires_grad = True

                # 计算fisher信息矩阵
                for step, batch in enumerate(epoch_iterator):
                    batch = tuple(t.to('cuda:0') for t in batch)

                    for i in range(batch_num):
                        inputs = batch[0][i].unsqueeze(0)
                        inp_kwargs = {
                            "attention_mask": None,
                            "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
                        }
                        # 前向传播
                        outputs = inputs
                        for j, model_part in enumerate(model_parts):
                            if j == 0:  # Embedding layer
                                device = 'cuda:0'
                                outputs = model_part(outputs.to(device))
                            elif j == len(model_parts) - 1:  # Post layer
                                device = f'cuda:{num_gpus - 1}'
                                outputs = model_part(outputs.to(device))
                            else:  # Transformer layers
                                device = f'cuda:{(j-1) % num_gpus}'
                                for item in inp_kwargs:
                                    if inp_kwargs[item] is not None:
                                        inp_kwargs[item] = inp_kwargs[item].to(device)
                                for layer in model_part:
                                    outputs = layer(outputs.to(device), **inp_kwargs)[0]

                        model.lm_head.to(f'cuda:{num_gpus - 1}')
                        logits = model.lm_head(outputs)
                        inputs = inputs.to(f'cuda:{num_gpus - 1}')
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = inputs[:, 1:]

                        # 计算添加扰动后的困惑度大小
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        if name not in modified_perplexitys[block_number]:
                            modified_perplexitys[block_number][name] = loss.to("cpu")
                        else:
                            modified_perplexitys[block_number][name] += loss.to("cpu")
                        # loss = perplexity - original_perplexitys[step]

                        # 反向传播
                        for model_part in model_parts:
                            model_part.zero_grad()

                        loss.backward()

                        # 计算 Fisher 信息矩阵
                        if param.grad is not None:
                            if name not in modified_fisher[block_number]:
                                modified_fisher[block_number][name] = torch.sum(param.grad.detach() ** 2).to("cpu")
                            else:
                                modified_fisher[block_number][name] += torch.sum(param.grad.detach() ** 2).to("cpu")
                # 恢复原始参数
                with torch.no_grad():
                    param.sub_(delta_theta)
                del outputs, logits, shift_logits, shift_labels, loss, inputs, delta_theta
                torch.cuda.empty_cache()
            del target_layers
            block_number += 1
    
    # Nomalize modeified fisher information and modified perplexitys
    for block_id,_ in enumerate(modified_fisher):
        for name in modified_fisher[block_id].keys():
            modified_fisher[block_id][name] = modified_fisher[block_id][name]/(batch_num*len(dataset))
            modified_perplexitys[block_id][name] = modified_perplexitys[block_id][name]/(batch_num*len(dataset))

    # calculate estimate fisher infomation
    for block_id, _ in enumerate(modified_fisher):
        est_fisher_info[block_id] = {}
        for name in modified_fisher[block_id].keys():
            est_fisher_info[block_id][name] = (modified_fisher[block_id][name] - original_fisher[block_id][name])/ original_fisher[block_id][name]
    
    for model_part in model_parts:
        model_part.to('cpu')
    
    return est_fisher_info, modified_perplexitys, original_perplexity


# 计算fisher信息矩阵4 (梯度传递采用全模型传递方式,添加扰动方式为逐层添加，扰动数据生成方式为扰动幅度)
def evaluate_fisher_information_with_random_perturb(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    # model.to(args.device)
    est_fisher_info = {}
    original_perplexitys = []
    batch_num = dataset[0][0].size()[0] 

    epoch_iterator = tqdm(dataset, desc="Iteration")
    # 计算原始困惑度
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to('cuda:0') for t in batch)
            
            batch_num = batch[0].size()[0]

            for i in range(batch_num):
                inputs = batch[0][i].unsqueeze(0)
                inp_kwargs = {
                    "attention_mask": None,
                    "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
                }
                # 前向传播
                outputs = inputs
                for j, model_part in enumerate(model_parts):
                    if j == 0:  # Embedding layer
                        device = 'cuda:0'
                        outputs = model_part(outputs.to(device))
                    elif j == len(model_parts) - 1:  # Post layer
                        device = f'cuda:{num_gpus - 1}'
                        outputs = model_part(outputs.to(device))
                    else:  # Transformer layers
                        device = f'cuda:{(j-1) % num_gpus}'
                        for item in inp_kwargs:
                            if inp_kwargs[item] is not None:
                                inp_kwargs[item] = inp_kwargs[item].to(device)
                        for layer in model_part:
                            outputs = layer(outputs.to(device), **inp_kwargs)[0]

                model.lm_head.to(f'cuda:{num_gpus - 1}')
                logits = model.lm_head(outputs)
                inputs = inputs.to(f'cuda:{num_gpus - 1}')
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]

                # 计算原始困惑度
                loss_fn = nn.CrossEntropyLoss()
                perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                original_perplexitys.append(perplexity)
                del outputs, logits, shift_logits, shift_labels, perplexity, inputs
            
            batch = tuple(t.to('cpu') for t in batch)
    
    print(original_perplexitys)
    for model_part in model_parts:
        model_part.train()
    block_number = 0
    for i,model_part in enumerate(model_parts):
        if i == 0 or i == len(model_parts) - 1:
            continue
        for seq_layer in model_part:
            grads = {}
            est_fisher_info[block_number] = {}
            target_layers = find_layers(seq_layer)
            # 逐层添加扰动并进行fisher info的计算
            for name, param in seq_layer.named_parameters():
                layer_name = name.rsplit('.', 1)[0]
                if layer_name not in target_layers.keys():
                    continue

                # 扰动噪声 δθ，作为可求导的变量
                with torch.no_grad():
                    delta_theta = random_data(param)
                
                handle = delta_theta.register_hook(lambda grad, name=name: grads.setdefault(name, grad.clone()))
                param.requires_grad = False
                param.add_(delta_theta)

                # 计算fisher信息矩阵
                for step, batch in enumerate(epoch_iterator):
                    batch = tuple(t.to('cuda:0') for t in batch)

                    for i in range(batch_num):
                        inputs = batch[0][i].unsqueeze(0)
                        inp_kwargs = {
                            "attention_mask": None,
                            "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
                        }
                        # 前向传播
                        outputs = inputs
                        for j, model_part in enumerate(model_parts):
                            if j == 0:  # Embedding layer
                                device = 'cuda:0'
                                outputs = model_part(outputs.to(device))
                            elif j == len(model_parts) - 1:  # Post layer
                                device = f'cuda:{num_gpus - 1}'
                                outputs = model_part(outputs.to(device))
                            else:  # Transformer layers
                                device = f'cuda:{(j-1) % num_gpus}'
                                for item in inp_kwargs:
                                    if inp_kwargs[item] is not None:
                                        inp_kwargs[item] = inp_kwargs[item].to(device)
                                for layer in model_part:
                                    outputs = layer(outputs.to(device), **inp_kwargs)[0]

                        model.lm_head.to(f'cuda:{num_gpus - 1}')
                        logits = model.lm_head(outputs)
                        inputs = inputs.to(f'cuda:{num_gpus - 1}')
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = inputs[:, 1:]

                        # 计算原始困惑度
                        loss_fn = nn.CrossEntropyLoss()
                        perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        loss = perplexity - original_perplexitys[step]

                        # 反向传播
                        for model_part in model_parts:
                            model_part.zero_grad()
                        loss.backward()

                        # 计算 Fisher 信息矩阵
                        if delta_theta.grad is not None:
                            if name not in est_fisher_info[block_number]:
                                est_fisher_info[block_number][name] = torch.sum(delta_theta.grad.detach() ** 2).to("cpu")
                            else:
                                est_fisher_info[block_number][name] += torch.sum(delta_theta.grad.detach() ** 2).to("cpu")
                # 恢复原始参数
                with torch.no_grad():
                    param.sub_(delta_theta)
                
                handle.remove()
                del outputs, logits, shift_logits, shift_labels, perplexity, inputs, delta_theta, loss, handle
                torch.cuda.empty_cache()
            del grads, target_layers
            block_number += 1
    
    for block_id,_ in enumerate(est_fisher_info):
        for name in est_fisher_info[block_id].keys():
            est_fisher_info[block_id][name] = est_fisher_info[block_id][name]/(batch_num*len(dataset))
        
    for model_part in model_parts:
        model_part.to('cpu')
    
    return est_fisher_info


# 计算fisher信息矩阵3 (梯度传递采用全模型传递方式,添加扰动方式为全组件添加，扰动数据生成方式为扰动幅度)
def evaluate_fisher_information_with_total_random_perturb(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    for model_part in model_parts:
        model_part.train()
    # model.to(args.device)
    est_fisher_info = {}
    perplexitys = [] 
    delta_thetas = {}

    epoch_iterator = tqdm(dataset, desc="Iteration")
    # 计算原始困惑度
    for step, batch in enumerate(epoch_iterator):

        # 将 batch 中的每个张量移动到第一个设备上
        batch = tuple(t.to('cuda:0') for t in batch)
        
        batch_num = batch[0].size()[0]

        for i in range(batch_num):
            inputs = batch[0][i].unsqueeze(0)
            inp_kwargs = {
                "attention_mask": None,
                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
            }
            # 前向传播
            outputs = inputs
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Embedding layer
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                elif j == len(model_parts) - 1:  # Post layer
                    device = f'cuda:{num_gpus - 1}'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{(j-1) % num_gpus}'
                    for item in inp_kwargs:
                        if inp_kwargs[item] is not None:
                            inp_kwargs[item] = inp_kwargs[item].to(device)
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            model.lm_head.to(f'cuda:{num_gpus - 1}')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # 计算原始困惑度
            loss_fn = nn.CrossEntropyLoss()
            perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexitys.append(perplexity)
            print(f'perplexity:{perplexity}')
    
    print(perplexitys)
    
    # 添加扰动
    delta_thetas = add_perturb(model,args)

    # 添加扰动后的fisher info计算
    for step, batch in enumerate(epoch_iterator):

        # 将 batch 中的每个张量移动到第一个设备上
        batch = tuple(t.to('cuda:0') for t in batch)
        
        batch_num = batch[0].size()[0]

        for i in range(batch_num):
            inputs = batch[0][i].unsqueeze(0)
            inp_kwargs = {
                "attention_mask": None,
                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
            }
            # 前向传播
            outputs = inputs
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Embedding layer
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                elif j == len(model_parts) - 1:  # Post layer
                    device = f'cuda:{num_gpus - 1}'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{(j-1) % num_gpus}'
                    for item in inp_kwargs:
                        if inp_kwargs[item] is not None:
                            inp_kwargs[item] = inp_kwargs[item].to(device)
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            model.lm_head.to(f'cuda:{num_gpus - 1}')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # 计算原始困惑度
            loss_fn = nn.CrossEntropyLoss()
            perplexity = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = perplexity - perplexitys[step]
            
            # 反向传播
            for model_part in model_parts:
                model_part.zero_grad()
            loss.backward()

            # 计算 Fisher 信息矩阵
            for name, delta_theta in enumerate(delta_thetas):
                if delta_theta[name].grad is not None:
                    if name not in est_fisher_info:
                        est_fisher_info[name] = 0
                    est_fisher_info[name] += torch.sum(delta_theta[name].grad.detach() ** 2).to("cpu")

    
    est_fisher_info = {n: p/batch_num for n, p in est_fisher_info.items()}
        
    for model_part in model_parts:
        model_part.to('cpu')
    
    remove_perturb(model,delta_thetas,args)

    return est_fisher_info


# 计算fisher信息矩阵2 (梯度传递采用全模型传递方式,无扰动添加)
def evaluate_fisher_information_no_perturb(model, dataset, args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    for model_part in model_parts:
        model_part.train()
    # model.to(args.device)
    est_fisher_info = {}

    epoch_iterator = tqdm(dataset, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):

        # 将 batch 中的每个张量移动到第一个设备上
        batch = tuple(t.to('cuda:0') for t in batch)
        
        batch_num = batch[0].size()[0]

        for i in range(batch_num):
            inputs = batch[0][i].unsqueeze(0)
            inp_kwargs = {
                # "attention_mask": batch[1][i].unsqueeze(0),
                # "attention_mask": create_attention_mask(len(batch[0][i])).to(args.device),
                "attention_mask": None,
                "position_ids": create_position_ids(len(batch[0][i])).to(args.device)
            }
            # 前向传播
            outputs = inputs
            for j, model_part in enumerate(model_parts):
                if j == 0:  # Embedding layer
                    device = 'cuda:0'
                    outputs = model_part(outputs.to(device))
                elif j == len(model_parts) - 1:  # Post layer
                    device = f'cuda:{num_gpus - 1}'
                    outputs = model_part(outputs.to(device))
                else:  # Transformer layers
                    device = f'cuda:{(j-1) % num_gpus}'
                    for item in inp_kwargs:
                        if inp_kwargs[item] is not None:
                            inp_kwargs[item] = inp_kwargs[item].to(device)
                    for layer in model_part:
                        outputs = layer(outputs.to(device), **inp_kwargs)[0]

            # outputs = model(inputs, **inp_kwargs)
            # logits = outputs.logits
            model.lm_head.to(f'cuda:{num_gpus - 1}')
            logits = model.lm_head(outputs)
            inputs = inputs.to(f'cuda:{num_gpus - 1}')
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # 计算损失
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 反向传播
            for model_part in model_parts:
                model_part.zero_grad()
                
            loss.backward()

            # 计算 Fisher 信息矩阵
            for i, model_part in enumerate(model_parts):
                for n, p in model_part.named_parameters():
                    if p.requires_grad:
                        if p.grad is not None:
                            if f"{i}.{n}" not in est_fisher_info:
                                est_fisher_info[f"{i}.{n}"] = 0
                            est_fisher_info[f"{i}.{n}"] += torch.sum(p.grad.detach() ** 2).to("cpu")

        est_fisher_info = {n: p/batch_num for n, p in est_fisher_info.items()}
        
    for model_part in model_parts:
        model_part.to('cpu')
    return est_fisher_info


# 计算fisher信息矩阵1 (梯度传递采用逐模块传递方式,添加扰动方式为逐层添加，扰动数据生成方式为扰动幅度)
def evaluate_fisher_information_with_random_perturb_blockly(seq_layer, inps, inp_kwargs):
    torch.autograd.set_detect_anomaly(True)
    est_fisher_info = {}
    outs_original = torch.ones_like(inps)
    tmp_inp = copy.deepcopy(inps)
    target_layers = find_layers(seq_layer)

    with torch.no_grad():
        for j in range(len(tmp_inp)):
            outs_original[j] = seq_layer(tmp_inp[j].unsqueeze(0), **inp_kwargs)[0]

    for name, param in seq_layer.named_parameters():
        layer_name = name.rsplit('.', 1)[0]
        if layer_name not in target_layers.keys():
            continue

        # 扰动噪声 δθ，作为可求导的变量
        with torch.no_grad():
            delta_theta = random_data(param)
        
        grads = {}
        handle = delta_theta.register_hook(lambda grad, name=name: grads.setdefault(name, grad.clone()))

        param.requires_grad = False
        param.add_(delta_theta)

        outs_after = torch.ones_like(inps)
        for j in range(len(tmp_inp)):
            outs_after[j] = seq_layer(tmp_inp[j].unsqueeze(0), **inp_kwargs)[0]

        loss = torch.nn.functional.mse_loss(outs_original, outs_after)

        seq_layer.zero_grad()
        loss.backward()

        # 计算损失对参数的Fisher information
        with torch.no_grad():
            grad = grads[name]
            grad_squared = (grad.detach().to(torch.float32) ** 2)
            est_fisher = (grad_squared.trace() / grad_squared.shape[0] / len(tmp_inp))
            logger.info(f"{name} fisher info : {est_fisher}")
            est_fisher_info[name] = est_fisher.to('cpu')

        # 恢复原始参数
        handle.remove()
        with torch.no_grad():
            param.sub_(delta_theta)

    del outs_original, delta_theta, loss, target_layers, seq_layer, tmp_inp, inp_kwargs
    torch.cuda.empty_cache()

    unreleased_tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]
    gc.collect()
    if unreleased_tensors:
        print(f"Unreleased tensors: {len(unreleased_tensors)}")

    return est_fisher_info


# 计算fisher信息矩阵0 (梯度传递采用逐模块传递方式,添加扰动方式为全组件添加，扰动数据生成方式为扰动幅度）
def evaluate_fisher_information_with_total_random_perturb_blockly(seq_layer, inps, inp_kwargs):
    torch.autograd.set_detect_anomaly(True)
    grads = {}
    est_fisher_info = {}
    outs_original = torch.ones_like(inps)
    outs_after = torch.ones_like(inps)
    tmp_inp = copy.deepcopy(inps)
    target_layers = find_layers(seq_layer)

    def get_grad(grad, name):
        grads[name] = grad.clone()

    # 获取扰动前的输出
    with torch.no_grad():
        for j in range(len(tmp_inp)):
            outs_original[j] = seq_layer(tmp_inp[j].unsqueeze(0), **inp_kwargs)[0]

    # 扰动噪声 δθ，作为可求导的变量
    with torch.no_grad():
        delta_thetas = {name.rsplit('.', 1)[0]: random_data(param) for name, param in seq_layer.named_parameters() if name.rsplit('.', 1)[0] in target_layers.keys()}
    handles = []

    for name, delta_theta in delta_thetas.items():
        handle = delta_theta.register_hook(lambda grad, name=name: get_grad(grad, name))
        handles.append(handle)

    # 添加扰动
    for name, para in seq_layer.named_parameters():
        para.requires_grad=False
        if name.rsplit('.', 1)[0] not in target_layers.keys():
            continue
        para.add_(delta_thetas[name.rsplit('.', 1)[0]])

    # 获取扰动后的输出
    for j in range(len(tmp_inp)):
        outs_after[j] = seq_layer(tmp_inp[j].unsqueeze(0), **inp_kwargs)[0]

    # 计算扰动前后输出的差异作为损失
    loss = torch.nn.functional.mse_loss(outs_original, outs_after)

    seq_layer.zero_grad()
    loss.backward()

    # 计算损失对参数的Fisher information, 注意梯度中可能存在较大的值，需要先转换成float32避免溢出
    for name, grad in grads.items():
        with torch.no_grad():
            # 使用.detach()获取梯度的副本，避免影响原始梯度
            grad_squared = (copy.deepcopy(grad).detach().to(torch.float32)**2)
            est_fisher = (grad_squared.trace() / grad_squared.shape[0] / len(tmp_inp))
            logger.info(f"{name} fisher info : {est_fisher}")
            if name not in est_fisher_info:
                est_fisher_info[name] = est_fisher.to('cpu')
            del grad_squared, est_fisher

    for handle in handles:
        handle.remove()

    # 恢复原始参数
    with torch.no_grad():
        for n, p in seq_layer.named_parameters():
            if n in delta_thetas:
                p.data.sub_(delta_thetas[n])

    # 释放所有资源
    del outs_original, outs_after, delta_thetas, loss, target_layers, seq_layer, tmp_inp, inp_kwargs, grads
    torch.cuda.empty_cache()

    unreleased_tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]
    gc.collect()
    if unreleased_tensors:
        print(f"Unreleased tensors: {len(unreleased_tensors)}" )
    
    return est_fisher_info


# 为模型全部目标层添加扰动 3
def add_perturb(model,args):
    delta_thetas = {}
    layers, pre_layers, post_layers = parsing_layers(model, args.meta)
    for step, seq_layer in enumerate(layers):
        delta_thetas[step] = {}
        target_layers = find_layers(seq_layer)
        for name, param in seq_layer.named_parameters():
            layer_name = name.rsplit('.', 1)[0]
            if layer_name not in target_layers.keys():
                continue

            # 扰动噪声 δθ，作为可求导的变量
            with torch.no_grad():
                delta_theta = random_data(param)

            delta_theta.requires_grad = True

            # handle = delta_theta.register_hook(lambda grad, name=name: get_grad(grad, name))
            # handles.append(handle)

            param.requires_grad = False
            param.add_(delta_theta)
            delta_thetas[step][name] = delta_theta

    return delta_thetas


# 复原模型全部目标层的参数 3
def remove_perturb(model,detla_thetas,args):
    layers, pre_layers, post_layers = parsing_layers(model, args.meta)
    for step, seq_layer in enumerate(layers):
        target_layers = find_layers(seq_layer)
        for name, param in seq_layer.named_parameters():
            layer_name = name.rsplit('.', 1)[0]
            if layer_name not in target_layers.keys():
                continue
            param.data.sub_(detla_thetas[step][name])
