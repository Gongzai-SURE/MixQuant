import time
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import gc
import json
import copy
import random
from mixq.linear import *
from mixq.quant import *
from mixq.utils.misc import *
from mixq.utils.FisherInfo import *
from .allocate import *


def calculate_bit(quantizers):
    present_bit = 0
    if len(quantizers) == 0:
        return 16
    for key in quantizers:
        present_bit += quantizers[key].bits
    # 返回平均bit位数
    present_bit = present_bit / len(quantizers)
    logger.info("present bit: {}".format(present_bit))
    return present_bit

def sorted_layer_score():
    pass

# @torch.no_grad()
def layerwise_quantize(model, dataloader, args):
    quantizers = {}

    if args.strategy == 'fisher':
        logger.info('Using fisher information strategy to quantize model.')
        layer_fisher(model, dataloader, args, quantizers)
    elif args.strategy == 'threshold':
        logger.info('Using threshold strategy to quantize model.')
        layer_threshold(model, dataloader, args, quantizers)
    elif args.strategy == 'activation':
        logger.info('Using activation strategy to quantize model.')
        layer_activation(model, dataloader, args, quantizers)
    elif args.strategy == 'random':
        logger.info('Using random strategy to quantize model.')
        layer_random(model, dataloader, args, quantizers)
    else:
        raise NotImplementedError(f"Strategy {args.strategy} is not implemented.")
    
    return quantizers

def layer_fisher(model, dataloader, args, quantizers):
    meta = args.meta
    # 获取模型层的fisher info 结果(本地加载或者新构造计算)
    if meta['fisher'] is None:
        fisher_information, modified_perplexitys, original_perplexity = evaluate_fisher_information_with_quant_perturb_sub(model, dataloader, args)
        model_name = args.model.split('/')[-1]
        time =  get_current_time()
        FI_filename = f'/root/autodl-tmp/methods/mix_quantize/model_info/{model_name}/fisher_data_{args.seqlen}_{args.nsamples}_{time}.json'
        MP_filename = f'/root/autodl-tmp/methods/mix_quantize/model_info/{model_name}/modified_perplexitys_{args.seqlen}_{args.nsamples}_{time}_{original_perplexity}.json'
        save_data(fisher_information, FI_filename)
        save_data(modified_perplexitys, MP_filename)
    else:
        fisher_information = meta['fisher']
    
    layers, pre_layers, post_layers = parsing_layers(model, meta)
        
    # 根据 Fisher 结果分配 bit 位数
    
    strategy = Greedy_allocation_list(args.wbits, args.target_bit, len(fisher_information)*len(fisher_information[0]))
    layer_bits = get_bits_list(fisher_information, strategy)
    
    # 按照layer_bits逐层量化每一层
    progress_bar = tqdm(range(len(layers)), desc="Quantizing")
    for i in progress_bar:
        logger.info(f"Layer {i} quantizing.")

        layers[i].to(args.device)
        target_layers = find_layers(layers[i])
        layer_bit = layer_bits[i]

        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(target_layers.keys())]
        
        for names in sequential:
            subset = {n: target_layers[n] for n in names}

            mixq_linear = {}
            # 初始化 Mix_Linear and quantizer类
            for name in subset:
                mixq_linear[name] = Mixq_Linear(subset[name], layer_bit[name])
                mixq_linear[name].quantizer = Quantizer(
                    layer_bit[name], perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                )
            
            for name in subset:
                W = subset[name].weight.data.clone().to(torch.float)
                W_quant = W
                W_quant,sum_frob_norm_error = mixq_linear[name].fasterquant(quantbit = layer_bit[name], groupsize=args.groupsize)
                # 修改model对应层的权重为 W_quant
                subset[name].weight.data = W_quant
                # 保存量化bit位数    
                mixq_linear[name].layer.weight.data = W_quant
                mixq_linear[name].bits = layer_bit[name]
                mixq_linear[name].free()

                del W
                del W_quant
                quantizers[f"{meta['prefix']}.{i}.{name}"] = mixq_linear[name].quantizer
                mixq_linear[name].free()
                torch.cuda.empty_cache()

        # 量化后结果计算与资源释放
        for name in list(target_layers.keys()):
            quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

        del mixq_linear
        layers[i].to('cpu') 
        torch.cuda.empty_cache()
        progress_bar.set_postfix({"layer": i})

def layer_random(layers, inps, inp_kwargs, meta, args, quantizers):
    # 随机分配 bit 位数
    progress_bar = tqdm(range(len(layers)), desc="Quantizing")
    for i in progress_bar:
        logger.info(f"Layer {i} quantizing.")

        layers[i].to(args.device)
        target_layers = find_layers(layers[i])
        layer_bit = args.wbits[random.randint(0, len(args.wbits))]
        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(target_layers.keys())]
        
        for names in sequential:
            subset = {n: target_layers[n] for n in names}

            mixq_linear = {}
            # 初始化 Mix_Linear and quantizer类
            for name in subset:
                mixq_linear[name] = Mixq_Linear(subset[name], layer_bit)
                mixq_linear[name].quantizer = Quantizer(
                    layer_bit, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                )
            
            for name in subset:
                W = subset[name].weight.data.clone().to(torch.float)
                W_quant = W
                W_quant,sum_frob_norm_error = mixq_linear[name].fasterquant(quantbit = layer_bit, groupsize=args.groupsize)
                # 修改model对应层的权重为 W_quant
                subset[name].weight.data = W_quant
                # 保存量化bit位数    
                mixq_linear[name].layer.weight.data = W_quant
                mixq_linear[name].bits = layer_bit
                mixq_linear[name].free()

                del W
                del W_quant
                quantizers[f"{meta['prefix']}.{i}.{name}"] = mixq_linear[name].quantizer
                mixq_linear[name].free()
                torch.cuda.empty_cache()

        # 量化后结果计算与资源释放
        for name in list(target_layers.keys()):
            quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

        del mixq_linear
        layers[i].to('cpu') 
        torch.cuda.empty_cache()
        progress_bar.set_postfix({"layer": i})

def layer_threshold(layers, inps, inp_kwargs, meta, args, quantizers, layer_score):
    # 存储经过该block的激活值,其中outs_original 为原始激活值， 记录每一个block量化后的激活值损失
    outs_original = torch.zeros_like(inps)
    inps_quantize = inps.clone()
    layer_score = []
    mixq_layers = args.meta['mixq_layers']
    quantizers = {}
    present_bit = 16
    logger.info(f"layer_loss_threshold composed by {args.pentalty[0]} * layer params number + {args.pentalty[1]} * layer amplitude + {args.pentalty[2]} * bit_differ.")
    
    # 按照块粒度进行敏感性研判(判断每一个块的敏感程度大小)
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(args.device)
        target_layers = find_layers(layer)
        
        # outs_quantize 为量化后激活值，与outs_original进行比较计算
        outs_quantize = torch.zeros_like(inps)
        # 计算并保存原始激活值结果
        for j in range(args.nsamples):
            outs_original[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
        
        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(target_layers.keys())]
        
        
        # 根据层的静态数据统计特征来决定层的带宽bit位数
        for names in sequential:
            subset = {n: target_layers[n] for n in names}
            #计算当前bit位数
            present_bit = calculate_bit(quantizers)

            mixq_linear = {}
            # 初始化 Mix_Linear and quantizer类
            for name in subset:
                mixq_linear[name] = Mixq_Linear(subset[name], args.wbits)
                mixq_linear[name].quantizer = Quantizer(
                    args.wbits[-1], perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                )
                mixq_linear[name].get_layer_loss_threshold(subset[name],args.pentalty,present_bit-args.target_bit)
            
            # 搜索量化空间，要求量化精度的量化损失不超过量化阈值并且尽可能小
            for name in subset:
                logger.info(f"{meta['prefix']}.{i}.{name} layer loss threshold: {mixq_linear[name].quant_loss_threshold}.")
                W = subset[name].weight.data.clone().to(torch.float)
                W_quant = W
                # 遍历可选的bit位数，并排除16bit的情况
                for bit in args.wbits:
                    if bit == 16:
                        mixq_linear[name].quantizer = Quantizer(
                            bit, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                        )   
                        continue
                    W_quant,sum_frob_norm_error = mixq_linear[name].fasterquant(quantbit=bit, groupsize=args.groupsize)
                
                    if sum_frob_norm_error < mixq_linear[name].quant_loss_threshold:
                        # 修改model对应层的权重为 W_quant
                        subset[name].weight.data = W_quant
                        # 保存量化bit位数    
                        mixq_linear[name].layer.weight.data = W_quant
                        mixq_linear[name].bits=bit
                        mixq_linear[name].free()
                        break

                del W
                del W_quant
                quantizers[f"{meta['prefix']}.{i}.{name}"] = mixq_linear[name].quantizer
                mixq_linear[name].free()
                # del temp_quan tizer
                # del frob_norm_error, sum_frob_norm_error
                torch.cuda.empty_cache()    

        # 量化后结果计算与资源释放
        for name in list(target_layers.keys()):
            quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()
            
        for j in range(args.nsamples):
            outs_quantize[j] = layer(inps_quantize[j].unsqueeze(0), **inp_kwargs)[0]

        # 记录一下状态字典
        differ_dict = {
            'block_id': i,
            'original': outs_original.cpu(),
            'quantize': outs_quantize.cpu(),
            'Average_error_ratio': ((outs_original - outs_quantize).sum()/outs_original.numel())/(outs_original.sum()/outs_original.numel()).cpu(),
            'rmse': torch.sqrt(torch.mean((outs_original-outs_quantize)**2)).cpu()
        }
        logger.info(f"\nblock{i} quantization_loss:{(outs_original - outs_quantize).sum():.4f}  \
                      \nblock{i} quantization_RMSE:{torch.sqrt(torch.mean((outs_original-outs_quantize)**2)).cpu():.4f}.")
        layer_score.append(differ_dict)

        layer_score_rank = sorted(layer_score, key=lambda x: x['rmse'])
        inps_quantize = outs_quantize.clone()

        layers[i] = layer.cpu()
        del differ_dict
        del outs_quantize
        del layer
        del mixq_linear 
        torch.cuda.empty_cache()

        inps, outs_original = outs_original, inps

def layer_activation(layers, inps, inp_kwargs, meta, args, quantizers, layer_score):
    # 存储经过该block的激活值,其中outs_original 为原始激活值，layer_score 记录每一个block量化后的激活值损失
    outs_original = torch.zeros_like(inps)
    inps_quantize = inps.clone()

    mixq_layers = args.meta['mixq_layers']
    present_bit = 16

    # 按照块粒度进行敏感性研判(判断每一个块的敏感程度大小)
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(args.device)
        target_layers = find_layers(layer)
        
        # outs_quantize 为量化后激活值，与outs_original进行比较计算
        outs_quantize = torch.zeros_like(inps)
        # 计算并保存原始激活值结果
        for j in range(args.nsamples):
            outs_original[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
        
        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(target_layers.keys())]

        # 计算量化前后actionvation的差值与bit位数变化的比值关系(Δactivation/Δbit)作为该模块每一层的量化敏感度
        for bit in args.wbits:
            for names in sequential:
                subset = {n: target_layers[n] for n in names}

                mixq_linear = {}
                # 初始化 Mix_Linear and quantizer类
                for name in subset:
                    mixq_linear[name] = Mixq_Linear(subset[name], args.wbits)
                    mixq_linear[name].quantizer = Quantizer(
                        args.wbits[-1], perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                    )
                
                for name in subset:
                    logger.info(f"{meta['prefix']}.{i}.{name} layer loss threshold: {mixq_linear[name].quant_loss_threshold}.")
                    W = subset[name].weight.data.clone().to(torch.float)
                    W_quant = W
                    # 遍历可选的bit位数，并排除16bit的情况
                    if bit == 16:
                        mixq_linear[name].quantizer = Quantizer(
                            bit, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse')
                        )   
                        continue
                    W_quant,sum_frob_norm_error = mixq_linear[name].fasterquant(quantbit=bit, groupsize=args.groupsize)
                
                    if sum_frob_norm_error < mixq_linear[name].quant_loss_threshold:
                        # 修改model对应层的权重为 W_quant
                        subset[name].weight.data = W_quant
                        # 保存量化bit位数    
                        mixq_linear[name].layer.weight.data = W_quant
                        mixq_linear[name].bits=bit
                        mixq_linear[name].free()
                        break

                    del W
                    del W_quant
                    quantizers[f"{meta['prefix']}.{i}.{name}"] = mixq_linear[name].quantizer
                    mixq_linear[name].free()
                    torch.cuda.empty_cache()

            # 量化后结果计算与资源释放
            for name in list(target_layers.keys()):
                quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()
                
            for j in range(args.nsamples):
                outs_quantize[j] = layer(inps_quantize[j].unsqueeze(0), **inp_kwargs)[0]

            # 记录一下状态字典
            differ_dict = {
                'block_id': i,
                'Average_error_ratio': ((outs_original - outs_quantize).sum()/outs_original.numel())/(outs_original.sum()/outs_original.numel()).cpu(),
                'rmse': torch.sqrt(torch.mean((outs_original-outs_quantize)**2)).cpu()
            }
            logger.info(f"\nblock{i} quantization_loss:{(outs_original - outs_quantize).sum():.4f}  \
                        \nblock{i} quantization_RMSE:{torch.sqrt(torch.mean((outs_original-outs_quantize)**2)).cpu():.4f}.")
            
            layer_score.append(differ_dict)

            layer_score_rank = sorted(layer_score, key=lambda x: x['rmse'])
            inps_quantize = outs_quantize.clone()

            layers[i] = layer.cpu()
            del differ_dict
            del outs_quantize
            del layer
            del mixq_linear 
            torch.cuda.empty_cache()

            inps, outs_original = outs_original, inps