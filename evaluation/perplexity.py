import torch
import torch.nn as nn


from tqdm import tqdm
from loguru import logger
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from mixq.quant import *
from mixq.utils.misc import parsing_layers,create_position_ids
from mixq.utils.multi_gpus import split_model_across_gpus,model_multigpu
import gc

# 计算模型在特定数据集上的困惑度
@torch.no_grad()
def eval_ppl(model, dataset, dev, args):
    meta = args.meta
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.init()

    dataset = dataset.input_ids
    nsamples = dataset.numel() // args.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if args.use_multi_gpu:
        ppl = multi_gpus_inference(model, dataset, nsamples, args)
        return ppl

    layers, pre_layers, post_layers = parsing_layers(model, meta)

    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {kw:None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            for key in cache:
                if key == 'i':
                    cache['i'] += 1
                else:
                    cache[key] = kwargs[key]
            raise ValueError

    layers[0] = Catcher(layers[0])

    
    model.to(dev)
    for i in range(nsamples):
        batch = dataset[:, (i * args.seqlen):((i + 1) * args.seqlen)].to(dev)
        try:
            model(batch) 
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.to('cpu')

    layers[0] = layers[0].cpu()
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    for post_layer in post_layers:
        post_layer = post_layer.to(dev)
    
    model.lm_head = model.lm_head.to(dev)

    dataset = dataset.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if post_layer in post_layers:
            hidden_states = post_layer(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = dataset[
            :, (i * args.seqlen):((i + 1) * args.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
    logger.info(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()


def multi_gpus_inference(model, dataset, nsamples,args):
    num_gpus = torch.cuda.device_count()
    meta = args.meta
    
    # 卸载模型到多个gpu上
    model_parts = split_model_across_gpus(model, meta, num_gpus)
    for model_part in model_parts:
        model_part.eval()
        
    nlls = []
    for i in tqdm(range(nsamples)):
        inputs = dataset[:, (i * args.seqlen):((i + 1) * args.seqlen)].to("cuda:0")
        inp_kwargs = {
            "attention_mask": None,
            "position_ids": create_position_ids(args.seqlen).to(args.device)
        }
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
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = dataset[:, (i * args.seqlen):((i + 1) * args.seqlen)][:, 1:].to(f'cuda:{num_gpus - 1}')

        # 计算原始困惑度
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
    logger.info(ppl.item())
    return ppl.item()



def multi_gpus_inference0(model, input_ids, args):
    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    model_multigpu(model, gpus, args)
    meta = args.meta
    layers, _, _ = parsing_layers(model, meta)
    
    dev = model.gpus[0] if hasattr(model, 'gpus') else model.device


    input_ids = input_ids.to(dev)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i): # for memory collect
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    
    for i, layer in enumerate(layers):
        layer.register_forward_hook(clear_past(i))

    loss = nn.CrossEntropyLoss()
    tot = 0.
    torch.cuda.empty_cache()
    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
            
    with torch.no_grad():
        for i in range(input_ids.numel()):
            
            out = model(input_ids[:, i].reshape(1,-1),
                        past_key_values=cache['past'])
            sync()
            
            if i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(dev), input_ids[:, (i + 1)].to(dev)).float()
            # print(i, t)
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        
    return torch.exp(tot / (input_ids.numel() - 1)).item()