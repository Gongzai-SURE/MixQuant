import torch
import torch.nn as nn

from tqdm import tqdm
from loguru import logger
import sys
sys.path.append('/root/autodl-tmp/methods/')
from mix_quantize.mixq.quant import *
from mix_quantize.mixq.utils.misc import *

# 计算模型在特定数据集上的困惑度
@torch.no_grad()
def eval_ppl(model, dataset, dev, seqlen, args):
    meta = args.meta
    logger.info('Evaluating ...')

    dataset = dataset.input_ids
    nsamples = dataset.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers, pre_layers, post_layers = parsing_layers(model, meta)

    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
        batch = dataset[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
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
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

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
            :, (i * seqlen):((i + 1) * seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    logger.info(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()