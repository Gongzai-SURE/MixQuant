import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
from ..utils import NativeScalerWithGradNormCount
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    model,
    args,
    dataloader,
    act_scales,
    act_shifts,
    allocation=None,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    dev = args.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.model.split('/')[-1].lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.model.split('/')[-1].lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.model.split('/')[-1].lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.model.split('/')[-1].lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    model.to(dev)
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    model.to('cpu')
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.model.split('/')[-1].lower() or "mixtral" in args.model.split('/')[-1].lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.model.split('/')[-1].lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(1,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    # if args.resume:
    #     omni_parameters = torch.load(args.resume)
    # else:
    # omni_parameters = {}

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.model.split('/')[-1].lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for j in range(args.nsamples):
                    fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        # if args.aug_loss:
                        #     fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = False
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
                                
        # if args.resume:
        #     qlayer.load_state_dict(omni_parameters[i], strict=False)
        
        with torch.no_grad():
            qlayer.float()      # required for AMP training
        # create optimizer
        optimizer = torch.optim.AdamW(
            [{"params":let_parameters(qlayer, use_shift),"lr":5e-3}, {"params":lwc_parameters(qlayer),"lr":1e-2}],weight_decay=0)
        loss_scaler = NativeScalerWithGradNormCount()
        
        for epochs in range(5):
            loss_list = []
            norm_list = []
            for j in range(args.nsamples//args.batch_size):    
                index = j * args.batch_size
                # obtain output of quantization model
                with traincast():
                    smooth_and_quant_temporary(qlayer, args, is_llama)
                    quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                    loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                    # if args.aug_loss:
                    #     loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                    
                loss_list.append(loss.detach().cpu())
                optimizer.zero_grad()
                norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                norm_list.append(norm.data)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} ")
        clear_temp_variable(qlayer)
        del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        
        # update input of quantization model
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            with traincast():
                for j in range(args.nsamples):
                    quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        register_scales_and_zeros(qlayer)
        layers[i] = qlayer.to("cpu")
        # omni_parameters[i] = omni_state_dict(qlayer)
        # torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))   
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    # del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

