import torch
import torch.nn as nn
from typing import Union,Optional
from transformers import AutoModelForCausalLM,EetqConfig
from types import SimpleNamespace
from collections import OrderedDict
import os
import sys
sys.path.append('../../')
from mixq.utils.misc import find_layers
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*do_sample.*")

def get_hfmodel(model_name_or_path: str,
                dtype='auto',
                quantilized = False,
                device_map='cpu',
                trust_remote_code=False,
                ):
    # for faster model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if quantilized:
        quantization_config = EetqConfig("int8")
        model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code, )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=dtype,
            device_map=device_map, 
            trust_remote_code=trust_remote_code, 
        )
    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    return model.half()


def move(model):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            from accelerate import infer_auto_device_map,dispatch_model
            max_mem = {i: "15GiB" for i in range(torch.cuda.device_count())}
            device_map = infer_auto_device_map(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],  # 如果是 LLaMA
            )
            model = dispatch_model(model, device_map=device_map)

            # 获取模型首个参数的 device（generate 时需要）
            first_device = next(model.parameters()).device
        else:
            model = model.to("cuda:0")
            first_device = torch.device("cuda:0")
    else:
        model = model.to("cpu")
        first_device = torch.device("cpu")

    return model, first_device

def load_model(model_name_or_path,
               checkpoint_path,
               faster: Optional[bool] = True,
               device: Optional[Union[int, str, torch.device]] = 'cuda:0',
               cpu_load: Optional[bool] = True,
               ):
    if not isinstance(device, torch.device) and device not in ['auto', 'cpu']:
        device = torch.device(device)
    device_map = 'cpu' if cpu_load else device
    ckpt = torch.load(checkpoint_path)
    dtype = ckpt['dtype']
    wbits = ckpt['bits']
    model = get_hfmodel(model_name_or_path,
                        dtype=dtype,
                        device_map=device_map)
    
    if ckpt['packing']:
        print(f"Loading packed model {checkpoint_path} ....")
        
        n_out_dict = ckpt['n_out_dict']
        make_quant(model, n_out_dict, wbits)
        
        # support old format
        for n, v in ckpt['model_state_dict'].items():
            if n.endswith('oweight') and v.shape[0] > v.shape[1]:
                ckpt['model_state_dict'][n] = v.t().contiguous()
                
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        qlayers = find_layers(model, [QuantLinear])
        for name in qlayers:
            qlayers[name].set_kernel(faster)
    else:
        print(f"Loading fake quantized model {checkpoint_path} ....")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
            
    if cpu_load and device not in ['auto', 'cpu']:
        model = model.to(device)
    
    del ckpt
    import gc; gc.collect()
    torch.cuda.empty_cache()
    
    print("Done.")
    return model
    
def save_model(model, 
               quantizers,
               save_path,
               packing:bool,
               fake:bool):
    
    def make_state_dict(model, quantizers):
        model_state_dict = OrderedDict()
        for name, module in model.named_modules():
            if name in quantizers:
                state_dict = module.state_dict()
                for key in state_dict.keys():
                    model_state_dict[name+'.'+key] = state_dict[key]

        return model_state_dict
    
    dtype = model.dtype
    wbits = list(quantizers.values())[0].bits
    
    if fake:
        ckpt_path = save_path.replace('.pt', '_fake.pt')
        # model_state_dict = make_state_dict(model, quantizers)
        model_state_dict = model.state_dict()
        out_ids_dict = {name : quantizers[name].out_ids for name in quantizers}
        
        torch.save({
            'model_state_dict': model_state_dict,
            'out_ids_dict': out_ids_dict,
            'packing': False,
            'dtype' : dtype,
            'bits' : wbits,
            }, ckpt_path)

        print(f"fake quantized model is saved to {ckpt_path}")

    if packing:
        assert wbits in [3, 4], f"{wbits}bits is not supported."
        
        n_out_dict = {n: SimpleNamespace(n_out=quantizers[n].n_out) for n in quantizers}

        model_state_dict = make_state_dict(model, quantizers)
        model_state_dict = model.state_dict()
        
        torch.save({
            'model_state_dict': model_state_dict,
            'n_out_dict': n_out_dict,
            'packing': True,
            'dtype' : dtype,
            'bits' : wbits, 
            }, save_path)
        print(f"{wbits}bit quantized packing model is saved to {save_path}")
