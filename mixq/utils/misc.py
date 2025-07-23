import torch
import torch.nn as nn
import json
from datetime import datetime
from loguru import logger
import numpy as np
import os
from pathlib import Path
from datetime import datetime

#from eetq.modules.qlinear import EetqLinear

layer_list = ['q','k','v','qkv','o','out','dense','fc1','fc2','up','gate','down']

def find_layers(module, layers=[nn.Linear], name=''):
    # logger.info(type(module))
    if type(module) in layers:
        return {name: module}
        # return {name: module.weight}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def standardize_keys(input_dict):
    keys_to_keep = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ]
    standardized_dict = {}

    counter = 0
    
    # 遍历原字典
    for key, value in input_dict.items():
        if len(key.split('.'))>=4:
            core_key = key.split('.')[2] + '.' + key.split('.')[3]
        else:
            core_key = key.split('.')[1]
        if core_key in keys_to_keep:
            if core_key == keys_to_keep[0]:
                standardized_dict[counter] = {}
                standardized_dict[counter][core_key] = value.item()
            elif core_key == keys_to_keep[-1]:
                standardized_dict[counter][core_key] = value.item()
                counter += 1
            else:
                standardized_dict[counter][core_key] = value.item()
    
    return standardized_dict

def parsing_layers(model, meta):
    from collections import OrderedDict
    results = OrderedDict({'layers':None,'pre_layers':[],'post_layers':[]})
    for data_name in results.keys():
        data = meta[data_name]
        if isinstance(data, list):
            for data_ in data:
                root_attr = model
                attrs = data_.split('.')[1:]
                for attr in attrs:
                    root_attr = getattr(root_attr,attr)
                results[data_name].append(root_attr)
        else: # str
            root_attr = model
            attrs = data.split('.')[1:]
            for attr in attrs:
                root_attr = getattr(root_attr,attr)
            results[data_name] = root_attr

    return results.values()

def interpret_dtype(dtype):
    if isinstance(dtype, str):
        if dtype in ['float16', 'fp16']:
            return torch.half
        elif dtype in ['bfloat16', 'bf16']:
            return torch.bfloat16
        elif dtype in ['float', 'float32', 'fp32', 'fp']:
            return torch.float32
        elif dtype == 'auto':
            return dtype
        else:
            raise ValueError
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif dtype is None:
        return 'auto' # for torch_dtype in AutoModelLM.from_pretrained
    else:
        raise ValueError

def seed_all(seed):
    import random
    import os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_args(logger, args):
    important_args = {
        "Model Settings": {
            "model": args.model,
            "groupsize": args.groupsize
        },
        "Dataset Settings": {
            "val_dataset": args.dataset,
            "dataset_dir": args.dataset_dir,
            "nsamples": args.nsamples
        },
        "Quantization Strategy": {
            "quant_method": args.quant_method,
            "wbits": args.wbits,
            "target_bit": args.target_bit,
            "strategy": args.strategy,
            "allocate_strategy": args.allocate_strategy,
            "sameLayerReset": args.sameLayerReset,
            "alpha": args.alpha,
            "top_r": args.top_r,
            "test_bit": args.test_bit
        },
        "Runtime Settings": {
            "device": args.device,
            "seed": args.seed,
        },
        "Evaluation": {
            "no_eval": args.no_eval,
            "reasoning": args.reasoning
        }
    }

    # 构建表格字符串
    table = "\n"+"="*50 + "\n"
    table += "Experiment Configuration\n"
    table += "="*50 + "\n"
    
    for category, params in important_args.items():
        table += f"\n[{category}]\n"
        table += "-"*50 + "\n"
        for key, value in params.items():
            table += f"{key:>20}: {str(value):<30}\n"
    
    table += "="*50 + "\n"
    
    logger.info(table)

# check args
def processing_arguments(args):
    import json
    if args.logfile:
        logger.add(args.logfile, encoding="utf-8")
    else: # logging.在指定目录下生成日志文件
        model_name = args.model.split("/")[-1]
        log_path = f'/root/autodl-tmp/methods/mix_quantize/logs/{model_name}'
        Path(log_path).mkdir(parents=True, exist_ok=True)
        if args.original:
            logger_file_name = f'Original_{get_current_time()}'
        else:
            if args.strategy:
                logger_file_name = f'{args.strategy}_{args.target_bit}_{args.allocate_strategy}_{args.alpha}_{args.quant_method}_{get_current_time()}'
            elif args.strategy is None and args.quant_method:
                logger_file_name = f'{args.quant_method}_{args.target_bit}_{get_current_time()}'
            else:
                logger_file_name = f'Test_{get_current_time()}'
        logger.add(f"{log_path}/{logger_file_name}.log", encoding="utf-8")
        log_args(logger, args)

    if args.device is None:
        if torch.cuda.is_available():
            args.device = torch.device('cuda')
            logger.info(f"Number of CUDA devices available: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 1:
                args.use_multi_gpu = True
            else:
                args.use_multi_gpu = False
        else:
            args.device = torch.device('cpu')
            logger.info(f"Process will be running on CPU.")
    else:
        args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.wbits:
        assert len(args.wbits) >= 2, 'Please give two or more values for wbits.'
        args.wbits = sorted(args.wbits)
    else:
        AssertionError('Please give wbits.')
    
    if args.save_path:
        if not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))

        # if not (args.save_path.endswith('.pth') or args.save_path.endswith('.pt')):
        #     raise ValueError("The save path '--args.save' must end in .pth or .pt.")
    
    
    with open('/root/autodl-tmp/methods/mix_quantize/model_config.json') as f:
        metas = json.load(f)

    args.dtype = interpret_dtype(args.dtype)
    
    # model config
    if 'opt' in args.model:
        meta = metas['opt']
        if '350m' in args.model:
            meta['pre_layers'].append('model.model.decoder.project_in')
            meta['post_layers'].append('model.model.decoder.project_out')
        else:
            meta['post_layers'].append('model.model.decoder.final_layer_norm')
    elif 'llama' in args.model or 'vicuna' in args.model:
        meta = metas['llama']
    elif 'bloom' in args.model:
        meta = metas['bloom']
    elif 'falcon' in args.model:
        meta = metas['falcon']
        args.trust_remote_code = True
        if args.percdamp < 1.0:
            logger.info(f"In the falcon model, change --percdamp from {args.percdamp} to 1.0 for numerical stability.")
            args.percdamp = 1.0
    elif 'qwen' in args.model:
        meta = metas['qwen']
    else:
        raise NotImplementedError(f"{args.model} model is not implemented.")
    
    
    map_layer = meta['map_layer']
    layers_mixq = {l:False for l in map_layer.values()}
    if args.layers is None: # apply mixq on all layers
        for l in layers_mixq:
            layers_mixq[l] = True
    else:
        for l in args.layers:
            if l in map_layer:
                layers_mixq[map_layer[l]] = True
            else:
                raise ValueError(f"{args.model} model doesn't have \'{l}\' layer. available layers : {list(map_layer.keys())}")
    for l in layers_mixq:
        if not layers_mixq[l]:
            meta['ratios'][l] = 0.0

    if args.load_fisher or args.load_ppl:
        model_name = args.model.split('/')[-1]
        meta['fisher'] = None
        meta['ppl'] = None
        try:
            files = [
                ('fisher', find_latest_file_by_keyword(f'/root/autodl-tmp/methods/mix_quantize/model_info/{model_name}', 'fisher_data')),
                ('ppl', find_latest_file_by_keyword(f'/root/autodl-tmp/methods/mix_quantize/model_info/{model_name}', 'modified_perplexities'))
            ]
            for data_type, file_path in files:
                if (data_type == 'fisher' and args.load_fisher) or (data_type == 'ppl' and args.load_ppl):
                    if file_path:
                        logger.info(f"Loading {data_type} data from {file_path}")
                        with open(file_path) as f:
                            json_data = json.load(f)
                            data = []
                            for block in json_data.values():
                                for value in block.values():
                                    if isinstance(value, float):
                                        data.append(value)
                                    elif isinstance(value, dict):
                                        data.extend(value.values())
                            meta[data_type] = np.array(data)
        except Exception as e:
            logger.warning(f"Error loading data: {e}")
    else:
        meta['fisher'] = None
        meta['ppl'] = None

    
    meta['mixq_layers'] = layers_mixq

    return meta

def get_current_time() -> str:
    current_time = datetime.now()
    return current_time.strftime("%Y-%m-%d-%H-%M-%S")

def find_latest_file_by_keyword(folder_path, keyword):
    # 获取文件夹中所有包含关键词的文件
    files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and keyword in f
    ]

    if not files:
        return None

    def extract_timestamp(filename):
        try:
            parts = os.path.basename(filename).split('_')
            date_time_part = parts[-1].split('.')[0]  # 去掉.json
            return datetime.strptime(date_time_part, "%Y-%m-%d-%H-%M-%S")
        except (IndexError, ValueError):
            return datetime.min

    latest_file = max(files, key=extract_timestamp)

    if extract_timestamp(latest_file) == datetime.min:
        return None
    else:
        return latest_file

def save_data(data, filename):
    """
    Save one-dimensional or multi-dimensional data to a file.

    Parameters:
        data: The data to save, can be 1D or multi-dimensional (list, dict, nested structures).
        filename: The name of the file to save the data.
    """
    def make_serializable(item):
        """
        Recursively convert items to a JSON-serializable format.
        """
        if isinstance(item, dict):
            return {key: make_serializable(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [make_serializable(element) for element in item]
        elif hasattr(item, 'item'):  # Handles PyTorch tensors, NumPy scalars, etc.
            return item.item()
        else:
            return item

    try:
        # Convert data to a JSON-serializable format
        serializable_data = make_serializable(data)

        # Save the serializable data to the file
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=4)

        logger.info(f"Data saved in {filename}.")
    except Exception as e:
        logger.error(f"Failed to save data to {filename}: {e}")
        raise


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