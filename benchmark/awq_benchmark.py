from loguru import logger
import torch
import torch.nn as nn
from tqdm import tqdm
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
# from ..evaluation.perplexity import eval_ppl
import random
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mix_quantize.awq.quantize.quantizer import AwqQuantizer


seed = random.seed(3)
model_path = '/root/autodl-tmp/models/llama2-7b'
quant_path = '/root/autodl-tmp/models/llama2-7b-awq'

def get_wikitext2(nsamples=32, seed=0, seqlen=1024, local_dir="/root/autodl-tmp/datasets/wikitext2"):
    # 加载数据集
    if local_dir is not None:
        dataset = load_from_disk(local_dir)["train"]
    else:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    random.seed(seed)
    
    # 预处理：过滤掉过短的文本并收集足够长的文本块
    valid_texts = []
    total_length = 0
    
    # 先收集所有足够长的文本
    for text in dataset['text']:
        if len(text) >= 2*seqlen + 1:
            valid_texts.append(text)
            total_length += len(text)
    
    # 如果有效文本不足，可以重复使用
    if len(valid_texts) < nsamples:
        repeat_times = (nsamples // len(valid_texts)) + 1
        valid_texts = valid_texts * repeat_times
    
    # 随机选择起始点
    trainloader = []
    for i in range(nsamples):
        text = valid_texts[i % len(valid_texts)]
        n = len(text)
        
        # 随机选择起始点
        start = random.randint(0, n - seqlen - 1)
        end = start + seqlen
        inp = text[start:end]
        
        # 目标文本
        target_end = min(end + seqlen, n)
        tar = text[end:target_end]
        
        trainloader.append(f"{inp}\n{tar}")
    
    return trainloader

# inferrence
@torch.no_grad()
def evaluate_perplexity(model, tokenizer,dev):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_from_disk("/root/autodl-tmp/datasets/wikitext2")["test"]
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids

    seqlen = 1024
    model.to(dev)
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index]
            with torch.no_grad():
                logits = model(batch.to(dev)).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:].to(dev)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()

if __name__ == "__main__":
    model_name = model_path.split("/")[-1]
    logger.add(f"mix_quantize/benchmark/awq_{model_name}", encoding="utf-8")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading {model_name} model ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # --------------Load model----------------
    awq_model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )

    model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype='auto',
            device_map='cpu'
        )

    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "gemm" }
    
    logger.info(f"Quantization config: {quant_config}")

    # 测试原精度
    # start_time = time.time()
    # logger.info("Start evaluating original model ...")
    # ppl = evaluate_perplexity(awq_model.model, tokenizer,dev)
    # logger.info(f"Original model Perplexity: {ppl},using time: {time.time() - start_time:.2f} seconds")

    train_dataset = get_wikitext2(nsamples=32, seed=0, seqlen=1024, local_dir="/root/autodl-tmp/datasets/wikitext2")
    
    # -------------Quantize-----------------------

    logger.info("Start quantization ...")
    start_time = time.time()

    # awq_model.quantize(tokenizer,
    #            quant_config=quant_config,
    #            calib_data=train_dataset,
    #            n_parallel_calib_samples=1,
    #            max_calib_samples=128,
    #            max_calib_seq_len=4000,
    #            apply_clip=False)
    

    awq_quant = AwqQuantizer(awq_model,
                             model,
                             tokenizer,
                             quant_config["w_bit"], 
                             quant_config["q_group_size"], 
                             quant_config["zero_point"], 
                             quant_config["version"],
                             calib_data="wikitext2",
                             apply_clip = False
                            )
    awq_quant.quantize()

    
    logger.info(f"Quantization time: {time.time() - start_time:.2f} seconds")
    # --------------evaluate perplexity--------------
    # awq_model.save_quantized(quant_path) 
    # tokenizer.save_pretrained(quant_path)
    # start_time = time.time()
    # model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
    # logger.info("Start evaluating quantized model ...")
    ppl = evaluate_perplexity(awq_model.model, tokenizer,dev)
    # logger.info(f"AWQ quantized model Perplexity: {ppl}, using time: {time.time() - start_time:.2f} seconds")

