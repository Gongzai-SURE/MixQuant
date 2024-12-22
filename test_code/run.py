from transformers import AutoModelForCausalLM,EetqConfig,AutoTokenizer
import argparse
import time
import torch
from EETQ import quant_weights, preprocess_weights, w8_a16_gemm

import os
#设置GPU个数
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def quantize_and_preprocess_weights(weight, scales=None):
    data_type = weight.dtype
    int8_weight = torch.t(weight).contiguous().cpu()
    if data_type == torch.int8:
        assert scales is not None   # need scales for real quantization
        int8_weight = preprocess_weights(int8_weight)
    elif data_type == torch.float16:
        int8_weight, scales = quant_weights(int8_weight, torch.int8, False)
    else:
        raise ValueError("Unsupported data type: {}".format(data_type))
    return int8_weight, scales



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,default="/model/llama3-8b-eetq",
        help='hugging face model to load'
    )

    args = parser.parse_args()

    weight =  torch.rand(256, 1024).to(torch.float16)
    weight_quant, scales = quantize_and_preprocess_weights(weight)

    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path)
    quantization_config = EetqConfig("int8")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", quantization_config=quantization_config)
    t_cost = time.time() - t0
    print('EETQ model load time: %.2f s' % t_cost)
    # 尝试模型推理
    domo_prompt = "How to be a good person?"
    inputs = tokenizer(domo_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #生成长度为1000的输出，并统计生成时间
    t0 = time.time()
    outputs = model.generate(**inputs,min_length=300,max_length=1000,do_sample=True,top_k=50,top_p=0.95,\
                             num_return_sequences=1,temperature=0.01,no_repeat_ngram_size=20,early_stopping=True)
    #计算每秒生成的token数量
    t_cost = time.time() - t0
    avg_token_per_sec = len(outputs[0]) / t_cost
    print(tokenizer.decode(outputs[0], skip_special_tokens=True)+"\n"+ "Time cost: %.2f s" % t_cost + \
          "\n" + "Avg token per sec: %.2f" % avg_token_per_sec)
    print("模型推理成功")
