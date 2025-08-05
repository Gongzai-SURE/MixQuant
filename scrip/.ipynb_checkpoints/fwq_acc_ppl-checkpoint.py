import subprocess

model_list = ["llama2-13b"]  #"qwen2.5-7b", "qwen2.5-14b","mistral-7b" ,"llama2-13b"
target_bits = [5,6]
quant_methods = ["awq"]
wbits = {
    "5": "4,5,6",
    "6": "5,6,7",
}

for model in model_list:
    for target_bit in target_bits:
        for quant_method in quant_methods:
            # 构建命令
            command = [
                "python",
                "/root/autodl-tmp/methods/mix_quantize/main.py",
                "--model", f"/root/autodl-tmp/models/{model}",
                "--quant_method", quant_method,
                "--target_bit", str(target_bit),
                "--strategy","fisher",
                "--load_fisher",
                "--alpha","50",
                "--allocate_strategy","rl",
                "--sameLayerReset",
                "--groupsize","64",
                "--wbits", wbits[str(target_bit)],
                # "--evaluate_ppl",
                "--reasoning"
            ]
            # 打印要执行的命令（可选）
            print("执行命令:", " ".join(command))
            
            # 执行命令
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"命令执行失败: {e}")
                # 执行下一条命令或退出
                continue

            print(f"模型 {model} 使用 {quant_method} 方法和目标位数 {target_bit} 的量化测试已完成。")