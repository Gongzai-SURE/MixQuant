import subprocess

model_list = ["llama2-13b"]  # "llama2-7b",
target_bits = [5,6]
quant_methods = ["nearest", "gptq", "awq"]

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
                "--reasoning"
            ]
            
            # 打印要执行的命令（可选）
            print("执行命令:", " ".join(command))
            
            # 执行命令
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"命令执行失败: {e}")
                # 可以选择继续执行下一条命令或退出
                continue

            print(f"模型 {model} 使用 {quant_method} 方法和目标位数 {target_bit} 的量化测试已完成。")