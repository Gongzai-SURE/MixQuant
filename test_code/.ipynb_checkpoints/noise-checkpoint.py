import torch

ly = torch.randn(5, 5)  # 示例张量

# 创建与 'ly' 形状相同的噪声，并将其缩放0.01倍
noise = torch.randn_like(ly) * 0.01
print(noise)

# 将噪声添加到 'ly' 中
perturbed_ly = ly + noise