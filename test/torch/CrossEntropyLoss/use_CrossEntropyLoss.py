import torch
import torch.nn as nn

# **假设有3个类别（C=3）**
logits = torch.tensor([[2.0, 1.0, 0.1],  # 这是模型的原始输出（logits）
                       [0.5, 2.0, 1.5]])

labels = torch.tensor([0, 1])  # 真实类别（第一样本类别0，第二样本类别1）

# **交叉熵损失**
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

print(f"CrossEntropyLoss: {loss.item():.4f}")



if __name__ == '__main__':
    print("CrossEntropyLoss calculate")