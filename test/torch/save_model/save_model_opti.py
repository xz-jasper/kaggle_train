import torch
import torch.nn as nn

# 定义一个简单模型
model = nn.Linear(2, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 保存模型和优化器
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "checkpoint.pth")

# 重新加载
checkpoint = torch.load("checkpoint.pth",weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == '__main__':
    print("model")