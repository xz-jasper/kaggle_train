import torch
import torch.nn as nn

# 定义一个简单模型
model = nn.Linear(2, 1)

# 保存 `state_dict`
torch.save(model.state_dict(), "model.pth")

# 重新加载模型
new_model = nn.Linear(2, 1)  # 需要重新定义结构
new_model.load_state_dict(torch.load("model.pth", weights_only=True))
new_model.eval()  # 切换到推理模式



if __name__ == '__main__':
    print("save param")