import torch
import torch.nn as nn

# 定义一个简单模型
model = nn.Linear(2, 1)

# 保存 `state_dict`
torch.save(model, "full_model.pth")  # 保存整个模型
loaded_model = torch.load("full_model.pth",weights_only=False)  # 直接加载
loaded_model.eval()



if __name__ == '__main__':
    print("save param")