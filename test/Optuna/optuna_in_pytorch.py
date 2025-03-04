import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import optuna


# **1. 定义神经网络**
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# **2. 目标函数（Optuna 会调用这个函数进行优化）**
def objective(trial):
    # **搜索超参数**
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-1)  # 学习率
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)  # 批量大小

    # **3. 数据加载**
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # **4. 初始化模型、损失函数、优化器**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # **5. 训练模型**
    model.train()
    for epoch in range(3):  # 训练 3 轮（可以调大）
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # **6. 返回损失值，Optuna 以最小化它为目标**
    return total_loss / len(train_loader)


# **7. 运行 Optuna 进行超参数优化**
study = optuna.create_study(direction="minimize")  # 目标是最小化损失
study.optimize(objective, n_trials=10)  # 进行 10 次搜索

# **8. 输出最佳超参数**
print("Best hyperparameters:", study.best_params)


if __name__ == '__main__':
    print("optuna in pytorch")