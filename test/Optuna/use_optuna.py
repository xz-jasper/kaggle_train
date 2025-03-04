import optuna

# 目标函数（优化目标）
def objective(trial):
    x = trial.suggest_float("x", -10, 10)  # 搜索范围 [-10, 10]
    return (x - 2) ** 2  # 目标是让 f(x) = (x-2)^2 最小

# 创建 Optuna study 并运行优化
study = optuna.create_study(direction="minimize")  # 目标是最小化函数值
study.optimize(objective, n_trials=100)  # 运行 100 次尝试

# 输出最佳参数
print(f"Best parameter: {study.best_params}, Best value: {study.best_value}")


if __name__ == '__main__':
    print("use optuna")