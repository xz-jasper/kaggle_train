from tqdm import tqdm
import time

for i in tqdm(range(10)):
    time.sleep(0.5)  # 模拟耗时操作


if __name__ == '__main__':
    print("usr tqdm")