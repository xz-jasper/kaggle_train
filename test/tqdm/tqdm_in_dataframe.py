import pandas as pd
from tqdm import tqdm

tqdm.pandas()  # 让 tqdm 兼容 pandas
df = pd.DataFrame({'values': range(10)})

# 在 apply() 里使用 tqdm
df['values'] = df['values'].progress_apply(lambda x: x**2)


if __name__ == '__main__':
    print("tqdm in dataframe")