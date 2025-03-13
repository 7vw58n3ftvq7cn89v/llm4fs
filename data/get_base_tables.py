from ucimlrepo import fetch_ucirepo
import pandas as pd
import os
from pathlib import Path

# 创建base_tables文件夹（如果不存在）
os.makedirs('base_tables', exist_ok=True)

# 获取数据集
recipe_reviews = fetch_ucirepo(id=911) 
  
# 获取特征和目标值
X = recipe_reviews.data.features 
y = recipe_reviews.data.targets 

# 合并特征和目标值
df = pd.concat([X, y], axis=1)

# 保存为CSV文件
output_path = Path('base_tables/recipe_reviews.csv')
df.to_csv(output_path, index=False)
print(f'数据集已保存至: {output_path}')
print(f'数据集形状: {df.shape}')
print(df.head())