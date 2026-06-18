import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.feature_extraction import DictVectorizer

df_jan2023=pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
print(df_jan2023.head())

#Q1
print(len(df_jan2023.columns))

#Q2
# 1. 计算行程时长（分钟）
df_jan2023['duration'] = (df_jan2023['tpep_dropoff_datetime'] - df_jan2023['tpep_pickup_datetime']).dt.total_seconds() / 60
std_dev = df_jan2023['duration'].std()
print(f"1月份行程时长的标准差为: {std_dev}")

#Q3
# 2. 筛选出符合条件的记录 (1 <= duration <= 60)
df_filtered = df_jan2023[(df_jan2023['duration'] >= 1) & (df_jan2023['duration'] <= 60)]

# 3. 计算剩余记录的占比
fraction = len(df_filtered) / len(df_jan2023)

print(f"剩余记录的比例为: {fraction:.2%}")

# Q4
# 4. 显式转为字符串，并确保是 Categorical 特征
features = ['PULocationID', 'DOLocationID']
df_filtered[features] = df_filtered[features].astype(str)

# 5. 转为字典列表
train_dicts = df_filtered[features].to_dict(orient='records')

# 6. Fit 向量化器
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

# 7. 查看矩阵维度
print(f"训练矩阵的形状: {X_train.shape}")

# 8. 提取目标变量
target = 'duration'
y_train = df_filtered[target].values
print(f"目标变量的形状: {y_train.shape}")

# 1. 下载并读取 2 月份的数据
# 假设 URL 格式遵循纽约出租车数据的通用规律
df_val = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')

# 2. 预处理 2 月份数据 (必须与 1 月份的预处理步骤完全一致)
# 计算 duration
df_val['duration'] = (df_val['tpep_dropoff_datetime'] - df_val['tpep_pickup_datetime']).dt.total_seconds() / 60

# 过滤异常值 (保持 1 到 60 分钟)
df_val = df_val[(df_val['duration'] >= 1) & (df_val['duration'] <= 60)].copy()

# 转为字符串并提取特征
features = ['PULocationID', 'DOLocationID']
df_val[features] = df_val[features].astype(str)

# 3. 转化为字典列表
val_dicts = df_val[features].to_dict(orient='records')

# 4. 使用 Q4 训练好的 dv 对数据进行转换 (注意：这里只用 transform)
X_val = dv.transform(val_dicts)

# 5. 生成目标变量
y_val = df_val['duration'].values