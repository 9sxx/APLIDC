import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../data/Credit Card Fraud Detection/creditcard.csv"

# 读取数据
data = pd.read_csv(path_train)

# 删除缺失值超过50%的列
threshold = 0.5 * len(data)
data = data.loc[:, data.isnull().mean() < threshold]

# 使用剩余特征中的众数填充缺失值
for col in data.columns:
    if data[col].isnull().any():
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

# 标签编码非数值型特征
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
