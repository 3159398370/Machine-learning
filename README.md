# Machine-learning
#机器学习实战-房价预测完整案例(源csdn)复刻并修改代码可用性
###原项目-https://blog.csdn.net/qq_43965708/article/details/116483085 机器学习实战——房价预测完整案例
# 房价预测项目大纲

---

## 1. 数据获取与加载

- `HOUSING_PATH` 数据集存储路径
- `fetch_housing_data()` 数据下载
  ├─ 检查本地文件存在性
  ├─ 创建数据集目录
  ├─ 下载housing.tgz
  └─ 解压并清理压缩包
- `load_housing_data()` 数据加载
  └─ 使用Pandas读取CSV

---

## 2. 数据探索分析

### 2.1 数据概览

- `head()` 查看首5行
- `info()` 结构分析
- `describe()` 统计分布

### 2.2 可视化分析

- 地理分布散点图
  ├─ 基础版（低透明度）
  └─ 优化版（人口/房价叠加）
- 收入-房价关系图

### 2.3 相关性分析

- 数值型特征相关矩阵
- 类别型特征编码（`OrdinalEncoder`）

---

## 3. 数据预处理

### 3.1 数据清洗

- `SimpleImputer` 中位数填充
- `income_cat` 分层抽样

### 3.2 特征工程

- 组合特征生成
  ├─ rooms_per_household
  └─ population_per_household
- `CombinedAttributesAdder` 自定义转换器

### 3.3 流水线处理

- `num_pipeline` 数值处理
  ├─ 缺失值填充
  ├─ 组合特征添加
  └─ 标准化（`StandardScaler`）
- `full_pipeline` 全量处理
  ├─ 数值列处理
  └─ 类别列独热编码

---

## 4. 模型训练与评估

### 4.1 基准模型

- 线性回归（`LinearRegression`）
- 决策树（`DecisionTreeRegressor`）

### 4.2 高级模型

- 随机森林（`RandomForestRegressor`）

### 4.3 交叉验证

- 10折验证（`cross_val_score`）
- RMSE评估指标

---

## 5. 模型优化

### 5.1 超参数调优

- 网格搜索（`GridSearchCV`）
- 随机搜索（`RandomizedSearchCV`）

### 5.2 特征分析

- `feature_importances_` 重要性排序

---

## 6. 最终评估

- 测试集RMSE计算
- 95%置信区间（`stats.t.interval`）
