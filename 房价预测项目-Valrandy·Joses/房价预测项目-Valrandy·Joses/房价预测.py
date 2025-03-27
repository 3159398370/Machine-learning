import urllib.request
import os
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.expand_frame_repr', False)  # 禁止自动换行
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
# 常量定义
HOUSING_PATH = Path("datasets/housing")
HOUSING_TGZ = HOUSING_PATH / "housing.tgz"
HOUSING_CSV = HOUSING_PATH / "housing.csv"
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"

def fetch_housing_data():
    # 如果CSV文件已存在，直接跳过
    if HOUSING_CSV.exists():
        print("数据文件已存在，跳过下载")
        return

    # 创建目录（自动处理路径）
    HOUSING_PATH.mkdir(parents=True, exist_ok=True)

    # 下载压缩包
    print("开始下载数据集...")
    urllib.request.urlretrieve(HOUSING_URL, HOUSING_TGZ)

    # 解压文件
    with tarfile.open(HOUSING_TGZ) as tgz:
        tgz.extractall(path=HOUSING_PATH)

    HOUSING_TGZ.unlink()

# 执行数据获取
fetch_housing_data()
# 定义一个函数，用于加载房价数据
def load_housing_data(housing_path=HOUSING_PATH):
    # 将数据路径与文件名拼接
    csv_path = os.path.join(housing_path, "housing.csv")
    # 返回读取的csv文件
    return pd.read_csv(csv_path)

housing = load_housing_data()

# 打印数据集的前5行
print(housing.head())
print("*"*100)
print(housing.head(5))

#使用 info() 获取数据集的简单描述。包括总行数、每个属性的类型和非空值的数量等。
print( housing.info())

#使用 describe() 获取数值属性的描述，注意统计时的空值会被忽略。
print(housing.describe())
print("*"*100)
import matplotlib.pyplot as plt
os.makedirs("./images/end_to_end_project/", exist_ok=True)
# 绘制直方图
print(housing.hist(bins=50, figsize=(20,15)))
# 保存图片
plt.savefig("./images/end_to_end_project/attribute_histogram_plots.png", dpi=300)
print("*"*100)
#2. 划分测试集

# 根据income_cat数据绘制直方图

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# 根据income_cat数据绘制直方图
print(housing["income_cat"].hist())
plt.savefig("./images/end_to_end_project/income_distribution.png", dpi=300, bbox_inches='tight')
# 导入StratifiedShuffleSplit模块
#StratifiedShuffleSplit是Scikit-learn库中的一个模块，用于将数据集划分为训练集和测试集，
# 同时保持每个子集的类分布与原始数据集相同。这对于分类问题非常重要，
# 因为如果测试集的类分布与原始数据集不同，那么模型的性能评估可能会不准确
from sklearn.model_selection import StratifiedShuffleSplit

# 创建StratifiedShuffleSplit对象，设置n_splits为1，test_size为0.2，random_state为42
s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# 使用StratifiedShuffleSplit对象对housing和housing["income_cat"]进行分割，得到train_index和test_index
for train_index, test_index in s.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 对strat_train_set和strat_test_set进行操作，删除income_cat列
for dataset in (strat_train_set, strat_test_set):
    dataset.drop("income_cat", axis=1, inplace=True)
# 复制strat_train_set到housing_train
housing_train = strat_train_set.copy()
print(housing_train)
#3. 可视化获取更多信息
# 显示中文
plt.rcParams['font.family'] = 'SimHei'
# 显示负号
plt.rcParams['axes.unicode_minus'] = False

housing_train.plot(kind="scatter", x="longitude", y="latitude")
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig("./images/end_to_end_project/bad_visualization_plot.png", dpi=300)
housing_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig("./images/end_to_end_project/better_visualization_plot.png", dpi=300)
#为了更加图像能够凸显更多的信息，将每个区域的人口数量作为图中每个圆的半径大小，房价中位数表示圆的颜色，使用预定义颜色表 "jet" ，颜色由（低房价）到红（高房价）
housing_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3,
                   s=housing_train["population"]/50, label="population", figsize=(10,7),
                   c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                   sharex=False)
plt.legend()
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig("./images/end_to_end_project/housing_prices_scatterplot.png", dpi=300)
#4. 寻找相关性

print("\n数据列类型:")
print(housing_train.dtypes)

# 显示非数值列
non_numeric = housing_train.select_dtypes(exclude=[np.number]).columns
print(f"\n非数值型列: {list(non_numeric)}")

# 对分类变量进行编码
if len(non_numeric) > 0:
    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder()
    housing_encoded = housing_train.copy()
    housing_encoded[non_numeric] = encoder.fit_transform(housing_train[non_numeric])

    # 使用编码后的数据计算全量相关性
    full_corr = housing_encoded.corr()
    print("\n包含编码分类特征的相关性矩阵:")
    print(full_corr)

# 此处应使用编码后的数据框
corr_matrix = housing_encoded.corr()
print(corr_matrix)
#具体看每个属性与房价中位数的相关性。
# 计算房价中位数与每个属性的相关性
corr_series = corr_matrix["median_house_value"].sort_values(ascending=False)
# 打印每个属性与房价中位数的相关性
print(f"每个属性与房价中位数的相关性为:\n{corr_series}")
#使用 pandas 的 scatter_matrix 函数，他会绘制出每个数值属性相对于其他数值属性的相关性
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing_train[attributes], figsize=(12, 8))
plt.savefig("./images/end_to_end_project/scatter_matrix_plot.png", dpi=300)
#与房价中位数最相关的属性是收入中位数，我们将上图中的收入中位数散点图单独拿出来
housing_train.plot(kind="scatter", x="median_income", y="median_house_value",
                   alpha=0.2)
plt.axis([0, 16, 0, 550000])
plt.savefig("./images/end_to_end_project/income_vs_house_value_scatterplot.png", dpi=300)
#5. 属性组合
#在把训练集投入模型之前，我们应该尝试各种属性的组合，例如相比于一个区域的房间总数 （total_rooms） 与家庭数量 （households） 、
# 我们可能更需要的是单个家庭的房间数量 （rooms_per_household） 。下面尝试创建一些新的组合属性
housing_train["rooms_per_household"] = housing_train["total_rooms"]/housing_train["households"]
housing_train["bedrooms_per_room"] = housing_train["total_bedrooms"]/housing_train["total_rooms"]
housing_train["population_per_household"]=housing_train["population"]/housing_train["households"]

# 计算相关矩阵
print("\n数据列类型:")
print(housing_train.dtypes)

# 识别非数值列
non_numeric = housing_train.select_dtypes(exclude=[np.number]).columns
print(f"\n非数值型列: {list(non_numeric)}")

# 编码分类变量
if len(non_numeric) > 0:
    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder()
    housing_encoded = housing_train.copy()

    #  确保转换后的列覆盖到副本
    housing_encoded[non_numeric] = encoder.fit_transform(housing_train[non_numeric])

    # 验证编码后的数据类型
    print("\n编码后的数据列类型:")
    print(housing_encoded.dtypes)

    # 计算相关性矩阵
    corr_matrix = housing_encoded.corr()

else:
    # 如果没有非数值列，直接使用原始数据
    corr_matrix = housing_train.corr()

#打印相关性矩阵
with pd.option_context("display.max_columns", None, "display.width", 1000):
    print("\n最终相关性矩阵:")
    print(corr_matrix)

# 分析目标变量相关性
print("\n每个属性与房价中位数的相关性排序:")
sorted_corr = corr_matrix["median_house_value"].sort_values(ascending=False)
print(sorted_corr.to_string())
#6.1 数据清洗
#填补 total_bedrooms 属性中的缺失值。
# 获取删除标签列后的数据集
housing = strat_train_set.drop("median_house_value", axis=1)
# 标签列
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer

# 创建SimpleImputer实例，设置中位数替换
imputer = SimpleImputer(strategy="median")
# 创建一个没有文本属性"ocean_proximity"的数据集
housing_num = housing.drop("ocean_proximity", axis=1)
# 将imputer实例适配到housing_num
imputer.fit(housing_num)
# 执行中位数替换缺失值的转化。
X = imputer.transform(housing_num)
# 将numpy数组转换成DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
print("\n属性中的缺失值补齐为：")
print(housing_tr)
#对于文本属性 ocean_proximity ，我们先看一下它的大致内容
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.value_counts())
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
# 拟合并转换
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# 查看转换后的数据
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)
#创建一个二进制属性,独热编码可以使用 sklearn 中的 OneHotEncoder 编码器，将整数类别值转换为独热向量
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print("作用：直观显示完整的独热编码结果")
print(housing_cat_1hot.toarray())
#6.3 自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin

# 选取列名
col_names = ["total_rooms", "total_bedrooms", "population", "households"]
rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(c) for c in col_names]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        # 根据超参数add_bedrooms_per_room判断是否需要添加该组合属性
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# 将housing_extra_attribs从array转为DataFrame
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
print(housing_extra_attribs.head())
#流水线式数据转换
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 创建一个流水线，包含三个步骤
num_pipeline = Pipeline([
    # 中位数替换缺失值
    ('imputer', SimpleImputer(strategy="median")),
    # 添加组合属性
    ('attribs_adder', CombinedAttributesAdder()),
    # 归一化，统一量纲
    ('std_scaler', StandardScaler()),
])

# 对housing_num进行流水线式数据转换
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)

#构造一个能够处理所有列的转换器
from sklearn.compose import ColumnTransformer

# 获得数值列名称列表
num_attribs = list(housing_num)
# 获得类别列名称列表
cat_attribs = ["ocean_proximity"]

# 元组中的三个参数分别代表：名称（自定），转换器，以及一个该转换器能够应用的列名字（或索引）的列表
full_pipeline = ColumnTransformer([
    # 数值属性列转换器
    ("num", num_pipeline, num_attribs),
    # 文本属性列转换器
    ("cat", OneHotEncoder(), cat_attribs),
])
# 将ColumnTranformer应用到房屋数据
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
#7. 选择和训练模型
#首先，先训练一个线性回归模型。
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
# 模型训练
lin_reg.fit(housing_prepared, housing_labels)
#使用 5 个训练集的实例进行测试。
# 在几个训练实例上应用完整的预处理
some_data = housing.iloc[:5]    # 测试集
some_labels = housing_labels.iloc[:5]    # 测试标签
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
# 均方误差
lin_mse = mean_squared_error(housing_labels, housing_predictions)
# 均方根误差
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
#尝试一个更强大的模型 DecisionTreeRegressor ，它能够从数据中只好到复杂的非线性关系
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
# 模型训练
tree_reg.fit(housing_prepared, housing_labels)
# 模型预测
housing_predictions = tree_reg.predict(housing_prepared)
# 均方误差
tree_mse = mean_squared_error(housing_labels, housing_predictions)
# 均方根误差
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#7.2 使用交叉验证来更好地进行评估
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(temp_scores):
    print("Scores:", temp_scores)
    print("Mean:", temp_scores.mean())
    print("Standard deviation:", temp_scores.std())

print(display_scores(tree_rmse_scores))

#再使用交叉验证计算一下线性回归模型的评分。
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(display_scores(lin_rmse_scores))

#接下来尝试最后一个模型 RandomForestRegressor 随机森林
from sklearn.ensemble import RandomForestRegressor

#创建一个随机森林回归器，设置树的数量为100
forest_reg = RandomForestRegressor(n_estimators=100)
#使用训练数据拟合模型
forest_reg.fit(housing_prepared, housing_labels)

#使用模型对训练数据进行预测
housing_predictions = forest_reg.predict(housing_prepared)
#计算均方误差
forest_mse = mean_squared_error(housing_labels, housing_predictions)
#计算均方根误差
forest_rmse = np.sqrt(forest_mse)
#打印均方根误差
print(forest_rmse)

#根据 RMSE 的结果来看，比较容易接受，接下来看一下交叉验证的评分。
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(display_scores(forest_rmse_scores))

#8. 微调模型
#8.1 网格搜索
#使用 Scikit-Learn 的 GridSearchCV 来寻找最佳的超参数组合
from sklearn.model_selection import GridSearchCV

param_grid = [
    # 尝试3×4=12种超参数组合
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # 之后设置bootstrap=False，再尝试2×3=6种超参数组合
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# 训练5次，总共（12+6）×5=90次
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
print(grid_search.fit(housing_prepared, housing_labels))
# 查看最佳参数
print(grid_search.best_params_)
#由于超参数的最佳值都处于取值范围的右边界，可能需要再扩大取值范围，继续寻找。
param_grid = [
    {'n_estimators': [30, 50, 70, 90], 'max_features': [7, 8, 9]},
]
#使用GridSearchCV进行网格搜索，寻找最佳的超参数组合
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
#将训练集和标签传入GridSearchCV进行训练
grid_search.fit(housing_prepared, housing_labels)
#接下来看看当前的最佳估算器（输出只显示非默认的参数）。
print(grid_search.best_params_)
#GridSearchCV 计算的各种超参数组合的评分
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
#8.2 随机搜索
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


param_distribs = {
    # 均匀离散随机变量
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=7, high=9),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
print(rnd_search.fit(housing_prepared, housing_labels))

#看一下 RandomizedSearchCV 计算的各种超参数组合的评分。
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
#8.3 分析最佳模型及其误差
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
#将这些重要性分数显示在对应的属性名称旁边。
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder: OneHotEncoder()
cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs: ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))
#8.4 通过测试集评估系统
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# 数据处理
X_test_prepared = full_pipeline.transform(X_test)
# 模型预测
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

from scipy import stats
#使用 scipy.stats.t.interval() 计算泛化误差的 95% 置信区间
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                               loc=squared_errors.mean(),
                               scale=stats.sem(squared_errors)))

      )