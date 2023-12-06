import os
import tarfile
# import urllib #これだとエラーが出た．何を使うかも指定してimport しないといけない．
import urllib.request

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_RUL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'
def fetch_housing_data(housing_url=HOUSING_RUL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()
# housing.head() #jupyter notebookならこれでよい
print(housing.head())

print(housing.info())

print(housing['ocean_proximity'].value_counts())

print(housing.describe())

import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(25, 15))
# plt.savefig('attribute_histogram_plots')
# plt.show()

import numpy as np
np.random.seed(42)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))


housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing['income_cat'].hist()
# plt.show()

#層化抽出
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]
    print(start_test_set['income_cat'].value_counts() / len(start_test_set))

#income_cat 属性を取り除き，データをもとの状態に戻す
for set_ in (start_train_set, start_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

#元データを壊さないようコピー
housing = start_train_set.copy()
#alphaで密度の高い所を可視化
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
#              s=housing['population']/100, label='population', figsize=(10, 7),
#              c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.show()

# corr_matrix = housing.corr()# エラーが取れません
# print(corr_matrix['median_house_value'].sort_values(ascending=False))
# print(corr_matrix)

from pandas.plotting import scatter_matrix

# attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing['room_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_househols'] = housing['population'] / housing['households']

# corr_matrix = housing.corr()

housing = start_train_set.drop('median_house_value', axis=1)
housing_labels = start_train_set['median_house_value'].copy()

# 2.5.1 cleaning data
# housing.dropna(subset=['total_bedrooms']) #option 1
# housing.drop('total_bedrooms', axis=1) #option2
median = housing['total_bedrooms'].median() #option 3
housing['total_bedrooms'].fillna(median, inplace=True)

# scikit-learnで欠損値をうまく処理してくれるクラス SimpleImputer
from sklearn.impute import SimpleImputer
# インスタンスの作成
imputer = SimpleImputer(strategy='median')
# 中央値は数値属性出なければ計算できない　→　テキスト属性のocean_proximityを取り除いたデータのコピーを作る
housing_num = housing.drop('ocean_proximity', axis=1)
# 訓練データにimputerインスタンスを適合
imputer.fit(housing_num)
print(imputer.statistics_)#すでに中央値が計算されている．
print(housing_num.median().values)#上と結果は一緒

# 欠損値を中央値で置き換えて訓練データを変換する
X = imputer.transform(housing_num)
# 返り値（X）は変換された特徴量を格納するNumpy配列
# PandasのDataFrameに戻す
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
print(housing_tr.head())

housing_cat = housing[['ocean_proximity']]
print(housing_cat.head(10))

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.shape)
print(type(housing_cat_1hot))

print(housing_cat_1hot.toarray())


# 2.5.3 カスタム変換器
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_rooms = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, households_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)

##変換パイプラインの例
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)



