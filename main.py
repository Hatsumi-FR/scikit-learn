import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
HOUSING_PATH = "housing.csv"
from sklearn.model_selection import train_test_split


def load_data(path=HOUSING_PATH):
    return pd.read_csv(path)

def split_dataset(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    testset_size = int(len(datas)) * test_ratio
    test_indices = shuffled_indices[:testset_size]
    train_indices = shuffled_indices [testset_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    datas = load_data()
    datas["income_cat"] = np.ceil(datas["median_income"] / 1.5)
    # datas["income_cat"].where(datas["income_cat"] < 5, 5.0, inplace= True)
    print(datas.info())
    d = datas["ocean_proximity"].value_counts()
    # print(d)
    print(datas.describe())
    datas.hist(bins=50, figsize=(20,15))
    datas.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.4, s=datas["population"]/100, label="population", figsize=(10,7), c ="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
    plt.legend()
    plt.show()
    train_set, test_set = train_test_split(datas, test_size=0.2, random_state=42)
    print(train_set)