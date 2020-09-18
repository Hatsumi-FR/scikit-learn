import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from mlxtend.plotting import plot_pca_correlation_graph


FILE_PATH = "housing.csv"
def circleOfCorrelations(corr, variance_ratio, variable_text):
    plt.Circle((0,0),radius=10, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    #fig = plt.gcf()
    fig = plt.figure(figsize=(15,15))
    fig.gca().add_artist(circle1)
    for idx in range(len(corr[0])):
        x = corr[0][idx]
        y = corr[1][idx]
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(variable_text[idx], xy=(x,y), fontsize=15)
    plt.xlabel("dimension 1 ({0:.1f}%)".format(pca.explained_variance_ratio_[0]*100))
    plt.ylabel("dimension 2 ({0:.1f}%)".format(pca.explained_variance_ratio_[1]*100))
    plt.plot([-1,1], [0,0], 'r')
    plt.plot([0,0], [-1,1], 'r')
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations")
    plt.show()
def load_data(path=FILE_PATH):
    '''
    Load datas from CSV
    '''
    return pd.read_csv(path)


if __name__ == '__main__':
    datas = load_data()
    datas_numeric = datas.drop("ocean_proximity", axis=1)
    encoder = LabelEncoder()
    num_labels = list(datas_numeric)
    cat_labels = ["ocean_proximity"]
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('normalize', Normalizer()),
        ('sclaer', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", numeric_pipeline, num_labels),
        # ("cat", OneHotEncoder(), cat_labels),
    ])
    numeric_datas_transformed = numeric_pipeline.fit_transform(datas_numeric)
    all_datas_trained = full_pipeline.fit_transform(datas)
    # cat_encoder = full_pipeline.named_transformers_["cat"]
    # cat_one_hot_attribs = list(cat_encoder.categories_[0])
    pca = PCA(n_components=all_datas_trained.shape[1])
    pca.fit(all_datas_trained)
    plt.bar(range(1, pca.explained_variance_ratio_.shape[0] + 1), pca.explained_variance_ratio_)
    # plt.show()
    # print(pca.explained_variance_ratio_.shape)
    # print(all_datas_trained)
    projection = pca.transform(all_datas_trained)
    plt.plot(projection[:, 0], all_datas_trained[:, 8], 'r.')
    plt.show()

    from sklearn.metrics import matthews_corrcoef
    X_corr = datas.corr()
    print(X_corr["median_house_value"].sort_values(ascending=False))
    corr = np.corrcoef(projection[:, 0], all_datas_trained[:, 8])
    print(corr)
    pc_0 = np.array([])
    pc_1 = np.array([])
    for i in range(len(all_datas_trained[0])):
        pc_0 = np.append(pc_0, np.corrcoef(projection[:, 0], all_datas_trained[:, i])[0, 1])
        pc_1 = np.append(pc_1, np.corrcoef(projection[:, 1], all_datas_trained[:, i])[0, 1])
    pc_corr = np.array([pc_0, pc_1])

    circleOfCorrelations(pc_corr, pca.explained_variance_ratio_, num_labels)