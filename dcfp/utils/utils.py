# utils.py
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from pandas.api.types import is_string_dtype
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import torch
import os, json

def get_data(csv_path:str, feature_selection:list=None, target:int=-1, disp:bool=False, stdf:bool=True, split_ratio:list=[.1,.3], save_fig:bool=False):
    """
    get_data()
        Descriptions: function for getting train, validation, test dataset
        Parameters:
            - csv_path : str : a path of dataset
            - feature_selection : list : feature index list
            - target : int : an index of target/label
            - disp : bool : option to display data plot
            - stdf : bool : option to standardize features data 
            - split_ratio : list : ratio list to split dataset, [.1, .3] means get test 10% data of dataset and get validation 30% data of remains
    """
    # load data
    data_name = csv_path.split("/")[-1]
    df = pd.read_csv(csv_path)
    # convert data to one-hot if it is not numeric
    df_dtype = df.dtypes
    for i, col_name in enumerate(df.columns):
        if is_string_dtype(df_dtype[i]) and col_name != df.columns[target]:
            df.iloc[:,i], _ = pd.factorize(df.iloc[:,i])
    df_ = df.copy()
    df.iloc[:,target], label = pd.factorize(df.iloc[:,target])
    # split train test
    if feature_selection is None:
        X, y = df.loc[:, df.columns != df.columns[target]].values, df.iloc[:,target].values
    else:
        X, y = df.iloc[:, feature_selection].values, df.iloc[:,target].values
    if len(split_ratio) == 2:
        _, data_test = train_test_split(df, test_size=split_ratio[0], random_state=42, shuffle=True)
        data_train, data_val = train_test_split(_, test_size=split_ratio[-1], random_state=42, shuffle=True)
    elif len(split_ratio) == 1:
        if split_ratio[0] != 1.:
            _, data_test = train_test_split(df, test_size=split_ratio[0], random_state=42, shuffle=True)
            data_train, data_val = train_test_split(_, test_size=split_ratio[-1], random_state=42, shuffle=True)
        else:
            data_test, data_train, data_val = df, df, df
    else:
        raise ValueError('Invalid values: plaese assign split ratio list with lenght not greater than 2')

    if feature_selection is None:
        X_train, X_val, X_test = data_train.loc[:, df.columns != df.columns[target]].values, data_val.loc[:, df.columns != df.columns[target]].values, data_test.loc[:, df.columns != df.columns[target]].values
    else:
        X_train, X_val, X_test = data_train.iloc[:, feature_selection].values, data_val.iloc[:, feature_selection].values, data_test.iloc[:, feature_selection].values
    y_train, y_val, y_test = data_train.iloc[:, target].values, data_val.iloc[:, target].values, data_test.iloc[:, target].values

    if stdf:
        # standardize features
        X_std, X_train_std, X_val_std, X_test_std = np.copy(X), np.copy(X_train), np.copy(X_val), np.copy(X_test)

        for i in range(0, X.shape[-1]):
            X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
            X_train_std[:,i] = (X_train[:,i] - X_train[:,i].mean()) / X_train[:,i].std()
            X_val_std[:,i] = (X_val[:,i] - X_val[:,i].mean()) / X_val[:,i].std()
            X_test_std[:,i] = (X_test[:,i] - X_test[:,i].mean()) / X_test[:,i].std()

        X, X_train, X_val, X_test = X_std, X_train_std, X_val_std, X_test_std

    if disp:
        print("\n--------- DISPLAY DATASET ---------")
        print('| train shape:\t\t {} |'.format(X_train.shape))
        print('| validation shape:\t {} |'.format(X_val.shape))
        print('| test shape:\t\t {} |'.format(X_test.shape))
        print("-----------------------------------\n")
    if save_fig:
        if torch.tensor(X, dtype=torch.float32).size()[-1] == 2:
            # data visualized
            plt.figure(figsize=(15,3))
            plt.subplot(141)
            plt.scatter(X[y==0, 0], X[y==0, 1], marker='o')
            plt.scatter(X[y==1, 0], X[y==1, 1], marker='x')
            plt.scatter(X[y==2, 0], X[y==2, 1], marker='.')
            plt.grid(linestyle='--')
            plt.title('Origin set')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[-1])
            plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
            plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

            plt.subplot(142)
            plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], marker='o')
            plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], marker='x')
            plt.scatter(X_train[y_train==2, 0], X_train[y_train==2, 1], marker='.')
            plt.grid(linestyle='--')
            plt.title('Train set')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[-1])
            plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
            plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

            plt.subplot(143)
            plt.scatter(X_val[y_val==0, 0], X_val[y_val==0, 1], marker='o')
            plt.scatter(X_val[y_val==1, 0], X_val[y_val==1, 1], marker='x')
            plt.scatter(X_val[y_val==2, 0], X_val[y_val==2, 1], marker='.')
            plt.grid(linestyle='--')
            plt.title('Validation set')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[-1])
            plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
            plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

            plt.subplot(144)
            plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], marker='o')
            plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], marker='x')
            plt.scatter(X_test[y_test==2, 0], X_test[y_test==2, 1], marker='.')
            plt.grid(linestyle='--')
            plt.title('Test set')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[-1])
            plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
            plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

            plt.tight_layout()
        else:
            seaborn.pairplot(df_, hue=df_.columns[target])
        if not os.path.exists('img'):
            os.mkdir('img')
        plt.savefig(f"img/{data_name}.png")
        if disp:
            plt.show()
    return ([X_train, X_val, X_test],[y_train, y_val, y_test], label)

def plot_decision_regions(X, y, classifier, label=[-1,1], resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    probas = classifier.forward(tensor)
    
    if probas.shape[1]==1:
      tredshold = torch.zeros_like(probas).add_(0.5)
      probas = torch.cat((probas, tredshold), 1)
    Z = np.argmax(probas.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min()-0.1, xx1.max()+0.1)
    plt.ylim(xx2.min()-0.1, xx2.max()+0.1)

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=label[idx])

def loadcseq(path:str):
    f = open(path, 'r')
    d = json.loads(f.read())
    f.close()
    return d

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'source', 'parkinsons.data')
    # feature_selection = [1, 2]
    feature_selection = None
    target = 17
    # path = os.path.join(os.getcwd(), 'source', 'sample_generated_data.csv')
    # feature_selection = [0, 1]
    # feature_selection = None
    X, y, label = get_data(path, feature_selection=feature_selection, target=target, disp=False)