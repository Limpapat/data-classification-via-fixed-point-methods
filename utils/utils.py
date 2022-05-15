# utils.py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import pandas as pd
import numpy as np
import os

def get_data(csv_path:str, fearture_selection:list=[0,1], disp:bool=False, stdf:bool=True, split_ratio:list=[.1,.3]):
    """
    get_data()
        Descriptions: function for getting train, validation, test dataset
        Parameters:
            - csv_path : str : a path of dataset
            - fearture_selection : list : feature index list
            - disp : bool : option to display data plot
            - stdf : bool : option to standardize features data 
            - split_ratio : list : ratio list to split dataset, [.1, .3] means get test 10% data of dataset and get validation 30% data of remains
    """
    # load data
    df = pd.read_csv(csv_path)

    # split train test
    X, y = df.iloc[:, fearture_selection].values, df.iloc[:,2].values
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

    X_train, X_val, X_test = data_train.iloc[:, fearture_selection].values, data_val.iloc[:, fearture_selection].values, data_test.iloc[:, fearture_selection].values
    y_train, y_val, y_test = data_train.iloc[:, 2].values, data_val.iloc[:, 2].values, data_test.iloc[:, 2].values

    if stdf:
        # standardize features
        X_std = np.copy(X)
        X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

        X_train_std = np.copy(X_train)
        X_train_std[:,0] = (X_train[:,0] - X_train[:,0].mean()) / X_train[:,0].std()
        X_train_std[:,1] = (X_train[:,1] - X_train[:,1].mean()) / X_train[:,1].std()

        X_val_std = np.copy(X_val)
        X_val_std[:,0] = (X_val[:,0] - X_val[:,0].mean()) / X_val[:,0].std()
        X_val_std[:,1] = (X_val[:,1] - X_val[:,1].mean()) / X_val[:,1].std()

        X_test_std = np.copy(X_test)
        X_test_std[:,0] = (X_test[:,0] - X_test[:,0].mean()) / X_test[:,0].std()
        X_test_std[:,1] = (X_test[:,1] - X_test[:,1].mean()) / X_test[:,1].std()

        X, X_train, X_val, X_test = X_std, X_train_std, X_val_std, X_test_std

    if disp:
        print("\n--------- DISPLAY DATASET ---------")
        print('| train shape:\t\t {} |'.format(X_train.shape))
        print('| validation shape:\t {} |'.format(X_val.shape))
        print('| test shape:\t\t {} |'.format(X_test.shape))
        print("-----------------------------------\n")
        # data visualized
        plt.figure(figsize=(15,3))
        plt.subplot(141)
        plt.scatter(X[y==0, 0], X[y==0, 1], marker='o')
        plt.scatter(X[y==1, 0], X[y==1, 1], marker='x')
        plt.scatter(X[y==2, 0], X[y==2, 1], marker='.')
        plt.grid(linestyle='--')
        plt.title('Origin set')
        plt.xlabel(df.columns[fearture_selection].values[0])
        plt.ylabel(df.columns[fearture_selection].values[1])
        plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
        plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

        plt.subplot(142)
        plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], marker='o')
        plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], marker='x')
        plt.scatter(X_train[y_train==2, 0], X_train[y_train==2, 1], marker='.')
        plt.grid(linestyle='--')
        plt.title('Train set')
        plt.xlabel(df.columns[fearture_selection].values[0])
        plt.ylabel(df.columns[fearture_selection].values[1])
        plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
        plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

        plt.subplot(143)
        plt.scatter(X_val[y_val==0, 0], X_val[y_val==0, 1], marker='o')
        plt.scatter(X_val[y_val==1, 0], X_val[y_val==1, 1], marker='x')
        plt.scatter(X_val[y_val==2, 0], X_val[y_val==2, 1], marker='.')
        plt.grid(linestyle='--')
        plt.title('Validation set')
        plt.xlabel(df.columns[fearture_selection].values[0])
        plt.ylabel(df.columns[fearture_selection].values[1])
        plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
        plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

        plt.subplot(144)
        plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], marker='o')
        plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], marker='x')
        plt.scatter(X_test[y_test==2, 0], X_test[y_test==2, 1], marker='.')
        plt.grid(linestyle='--')
        plt.title('Test set')
        plt.xlabel(df.columns[fearture_selection].values[0])
        plt.ylabel(df.columns[fearture_selection].values[1])
        plt.xlim([X[:,0].min()-0.1, X[:,0].max()+0.1])
        plt.ylim([X[:,1].min()-0.1,X[:,1].max()+0.1])

        plt.tight_layout()
        plt.show()
    return ([X_train, X_val, X_test],[y_train, y_val, y_test])

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

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'source', 'sample_generated_data.csv')
    X, y = get_data(path, disp=True)