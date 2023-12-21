import os
import matplotlib.pyplot as plt
from datetime import datetime
from sampling import sample_iid, sample_non_iid
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
import numpy as np

import colorsys


def get_dataset():
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    dir = os.path.dirname(__file__)


    seed = None

    data, labels = datasets.make_blobs(
        n_samples=1000,
        n_features=1,
        centers=2,
        random_state=seed,
        shuffle=False
    )

    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    train_dataset = np.array(data)

    lb = preprocessing.LabelBinarizer()
    lb.fit(labels)
    labels = lb.transform(labels)
    train_dataset_labels = labels

    return train_dataset, train_dataset_labels

import numpy as np
import matplotlib.pyplot as plt
 
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def plot1D(ax, X, labels, title=None): 
    N = len(X)
    order = X.argsort(axis=0)
    X = X.reshape(-1)
    labels = labels.reshape(-1)
    order = order.reshape(-1)
    x = np.arange(X.shape[0]).tolist()
    y = X[order].tolist()
    c = labels[order].tolist()
    #print(c)
    
    m = {1:'o',2:'s',3:'D',0:'+'}
    cm = list(map(lambda x:m[x],c))
    
    scatter = mscatter(x, y, c=c, m=cm, ax=ax,cmap=plt.cm.RdYlBu)
     
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    #plt.show()
    if title: ax.set_title(title)
    
    return
    
def plot_PCA(ax, X, labels, pca_components=2, soft_clustering=True, title=None, random_state=None):  
    if X.shape[1] > 1:  
        if pca_components > 2: pca_components = 3
        else: pca_components = 2

        if random_state:
            random_state = int(random_state)
        pca = PCA(n_components=pca_components, random_state=random_state)
        pca.fit(X)
        pca_data = pca.transform(X)

        pc1 = pca_data[:, 0]
        pc2 = pca_data[:, 1]
        if pca_components == 3: pc3 = pca_data[:, 2]

        if not bool(soft_clustering):
            idxs = np.argmax(labels, axis=1)
            labels = np.zeros(labels.shape)
            for i in range(labels.shape[0]):
                labels[i, idxs[i]] = 1

        N = labels.shape[1]  
        HSV_tuples = [(i*1.0/N, 0.75, 0.75) for i in range(N)]
        RGB_tuples = np.array(list(map(lambda i: colorsys.hsv_to_rgb(*i), HSV_tuples)))
        colors = np.matmul(labels, RGB_tuples)

        if pca_components == 2:
            ax.set_aspect(1)
            ax.scatter(pc1, pc2, s=0.1, c=colors)
        else:
            ax.scatter(pc1, pc2, pc3, s=0.1, c=colors)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if pca_components == 3: ax.set_zlabel('PC3')

        if title: ax.set_title(title)   
    else:
        print('Data have only 1 feature. PCA cannot be applied.')

    return

def plot_metric(metric, n_iterations, output_dir, xLabel, yLabel):
    filename = str(yLabel).lower().replace('-', '_') + '.png'
    dir_name = os.path.join(output_dir, filename)

    ax = plt.figure().gca()

    if n_iterations != len(metric):
        x = np.arange(start=0, stop=n_iterations+1)
    else:
        x = np.arange(start=1, stop=n_iterations+1)
    y = metric
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.plot(x, y)
    
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(yLabel + ' vs. ' + xLabel)
    plt.tight_layout()

    plt.savefig(dir_name, dpi=150)
    plt.close()

    return

def prepare_output_dir():
    dir = os.path.dirname(__file__)
    path = '../output'
    dir_name = os.path.join(dir, path)
    os.makedirs(dir_name, exist_ok=True)

    date = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    output_dir = os.path.join(dir, path, date)
    os.makedirs(output_dir) 

    return output_dir