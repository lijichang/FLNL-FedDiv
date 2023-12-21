from utils import get_dataset
from sklearn import datasets, preprocessing
import numpy as np

def get_dataset():
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

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

train_dataset, init_data_labels = get_dataset()

init_data = (train_dataset - min(train_dataset)) / (max(train_dataset) - min(train_dataset))
