import copy
import numpy as np

class Client():
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = np.array(dataset)

    def fit(self, global_model, local_epochs:int = 1):
        local_model = copy.deepcopy(global_model)
        local_model.fit(self.dataset, epochs=local_epochs)
        return local_model.history_