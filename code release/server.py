import numpy as np
from gmm import GaussianMixture

class Server():
    def __init__(self, init_dataset, components=2, local_epochs=3, seed=False, init_params='random'):
        self.random_state = None
        if seed:
            self.random_state = int(seed)
        
        # Initialize GaussianMixture model
        self.model = GaussianMixture(
            X=init_dataset,
            n_components=components,
            random_state=self.random_state,
            is_quiet=True,
            init_params=init_params,
        )

        self.init_params = init_params
        self.components = components
        self.init_dataset = init_dataset
        self.local_epochs = local_epochs
        self.selected_clients = {}

    def _set_parameters_from_clients_models(self, server_cached_local_filter):
        # Extract parameters from client models
        self.clients_means = []
        self.clients_covariances = []
        self.clients_weights = []

        for client_id in server_cached_local_filter:
            parameters = server_cached_local_filter[client_id]['parameters']
            self.clients_means.append(parameters['means'][-1])
            self.clients_covariances.append(parameters['covariances'][-1])
            self.clients_weights.append(parameters['weights'][-1])

        # Convert to NumPy arrays
        self.clients_means = np.array(self.clients_means)
        self.clients_covariances = np.array(self.clients_covariances)
        self.clients_weights = np.array(self.clients_weights)

        return

    def set_parameters_from_clients_models(self, server_cached_local_filter):
        # Same as _set_parameters_from_clients_models function
        self._set_parameters_from_clients_models(server_cached_local_filter)

        return
    
    def start_round(self, selected_clients, server_cached_local_filter):
        # Fit the model for selected clients
        for client in selected_clients:
            server_cached_local_filter[client.id] = client.fit(self.model, self.local_epochs)
        return server_cached_local_filter

    # def weighted_average_clients_models(self, dict_len):
    #     dict_len = np.array(dict_len)
    #     gamma = dict_len * 1.0 / np.sum(dict_len)

    #     # Calculate weighted average model parameters
    #     for k in range(len(dict_len)):
    #         self.clients_means[k] = self.clients_means[k] * pow(gamma[k], 1)
    #         self.clients_covariances[k] = self.clients_covariances[k] * pow(gamma[k], 2)
    #         self.clients_covariances * pow(gamma, 2)
    #         self.clients_weights[k] = self.clients_weights[k] * pow(gamma[k], 1)

    #     # Update model parameters
    #     self._update_model_parameters()

    #     return

    def update_server_model(self):
        # Update server-side model
        self.model = GaussianMixture(
            X=self.init_dataset,
            n_components=self.components,
            random_state=self.random_state,
            is_quiet=True,
            init_params=self.init_params,
            weights_init=self.avg_clients_weights,
            means_init=self.avg_clients_means,
            precisions_init=self.avg_clients_precisions
        )

        return

    def _update_model_parameters(self):
        # Update model parameters
        self.avg_clients_precisions_cholesky = self.model.compute_precision_cholesky(
            self.avg_clients_covariances, self.model.covariance_type
        )
        
        params = (self.avg_clients_weights, self.avg_clients_means, self.avg_clients_covariances, self.avg_clients_precisions_cholesky)
        self.model.set_parameters(params)

        self.avg_clients_precisions = self.model.precisions_

        return
    

    def weighted_average_clients_models(self, dict_len):
        
        dict_len = np.array(dict_len)
        gamma = dict_len * 1.0 / np.sum(dict_len)
        
        for k in range(len(dict_len)):
            self.clients_means[k] = self.clients_means[k] * pow(gamma[k], 1)
            self.clients_covariances[k] = self.clients_covariances[k] * pow(gamma[k], 2)
            self.clients_covariances * pow(gamma, 2)
            self.clients_weights[k] = self.clients_weights[k] * pow(gamma[k], 1)
            
        
        self.avg_clients_means = np.sum(self.clients_means, axis=0)
        self.avg_clients_covariances = np.sum(self.clients_covariances, axis=0)
        self.avg_clients_weights = np.sum(self.clients_weights, axis=0)
        
        self.avg_clients_precisions_cholesky = self.model.compute_precision_cholesky(self.avg_clients_covariances, self.model.covariance_type)
        
        params = (self.avg_clients_weights, self.avg_clients_means, self.avg_clients_covariances, self.avg_clients_precisions_cholesky)
        self.model.set_parameters(params)

        self.avg_clients_precisions = self.model.precisions_

        return