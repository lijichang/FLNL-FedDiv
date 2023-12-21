import os
import numpy as np
import sklearn.mixture
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from utils import plot_PCA, plot1D


class GaussianMixture(sklearn.mixture.GaussianMixture):

    def __init__(self, X, n_components=3, covariance_type='full',
                 weights_init=None, means_init=None, precisions_init=None, covariances_init=None,
                 init_params='kmeans', tol=1e-3, random_state=None, is_quiet=False):

        do_init = (weights_init is None) and (means_init is None) and (precisions_init is None) and (covariances_init is None)
        if do_init:
            _init_model = sklearn.mixture.GaussianMixture(
                n_components=n_components,
                tol=tol,
                covariance_type=covariance_type,
                random_state=random_state,
                init_params=init_params
            )

            # Responsibilities are found through KMeans or randomly assigned, from responsibilities the gaussian parameters are estimated (precisions_ is not calculated)
            _init_model._initialize_parameters(X, np.random.RandomState(random_state))
            # The gaussian parameters are fed into _set_parameters() which computes also precisions_ (the others remain the same)
            _init_model._set_parameters(_init_model._get_parameters())

            weights_init = _init_model.weights_
            means_init = _init_model.means_
            precisions_init = _init_model.precisions_
            covariances_init = _init_model.covariances_

        super().__init__(
            n_components=n_components,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            tol=tol,
            init_params=init_params,
            covariance_type=covariance_type,
            random_state=random_state,
            warm_start=True,
            max_iter=1
        )
        self._is_quiet = is_quiet
        # The gaussian parameters are recomputed by KMeans or randomly, but since the init parameters are given they are discarded (covariances_ is not generated)
        self._initialize_parameters(X, np.random.RandomState(random_state))
        # covariances_ is copied from the initial model (since it has it)
        self.covariances_ = covariances_init
        # precisions_ is computed as before
        self._set_parameters(self._get_parameters())

    def fit(self, X, epochs=1, labels=None, args=None, output_dir=None):
        self.history_ = {
            'epochs': epochs,
            'converged': [],
            'metrics': {
                'aic': [],
                'bic': [],
                'll': [],
            },
            'parameters': {
                'means': [],
                'covariances': [],
                'weights': []
            }
        }

        if not self._is_quiet:
            self.plot(X, labels, args, output_dir)

        pbar = tqdm(range(epochs), disable=self._is_quiet)
        for epoch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch+1, epochs))

            super().fit(X)

            self.history_['converged'].append(self.converged_)

            self.history_['metrics']['aic'].append(self.aic(X))
            self.history_['metrics']['bic'].append(self.bic(X))
            self.history_['metrics']['ll'].append(self.score(X))

            self.history_['parameters']['means'].append(self.means_)
            self.history_['parameters']['weights'].append(self.weights_)
            self.history_['parameters']['covariances'].append(self.covariances_)

            if not self._is_quiet and (epoch+1) % args.plots_step == 0:
                self.plot(X, labels, args, output_dir, 'epoch', epoch)

        if not self._is_quiet:
            if self.converged_:
                print('\nThe model successfully converged.')
            else:
                print('\nThe model did NOT converge.')

        return self.history_

    def predict(self, X):
        predicted_labels = self.predict_proba(X).tolist()
        predicted_labels = np.array(predicted_labels)

        return predicted_labels

    def get_parameters(self):
        parameters = self._get_parameters()

        return parameters

    def set_parameters(self, params):
        self._set_parameters(params)

        return

    def plot(self, X, labels, output_dir, filename=None, iteration=None, plots_3d=0, soft=1):
        path = './model'
        dir_name = os.path.join(output_dir, path)
        os.makedirs(dir_name, exist_ok=True)

        if iteration is None:
            filename = 'init'
        else:
            filename = '{}_{}'.format(filename, iteration+1)
        dir_name = os.path.join(dir_name, filename)
        
        if X.shape[1] <= 2:
            plots_3d = 0

        

        if plots_3d == 2:
            predicted_labels = self.predict(X)

            #3D
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            pca_components = 3

            plot_PCA(ax1, X, labels, pca_components, soft, 'Ground Truth', random_state=self.random_state)
            plot_PCA(ax2, X, predicted_labels, pca_components, soft, 'Predicted Clusters', random_state=self.random_state)
            dir_name_3d = dir_name + '_3d'
            fig.savefig(dir_name_3d, dpi=150)
            plt.close(fig)
            
            #2D
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            pca_components = 2

            plot_PCA(ax1, X, labels, pca_components, soft, 'Ground Truth', random_state=self.random_state)
            plot_PCA(ax2, X, predicted_labels, pca_components, soft, 'Predicted Clusters', random_state=self.random_state)
            dir_name_2d = dir_name + '_2d'
            fig.savefig(dir_name_2d, dpi=150)
            plt.close(fig)
        elif plots_3d == 3:
            fig = plt.figure(figsize=plt.figaspect(0.5))

            if bool(plots_3d) == True:
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                pca_components = 3
            else:
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                pca_components = 2

            plot_PCA(ax1, X, labels, pca_components, soft, 'Ground Truth', random_state=self.random_state)
            plot_PCA(ax2, X, self.predict(X), pca_components, soft, 'Predicted Clusters', random_state=self.random_state)
            fig.savefig(dir_name, dpi=150)
            plt.close(fig)

        else:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            plot1D(ax=ax1, X=X, labels=labels, title='Ground Truth')
            
            predicted = self.predict(X) 
            
            labels = predicted[:, self.means_.argmin()] > 0.5
            
            predicted_labels = np.ones((len(labels), 1), dtype=np.int32())
            predicted_labels[labels==False] = 0
            
            #predicted_labels = labels   #predicted.argmax(axis=1)
            #predicted_labels = predicted_labels.reshape(-1)
            #predicted = predicted.reshape(-1)
            #predicted_labels[predicted>=0.5] = 1
            #predicted_labels[predicted<0.5] = 0
            #predicted_labels = predicted_labels.reshape(-1, 1)
            plot1D(ax=ax2, X=X, labels=predicted_labels, title='Predicted Clusters')
            fig.savefig(dir_name, dpi=150)
            plt.close(fig)
        return

    def compute_precision_cholesky(self, covariances, covariance_type):
        """Compute the Cholesky decomposition of the precisions.
        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar.")

        if covariance_type == 'full':
            n_components, n_features, _ = covariances.shape
            precisions_chol = np.empty((n_components, n_features, n_features))
            for k, covariance in enumerate(covariances):
                try:
                    cov_chol = linalg.cholesky(covariance, lower=True)
                except linalg.LinAlgError:
                    raise ValueError(estimate_precision_error_message)
                precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                             np.eye(
                                                                 n_features),
                                                             lower=True).T
        elif covariance_type == 'tied':
            _, n_features = covariances.shape
            try:
                cov_chol = linalg.cholesky(covariances, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                      lower=True).T
        else:
            if np.any(np.less_equal(covariances, 0.0)):
                raise ValueError(estimate_precision_error_message)
            precisions_chol = 1. / np.sqrt(covariances)
        return precisions_chol
