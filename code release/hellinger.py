import numpy as np
import math

class NormalDistribution():
    def __init__(self, means, covariances):
        self.means = means
        self.covariances = covariances
        self.covariances_det = np.linalg.det(covariances)

def get_hellinger_multivariate(distribution_1:NormalDistribution, distribution_2:NormalDistribution):
    covariances_sum = (distribution_1.covariances + distribution_2.covariances) / 2

    numerator = math.pow(distribution_1.covariances_det, 1/4) * math.pow(distribution_2.covariances_det, 1/4)
    denominator = math.pow(np.linalg.det(covariances_sum), 1/2)

    means_difference = distribution_1.means - distribution_2.means
    inverse_covariances_sum = np.linalg.pinv(covariances_sum)
    exponent = -(np.matmul(np.matmul(np.transpose(means_difference), inverse_covariances_sum), means_difference)) * 1/8

    hellinger_squared_distance = 1 - (numerator / denominator) * math.exp(exponent)

    return hellinger_squared_distance
