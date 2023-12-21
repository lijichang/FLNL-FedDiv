# python version 3.7.1
# -*- coding: utf-8 -*-
import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from scipy.special import softmax

from client import Client as Local_Noise_Filter
from model.build_model import build_model
from save_pyfiles import save_runfile
from server import Server as Global_Noise_Filter
from train_data import init_data
from util.dataset import get_dataset
from util.fedavg import FedAvg
from util.local_training import LocalUpdate, globaltest
from util.options import args_parser
from util.util import add_noise, get_output

np.set_printoptions(threshold=np.inf)

def causal_inference(current_logit, phat, xi=0.5):
    """
    Adjust logits based on prior probabilities to perform de-biasing.

    Args:
        current_logit (array-like): The logits from the current model prediction.
        phat (array-like): The estimated prior probabilities per class.
        xi (float, optional): The scaling factor for the log prior probabilities.

    Returns:
        numpy.ndarray: The de-biased probabilities after applying softmax.
    """
    # Adjust the logits by subtracting a scaled log of the estimated prior probabilities
    adjusted_logit = current_logit - xi * np.log(phat)
    # Compute de-biased probabilities using softmax
    de_biased_prob = softmax(adjusted_logit, axis=1)

    return de_biased_prob

def initial_phat(class_num=1000):
    """
    Initialize a mean probability vector with equal probabilities for each class.

    Args:
        class_num (int): The number of classes.

    Returns:
        numpy.ndarray: A 2D array representing the initialized mean probability vector.
    """
    # Create a 2D numpy array with shape (1, class_num)
    # Each element is set to 1/class_num to represent equal probability for each class
    phat = (np.ones([1, class_num], dtype=np.float32) / class_num)
    
    return phat

def update_phat(probs, phat, momentum, phat_mask=None):
    """
    Update the running estimate of the mean probability vector.

    Args:
        probs (array-like): Array of probability vectors for instances.
        phat (array-like): The current estimate of the mean probability vector.
        momentum (float): The momentum coefficient for exponential weighting.
        phat_mask (array-like, optional): An optional mask to apply to the probabilities
                                          before averaging.

    Returns:
        numpy.ndarray: The updated estimate of the mean probability vector.
    """
    # If a mask is provided, apply it to the probabilities.
    if phat_mask is not None:
        # Apply the mask and sum across the batch dimension.
        # Reshape the mask to be compatible with the probability array.
        mean_prob = (probs * phat_mask.reshape(-1, 1)).sum(axis=0) / phat_mask.sum()
    else:
        # Calculate the mean of the probability vectors across the batch dimension.
        mean_prob = probs.mean(axis=0)
    
    # Update phat with a weighted sum of the old value and the new mean probabilities.
    phat = momentum * phat + (1 - momentum) * mean_prob

    return phat

def client_cached_phat_update(args, de_bias, local_logits, idx, sample_idx):
    """
    Update the cached debiased probability estimates (phat) for a client based on local model logits.
    
    Args:
        args: A configuration object containing hyperparameters.
        de_bias: A dictionary containing cached debiased probability estimates and other debiasing information.
        local_logits: The logits output by the local model for the current batch of data.
        idx: The index of the current client or data partition.
        sample_idx: The indices of the samples in the current batch.
    
    Returns:
        de_bias: The updated debiasing dictionary with new probability estimates and labels.
    """

    # Apply causal inference to obtain debiased predictions using the local logits and cached phat
    de_biased_preds = causal_inference(local_logits, de_bias['phat'][idx], xi=args.xi)

    # Find the maximum probability from the debiased predictions for each sample
    de_biased_max_probs = np.max(de_biased_preds, axis=-1)

    # Update the cached debiased probabilities for the given sample indices
    de_bias['de_biased_probs'][sample_idx] = de_biased_max_probs

    # Update the cached debiased labels for the given sample indices
    de_bias['de_biased_labels'][sample_idx] = np.argmax(de_biased_preds, axis=-1)

    # Create a mask for samples where the maximum debiased probability is greater than a confidence threshold (0.85)
    prob_mask = de_biased_max_probs > 0.85

    # If there are samples above the confidence threshold, update phat for the given client or data partition
    if prob_mask.sum() > 0:
        # Update phat using a moving average with momentum; only use logits from confident samples
        de_bias['phat'][idx] = update_phat(softmax(local_logits, axis=-1)[prob_mask], de_bias['phat'][idx], momentum=args.m)

    # Return the updated debiasing information
    return de_bias

def calculate_normalized_loss(loss):
    """
    Normalize the loss values to a range [0, 1].

    Args:
        loss (list or array): A list or numpy array of loss values to be normalized.

    Returns:
        numpy.ndarray: The normalized loss values.
    """
    # Convert the input to a numpy array if it isn't one already
    loss_array = np.array(loss)
    
    # Find the minimum loss value
    min_loss = np.min(loss_array)
    
    # Find the maximum loss value
    max_loss = np.max(loss_array)
    
    # Apply the normalization formula
    normalized_loss = (loss_array - min_loss) / (max_loss - min_loss)
    
    return normalized_loss

def federated_filter_aggregation(server_cached_local_filter, dict_clients, global_noise_filter):
    """
    Aggregate local noise filters to update the global noise filter in a federated learning system.

    Args:
        server_cached_local_filter: A dictionary containing local filter states cached on the server,
                                    with keys being client identifiers.
        dict_clients: A dictionary mapping client indices to their respective data points or data sizes.
        global_noise_filter: The global noise filter object that needs to be updated.

    This function does not return a value; it updates the global noise filter in place.
    """

    # Initialize a list to store the number of data points associated with each client's local filter
    local_filter_lens_per_rnd = []

    # Iterate over the keys (client indices) in the server-cached local filter dictionary
    for idx in server_cached_local_filter.keys():
        # Append the number of data points for the current client's local filter to the list
        local_filter_lens_per_rnd.append(len(dict_clients[idx]))

    # Set the parameters of the global noise filter from the local filters cached on the server
    global_noise_filter.set_parameters_from_clients_models(server_cached_local_filter)

    # Perform a weighted average of the local filters based on the number of data points per client/client
    global_noise_filter.weighted_average_clients_models(local_filter_lens_per_rnd)

    # Update the global noise filter's model (e.g., coefficients, weights) based on the aggregated information
    global_noise_filter.update_server_model()

def federated_model_aggregation(netglob, w_locals, idxs_clients, dict_clients):
    """
    Aggregate local model weights to update the global model in a federated learning setting.

    Args:
        netglob: The global model whose parameters are to be updated.
        w_locals: A list of local model weights from different clients participating in the federated learning round.
        idxs_clients: A list of indices representing the clients who have contributed to the training in the current round.
        dict_clients: A dictionary mapping client indices to their respective data points or data sizes.

    Returns:
        This function does not return a value; it updates the global model in place.
    """
    
    # Calculate the number of data points for each client that participated in the federated learning round
    dict_len = [len(dict_clients[idx]) for idx in idxs_clients]

    # Aggregate the local weights using a federated averaging algorithm, weighted by the number of data points
    w_glob = FedAvg(w_locals, dict_len)

    # Load the aggregated weights into the global model, using a deep copy to ensure no references are shared
    netglob.load_state_dict(copy.deepcopy(w_glob))
    
def local_filter_training(loss, idx, global_noise_filter, server_cached_local_filter):
    """
    Update the local noise filter for each client based on the loss values of local samples.

    Args:
        loss: The loss value or values obtained from local model training.
        idx: The index representing the specific local model or data partition.
        global_noise_filter: The global noise filter that aggregates knowledge learned from all local models.
        server_cached_local_filter: The current state of the noise filter cached on the server for this local model.
    
    Returns:
        global_noise_filter: The global noise filter object, potentially updated with the new round of filtering.
        server_cached_local_filter: The updated state of the noise filter for this local model, as cached on the server.
    """

    # Calculate a normalized version of the loss, which can be used for noise filtering
    normalized_loss = calculate_normalized_loss(loss)

    # Create a local noise filter object for the current local model using its index and the normalized loss
    local_noise_filter = Local_Noise_Filter(idx, normalized_loss.reshape(-1, 1))

    # Update the server's cached local filter by starting a new round in the global noise filter
    # This likely involves aggregating information from all local models to improve the server's filter
    server_cached_local_filter = global_noise_filter.start_round([local_noise_filter], server_cached_local_filter)

    # Return the potentially updated global noise filter and the new state of the local noise filter cached on the server
    return global_noise_filter, server_cached_local_filter

def local_model_training(args, w_locals, net_local, netglob, revised_sample_idx, dataset_train):
    """
    Train a local model using a subset of training data, and collect the local model weights for aggregation.
    
    Args:
        args: A configuration object containing hyperparameters and device information.
        w_locals: A list to collect the local model weights from each participant.
        net_local: The local model that will be trained.
        netglob: The global model whose weights are used as the starting point for training the local model.
        revised_sample_idx: The indices of the training samples selected for local model training.
        dataset_train: The training dataset which includes train_labels.
    
    Returns:
        w_locals: Updated list with the local model weights after training.
        net_local: The trained local model.
        dataset_train: The training dataset which may have updated labels.
    """

    # Store the original training labels
    y_train_given = np.array(dataset_train.train_labels)

    # If the size of the selected sample index set is larger than the batch size, perform local training
    if len(revised_sample_idx) > args.local_bs:
        # Initialize a LocalUpdate object with the selected indices and training data
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=revised_sample_idx)
        # Perform local model weights update
        w_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), epoch=args.local_ep)
        # Append the updated weights to the list of local weights
        w_locals.append(copy.deepcopy(w_local))
    else:
        # If not enough data for a batch, use the global model's weights directly
        w_local = copy.deepcopy(netglob.state_dict())
        w_locals.append(w_local)

    # Load the updated local weights into the local model
    net_local.load_state_dict(copy.deepcopy(w_local))

    # Restore the original training labels to the dataset
    dataset_train.train_labels = y_train_given

    # Return the updated list of local weights, the local model, and the dataset
    return w_locals, net_local, dataset_train

def local_data_splitting(loss, global_noise_filter):
    """
    This function splits local data into clean and noisy subsets based on the provided loss
    using a global noise filter model.
    
    Args:
        loss: A numpy array containing loss values of the local model's predictions.
        global_noise_filter: An object containing a pre-trained model for filtering noise.
    
    Returns:
        local_split: A dictionary containing the predicted clean and noisy flags for the samples,
                     and the estimated noise level in the local data.
    """
    
    # Calculate normalized loss, which is a preprocessing step before using the noise filter model
    normalized_loss = calculate_normalized_loss(loss)
    
    # Predict the probability of being clean for each data point using the global noise filter model
    # The model is presumably a Gaussian Mixture Model or similar, where 'means_' is an attribute
    prob_clean = global_noise_filter.model.predict(normalized_loss.reshape(-1, 1))
    
    # Select the probability of being clean associated with the component of the model with the lowest mean
    # This assumes that the clean data is associated with the component that has the lowest mean loss
    prob_clean = prob_clean[:, global_noise_filter.model.means_.argmin()]
    
    # Determine which data points are predicted clean based on the probability threshold (e.g., higher than 50%)
    pred_clean = prob_clean > 0.50
    
    # The noisy predictions are simply the complement of the clean predictions
    pred_noisy = ~pred_clean
    
    # Estimate the noise level in the dataset as the proportion of data points predicted to be noisy
    estimated_noisy_level = 1.0 - np.sum(pred_clean) / len(pred_clean)

    # Compile the results into a dictionary
    local_split = {
        'pred_clean': pred_clean,
        'pred_noisy': pred_noisy,
        'estimated_noisy_level': estimated_noisy_level,
    }
    
    # Return the dictionary containing the split information
    return local_split

def predictive_consistency_sampling(de_bias, noisy_indices, clean_indices, clean_pseudo_labels, noisy_pseudo_labels):
    # Retrieve de-biased labels for noisy and clean samples
    de_bias_labels_noisy = de_bias['de_biased_labels'][noisy_indices]
    de_bias_labels_clean = de_bias['de_biased_labels'][clean_indices]
    
    # Check which clean predictions match the de-biased labels
    de_bias_clean_predictions = clean_pseudo_labels == de_bias_labels_clean
    de_bias_clean_indices = clean_indices[de_bias_clean_predictions]
    
    # Determine which noisy predictions match the de-biased labels and have high confidence
    de_bias_noisy_predictions = (de_bias_labels_noisy == noisy_pseudo_labels)

    return de_bias_clean_indices, de_bias_noisy_predictions

def relabeling_and_reselection(args, local_split, idx, sample_idx, idx_cnt, local_output, dataset_train, de_bias):
    """
    This function relabels noisy labels and reselects a subset of the data based on the model's predictions
    and a de-biasing mechanism.
    
    Args:
        args: Configuration parameters including confidence threshold and clean set threshold.
        local_split: A dictionary containing local information about predicted clean and noisy labels,
                     and the estimated noise level.
        idx: The index of the current local model or client.
        sample_idx: The indices of the samples assigned to the current local model.
        idx_cnt: A count of how many times each index has been processed.
        local_output: The output probabilities from the local model.
        dataset_train: The training dataset object which includes train_labels.
        de_bias: A dictionary containing de-biased labels and probabilities.
    
    Returns:
        revised_sample_idx: A set of indices indicating the samples selected after relabeling.
        dataset_train: The training dataset which may have updated labels.
        de_bias: The updated de-biasing dictionary.
    """
    
    # loading local data split
    pred_clean = local_split['pred_clean']
    pred_noisy = local_split['pred_noisy']
    estimated_noisy_level = local_split['estimated_noisy_level']

    y_train_given = np.array(dataset_train.train_labels)

    # Initialize de-biased probabilities and labels if this is the first time processing this index
    if idx_cnt[idx] == 0:
        de_bias['de_biased_probs'][sample_idx] = np.max(local_output, 1)
        de_bias['de_biased_labels'][sample_idx] = np.argmax(local_output, 1)

    # Select indices from the original sample set based on prediction cleanliness
    clean_indices = sample_idx[pred_clean]
    
    # Get pseudo-labels by finding the index of the maximum value in the model output
    pseudo_labels = np.argmax(local_output, axis=1)
    
    # Get pseudo-labels for clean samples
    clean_pseudo_labels = pseudo_labels[pred_clean]
    
    # Select the maximum output probabilities for noisy predictions
    max_prob_predictions = np.max(local_output, axis=1)

    # Noisy sample relabeling to assign pseudo-labels for noisy samples with confidence higher than threshold
    pred_noisy = pred_noisy & (max_prob_predictions > args.conf)
    noisy_indices = sample_idx[pred_noisy]
    noisy_pseudo_labels = pseudo_labels[pred_noisy]

    # Perform labeled sample re-selection via Predictive Consistency based Sampler (PCS)
    de_bias_clean_indices, de_bias_noisy_predictions = predictive_consistency_sampling(de_bias, noisy_indices, clean_indices, clean_pseudo_labels, noisy_pseudo_labels)
    
    # Select the indices for noisy predictions that are confident and match the de-biased labels
    relabel_indices = noisy_indices[de_bias_noisy_predictions]
    # Get new labels for confident noisy predictions
    new_labels_for_noisy = noisy_pseudo_labels[de_bias_noisy_predictions]
    
    # Update the given training labels with new labels for confident noisy predictions
    y_train_noisy_new_de_bias = np.array(y_train_given)
    y_train_noisy_new_de_bias[relabel_indices] = new_labels_for_noisy
    
    # Determine revised sample indices based on the number of times the index has been processed
    if idx_cnt[idx] >= 5:
        if estimated_noisy_level > 0.20:
            revised_sample_idx = set(de_bias_clean_indices) | set(relabel_indices)
        else:
            revised_sample_idx = set(de_bias_clean_indices)
    else:
        revised_sample_idx = set(clean_indices)
    
    # Update the sample labels if estimated noise level is above the threshold (default: 0.1)
    if estimated_noisy_level <= args.clean_set_thres:
        revised_sample_idx = set(sample_idx)
    else:
        dataset_train.train_labels = y_train_noisy_new_de_bias
    
    # Return the revised indices, updated training dataset, and de-biasing dictionary
    return revised_sample_idx, dataset_train, de_bias

def create_de_bias(args, dataset_train):
    """
    Initialize the de-biasing structure with default values.

    Args:
        args (Namespace): Configuration containing hyperparameters and settings.
        dataset_train (Dataset): The training dataset which has an attribute `train_labels`.

    Returns:
        de_bias (dict): A dictionary containing de-biasing information:
                        - 'de_biased_labels': The adjusted label values for the training dataset.
                        - 'de_biased_probs': The adjusted probability values for the training dataset.
                        - 'phat': A dictionary storing the initial estimated probability distribution for each class, for each client.
    """

    # Initialize the de-biasing dictionary with zeros for labels and probabilities
    de_bias = {
        'de_biased_labels': np.zeros(len(dataset_train.train_labels)),
        'de_biased_probs': np.zeros(len(dataset_train.train_labels)),
        'phat': {}
    }
    
    # Initialize 'phat' for each client with a function 'initial_phat' which is assumed to return
    # an initial probability distribution over the classes.
    for usr in range(args.num_clients):
        de_bias['phat'][usr] = initial_phat(args.num_classes)
        
    return de_bias

def local_model_warmup(dataset_train, sample_idx, net_local, w_locals):
    """
    Perform a warm-up training on the local model using the client's dataset.

    Args:
        dataset_train (Dataset): The training dataset.
        sample_idx (list or ndarray): Indices of samples selected for the local client.
        net_local (torch.nn.Module): The local model to be trained.
        w_locals (list): A list of model weights from all clients.

    Returns:
        w_locals (list): Updated list of model weights including the weights from the newly trained local model.
    """

    # Initialize local update process with the given arguments, dataset, and indices
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)

    # Perform local model weight update using a deep copy of the local model moved to the device
    w = local.update_weights(net=copy.deepcopy(net_local).to(args.device), epoch=args.local_ep)

    # Load the updated weights into the local model
    net_local.load_state_dict(copy.deepcopy(w))

    # Append the updated weights to the list of local model weights
    w_locals.append(copy.deepcopy(w))

    # Return the updated list of weights
    return w_locals

def initialize_seeds(seed):
    """
    Initialize seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def open_accuracy_file(txtpath):
    """
    Open a file for logging accuracy.
    """
    return open(os.path.join(txtpath, 'acc.txt'), 'a')

def warmup_training_phase(args, dataset_train, dataset_test, real_noise_level, netglob, net_local, dict_clients, f_acc):
    """
    Perform the initial training phase for federated learning.

    Args:
        args (Namespace): Configuration containing hyperparameters and settings.
        dataset_train (Dataset): The training dataset.
        dataset_test (Dataset): The test dataset to evaluate global model accuracy.
        real_noise_level (list): A list containing the actual noise levels for each client.
        netglob (torch.nn.Module): The global model that is shared across all clients.
        net_local (torch.nn.Module): The local model that is trained by individual clients.
        dict_clients (dict): A dictionary mapping client IDs to their respective data indices.
        f_acc (file object): The file object for logging the accuracy of the model.
    """
    cnt = 0  # Counter to keep track of the total number of training steps done across all clients

    # Iterate through the number of iterations specified for phase 1
    for iteration in range(args.iteration1):
        # Initialize a uniform probability distribution over the clients
        prob = [1 / args.num_clients] * args.num_clients

        # Perform training with a fraction of clients for each sub-iteration
        for _ in range(int(1 / args.frac1)):
            # Randomly select a subset of clients based on the probability distribution
            idxs_clients = np.random.choice(range(args.num_clients), int(args.num_clients * args.frac1), p=prob)
            w_locals = []  # List to store local model weights

            # Iterate through the selected clients
            for _, idx in enumerate(idxs_clients):
                prob[idx] = 0  # Set the selected client's probability to zero to avoid reselection
                if sum(prob) > 0:
                    # Recalculate probabilities to maintain a distribution over the remaining clients
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                # Load the global model parameters into the local model
                net_local.load_state_dict(netglob.state_dict())

                # Get the client's data indices and create a subset dataset for the client
                sample_idx = np.array(list(dict_clients[idx]))

                # Perform local model warmup training with the client's data
                w_locals = local_model_warmup(dataset_train, sample_idx, net_local, w_locals)

                # Test the local model on the global test dataset and record the accuracy
                acc_local_warmup = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                f_acc.write(f"Warmup_Training_Phase: iteration {iteration:2d}, count {cnt + 1:3d}, client {idx:03d} ({real_noise_level[idx]:.4f}), acc: {acc_local_warmup:.4f} \n")
                f_acc.flush()  # Flush the file buffer to ensure accuracy is written to file
                
                cnt += 1  # Increment the training step counter

            # Aggregate the local model weights to update the global model
            federated_model_aggregation(netglob, w_locals, idxs_clients, dict_clients)

def model_training_phase(args, dataset_train, dataset_test, netglob, net_local, m_clients, prob_clients, dict_clients, criterion, f_acc, global_noise_filter, de_bias):
    """
    Perform model fine-tuning phase in federated learning.

    Args:
        args (Namespace): Configuration containing hyperparameters and settings.
        dataset_train (Dataset): The training dataset.
        dataset_test (Dataset): The testing dataset.
        netglob (torch.nn.Module): The global model that is shared across all clients.
        net_local (torch.nn.Module): The local model that is trained by individual clients.
        dict_clients (dict): A dictionary mapping client IDs to their respective data indices.
        criterion: The loss criterion used for model training.
        f_acc (file object): The file object for logging the accuracy of the model.
        global_noise_filter: A filter used to handle noisy data samples in the dataset.
        de_bias: A mechanism to de-bias the model training process.
    """
    cnt = 0  # Counter for the number of local training updates
    idx_cnt = {n: 0 for n in range(args.num_clients)}  # Count of updates per client
    server_cached_local_filter = {}  # Cache for the local filters

    de_bias = create_de_bias(args, dataset_train)  # Initialize de-biasing mechanism

    for rnd in range(args.rounds1 + args.rounds2):
        args.current_ep = rnd  # Update the current epoch/round in args

        # Randomly select a subset of clients
        idxs_clients = np.random.choice(range(args.num_clients), m_clients, replace=False, p=prob_clients)

        w_locals = []  # List to store local model weights
        
        for idx in idxs_clients:
            cnt += 1

            # Load local data and model
            sample_idx = np.array(list(dict_clients[idx]))
            dataset_client = Subset(dataset_train, sample_idx)
            loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
            net_local.load_state_dict(netglob.state_dict())
            
            # Obtain output from the local model before local training
            local_output, loss, _ = get_output(loader, net_local.to(args.device), args, criterion)

            # Split local data based on loss values indicating potential noise
            local_split = local_data_splitting(loss, global_noise_filter)
            
            # Relabel noisy samples and reselect labeled samples
            revised_sample_idx, dataset_train, de_bias = relabeling_and_reselection(
                args, local_split, idx, sample_idx, idx_cnt, local_output, dataset_train, de_bias
            )

            # Train local model with revised samples
            w_locals, net_local, dataset_train = local_model_training(
                args, w_locals, net_local, netglob, revised_sample_idx, dataset_train
            )

            # Evaluate the local model after local training
            _, loss, local_logits = get_output(loader, net_local.to(args.device), args, criterion)
            
            # Test local model accuracy on the global test set
            acc_local = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
            f_acc.write(f"\nFederated_Model_Training_Phase: round {rnd:03d}-count {cnt:04d}-client {idx:03d}, testing acc: {acc_local:.4f}\n")
            f_acc.flush()

            # Update client-cached p_hat with new logits
            de_bias = client_cached_phat_update(args, de_bias, local_logits, idx, sample_idx)

            # Train the local filter to handle noisy data
            global_noise_filter, server_cached_local_filter = local_filter_training(
                loss, idx, global_noise_filter, server_cached_local_filter
            )

            idx_cnt[idx] += 1  # Increment the count for the current client

        # Aggregate local model weights to update the global model
        federated_model_aggregation(netglob, w_locals, idxs_clients, dict_clients)

        # Aggregate the filters from different clients to update the global noise filter
        federated_filter_aggregation(server_cached_local_filter, dict_clients, global_noise_filter)

        # Evaluate the global model accuracy on the global test set
        acc_global = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        f_acc.write(f"\nFederated_Model_Training_Phase: round {rnd:03d}, test acc {acc_global:.4f}\n")
        f_acc.flush()

def main(args, txtpath):
    initialize_seeds(args.seed)

    # Load datasets and initialize models
    dataset_train, dataset_test, dict_clients = get_dataset(args)
    netglob = build_model(args)
    net_local = build_model(args)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Add noise to the training labels
    y_train = np.array(dataset_train.train_labels)
    y_train_noisy, real_noise_level = add_noise(args, y_train, dict_clients)
    dataset_train.train_labels = y_train_noisy

    # Open file to log accuracy
    f_acc = open_accuracy_file(txtpath)

    ## Perform federated model warmup training phase
    warmup_training_phase(args, dataset_train, dataset_test, real_noise_level, netglob, net_local, dict_clients, f_acc)

    ## Prepare for federated model fine tuning phase
    m_clients = max(int(args.frac2 * args.num_clients), 1)
    prob_clients = [1 / args.num_clients for _ in range(args.num_clients)]
    global_noise_filter = Global_Noise_Filter(init_dataset=init_data, components=2, seed=args.seed, init_params='random')
    de_bias = create_de_bias(args, dataset_train)

    # Perform federated model fine tuning phase
    model_training_phase(args, dataset_train, dataset_test, netglob, net_local, m_clients, prob_clients, dict_clients, criterion, f_acc, global_noise_filter, de_bias)

    # Close the accuracy file
    f_acc.close()

if __name__ == '__main__':

    # Import the datetime module for timestamping
    import datetime

    # Parse command-line arguments
    args = args_parser()
    # Set the process ID in the arguments for later reference
    args.pid = str(os.getpid())
    print(args)

    # Set CUDA visibility to the specified GPU in the arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Define the root path for saving records
    rootpath = "./record/"

    # Get the current time for timestamping the run
    run_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # Create a directory to save text files if it doesn't already exist
    if not os.path.exists(rootpath + 'txtsave/'):
        os.makedirs(rootpath + 'txtsave/')

    # Initialize the folder name with the current time
    folder = "{}".format(run_time)

    # Add a remark to the folder name if provided
    folder += "-rm-{}".format(args.remark)

    # Add dataset, model, and noise levels to the folder name
    folder += '-%s_%s_NL_%.1f_LB_%.1f' % (args.dataset, args.model, args.level_n_system, args.level_n_lowerb)

    # Add IID or non-IID information to the folder name based on the arguments
    if args.iid:
        folder += "_IID"
    else:
        folder += "_nonIID_p_%.1f_dirich_%.1f" % (args.non_iid_prob_class, args.alpha_dirichlet)

    # Add fine-tuning indicator to the folder name if applicable
    if args.fine_tuning:
        folder += "_FT"
    # Add correction indicator to the folder name if applicable
    if args.correction:
        folder += "_CORR"
    # Add mixup information to the folder name if the mixup argument is provided
    if args.mixup:
        folder += "_Mix_%.1f" % (args.alpha)

    # Construct the full path for saving text files
    txtpath = os.path.join(rootpath, folder)

    # Save the configuration and path information to a run file
    txtpath = save_runfile(txtpath, args)

    print("WHO ARE YOU.")

    # Call the main function with the parsed arguments and the path for text saving
    main(args, txtpath)
