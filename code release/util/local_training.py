# python version 3.7.1
# -*- coding: utf-8 -*-

from matplotlib.pyplot import axis
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    # Sample lambda from a Beta distribution, or use 1 if alpha is not greater than 0
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # Get the batch size from the input tensor
    batch_size = x.size()[0]

    # Generate a random permutation of the batch indices
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # Perform mixup by combining images according to lambda (lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Get the corresponding pairs of targets
    y_a, y_b = y, y[index]

    # Return the mixed inputs, pair of targets, and the lambda value
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # Calculate the mixup loss for the first component of the mix
    loss_a = criterion(pred, y_a) * lam
    # Calculate the mixup loss for the second component of the mix
    loss_b = criterion(pred, y_b) * (1 - lam)
    # The final loss is the sum of the individual losses
    return loss_a + loss_b


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        """
        Initialize the DatasetSplit instance.
        
        Parameters:
        - dataset (Dataset): The original dataset which is to be split.
        - idxs (list or iterable): The indices of the samples that will form the subset.
        """
        self.dataset = dataset  # The original dataset
        self.idxs = list(idxs)  # Convert the indices to a list in case it's another iterable

    def __len__(self):
        # Return the length of the dataset subset.
        return len(self.idxs)  # The length is the number of indices

    def __getitem__(self, item):
        # Map the item index to the corresponding index in the original dataset
        image, label = self.dataset[self.idxs[item]]
        return image, label

def regularization_penalty(log_probs_mixed, args):
    # Create a uniform distribution prior across all classes
    prior = torch.ones(args.num_classes) / args.num_classes
    # Move the prior to the same device as the log_probs_mixed tensor
    prior = prior.to(log_probs_mixed.device)
    # Calculate mean predicted probabilities across the batch
    pred_mean = torch.softmax(log_probs_mixed, dim=1).mean(0)
    # Calculate the KL divergence between the prior and mean predictions
    penalty = torch.sum(prior * torch.log(prior / pred_mean))
    
    # Return the penalty scaled by the regularization coefficient
    return penalty * args.reg_coef

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args  # Store the passed arguments
        self.loss_func = nn.CrossEntropyLoss()  # Initialize the loss function (Cross Entropy Loss)
        self.sample_idxs = np.array(list(idxs))  # Convert the indices for data samples to a NumPy array
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))  # Create data loaders for training and test sets

    def train_test(self, dataset, idxs):
        # Create a DataLoader for the training set with a subset of indices
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # Create a DataLoader for the test set
        test = DataLoader(dataset, batch_size=128)
        return train, test  # Return the training and test DataLoaders

    def update_weights(self, net, epoch, lr=None):
        # Set the model to training mode
        net.train()
        # Initialize the optimizer with SGD and set the learning rate
        optimizer = torch.optim.SGD(net.parameters(), lr=lr or self.args.lr, momentum=self.args.momentum)
        
        # Iterate over the number of epochs
        for iter in range(epoch):
            batch_loss = []  # List to store loss of each batch
            
            # Loop over batches of data in the training DataLoader
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # Move the images and labels to the specified device (e.g., GPU)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                # If mixup augmentation is enabled
                if self.args.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(images, labels, self.args.alpha)
                    net.zero_grad()  # Zero the parameter gradients
                    log_probs = net(images)  # Get the log probabilities for the original images
                    log_probs_mixed = net(inputs)  # Get the log probabilities for the mixup images
                    # Calculate mixup loss
                    loss = mixup_criterion(self.loss_func, log_probs_mixed, targets_a, targets_b, lam)
                    
                    # If regularization is enabled, calculate regularization penalty and add to the loss
                    if self.args.reg:
                        penalty = regularization_penalty(log_probs_mixed, self.args)
                        loss += penalty * self.args.reg_coef

                else:
                    # Standard training without mixup
                    labels = labels.long()  # Ensure labels are of type long
                    net.zero_grad()  # Zero the parameter gradients
                    log_probs = net(images)  # Get the log probabilities for the images
                    loss = self.loss_func(log_probs, labels)  # Calculate the loss

                loss.backward()  # Perform backpropagation
                optimizer.step()  # Update the network weights

                batch_loss.append(loss.item())  # Append the loss of this batch to the list

        return net.state_dict()  # Return the state dictionary of the model after training

def globaltest(net, test_dataset, args):
    # Set the network to evaluation mode to turn off dropout and batch normalization
    net.eval()
    
    # Create a data loader for the test dataset
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    
    # Context manager that disables gradient computation
    with torch.no_grad():
        correct = 0  # Variable to store the number of correct predictions
        total = 0    # Variable to store the total number of labels
        
        # Iterate over the test data loader
        for images, labels in test_loader:
            # Move the images and labels to the device (CPU or GPU)
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            # Forward pass to get the outputs from the network
            outputs = net(images)
            # Slice the outputs if the number of output units exceeds the number of classes
            outputs = outputs[:, :args.num_classes]
            
            # Get the predicted classes by finding the max log-probability
            _, predicted = torch.max(outputs.data, 1)
            
            # Increment the total count
            total += labels.size(0)
            # Increment the correct count
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    acc = correct / total
    return acc



