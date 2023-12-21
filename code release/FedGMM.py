import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import copy

from client import Client
from server import Server
from utils import get_dataset, plot_metric, prepare_output_dir
from utils import print_configuration, save_configuration
from args_parser import parse_args

if __name__ == '__main__':    
    args = parse_args(is_federated=True)
    if args.seed: 
        random.seed(int(args.seed))
        np.random.RandomState(int(args.seed))
    
    output_dir = prepare_output_dir()

    train_dataset, train_dataset_labels, clients_groups = get_dataset(args)
    
    train_dataset_old = train_dataset
    
    train_dataset = (train_dataset - min(train_dataset)) / (max(train_dataset) - min(train_dataset))

    print_configuration(args, train_dataset, True)
    save_configuration(args, train_dataset, output_dir, True)


    # Prepare server --> init_dataset is given by 0.5% of the train_dataset randomly sampled
    init_dataset_size = int(train_dataset.shape[0] * 0.005)
    init_dataset = train_dataset[np.random.choice(train_dataset.shape[0], init_dataset_size, replace=False)]
    server = Server(init_dataset=init_dataset, components=2, local_epochs=10, seed=False, init_params='random')
    
    server.plot(train_dataset, train_dataset_labels, output_dir)
    round_history = {}
    n_clients_round = 10
    for round in range(args.rounds):
        idxs_round_clients = np.random.choice(range(100), n_clients_round, replace=False)
        round_history = {}
        for idx in idxs_round_clients:
            local = train_dataset[np.array(list(clients_groups[idx]))]
            client = Client(idx, local)
            round_history = server.start_round([client], round_history)
        server.set_parameters_from_clients_models(round_history)
        
        server.average_clients_models(n_clients_round=n_clients_round)
        server.update_server_model()

        if (round+1) % args.plots_step == 0: server.plot(train_dataset, train_dataset_labels, output_dir, round)

    predicted_labels = server.model.predict(train_dataset)

    print('Done.')
