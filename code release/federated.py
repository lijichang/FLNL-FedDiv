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

    # Prepare clients
    clients = {}
    for idx_client in range(args.K):
        clients[idx_client] = Client(idx_client, train_dataset, clients_groups[idx_client])
        
        # clients_groups[idx_client] sample index in client idx

    # Prepare server --> init_dataset is given by 0.5% of the train_dataset randomly sampled
    init_dataset_size = int(train_dataset.shape[0] * 0.005)
    init_dataset = train_dataset[np.random.choice(train_dataset.shape[0], init_dataset_size, replace=False)]
    server = Server(args, init_dataset, clients, output_dir)
    
    server.plot(train_dataset, train_dataset_labels)
    round_history = {}
    for round in range(args.rounds):
        # zero test
        #server.start_round(round)
                            
        ## first test
        # idxs_round_clients = np.random.choice(range(server.n_clients), server.n_clients_round, replace=False)
        # lens = len(idxs_round_clients)
        # for cnt, idx in enumerate(idxs_round_clients):
        #     print(lens, cnt, idx)
        #     local_model = copy.deepcopy(server.model)
        #     local_dataset = np.array(train_dataset[list(clients_groups[idx])])
        #     #print('local_dataset', local_dataset, len(local_dataset), local_dataset.shape)
        #     #print('max data', max(local_dataset), 'min data', min(local_dataset))
        #     local_model.fit(local_dataset, epochs=server.args.local_epochs)
        #     round_history[idx] = local_model.history_
        
        # print('round_history', round_history, len(round_history))
        # server.set_parameters_from_clients_models(round_history)
        
        ## second test
        idxs_round_clients = np.random.choice(range(server.n_clients), server.n_clients_round, replace=False)
        selected_clients = []
        for idx in idxs_round_clients:
            client = Client(idx, train_dataset, clients_groups[idx])
            selected_clients.append(client)
        
        server.start_round_(round, selected_clients)
        
        server.average_clients_models(use_hellinger_distance=False, update_reference=False)
        server.update_server_model()

        if (round+1) % args.plots_step == 0: server.plot(train_dataset, train_dataset_labels, round)

    predicted_labels = server.model.predict(train_dataset)

    print('Done.')
