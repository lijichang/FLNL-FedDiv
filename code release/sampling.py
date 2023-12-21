import math
import numpy as np
# from torchvision import datasets, transforms

def sample_iid(dataset, n_clients):
    n_samples = int(len(dataset) / n_clients)
    clients_groups, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(n_clients):
        clients_groups[i] = set(np.random.choice(all_idxs, n_samples, replace=False))
        all_idxs = list(set(all_idxs) - clients_groups[i])

    return clients_groups

def sample_non_iid(dataset, n_clients, shards_per_client = 2):

    # shards per client
    shards_per_client = shards_per_client
    n_shards = n_clients * shards_per_client
    n_samples = math.floor(len(dataset) / n_shards)

    shards_idxs = [i for i in range(n_shards)]
    clients_groups = {i: np.array([]) for i in range(n_clients)}
    idxs = np.arange(n_shards * n_samples)
    print(idxs)

    for client_idx in range(n_clients):
        # Pick randomly the n shards for this client
        client_shards_idxs = np.random.choice(shards_idxs, shards_per_client, replace=False)
        # Remove the selected n shards from the available ones
        shards_idxs = list(set(shards_idxs) - set(client_shards_idxs))
        
        for shard_idx in set(client_shards_idxs):
            clients_groups[client_idx] = np.concatenate(
                (clients_groups[client_idx], idxs[shard_idx * n_samples : (shard_idx + 1) * n_samples]), 
                axis=0
            )

        clients_groups[client_idx] = clients_groups[client_idx].astype(int)

    return clients_groups

# def mnist_noniid_unequal(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30

#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)

#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:

#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         random_shard_size = random_shard_size-1

#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#     else:

#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#     return dict_users