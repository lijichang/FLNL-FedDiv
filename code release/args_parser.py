import argparse

def parse_args(is_federated):
    parser = argparse.ArgumentParser()
    parse_common(parser)

    if is_federated == True:
        parse_federated(parser)
    elif is_federated == False:
        parse_baseline(parser)
    else:
        print('Unspecified mode: federated or baseline?')

    args = parser.parse_args()
    
    return args

def parse_baseline(parser):
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of epochs of training.'
    )
    
    return

def parse_federated(parser):
    parser.add_argument(
        '--rounds', type=int, default=100, help='Number of rounds of training.'
    )
    parser.add_argument(
        '--local_epochs', type=int, default=10, help='Number of local epochs for each client at every round.'
    )
    parser.add_argument(
        '--K', type=int, default=100, help='Total number of clients.'
    )
    parser.add_argument(
        '--C', type=float, default=0.1, help='Fraction of clients to employ in each round. From 0 to 1.'
    )
    parser.add_argument(
        '--S', type=int, default=None, help='Number of shards for each client. If None data are assumed to be IID, otherwise are non-IID.'
    )

    return

def parse_common(parser):
    parser.add_argument(
        '--dataset', type=str, default='blob', help='Name of the dataset.'
    )
    parser.add_argument(
        '--components', type=int, default=2, help='Number of Gaussians to fit.'
    )
    parser.add_argument(
        '--seed', default=None, help='Number to have random consistent results across executions.'
    )
    parser.add_argument(
        '--init', type=str, default='random', help='Model initialization method: random or kmeans (over a fraction of the dataset).'
    )
    parser.add_argument(
        '--samples', type=int, default=10000, help='Number of samples to generate.'
    )
    parser.add_argument(
        '--features', type=int, default=1, help='Number of features for each generated sample.'
    )
    parser.add_argument(
        '--soft', type=int, default=1, help='Specifies if cluster bounds are soft or hard.'
    )
    parser.add_argument(
        '--plots_3d', type=int, default=0, help='Specifies if plots are to be done in 3D or 2D.'
    )
    parser.add_argument(
        '--plots_step', type=int, default=1, help='Specifies the number of rounds or epochs after which saving a plot.'
    )

    return
