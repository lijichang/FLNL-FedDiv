import os

command_federated = 'python {} --components=5 --rounds=50 --dataset=blob --K=100 --C=0.1 --plots_step=10 --seed=111 --init=kmeans'.format('.\\src\\federated.py')
command_baseline = 'python {} --components=5 --epochs=100 --dataset=blob --plots_step=20 --seed=111 --init=kmeans'.format('.\\src\\baseline.py')

for _ in range(10):
    os.system(command_federated)