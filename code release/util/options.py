# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=200, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac1', type=float, default=0.01, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.1, help="fration of selected clients in fine-tuning and usual training stage")

    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")
    
    # noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")

    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")

    # ablation study
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--reg_coef', type=float, default=1.0, help="0.1,1,5")
    parser.add_argument('--contrast', action='store_true')
    
    parser.add_argument('--alpha', type=float, default=1, help="0.1,1,5")
    parser.add_argument('--prediction_decay', type=float, default=0.6, help="0.6")

    parser.add_argument('--remark', default='', type=str)

    parser.add_argument('--server', default="sysu26", type=str, required=False, help="Server Name")
    parser.add_argument('--expid', default="000", type=str, required=False, help="EXP ID")
    parser.add_argument('--gpu', type=int, default=3, help="gpu")


    parser.add_argument('--current_ep', type=int, default=0, help="number of local epochs")
    parser.add_argument('--neg_loss', action='store_true') #norm_extremum
    parser.add_argument('--norm_extremum', action='store_true')
    parser.add_argument('--feature_dim', type=int, default=0, help="dimension of feature output")
    parser.add_argument('--lambda_c', type=float, default=0.025, help="0.1,1,5") # lambda_mse
    parser.add_argument('--lambda_mse', type=float, default=1.0, help="0.1,1,5") # lambda_mse
    parser.add_argument('--conf', type=float, default=0.0, help="0.1,1,5") # ramp_up_mult
    parser.add_argument('--ramp_up_mult', type=float, default=20.0, help="0.1,1,5") # ramp_up_mult

    parser.add_argument('--xi', type=int, default=0.2, help="")
    parser.add_argument('--m', type=int, default=0.5, help="")

    
    parser.add_argument('--pid', type=str, default='0', help="pid")
    return parser.parse_args()
