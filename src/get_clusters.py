import numpy as np
from kshape.core import KShapeClusteringCPU
from kshape.core_gpu import KShapeClusteringGPU
import argparse
from src.utils import read_dataset_from_npy, Logger
import os
import pickle as pkl
from Kmedoids import *

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']
arr_datasets = ['Adiac', 'Beef', 'CBF', 'CricketX', 'FaceAll', 'GunPoint', 'SyntheticControl', 'ECGFiveDays']

data_dir = './tmp'
log_dir = './logs'

def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='CharacterTrajectories')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--method', help='augmentation', type=str, default='jitter')

    parser.add_argument('--shot', help='shot', type=int, default=30)
    parser.add_argument('--K', help='K', type=int, default=3)
    parser.add_argument('--start_target', help='starting ind of target', type=int, default=0)
    parser.add_argument('--alpha', help='alpha', type=float, default=0.3)
    parser.add_argument('--use_prev', help='alpha', action='store_true', default=False)
    parser.add_argument('--tune', help='tune: limited number', action='store_true', default=False)
    parser.add_argument('--para', help='parameter of augmentation', type=float, default=0.1)
    parser.add_argument('--add_aug', help='add augmentation', action='store_false', default=True)

    parser.add_argument('--aug_size', help='ratio of augmentation data', type=float, default=0.5)


    return parser


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    model_name = args.model
    method = args.method

    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'multivariate_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    elif args.dataset in arr_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'arr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    else:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'ucr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))

    K = 10

    train_idx_Ds, test_idx_Ds = pkl.load(open(f"shadow_data/{args.dataset}_S{0}_noaug.pkl", "rb"))

    X_train = X[train_idx_Ds]

    num_clusters = len(X_train) // K
    if method == 'kshape':
        # swap axes 1 and 2
        #X_train = np.swapaxes(X_train, 1, 2)
        # GPU Model
        ksg = KShapeClusteringGPU(num_clusters, centroid_init='zero', max_iter=10)
        ksg.fit(X_train)

        labels = ksg.labels_
        cluster_centroids = ksg.centroids_.detach().cpu()

        # save the cluster centroids and labels
        if not os.path.exists(f"datasets/kshape/"):
            os.makedirs(f"datasets/kshape/")
        pkl.dump((cluster_centroids, labels), open(f"datasets/kshape/{args.dataset}_S{0}.pkl", "wb"))
    elif method == 'kmedoids':
        if not os.path.exists(f"datasets/kmedoids/"):
            os.makedirs(f"datasets/kmedoids/")
        labels, sse_all, j, closest_observations_prev = Kmedoids(X_train, num_clusters, 10, 1)
        pkl.dump((labels, sse_all, j, closest_observations_prev), open(f"datasets/kmedoids/{args.dataset}_S{0}.pkl", "wb"))
