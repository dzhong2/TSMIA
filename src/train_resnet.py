import os
import argparse

import numpy as np
import torch

from src.utils import read_dataset_from_npy, Logger
from src.resnet.model import ResNet, ResNetTrainer
import pickle as pkl
import pandas as pd

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']
arr_datasets = ['Adiac', 'Beef', 'CBF', 'CricketX', 'FaceAll', 'GunPoint', 'SyntheticControl', 'ECGFiveDays', 'Mallat','SwedishLeaf']

def train(X_train, y_train, X_test, y_test, device, logger, epochs=100, DP=-1):
    if len(y_test.shape) == 1:
        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    else:
        nb_classes = y_train.shape[1]

    input_size = X_train.shape[1]
    model = ResNet(input_size, nb_classes)
    model = model.to(device)
    trainer = ResNetTrainer(device, logger, DP)

    model = trainer.fit(model, X_train, y_train, epochs=epochs)
    acc = trainer.test(model, X_test, y_test)
    trainer.model = model

    return acc, trainer


def argsparser():
    parser = argparse.ArgumentParser("Active Timeseries classification")
    parser.add_argument('--dataset', help='Dataset name', default='Coffee')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dp', type=float, default=-1)
    parser.add_argument('--shot', help='shot', type=int, default=1)

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    model_name = 'resnet'
    load_pretrained = True

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print("--> Running on the GPU: ", args.gpu)
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    all_res = []
    # Seeding
    for t in range(10):
        # Seeding
        np.random.seed(args.seed + t)
        torch.manual_seed(args.seed + t)

        log_dir = os.path.join(log_dir, 'resnet_log_' + str(args.shot) + '_shot')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        out_path = os.path.join(log_dir, args.dataset + '_' + str(args.seed + t) + '.txt')

        with open(out_path, 'w') as f:
            logger = Logger(f)
            # Read data
            if args.dataset in multivariate_datasets:
                X, y, train_idx, test_idx = read_dataset_from_npy(
                    os.path.join(data_dir, 'multivariate_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
            elif args.dataset in arr_datasets:
                X, y, train_idx, test_idx = read_dataset_from_npy(
                    os.path.join(data_dir, 'arr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
            else:
                X, y, train_idx, test_idx = read_dataset_from_npy(
                    os.path.join(data_dir, 'ucr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))

            if not os.path.exists(f"{model_name}/{args.dataset}/"):
                os.makedirs(f"{model_name}/{args.dataset}/")

            # Train the model
            if args.dp == -1:
                DP_str = ""
                print("No DP")
            else:
                DP_str = f"_dp_{args.dp}"
                print("DP: ", args.dp)
            if not load_pretrained or not os.path.exists(f"{model_name}/{args.dataset}/trainer{DP_str}_{t}.pkl"):
                acc, trainer = train(X[train_idx], y[train_idx], X[test_idx], y[test_idx], device, logger, DP=args.dp)
                delattr(trainer, "logger")
                pkl.dump(trainer, open(f"{model_name}/{args.dataset}/trainer{DP_str}_{t}.pkl", "wb"))
            else:
                print("Load previous models")
                trainer = pkl.load(open(f"{model_name}/{args.dataset}/trainer{DP_str}_{t}.pkl", "rb"))
                trainer.model = trainer.model.to(device)

            train_pred = trainer.pred_proba(trainer.model, X[train_idx], y[train_idx])
            test_pred = trainer.pred_proba(trainer.model, X[test_idx], y[test_idx])

            df_res = pd.DataFrame(np.concatenate([train_pred, test_pred]), columns=np.unique(y))
            df_res['truth'] = np.concatenate([y[train_idx], y[test_idx]])
            df_res['predict'] = np.concatenate([train_pred.argmax(axis=1),
                                                test_pred.argmax(axis=1)])

            df_res['correct'] = df_res['truth'] == df_res['predict']
            df_res['member'] = np.concatenate([np.ones(len(train_idx)), np.zeros(len(test_idx))])
            print(df_res.groupby('member')['correct'].mean())

            df_res['index'] = np.concatenate([train_idx, test_idx])

            df_res.to_csv(f"{model_name}/{args.dataset}/target_results_{t}.csv", index=False)

            all_res.append(df_res.groupby('member')['correct'].mean().values)

    all_res = np.array(all_res)
    print(all_res.mean(axis=0))
    pass

