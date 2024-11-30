import copy

import numpy as np
import os
import argparse
import torch
from src.utils import read_dataset_from_npy, Logger
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy

from trainInformer import train as train_informer
from train_resnet import train as train_resnet
from train_fcn import train as train_fcn
import argparse




data_dir = './tmp'
log_dir = './logs'

train_dict = {'resnet': train_resnet, 'fcn': train_fcn, 'informer': train_informer}

def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='CharacterTrajectories')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--shot', help='shot', type=int, default=30)
    parser.add_argument('--K', help='K', type=int, default=3)
    parser.add_argument('--alpha', help='alpha', type=float, default=0.3)
    parser.add_argument('--use_prev', help='alpha', action='store_true', default=False)
    parser.add_argument('--para', help='parameter of augmentation', type=float, default=0.1)
    parser.add_argument('--method', help='augmentation', type=str, default='jitter')
    parser.add_argument('--add_aug', help='add augmentation', action='store_false', default=True)
    parser.add_argument('--aug_size', help='ratio of augmentation data', type=float, default=0.5)

    return parser


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    model_name = args.model
    load_pretrained = True
    split_series = False
    method = args.method
    para = args.para
    all_res = []
    multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']
    arr_datasets = ['Adiac', 'Beef', 'CBF', 'CricketX', 'FaceAll', 'GunPoint', 'SyntheticControl', 'ECGFiveDays',
                    'Mallat', 'SwedishLeaf']

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'multivariate_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    elif args.dataset in arr_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'arr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    else:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'ucr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    if method == 'window_warp':
        para = int(para)

    t = 0
    # load target and run, replace y with output of y
    target_trainer = pkl.load(open(f"{model_name}/{args.dataset}/trainer_{t}.pkl", "rb"))
    target_trainer.model = target_trainer.model.to(device)
    prob = target_trainer.pred_proba(target_trainer.model, X, y)
    y_ori = copy.deepcopy(y)
    y = prob#.argmax(axis=1)

    num_shadow = int(args.shot * args.aug_size)
    num_aug = int(args.shot * (1 - args.aug_size) * y.shape[1])

    # construct or load shadow model
    for shadow_index in range(10):
        np.random.seed(args.seed + shadow_index * 10)
        torch.manual_seed(args.seed + shadow_index * 10)
        agg_set = f"size={args.aug_size}" if args.add_aug else "_noaug"

        if not load_pretrained or not os.path.exists(f"shadow_data/{args.dataset}_S{shadow_index}{agg_set}.pkl"):
            df_train_test = pd.DataFrame({"id": np.arange(X.shape[0]), "label": y_ori})
            df_train_test = df_train_test.loc[~df_train_test['id'].isin(train_idx)]
            sizes = df_train_test.groupby('label').size()
            if sizes.min() < num_shadow:
                df_train = df_train_test.groupby('label').sample(n=sizes.min() // 2, random_state=args.seed + shadow_index * 10)
            else:
                df_train = df_train_test.groupby('label').sample(n=num_shadow, random_state=args.seed + shadow_index * 10)
            df_test = df_train_test[~df_train_test['id'].isin(df_train['id'])]
            train_idx_shadow = df_train['id'].values
            test_idx_shadow = df_test['id'].values
            test_idx_shadow = np.concatenate([test_idx_shadow, train_idx])

            # save train_idx and test_idx
            pkl.dump([train_idx_shadow, test_idx_shadow], open(f"shadow_data/{args.dataset}_S{shadow_index}{agg_set}.pkl", "wb"))
        else:
            train_idx_shadow, test_idx_shadow = pkl.load(open(f"shadow_data/{args.dataset}_S{shadow_index}{agg_set}.pkl", "rb"))
        out_path = os.path.join(log_dir, args.dataset + '_' + str(args.seed + shadow_index * 10) + '.txt')
        # train shadow model
        with open(out_path, 'w') as f:
            logger = Logger(f)
            train = train_dict[args.model]

            log_dir = os.path.join(log_dir, f'{args.model}_log_' + str(args.shot) + '_shot')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            out_path = os.path.join(log_dir, args.dataset + '_' + str(args.seed + shadow_index * 10) + '.txt')
            #acc, trainer = train(X[train_idx], y[train_idx], X[test_idx], y[test_idx], device, logger)

            if not load_pretrained or not os.path.exists(f"{model_name}/{args.dataset}/shadow_trainer_S{shadow_index}_{method}_{para}{agg_set}.pkl"):
                # construct train_shadow: full X_train and augmented X_train
                if not args.add_aug:
                    X_shadow_train = X[train_idx_shadow]
                    y_shadow_train = y[train_idx_shadow]

                    dict_data_aug = {'X': X, 'y': y, 'train_id': train_idx_shadow, 'test_id': test_idx_shadow}
                    pkl.dump(dict_data_aug, open(
                        f"shadow_data/{args.dataset}_{shadow_index}_S{shadow_index}{agg_set}.pkl",
                        "wb"))

                else:
                    print(f"loading {method} augmentation with para={para}")
                    X_aug_list = pkl.load(open(f"aug_data/{args.dataset}_{method}_{para}.pkl", "rb"))
                    aug_index = np.random.choice(test_idx, num_aug, replace=False)

                    X_aug_train = np.concatenate([X_aug[aug_index] for X_aug in X_aug_list])
                    y_aug_train = np.concatenate([y[aug_index] for _ in range(len(X_aug_list))])
                    X_aug_train_idx = np.random.choice(len(X_aug_train), num_aug, replace=False)
                    X_aug_train = X_aug_train[X_aug_train_idx]
                    y_aug_train = y_aug_train[X_aug_train_idx]
                    # save train_aug as dictionary
                    dict_data_aug = {'X': X, 'y': y, 'train_id': train_idx_shadow, 'test_id':test_idx_shadow,
                                    'X_aug': X_aug_train, 'y_aug': y_aug_train}
                    pkl.dump(dict_data_aug, open(f"shadow_data/{args.dataset}_{shadow_index}_{method}_{para}_S{shadow_index}{agg_set}.pkl", "wb"))
                    X_shadow_train = np.concatenate([X[train_idx_shadow], X_aug_train])
                    y_shadow_train = np.concatenate([y[train_idx_shadow], y_aug_train])

                acc, trainer = train(X_shadow_train, y_shadow_train, X[test_idx_shadow], y[test_idx_shadow], device, logger)
                delattr(trainer, "logger")
                pkl.dump(trainer, open(f"{model_name}/{args.dataset}/shadow_trainer_S{shadow_index}{agg_set}.pkl", "wb"))
            else:
                print("Load previous model")
                trainer = pkl.load(open(f"{model_name}/{args.dataset}/shadow_trainer_S{shadow_index}{agg_set}.pkl", "rb"))
                trainer.model = trainer.model.to(device)

            train_pred = trainer.pred_proba(trainer.model, X[train_idx_shadow], y[train_idx_shadow])
            test_pred = trainer.pred_proba(trainer.model, X[test_idx_shadow], y[test_idx_shadow])

            df_res = pd.DataFrame(np.concatenate([train_pred, test_pred]), columns=np.unique(y_ori))
            df_res['truth'] = np.concatenate([y_ori[train_idx_shadow], y_ori[test_idx_shadow]])
            df_res['predict'] = np.concatenate([train_pred.argmax(axis=1),
                                                test_pred.argmax(axis=1)])

            df_res['correct'] = df_res['truth'] == df_res['predict']
            df_res['member'] = np.concatenate([np.ones(len(train_idx_shadow)), np.zeros(len(test_idx_shadow))])
            print(df_res.groupby('member')['correct'].mean())

            df_res['index'] = np.concatenate([train_idx_shadow, test_idx_shadow])

            #df_res.to_csv(f"{model_name}/{args.dataset}/shadow_results_{shadow_index}_{method}_{para}.csv", index=False)