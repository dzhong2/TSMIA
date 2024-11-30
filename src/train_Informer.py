import os
import argparse

import numpy as np
import torch

from src.utils import read_dataset_from_npy, Logger
from src.Informer.model import InformerTrainer, Informer
import pickle as pkl
import pandas as pd
from dotted_dict import DottedDict

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']
arr_datasets = ['Adiac', 'Beef', 'CBF', 'CricketX', 'FaceAll', 'GunPoint', 'SyntheticControl', 'ECGFiveDays']

def train(X_train, y_train, X_test, y_test, device, logger, epochs=100):
    #nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    if len(np.concatenate((y_train, y_test), axis=0).shape) == 2:
        nb_classes = y_test.shape[1]
    else:
        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    input_size = X_train.shape[-1]
    input_ft = X_train.shape[-2]
    # init a config
    config = DottedDict()
    config['task_name'] = 'classification'
    config['pred_len'] = 0
    config['enc_in'] = input_ft
    config['dec_in'] = 7
    config['label_len'] = nb_classes
    config['d_model'] = 128
    config['embed'] = 'timeF'
    config['freq'] = 'h'
    config['dropout'] = 0.1
    config['factor'] = 1
    config['output_attention'] = False
    config['n_heads'] = 8
    config['d_ff'] = 256
    config['activation'] = 'gelu'
    config['e_layers'] = 3
    config['d_layers'] = 1
    config['distil'] = False
    config['seq_len'] = input_size
    config['c_out'] = 7
    config['num_class'] = nb_classes


    model = Informer(config)
    model = model.to(device)
    trainer = InformerTrainer(device, logger)

    model = trainer.fit(model, X_train, y_train, epochs=epochs)
    acc = trainer.test(model, X_test, y_test)
    trainer.model = model

    return acc, trainer


def argsparser():
    parser = argparse.ArgumentParser("Active Timeseries classification")
    parser.add_argument('--dataset', help='Dataset name', default='CharacterTrajectories')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--shot', help='shot', type=int, default=30)

    return parser


if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    model_name = 'informer'
    load_pretrained = False

    # Setup the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    all_res = []
    # Seeding
    for t in range(10):
        # Seeding
        np.random.seed(args.seed + t)
        torch.manual_seed(args.seed + t)

        log_dir = os.path.join(log_dir, 'fcn_log_' + str(args.shot) + '_shot')
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
            if not load_pretrained or not os.path.exists(f"{model_name}/{args.dataset}/trainer_{t}.pkl"):
                acc, trainer = train(X[train_idx], y[train_idx], X[test_idx], y[test_idx], device, logger)
                delattr(trainer, "logger")
                pkl.dump(trainer, open(f"{model_name}/{args.dataset}/trainer_{t}.pkl", "wb"))
            else:
                print("Load previous model")
                trainer = pkl.load(open(f"{model_name}/{args.dataset}/trainer_{t}.pkl", "rb"))
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