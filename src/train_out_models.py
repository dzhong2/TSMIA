import numpy as np
import os
import argparse
import torch
from src.utils import read_dataset_from_npy, Logger
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from trainInformer import train as train_informer
from train_resnet import train as train_resnet
from train_fcn import train as train_fcn
#from augment import augment_series
import copy

from models import MLP2Layer, train_model, MLP1Layer
from sklearn.preprocessing import StandardScaler
from torchmetrics.functional.pairwise import pairwise_cosine_similarity



data_dir = './tmp'
log_dir = './logs'

train_dict = {'resnet': train_resnet, 'fcn': train_fcn, 'informer': train_informer}

def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='GunPoint')
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
    load_pretrained = False
    split_series = False
    all_res = []
    multivariate_datasets = ['CharacterTrajectories', 'ECG', 'KickvsPunch', 'NetFlow']
    arr_datasets = ['Adiac', 'Beef', 'CBF', 'CricketX', 'FaceAll', 'GunPoint', 'SyntheticControl', 'ECGFiveDays']
    method = args.method
    para = args.para
    if method == 'window_warp':
        para = int(para)
    t=0
    np.random.seed(args.seed + t)
    torch.manual_seed(args.seed + t)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    agg_set = f"size={args.aug_size}" if args.add_aug else "_noaug"

    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'multivariate_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    elif args.dataset in arr_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'arr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))
    else:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'ucr_datasets_' + str(args.shot) + '_shot', args.dataset + '.npy'))

    #load target_trainer
    target_trainer = pkl.load(open(f"{model_name}/{args.dataset}/trainer_{t}.pkl", "rb"))
    target_trainer.model = target_trainer.model.to(device)

    prob = target_trainer.pred_proba(target_trainer.model, X, y)
    y_ori = copy.deepcopy(y)
    y = prob#.argmax(axis=1)


    # pick targets: 10 member and 10 non_members
    # todo: for now targets are not included in Ds, 10 members and 10 non-members
    res = []
    for shadow_index in range(10):
        # load shadow training set
        train_idx_Ds, test_idx_Ds = pkl.load(open(f"shadow_data/{args.dataset}_S{shadow_index}{agg_set}.pkl", "rb"))
        # vulnerable targets
        '''
        # load target
        
        '''

        # pick targets
        #members_not_in_Ds = np.random.choice(np.setdiff1d(train_idx, train_idx_Ds), 5, replace=False)
        #non_members_not_in_Ds = np.random.choice(np.setdiff1d(test_idx, train_idx_Ds), 5, replace=False)
        shadow_trainer = pkl.load(open(f"{model_name}/{args.dataset}/shadow_trainer_S{shadow_index}{agg_set}.pkl", "rb"))
        pred_all = shadow_trainer.pred_proba(shadow_trainer.model, X, y)
        cosine_matrix = pairwise_cosine_similarity(torch.Tensor(pred_all))
        threshold = 0.95
        neighbor_counts = (cosine_matrix > threshold).sum(axis=1)
        neighbor_rank = np.argsort(neighbor_counts)

        df_for_pick = pd.DataFrame({'ids': np.arange(len(y)),
                                   'member_target': [1 if i in train_idx else 0 for i in range(len(y))],
                                   'member_shadow': [1 if i in train_idx_Ds else 0 for i in range(len(y))],
                                   'rank': neighbor_rank})

        df_non_ds = df_for_pick[df_for_pick['member_shadow'] == 0]
        # pick top 5 with highest rank value
        df_picked = df_non_ds.sort_values('rank', ascending=True).groupby('member_target').head(10)
        targets = df_picked['ids'].values
        targets_mems = df_picked['member_target'].values


        #targets = np.concatenate([members_not_in_Ds, non_members_not_in_Ds])
        #target_mems = np.concatenate([np.ones(5), np.zeros(5)])

        for i in range(len(targets)):
            target = targets[i]
            target_mem = targets_mems[i]
            '''aug_set = []
            [aug_set.append(np.float32(augment_series(X[target], args.method, dataset=args.dataset, sigma=args.para))) for i in range(len(train_idx) // 10)]
            X_train_with_aug = np.concatenate([X[train_idx], np.array(aug_set)])
            y_train_with_aug = np.concatenate([y[train_idx], np.ones(len(train_idx) // 10) * y[target]]).astype(np.int64)
            '''


            # finetune 20 epochs
            shadow_model_copy = copy.deepcopy(shadow_trainer)
            shadow_model_copy.model = shadow_model_copy.model.to(device)
            '''
            shadow_trainer.fit(shadow_trainer.model,
                               X_train_with_aug,
                               y_train_with_aug, epochs=20)'''

            S_in, S_out = None, None
            if target in train_idx_Ds:
                pass
            else:
                S_out_trainer = shadow_model_copy
                # add original target in X_train_in
                # load shadow dataset with aug
                X_train_shadow_dict = pkl.load(open(f"shadow_data/{args.dataset}_{shadow_index}_{method}_{para}_S{shadow_index}{agg_set}.pkl", "rb"))
                if args.add_aug:
                    X_train_shadow = X_train_shadow_dict['X_aug']
                    y_train_shadow = X_train_shadow_dict['y_aug']

                    X_train_in = np.concatenate([X[train_idx_Ds], X_train_shadow, X[[target]]])
                    y_train_in = np.concatenate([y[train_idx_Ds], y_train_shadow, [y[target]]]).astype(np.int64)
                    y_train_in = target_trainer.pred_proba(target_trainer.model, X_train_in, y_train_in)
                    label_index = np.concatenate([y_ori[train_idx_Ds], y_train_shadow.argmax(axis=1), [y_ori[target]]]).astype(np.int64)

                    candidate_X = np.concatenate([X, X_train_shadow])
                    candidate_y = np.concatenate([y_ori, y_train_shadow.argmax(axis=1)])

                    # get output of S_in and S_out on all data
                    candidate_member = np.concatenate([np.array([1 if i in train_idx_Ds else 0 for i in range(len(X))]),
                                                       np.ones(len(y_train_shadow))])
                else:
                    X_train_in = np.concatenate([X[train_idx_Ds], X[[target]]])
                    y_train_in = np.concatenate([y[train_idx_Ds], [y[target]]]).astype(np.int64)
                    y_train_in = target_trainer.pred_proba(target_trainer.model, X_train_in, y_train_in)
                    label_index = np.concatenate(
                        [y_ori[train_idx_Ds], [y_ori[target]]]).astype(np.int64)

                    candidate_X = X
                    candidate_y = y_ori
                    candidate_member = np.array([1 if i in train_idx_Ds else 0 for i in range(len(X))])

                # further finetune for 20 epochs
                S_in_trainer = copy.deepcopy(shadow_model_copy)
                S_in_trainer.model = S_in_trainer.model.to(device)
                S_in_trainer.fit(S_in_trainer.model, X_train_in, y_train_in, epochs=20)
                agg_set_add = agg_set + f"add_target{target}"
                if args.add_aug:
                    pkl.dump(S_in_trainer, open(
                        f"{model_name}/{args.dataset}/shadow_trainer_S{shadow_index}_{method}_{para}{agg_set_add}.pkl", "wb"))
                else:
                    pkl.dump(S_in_trainer, open(
                        f"{model_name}/{args.dataset}/shadow_trainer_S{shadow_index}{agg_set_add}.pkl",
                        "wb"))
'''
            P_in = S_in_trainer.pred_proba(S_in_trainer.model, candidate_X, candidate_y)
            P_in_true = np.array([p[candidate_y[i]] for i, p in enumerate(P_in)])
            P_out = S_out_trainer.pred_proba(S_out_trainer.model, candidate_X, candidate_y)
            P_out_true = np.array([p[candidate_y[i]] for i, p in enumerate(P_out)])

            P_diff = P_in_true - P_out_true
            # impacted ids: the ones with positive P_diff
            df_pdiff = pd.DataFrame(P_diff)
            df_pdiff['mem'] = [1 if i in train_idx_Ds else 0 for i in range(len(candidate_X))]
            # pick members and non-members with the highest P_diff as impact ids
            impact_mems = df_pdiff[df_pdiff['mem'] == 1].sort_values(by=0, ascending=False).index[:len(candidate_X) // 10]
            impact_non_mems = df_pdiff[df_pdiff['mem'] == 0].sort_values(by=0, ascending=False).index[:len(candidate_X) // 10]
            impact_ids = np.concatenate([impact_mems, impact_non_mems]) # id in candidates!


            #impact_ids = np.where(P_diff > 0.05)[0]

            #X_train: P_in||P_out
            X_train_mia = []
            y_train_mia = []
            for id in impact_ids:
                if id == target:
                    continue
                X_train_mia.append(np.concatenate([P_in[id], P_out[id]]))
                #y_train_mia.append(1 * (id in train_idx_Ds))
                y_train_mia.append(candidate_member[id])

            df_mia = pd.DataFrame(X_train_mia)
            df_mia['y'] = y_train_mia
            #df_mia['id'] = impact_ids
            min_number = df_mia.groupby('y').size().min()
            df_mia = df_mia.groupby('y').sample(n=min_number, replace=False)

            X_train_mia = df_mia.drop(columns='y').values
            y_train_mia = df_mia['y'].values
            id_train_mia = df_mia.index.values


            # train attack model
            ss = StandardScaler()
            #X_train_mia = ss.fit_transform(X_train_mia)

            clf = MLP2Layer(in_dim=X_train_mia.shape[1], out_dim=2, layer_list=[200, 200], device=torch.device(device))
            clf.criterion = torch.nn.CrossEntropyLoss()
            clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
            clf.to(torch.device(device))
            clf = train_model(clf, X_train_mia, y_train_mia, X_train_mia, y_train_mia, max_patient=100, display=0)
            performance = clf.all_metrics(X_train_mia, y_train_mia, verbos=False)

            PT = target_trainer.pred_proba(target_trainer.model, candidate_X[impact_ids][id_train_mia], candidate_y[impact_ids][id_train_mia])
            attack_T_as_Sin = np.concatenate([PT, P_out[impact_ids][id_train_mia]], axis=1)
            #attack_T_as_Sin = ss.transform(attack_T_as_Sin)
            # pick non-members only
            non_mem_index = np.where(y_train_mia == 0)[0]
            #acc_T_as_in = clf.all_metrics(attack_T_as_Sin, y_train_mia, verbos=False)
            acc_T_as_in = clf.acc(attack_T_as_Sin[non_mem_index], y_train_mia[non_mem_index])

            attack_T_as_Sout = np.concatenate([P_in[impact_ids][id_train_mia], PT], axis=1)
            #attack_T_as_Sout = ss.transform(attack_T_as_Sout)
            acc_T_as_out = clf.acc(attack_T_as_Sout[non_mem_index], y_train_mia[non_mem_index])
            if target_mem == 1 and acc_T_as_in > acc_T_as_out:
                print(f"target: {target} is correctly inferred as member")
                correct = 1
            elif target_mem == 0 and acc_T_as_in < acc_T_as_out:
                print(f"target: {target} is correctly inferred as non-member")
                correct = 1
            else:
                print(f"target: {target} is incorrectly inferred")
                correct = 0

            res.append([target, target_mem, acc_T_as_in, acc_T_as_out, correct])
        df_res = pd.DataFrame(res, columns=['target', 'target_mem', 'acc_T_as_in', 'acc_T_as_out', 'correct'])
    df_res = pd.DataFrame(res, columns=['target', 'target_mem', 'acc_T_as_in', 'acc_T_as_out', 'correct'])'''



