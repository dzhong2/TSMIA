# given a pair of shadow models (Sin(T) and Sout), pick reference data
# construct attack feature: Sin(R)||Sout(R), sim(Sin(R), Sout(R)
# construct attack label: member of T
# check distribution

import pickle as pkl
import os
import seaborn as sns
import numpy as np
from models import MLP2Layer, train_model, MLP1Layer
from sklearn.preprocessing import StandardScaler
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from sklearn.model_selection import train_test_split
import torch
import scipy.stats as ss
import matplotlib.pyplot as plt

from src.utils import read_dataset_from_npy, Logger
from tqdm import tqdm
from sklearn import metrics
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock
import argparse
similarity_list = [cosine, euclidean, correlation, cityblock]

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'NetFlow']
arr_datasets = [ 'CricketX''ECGFiveDays']

def load_dist_matrix(dist):
    if dist == 'correlation':
        dist_matrix = pkl.load(open(f"datasets/correlate/{dataset}_corr.pkl", "rb"))
        dist_matrix = 1 - dist_matrix / 200
        for i in range(len(dist_matrix)):
            dist_matrix[i, i] = 0
    elif dist == 'dtw':
        dist_matrix = pkl.load(open(f"datasets/dtws/{dataset}_dtw.pkl", "rb"))
    elif dist == 'L2':
        dist_matrix = pkl.load(open(f"datasets/L2/{dataset}_l2.pkl", "rb"))
    elif dist == 'DFT':
        dist_matrix = pkl.load(open(f"datasets/DFT/{dataset}_dft.pkl", "rb"))
    elif dist == 'MTLC':
        dist_matrix = pkl.load(open(f"datasets/lag_corr/{dataset}_lag_corr.pkl", "rb"))
    elif dist == 'twed':
        dist_matrix = pkl.load(open(f"datasets/twed/{dataset}_twed.pkl", "rb"))
    else:
        dist_matrix = None
    return dist_matrix

def argsparser():
    parser = argparse.ArgumentParser("SimTSC")
    parser.add_argument('--dataset', help='Dataset name', default='CharacterTrajectories')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--num_ref', help='number of reference sample', type=int, default=3)
    parser.add_argument('--alpha', help='alpha', type=float, default=0.3)
    parser.add_argument('--use_prev', help='use pretrained model', action='store_true', default=False)
    parser.add_argument('--attack_options', help='XI, XS or XIXS', type=str, default='XIXS')

    return parser


def get_shadow_inputs(model_name, dataset, X, y, train_idx, test_idx, shadow_index=0):
    Sout_traininer = pkl.load(
        open(f"{model_name}/{dataset}/shadow_trainer_S{shadow_index}.pkl", "rb"))

    Sin_trainer_list = os.listdir(f"{model_name}/{dataset}")
    Sin_trainer_list = [f for f in Sin_trainer_list if 'shadow_trainer' in f and f"S{shadow_index}add_target" in f]

    # target model and get output
    target_trainer = pkl.load(open(f"{model_name}/{dataset}/trainer_0.pkl", "rb"))
    target_trainer.model = target_trainer.model.to(torch.device('cuda:0'))
    P_target_all = target_trainer.pred_proba(target_trainer.model, X, y)

    P_out_all = Sout_traininer.pred_proba(Sout_traininer.model, X, y)
    P_in_list = []
    P_out_list = []
    member_list = []
    added_list = []
    for Sin_trainer_file in tqdm(Sin_trainer_list):
        Sin_trainer = pkl.load(open(f"{model_name}/{dataset}/{Sin_trainer_file}", "rb"))
        # get reference samples

        P_in_all = Sin_trainer.pred_proba(Sin_trainer.model, X, y)

        P_in_list.append(P_in_all)
        P_out_list.append(P_out_all)
        added = int(Sin_trainer_file.split('target')[1].split('.pkl')[0])
        member_list.append(added in train_idx)
        added_list.append(added)

    pass
    # pick reference samples

    dist_dft = load_dist_matrix('DFT')
    dist_twed = load_dist_matrix('twed')
    dist_dtw = load_dist_matrix('dtw')

    for dist in ['None']:#, 'correlation', 'dtw', 'L2', 'DFT','MTLC', 'twed']:
        X_list = []
        y_list = []
        long_test_log_loss = []

        dist_matrix = load_dist_matrix(dist)

        if dist != 'None':
            dist_mean = dist_matrix.mean()
        dist_list = []
        distances_means = []
        for i in range(len(P_in_list)):
            use_sim_RS = True
            P_in = P_in_list[i]
            P_out = P_out_list[i]
            member = member_list[i]
            added = added_list[i]

            # pick reference samples
            if member == False:
                pass

            diff_P = P_in - P_out

            diff_P_list = np.array([diff_P[i, y[i]] for i in range(len(X))])
            threshold_P = diff_P_list[np.argsort(diff_P_list)[int(len(X) * 0.9)]]
            distances_mean = np.array([[dm[r, added] for dm in [dist_dtw, dist_twed, dist_dft]] for r in range(len(X))]).mean(axis=1)
            threshold_d = distances_mean[np.argsort(distances_mean)[int(len(X) * 0.1)]]



        X_all = np.concatenate(X_list)
        y_all = np.concatenate(y_list)

        X_mean_all = np.concatenate([[np.mean(x, axis=0)] for x in X_list], axis=0)
        y_mean_all = [y[0] for y in y_list]
        long_test_log_loss = np.concatenate(long_test_log_loss)
        # long's attack

        X_outs =X_all[:, P_out_all.shape[1]:]
        # calculate log loss
        log_loss = -np.log(X_outs[np.arange(len(X_outs)), y_all.astype(int)])

        ss = StandardScaler()
        X_all = ss.fit_transform(X_all)

        train_ft_ind = list(range(P_target_all.shape[1] * 2)) + list(range(P_target_all.shape[1] * 3, P_target_all.shape[1] * 3 + 3))
        test_ft_ind = list(range(P_target_all.shape[1] * 1)) + list(range(P_target_all.shape[1] * 2, P_target_all.shape[1] * 3 + 3))

        X_train_mia, X_test_mia, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
        X_train_mia_sub = X_train_mia[:, train_ft_ind]

        X_test_mia_sub = X_test_mia[:, test_ft_ind]
        # train attack model

        # concat mean solution
        ss = StandardScaler()
        X_mean_all = ss.fit_transform(X_mean_all)

        X_train_mean_mia, X_test_mean_mia, y_mean_train, y_mean_test = train_test_split(X_mean_all, y_mean_all, test_size=0.3, random_state=42)

        X_train_mean_mia_sub = X_train_mean_mia[:, train_ft_ind]
        X_test_mean_mia_sub = X_test_mean_mia[:, test_ft_ind]
        # similarities

        input_sims = []
        for t in range(X_all.shape[0]):
            P_in, P_out, P_target = X_all[t][:P_target_all.shape[1]], X_all[t][P_target_all.shape[1]:P_target_all.shape[1] * 2], X_all[t][P_target_all.shape[1] * 2:-3]
            tmp_input = []
            for sim in similarity_list:
                tmp_input.append(sim(P_in, P_out))
            for sim in similarity_list:
                tmp_input.append(sim(P_in, P_target))
            input_sims.append(tmp_input)

        X_mia_sim = np.array(input_sims)
        X_mia_sim = np.concatenate([X_mia_sim, X_all[:, -3:]], axis=1)
        ss = StandardScaler()
        X_mia_sim = ss.fit_transform(X_mia_sim)

        # attack
        train_ft_ind_sim = list(range(4)) + list(range(8, 11))
        test_ft_ind_sim = list(range(4, 11))
        X_train_sim_mia, X_test_sim_mia, y_sim_train, y_sim_test = train_test_split(X_mia_sim, y_all, test_size=0.3, random_state=42)
        X_train_sim_mia = X_train_sim_mia[:, train_ft_ind_sim]
        X_test_sim_mia = X_test_sim_mia[:, test_ft_ind_sim]

        sim_mean_X = []
        for xs in X_list:
            tmp = []
            for i in range(len(xs)):
                P_in, P_out, P_target = X_all[t][:P_target_all.shape[1]], X_all[t][P_target_all.shape[1]:P_target_all.shape[1] * 2], X_all[t][P_target_all.shape[1] * 2:-3]
                tmp_input = []
                for sim in similarity_list:
                    tmp_input.append(sim(P_in, P_out))
                for sim in similarity_list:
                    tmp_input.append(sim(P_in, P_target))
                tmp.append(tmp_input)
            sim_mean_X.append(np.mean(tmp, axis=0))
        sim_mean_X = np.array(sim_mean_X)
        sim_mean_X = np.concatenate([sim_mean_X, np.array(distances_means)], axis=1)
        sim_mean_y = [y[0] for y in y_list]
        ss = StandardScaler()
        sim_mean_X = ss.fit_transform(sim_mean_X)


        X_train_sim_mean_mia, X_test_sim_mean_mia, y_sim_mean_train, y_sim_mean_test = train_test_split(sim_mean_X, sim_mean_y, test_size=0.3, random_state=42)

        X_train_sim_mean_mia = X_train_sim_mean_mia[:, train_ft_ind_sim]
        X_test_sim_mean_mia = X_test_sim_mean_mia[:, test_ft_ind_sim]

        #return (X_all, y_all), (X_mean_all, y_mean_all), (X_mia_sim, y_all), (sim_mean_X, sim_mean_y)
        return (X_train_mia_sub, X_test_mia_sub, y_train, y_test)


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset



    # load dataset
    if dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'multivariate_datasets_30_shot', dataset + '.npy'))
    else:
        X, y, train_idx, test_idx = read_dataset_from_npy(
            os.path.join(data_dir, 'arr_datasets_30_shot', dataset + '.npy'))
    Xsys=get_shadow_inputs(model_name, dataset, X, y, train_idx, test_idx, shadow_index=si)
    solution_names = 'XI+XS'
    Xitrain, Xitest, yitrain, yitest = Xsys[0], Xsys[1], Xsys[2], Xsys[3]
    #Xi_concat = np.concatenate(Xi)
    #yi_concat = np.concatenate(yi)

    X_train_mia = np.concatenate(Xitrain)
    X_test_mia = np.concatenate(Xitest)
    y_train = np.concatenate(yitrain)
    y_test = np.concatenate(yitest)

    # attack
    #X_train_mia, X_test_mia, y_train, y_test = train_test_split(Xi_concat, yi_concat, test_size=0.3, random_state=42)
    clf = MLP2Layer(in_dim=X_train_mia.shape[1], out_dim=2, layer_list=[200, 200], device=torch.device('cuda:0'))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
    clf.to(torch.device('cuda:0'))
    clf = train_model(clf, X_train_mia, y_train, X_test_mia, y_test, max_patient=100, display=0)
    performance = clf.all_metrics(X_test_mia, y_test, verbos=False)
    print(f"attack performance of XIXS:\n {np.array(performance)[[0, -3]]}")

    Xitrain, Xitest, yitrain, yitest = ([Xsys[j][0][0] for j in range(len(Xsys))],
                                        [Xsys[j][0][1] for j in range(len(Xsys))],
                                        [Xsys[j][0][2] for j in range(len(Xsys))],
                                        [Xsys[j][0][3] for j in range(len(Xsys))])

    X_train_mia = np.concatenate(Xitrain)[:, :-3]
    X_test_mia = np.concatenate(Xitest)[:, :-3]
    y_train = np.concatenate(yitrain)
    y_test = np.concatenate(yitest)
    clf = MLP2Layer(in_dim=X_train_mia.shape[1], out_dim=2, layer_list=[200, 200], device=torch.device('cuda:0'))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
    clf.to(torch.device('cuda:0'))
    clf = train_model(clf, X_train_mia, y_train, X_test_mia, y_test, max_patient=100, display=0)
    performance = clf.all_metrics(X_test_mia, y_test, verbos=False)
    print(f"attack performance of XI only:\n {np.array(performance)[[0, -3]]}")

    # SP only
    #Xi, yi = [Xsys[j][2][0] for j in range(len(Xsys))], [Xsys[j][2][1] for j in range(len(Xsys))]
    #Xi_concat = np.concatenate(Xi)[:, :-3]
    #yi_concat = np.concatenate(yi)
    #X_train_mia, X_test_mia, y_train, y_test = train_test_split(Xi_concat, yi_concat, test_size=0.3, random_state=42)

    Xitrain, Xitest, yitrain, yitest = ([Xsys[j][2][0] for j in range(len(Xsys))],
                                        [Xsys[j][2][1] for j in range(len(Xsys))],
                                        [Xsys[j][2][2] for j in range(len(Xsys))],
                                        [Xsys[j][2][3] for j in range(len(Xsys))])

    X_train_mia = np.concatenate(Xitrain)[:, :-3]
    X_test_mia = np.concatenate(Xitest)[:, :-3]
    y_train = np.concatenate(yitrain)
    y_test = np.concatenate(yitest)
    clf = MLP2Layer(in_dim=X_train_mia.shape[1], out_dim=2, layer_list=[200, 200], device=torch.device('cuda:0'))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
    clf.to(torch.device('cuda:0'))
    clf = train_model(clf, X_train_mia, y_train, X_test_mia, y_test, max_patient=100, display=0)
    performance = clf.all_metrics(X_test_mia, y_test, verbos=False)
    print(f"attack performance of XI only :\n {np.array(performance)[[0, -3]]}")

    # SS only
    #Xi, yi = [Xsys[j][0][0] for j in range(len(Xsys))], [Xsys[j][0][1] for j in range(len(Xsys))]
    #Xi_concat = np.concatenate(Xi)[:, -3:]
    #yi_concat = np.concatenate(yi)
    #X_train_mia, X_test_mia, y_train, y_test = train_test_split(Xi_concat, yi_concat, test_size=0.3, random_state=42)

    Xitrain, Xitest, yitrain, yitest = ([Xsys[j][0][0] for j in range(len(Xsys))],
                                        [Xsys[j][0][1] for j in range(len(Xsys))],
                                        [Xsys[j][0][2] for j in range(len(Xsys))],
                                        [Xsys[j][0][3] for j in range(len(Xsys))])

    X_train_mia = np.concatenate(Xitrain)[:, -3:]
    X_test_mia = np.concatenate(Xitest)[:, -3:]
    y_train = np.concatenate(yitrain)
    y_test = np.concatenate(yitest)

    clf = MLP2Layer(in_dim=X_train_mia.shape[1], out_dim=2, layer_list=[200, 200], device=torch.device('cuda:0'))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
    clf.to(torch.device('cuda:0'))
    clf = train_model(clf, X_train_mia, y_train, X_test_mia, y_test, max_patient=100, display=0)
    performance = clf.all_metrics(X_test_mia, y_test, verbos=False)
    print(f"attack performance of SS-only:\n {np.array(performance)[[0, -3]]}")

    # CPSS with combination of SS (3 sims intotal, 6 combinations)
    #Xi, yi = [Xsys[j][0][0] for j in range(len(Xsys))], [Xsys[j][0][1] for j in range(len(Xsys))]

    Xitrain, Xitest, yitrain, yitest = ([Xsys[j][0][0] for j in range(len(Xsys))],
                                        [Xsys[j][0][1] for j in range(len(Xsys))],
                                        [Xsys[j][0][2] for j in range(len(Xsys))],
                                        [Xsys[j][0][3] for j in range(len(Xsys))])
    sim_names = ['DFT', 'twed', 'dtw']
    for sim_inds in [[0], [0, 1], [0,2], [1], [1,2], [2]]:
        CP_shape = Xitrain[0][:, :-3].shape[1]
        #Xi_concat = np.concatenate(Xi)[:, list(range(CP_shape)) + [CP_shape + i for i in sim_inds]]
        #yi_concat = np.concatenate(yi)

        #X_train_mia, X_test_mia, y_train, y_test = train_test_split(Xi_concat, yi_concat, test_size=0.3, random_state=42)
        X_train_mia = np.concatenate(Xitrain)[:, list(range(CP_shape)) + [CP_shape + i for i in sim_inds]]
        X_test_mia = np.concatenate(Xitest)[:, list(range(CP_shape)) + [CP_shape + i for i in sim_inds]]
        y_train = np.concatenate(yitrain)
        y_test = np.concatenate(yitest)

        clf = MLP2Layer(in_dim=X_train_mia.shape[1], out_dim=2, layer_list=[200, 200], device=torch.device('cuda:0'))
        clf.criterion = torch.nn.CrossEntropyLoss()
        clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
        clf.to(torch.device('cuda:0'))
        clf = train_model(clf, X_train_mia, y_train, X_test_mia, y_test, max_patient=100, display=0)
        performance = clf.all_metrics(X_test_mia, y_test, verbos=False)
        print(f"attack performance of CPSS with {[sim_names[sn] for sn in sim_inds]}:\n {np.array(performance)[[0, -3]]}")






