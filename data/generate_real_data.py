import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import argparse
import pdb
import h5py
from collections import defaultdict
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# METABRIC
parser = argparse.ArgumentParser(description='Data Gen')

parser.add_argument('--dataset', type=str, default='metabric', choices=['whas', 'metabric', 'gbsg', 'nacd',
                                                                        'sequence', 'support', 'mimic',
                                                                        'liver', 'stomach', 'bladder'])
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
                
    return datasets

if args.dataset in ['whas', 'metabric', 'gbsg', 'support']:
    if args.dataset == 'whas':
        dataset = load_datasets("./data/KSP/whas/whas_train_test.h5")
    
    elif args.dataset == 'metabric':
        dataset = load_datasets("./data/KSP/metabric/metabric_IHC4_clinical_train_test.h5")

    elif args.dataset == 'gbsg':
        dataset = load_datasets("./data/KSP/gbsg/gbsg_cancer_train_test.h5")

    else:
        dataset = load_datasets("./data/KSP/support/support_train_test.h5")
    
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    dataset_train_t = dataset_train['t']
    dataset_train_e = dataset_train['e']
    dataset_train_x = dataset_train['x']

    dataset_test_t = dataset_test['t']
    dataset_test_e = dataset_test['e']
    dataset_test_x = dataset_test['x']

    dataset_t = pd.DataFrame(np.concat([dataset_train_t, dataset_test_t]), columns=['time'])
    dataset_e = pd.DataFrame(np.concat([dataset_train_e, dataset_test_e]), columns=['status'])
    dataset_x = pd.DataFrame(np.concat([dataset_train_x, dataset_test_x]))
    # dataset_x.to_csv(f'./{args.dataset}.csv')
    # assert False
    mask = (dataset_t['time'] > 0)
    dataset_t = dataset_t[mask]
    dataset_e = dataset_e[mask]
    dataset_x = dataset_x[mask]

    dataset = pd.concat([dataset_t, dataset_e, dataset_x], axis=1)
    # dataset.to_csv(f'./{args.dataset}.csv')

    from sklearn.metrics import silhouette_score
    from sklearn_extra.cluster import KMedoids
    import gower

    # Gower distance matrix
    gower_dist = gower.gower_matrix(dataset_x)

    def run_kmedoids_and_silhouette(dist_matrix, min_k=2, max_k=10):
        results = []
        best_score = -1
        best_model = None

        for k in range(min_k, max_k+1):
            model = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
            labels = model.fit_predict(dist_matrix)

            if len(set(labels)) > 1:  # silhouette cannot compute with 1 cluster
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            else:
                score = -1

            results.append((k, score))

            if score > best_score:
                best_score = score
                best_model = model

            print(f"K={k}, silhouette={score:.4f}")

        return results, best_model

    results, best_model = run_kmedoids_and_silhouette(gower_dist, min_k=2, max_k=6)
    print("\nOptimal K:", best_model.n_clusters)
    print("Cluster assignments:", best_model.labels_)
    
    best_k = best_model.n_clusters
    labels = best_model.labels_
    dataset['labels'] = labels

    from sklearn.model_selection import train_test_split

    dataset["strata"] = dataset["status"].astype(str) + "_" + dataset["labels"].astype(str)

    from sklearn.model_selection import train_test_split

    dataset_train, dataset_test_temp = train_test_split(dataset, test_size=0.4, stratify=dataset['strata'], random_state=args.seed)
    dataset_valid, dataset_test = train_test_split(dataset_test_temp, test_size=0.5, stratify=dataset_test_temp['strata'], random_state=args.seed)

    dataset_train_t = torch.tensor(dataset_train['time'].values)
    dataset_train_e = torch.tensor(dataset_train['status'].values)
    dataset_train_x = dataset_train.drop(columns=['time', 'status', 'labels', 'strata'])

    dataset_valid_t = torch.tensor(dataset_valid['time'].values)
    dataset_valid_e = torch.tensor(dataset_valid['status'].values)
    dataset_valid_x = dataset_valid.drop(columns=['time', 'status', 'labels', 'strata'])

    dataset_test_t = torch.tensor(dataset_test['time'].values)
    dataset_test_e = torch.tensor(dataset_test['status'].values)
    dataset_test_x = dataset_test.drop(columns=['time', 'status', 'labels', 'strata'])
    
    scaler = StandardScaler()
    
    if args.dataset == 'whas':
        dataset_train_x_cont = dataset_train_x.iloc[:, [1, 3]]
        dataset_train_x_cont = torch.tensor(scaler.fit_transform(dataset_train_x_cont))
        dataset_train_x_cate = torch.tensor(dataset_train_x.iloc[:, [0, 2, 4, 5]].values)
        dataset_train_x = torch.concat([dataset_train_x_cont, dataset_train_x_cate], dim=1)

        dataset_train_labels = torch.tensor(dataset_train['labels'].values)

        dataset_valid_x_cont = dataset_valid_x.iloc[:, [1, 3]]
        dataset_valid_x_cont = torch.tensor(scaler.transform(dataset_valid_x_cont))
        dataset_valid_x_cate = torch.tensor(dataset_valid_x.iloc[:, [0, 2, 4, 5]].values)
        dataset_valid_x = torch.concat([dataset_valid_x_cont, dataset_valid_x_cate], dim=1)

        dataset_valid_labels = torch.tensor(dataset_valid['labels'].values)

        dataset_test_x_cont = dataset_test_x.iloc[:, [1, 3]]
        dataset_test_x_cont = torch.tensor(scaler.transform(dataset_test_x_cont))
        dataset_test_x_cate = torch.tensor(dataset_test_x.iloc[:, [0, 2, 4, 5]].values)
        dataset_test_x = torch.concat([dataset_test_x_cont, dataset_test_x_cate], dim=1)

        dataset_test_labels = torch.tensor(dataset_test['labels'].values)

    elif args.dataset == 'metabric':
        dataset_train_x_cont = dataset_train_x.iloc[:, [0, 1, 2, 3, 8]]
        dataset_train_x_cont = torch.tensor(scaler.fit_transform(dataset_train_x_cont))
        dataset_train_x_cate = torch.tensor(dataset_train_x.iloc[:, [4, 5, 6, 7]].values)
        dataset_train_x = torch.concat([dataset_train_x_cont, dataset_train_x_cate], dim=1)

        dataset_train_labels = torch.tensor(dataset_train['labels'].values)

        dataset_valid_x_cont = dataset_valid_x.iloc[:, [0, 1, 2, 3, 8]]
        dataset_valid_x_cont = torch.tensor(scaler.transform(dataset_valid_x_cont))
        dataset_valid_x_cate = torch.tensor(dataset_valid_x.iloc[:, [4, 5, 6, 7]].values)
        dataset_valid_x = torch.concat([dataset_valid_x_cont, dataset_valid_x_cate], dim=1)

        dataset_valid_labels = torch.tensor(dataset_valid['labels'].values)

        dataset_test_x_cont = dataset_test_x.iloc[:, [0, 1, 2, 3, 8]]
        dataset_test_x_cont = torch.tensor(scaler.transform(dataset_test_x_cont))
        dataset_test_x_cate = torch.tensor(dataset_test_x.iloc[:, [4, 5, 6, 7]].values)
        dataset_test_x = torch.concat([dataset_test_x_cont, dataset_test_x_cate], dim=1)

        dataset_test_labels = torch.tensor(dataset_test['labels'].values)

    elif args.dataset == 'gbsg':
        dataset_train_x_cont = dataset_train_x.iloc[:, 3:]
        dataset_train_x_cont = torch.tensor(scaler.fit_transform(dataset_train_x_cont))
        dataset_train_x_cate = torch.tensor(dataset_train_x.iloc[:, :3].values)
        dataset_train_x = torch.concat([dataset_train_x_cont, dataset_train_x_cate], dim=1)

        dataset_train_labels = torch.tensor(dataset_train['labels'].values)

        dataset_valid_x_cont = dataset_valid_x.iloc[:, 3:]
        dataset_valid_x_cont = torch.tensor(scaler.transform(dataset_valid_x_cont))
        dataset_valid_x_cate = torch.tensor(dataset_valid_x.iloc[:, :3].values)
        dataset_valid_x = torch.concat([dataset_valid_x_cont, dataset_valid_x_cate], dim=1)

        dataset_valid_labels = torch.tensor(dataset_valid['labels'].values)

        dataset_test_x_cont = dataset_test_x.iloc[:, 3:]
        dataset_test_x_cont = torch.tensor(scaler.transform(dataset_test_x_cont))
        dataset_test_x_cate = torch.tensor(dataset_test_x.iloc[:, :3].values)
        dataset_test_x = torch.concat([dataset_test_x_cont, dataset_test_x_cate], dim=1)

        dataset_test_labels = torch.tensor(dataset_test['labels'].values)

    else:
        dataset_train_x_cont = dataset_train_x.iloc[:, [0, 7, 8, 9, 10, 11, 12, 13]]
        dataset_train_x_cont = torch.tensor(scaler.fit_transform(dataset_train_x_cont))
        dataset_train_x_cate = torch.tensor(dataset_train_x.iloc[:, [1, 2, 3, 4, 5, 6]].values)
        dataset_train_x = torch.concat([dataset_train_x_cont, dataset_train_x_cate], dim=1)

        dataset_train_labels = torch.tensor(dataset_train['labels'].values)

        dataset_valid_x_cont = dataset_valid_x.iloc[:, [0, 7, 8, 9, 10, 11, 12, 13]]
        dataset_valid_x_cont = torch.tensor(scaler.transform(dataset_valid_x_cont))
        dataset_valid_x_cate = torch.tensor(dataset_valid_x.iloc[:, [1, 2, 3, 4, 5, 6]].values)
        dataset_valid_x = torch.concat([dataset_valid_x_cont, dataset_valid_x_cate], dim=1)

        dataset_valid_labels = torch.tensor(dataset_valid['labels'].values)

        dataset_test_x_cont = dataset_test_x.iloc[:, [0, 7, 8, 9, 10, 11, 12, 13]]
        dataset_test_x_cont = torch.tensor(scaler.transform(dataset_test_x_cont))
        dataset_test_x_cate = torch.tensor(dataset_test_x.iloc[:, [1, 2, 3, 4, 5, 6]].values)
        dataset_test_x = torch.concat([dataset_test_x_cont, dataset_test_x_cate], dim=1)

        dataset_test_labels = torch.tensor(dataset_test['labels'].values)

    print("dataset:", args.seed)
    print("mean time:", torch.mean(dataset_train_t), torch.mean(dataset_valid_t), torch.mean(dataset_test_t))
    print("train:", dataset_train_x.shape, "validation:", dataset_valid_x.shape, "test:", dataset_test_x.shape)
    print(str(args.dataset) + " split by", dataset_train_t.shape[0], dataset_valid_t.shape[0], dataset_test_t.shape[0])
    print("training censoring rate:", 1 - (dataset_train_e.sum()/dataset_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (dataset_valid_e.sum()/dataset_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (dataset_test_e.sum()/dataset_test_e.shape[0]).item())
    print("training labels:", pd.DataFrame(dataset_train_labels).value_counts())
    print("validation labels:", pd.DataFrame(dataset_valid_labels).value_counts())
    print("test labels:", pd.DataFrame(dataset_test_labels).value_counts())
    
    # torch.save(dataset_train_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_t.pt')
    # torch.save(dataset_train_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_e.pt')
    # torch.save(dataset_train_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_x.pt')
    
    # torch.save(dataset_valid_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_t.pt')
    # torch.save(dataset_valid_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_e.pt')
    # torch.save(dataset_valid_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_x.pt')
    
    # torch.save(dataset_test_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_t.pt')
    # torch.save(dataset_test_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_e.pt')
    # torch.save(dataset_test_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_x.pt')

    # torch.save(dataset_train_labels, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_labels.pt')
    # torch.save(dataset_valid_labels, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_labels.pt')
    # torch.save(dataset_test_labels, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_labels.pt')
    
elif args.dataset == 'nacd':
    nacd = pd.read_csv("./data/KSP/nacd/NACD.csv")
    nacd.dropna(inplace=True)
    nacd = nacd.reset_index(drop=True).sort_index(ascending=True)
    
    from sklearn.metrics import silhouette_score
    from sklearn_extra.cluster import KMedoids
    import gower

    # Gower distance matrix
    gower_dist = gower.gower_matrix(nacd.loc[:, 'GENDER':'ALBUMIN'])

    def run_kmedoids_and_silhouette(dist_matrix, min_k=2, max_k=10):
        results = []
        best_score = -1
        best_model = None

        for k in range(min_k, max_k+1):
            model = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
            labels = model.fit_predict(dist_matrix)

            if len(set(labels)) > 1:  # silhouette cannot compute with 1 cluster
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            else:
                score = -1

            results.append((k, score))

            if score > best_score:
                best_score = score
                best_model = model

            print(f"K={k}, silhouette={score:.4f}")

        return results, best_model

    results, best_model = run_kmedoids_and_silhouette(gower_dist, min_k=2, max_k=6)
    print("\nOptimal K:", best_model.n_clusters)
    print("Cluster assignments:", best_model.labels_)
    
    best_k = best_model.n_clusters
    labels = best_model.labels_
    nacd['labels'] = labels

    from sklearn.model_selection import train_test_split

    nacd["strata"] = nacd["status"].astype(str) + "_" + nacd["labels"].astype(str)

    nacd_train, nacd_test_temp = train_test_split(nacd, test_size=0.4, stratify=nacd['strata'], random_state=args.seed)
    nacd_valid, nacd_test = train_test_split(nacd_test_temp, test_size=0.5, stratify=nacd_test_temp['strata'], random_state=args.seed)

    nacd_train_t = torch.tensor(nacd_train['SURVIVAL'].values)
    nacd_train_e = torch.tensor(nacd_train['status'].values)
    nacd_train_x = nacd_train.loc[:, 'GENDER':'ALBUMIN']
    nacd_train_x_cont = nacd_train_x.loc[:, ['BMI', 'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES', 'PLATELET', 
                                             'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']]
    nacd_train_x_cate = torch.tensor(nacd_train_x.drop(columns=nacd_train_x_cont.columns).values)

    scaler = StandardScaler()
    nacd_train_x_cont = torch.tensor(scaler.fit_transform(nacd_train_x_cont))
    nacd_train_x = torch.concat([nacd_train_x_cont, nacd_train_x_cate], dim=1)

    nacd_train_labels = torch.tensor(nacd_train['labels'].values)

    nacd_valid_t = torch.tensor(nacd_valid['SURVIVAL'].values)
    nacd_valid_e = torch.tensor(nacd_valid['status'].values)
    nacd_valid_x = nacd_valid.loc[:, 'GENDER':'ALBUMIN']
    nacd_valid_x_cont = nacd_valid_x.loc[:, ['BMI', 'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES', 'PLATELET', 
                                             'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']]
    nacd_valid_x_cate = torch.tensor(nacd_valid_x.drop(columns=nacd_valid_x_cont.columns).values)
    nacd_valid_x_cont = torch.tensor(scaler.transform(nacd_valid_x_cont))
    nacd_valid_x = torch.concat([nacd_valid_x_cont, nacd_valid_x_cate], dim=1)

    nacd_valid_labels = torch.tensor(nacd_valid['labels'].values)

    nacd_test_t = torch.tensor(nacd_test['SURVIVAL'].values)
    nacd_test_e = torch.tensor(nacd_test['status'].values)
    nacd_test_x = nacd_test.loc[:, 'GENDER':'ALBUMIN']
    nacd_test_x_cont = nacd_test_x.loc[:, ['BMI', 'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES', 'PLATELET', 
                                           'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']]
    nacd_test_x_cate = torch.tensor(nacd_test_x.drop(columns=nacd_test_x_cont.columns).values)
    nacd_test_x_cont = torch.tensor(scaler.transform(nacd_test_x_cont))
    nacd_test_x = torch.concat([nacd_test_x_cont, nacd_test_x_cate], dim=1)

    nacd_test_labels = torch.tensor(nacd_test['labels'].values)
    
    print("dataset:", args.seed)
    print("mean time:", torch.mean(nacd_train_t), torch.mean(nacd_valid_t), torch.mean(nacd_test_t))
    print("train:", nacd_train_x.shape, "validation:", nacd_valid_x.shape, "test:", nacd_test_x.shape)
    print("NACD split by", nacd_train_t.shape[0], nacd_valid_t.shape[0], nacd_test_t.shape[0])
    print("training censoring rate:", 1 - (nacd_train_e.sum()/nacd_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (nacd_valid_e.sum()/nacd_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (nacd_test_e.sum()/nacd_test_e.shape[0]).item())
    print("training labels:", pd.DataFrame(nacd_train_labels).value_counts())
    print("validation labels:", pd.DataFrame(nacd_valid_labels).value_counts())
    print("test labels:", pd.DataFrame(nacd_test_labels).value_counts())
    
    torch.save(nacd_train_t, './data/nacd/' + str(args.seed) + '/nacd_train_t.pt')
    torch.save(nacd_valid_t, './data/nacd/' + str(args.seed) + '/nacd_valid_t.pt')
    torch.save(nacd_test_t, './data/nacd/' + str(args.seed) + '/nacd_test_t.pt')

    torch.save(nacd_train_e, './data/nacd/' + str(args.seed) + '/nacd_train_e.pt')
    torch.save(nacd_valid_e, './data/nacd/' + str(args.seed) + '/nacd_valid_e.pt')
    torch.save(nacd_test_e, './data/nacd/' + str(args.seed) + '/nacd_test_e.pt')

    torch.save(nacd_train_x, './data/nacd/' + str(args.seed) + '/nacd_train_x.pt')
    torch.save(nacd_valid_x, './data/nacd/' + str(args.seed) + '/nacd_valid_x.pt')
    torch.save(nacd_test_x, './data/nacd/' + str(args.seed) + '/nacd_test_x.pt')

    torch.save(nacd_train_labels, './data/nacd/' + str(args.seed) + '/nacd_train_labels.pt')
    torch.save(nacd_valid_labels, './data/nacd/' + str(args.seed) + '/nacd_valid_labels.pt')
    torch.save(nacd_test_labels, './data/nacd/' + str(args.seed) + '/nacd_test_labels.pt')

elif args.dataset == 'sequence':
    # split by 3/1/1
    sequence = pd.read_csv("./data/sequence/sequence.csv")
    
    from sklearn.metrics import silhouette_score
    from sklearn_extra.cluster import KMedoids
    import gower

    # Gower distance matrix
    gower_dist = gower.gower_matrix(sequence.drop(columns=['Time', 'Status']))

    def run_kmedoids_and_silhouette(dist_matrix, min_k=2, max_k=10):
        results = []
        best_score = -1
        best_model = None

        for k in range(min_k, max_k+1):
            model = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
            labels = model.fit_predict(dist_matrix)

            if len(set(labels)) > 1:  # silhouette cannot compute with 1 cluster
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            else:
                score = -1

            results.append((k, score))

            if score > best_score:
                best_score = score
                best_model = model

            print(f"K={k}, silhouette={score:.4f}")

        return results, best_model

    results, best_model = run_kmedoids_and_silhouette(gower_dist, min_k=2, max_k=6)
    print("\nOptimal K:", best_model.n_clusters)
    print("Cluster assignments:", best_model.labels_)
    
    best_k = best_model.n_clusters
    labels = best_model.labels_
    sequence['labels'] = labels

    from sklearn.model_selection import train_test_split

    sequence["strata"] = sequence["Status"].astype(str) + "_" + sequence["labels"].astype(str)

    sequence_train, sequence_test_temp = train_test_split(sequence, test_size=0.4, stratify=sequence['strata'], random_state=args.seed)
    sequence_valid, sequence_test = train_test_split(sequence_test_temp, test_size=0.5, stratify=sequence_test_temp['strata'], random_state=args.seed)

    sequence_train_t = torch.tensor(sequence_train['Time'].values)
    sequence_train_e = torch.tensor(sequence_train['Status'].values)
    sequence_train_x = sequence_train.drop(columns=['Time', 'Status']).values

    scaler = StandardScaler()
    sequence_train_x = torch.tensor(scaler.fit_transform(sequence_train_x))

    sequence_valid_t = torch.tensor(sequence_valid['Time'].values)
    sequence_valid_e = torch.tensor(sequence_valid['Status'].values)
    sequence_valid_x = sequence_valid.drop(columns=['Time', 'Status']).values
    sequence_valid_x = torch.tensor(scaler.transform(sequence_valid_x))

    sequence_test_t = torch.tensor(sequence_test['Time'].values)
    sequence_test_e = torch.tensor(sequence_test['Status'].values)
    sequence_test_x = sequence_test.drop(columns=['Time', 'Status']).values
    sequence_test_x = torch.tensor(scaler.transform(sequence_test_x))

    sequence_train_labels = sequence_train['labels'].values
    sequence_valid_labels = sequence_valid['labels'].values
    sequence_test_labels = sequence_test['labels'].values

    print("dataset:", args.seed)
    print("mean time:", torch.mean(sequence_train_t), torch.mean(sequence_valid_t), torch.mean(sequence_test_t))
    print("train:", sequence_train_x.shape, "validation:", sequence_valid_x.shape, "test:", sequence_test_x.shape)
    print("NB-SEQ split by", sequence_train.shape[0], sequence_valid.shape[0], sequence_test.shape[0])
    print("training censoring rate:", 1 - (sequence_train_e.sum()/sequence_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (sequence_valid_e.sum()/sequence_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (sequence_test_e.sum()/sequence_test_e.shape[0]).item())
    print("training labels:", pd.DataFrame(sequence_train_labels).value_counts())
    print("validation labels:", pd.DataFrame(sequence_valid_labels).value_counts())
    print("test labels:", pd.DataFrame(sequence_test_labels).value_counts())
    
    # torch.save(sequence_train_t, './data/sequence/' + str(args.seed) + '/sequence_train_t.pt')
    # torch.save(sequence_valid_t, './data/sequence/' + str(args.seed) + '/sequence_valid_t.pt')
    # torch.save(sequence_test_t, './data/sequence/' + str(args.seed) + '/sequence_test_t.pt')

    # torch.save(sequence_train_e, './data/sequence/' + str(args.seed) + '/sequence_train_e.pt')
    # torch.save(sequence_valid_e, './data/sequence/' + str(args.seed) + '/sequence_valid_e.pt')
    # torch.save(sequence_test_e, './data/sequence/' + str(args.seed) + '/sequence_test_e.pt')

    # torch.save(sequence_train_x, './data/sequence/' + str(args.seed) + '/sequence_train_x.pt')
    # torch.save(sequence_valid_x, './data/sequence/' + str(args.seed) + '/sequence_valid_x.pt')
    # torch.save(sequence_test_x, './data/sequence/' + str(args.seed) + '/sequence_test_x.pt')

    # torch.save(torch.tensor(sequence_train_labels), './data/sequence/' + str(args.seed) + '/sequence_train_labels.pt')
    # torch.save(torch.tensor(sequence_valid_labels), './data/sequence/' + str(args.seed) + '/sequence_valid_labels.pt')
    # torch.save(torch.tensor(sequence_test_labels), './data/sequence/' + str(args.seed) + '/sequence_test_labels.pt')

elif args.dataset == 'mimic':
    # split by 3/1/1
    mimic_train_t = pd.read_csv("./data/mimic/train_t.csv")
    mimic_train_x = pd.read_csv("./data/mimic/train_x.csv")
    mimic_train = pd.merge(mimic_train_t, mimic_train_x)
    mimic_train.dropna(inplace=True)

    mimic_test_t = pd.read_csv("./data/mimic/test_t.csv")
    mimic_test_x = pd.read_csv("./data/mimic/test_x.csv")
    mimic_test = pd.merge(mimic_test_t, mimic_test_x)
    mimic_test.dropna(inplace=True)

    mimic = pd.concat([mimic_train, mimic_test], axis=0)
    mimic = mimic.reset_index(drop=True).sort_index(ascending=True)

    mimic.loc[mimic['Glascow coma scale eye opening'] == 'Spontaneously', 'Glascow coma scale eye opening'] = '4 Spontaneously'
    mimic.loc[mimic['Glascow coma scale eye opening'] == 'To Speech', 'Glascow coma scale eye opening'] = '3 To speech'

    mimic.loc[mimic['Glascow coma scale motor response'] == 'Obeys Commands', 'Glascow coma scale motor response'] = '6 Obeys Commands'
    mimic.loc[mimic['Glascow coma scale motor response'] == 'Flex-withdraws', 'Glascow coma scale motor response'] = '4 Flex-withdraws'

    mimic.loc[mimic['Glascow coma scale verbal response'] == 'Oriented', 'Glascow coma scale verbal response'] = '5 Oriented'
    mimic.loc[mimic['Glascow coma scale verbal response'] == 'No Response-ETT', 'Glascow coma scale verbal response'] = '1 No Response'

    # print(mimic['Glascow coma scale eye opening'].value_counts())
    # print(mimic['Glascow coma scale motor response'].value_counts())
    # print(mimic['Glascow coma scale verbal response'].value_counts())

    mimic = pd.get_dummies(mimic, columns=['Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale verbal response'])

    mimic = mimic.loc[mimic['Diastolic blood pressure'] < 110, :]
    mimic = mimic.loc[mimic['Temperature'] >= 30, :]
    mimic = mimic.loc[mimic['pH'] < 10, :]
    mimic.reset_index(drop=True, inplace=True)
    mimic_x = mimic.loc[:, 'Diastolic blood pressure':'Glascow coma scale verbal response_5 Oriented']
    
    from sklearn.metrics import silhouette_score
    from sklearn_extra.cluster import KMedoids
    import gower

    # Gower distance matrix
    gower_dist = gower.gower_matrix(mimic_x)

    def run_kmedoids_and_silhouette(dist_matrix, min_k=2, max_k=10):
        results = []
        best_score = -1
        best_model = None

        for k in range(min_k, max_k+1):
            model = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
            labels = model.fit_predict(dist_matrix)

            if len(set(labels)) > 1:  # silhouette cannot compute with 1 cluster
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            else:
                score = -1

            results.append((k, score))

            if score > best_score:
                best_score = score
                best_model = model

            print(f"K={k}, silhouette={score:.4f}")

        return results, best_model

    results, best_model = run_kmedoids_and_silhouette(gower_dist, min_k=2, max_k=6)
    print("\nOptimal K:", best_model.n_clusters)
    print("Cluster assignments:", best_model.labels_)
    
    best_k = best_model.n_clusters
    labels = best_model.labels_

    from sklearn.model_selection import train_test_split

    mimic_train, mimic_test_temp, mimic_train_labels, mimic_test_labels_temp = train_test_split(mimic, labels, test_size=0.4, stratify=labels, random_state=args.seed)
    mimic_valid, mimic_test, mimic_valid_labels, mimic_test_labels = train_test_split(mimic_test_temp, mimic_test_labels_temp, test_size=0.5, stratify=mimic_test_labels_temp, random_state=args.seed)

    mimic_train_t = torch.tensor(mimic_train['Time'].values)
    mimic_train_e = torch.tensor(mimic_train['Status'].values)
    mimic_train_x = mimic_train.loc[:, 'Diastolic blood pressure':'Glascow coma scale verbal response_5 Oriented']

    scaler = StandardScaler()
    mimic_train_x_conti = torch.tensor(scaler.fit_transform(mimic_train_x.loc[:, 'Diastolic blood pressure':'pH']))
    mimic_train_x = torch.concat([mimic_train_x_conti, torch.tensor(mimic_train_x.loc[:, 'Glascow coma scale eye opening_1 No Response':'Glascow coma scale verbal response_5 Oriented'].values)], dim=1)
    
    mimic_valid_t = torch.tensor(mimic_valid['Time'].values)
    mimic_valid_e = torch.tensor(mimic_valid['Status'].values)
    mimic_valid_x = mimic_valid.loc[:, 'Diastolic blood pressure':'Glascow coma scale verbal response_5 Oriented']
    mimic_valid_x_conti = torch.tensor(scaler.transform(mimic_valid_x.loc[:, 'Diastolic blood pressure':'pH']))
    mimic_valid_x = torch.concat([mimic_valid_x_conti, torch.tensor(mimic_valid_x.loc[:, 'Glascow coma scale eye opening_1 No Response':'Glascow coma scale verbal response_5 Oriented'].values)], dim=1)
    
    mimic_test_t = torch.tensor(mimic_test['Time'].values)
    mimic_test_e = torch.tensor(mimic_test['Status'].values)
    mimic_test_x = mimic_test.loc[:, 'Diastolic blood pressure':'Glascow coma scale verbal response_5 Oriented']
    mimic_test_x_conti = torch.tensor(scaler.transform(mimic_test_x.loc[:, 'Diastolic blood pressure':'pH']))
    mimic_test_x = torch.concat([mimic_test_x_conti, torch.tensor(mimic_test_x.loc[:, 'Glascow coma scale eye opening_1 No Response':'Glascow coma scale verbal response_5 Oriented'].values)], dim=1)
    
    print("Dataset:", args.seed)
    print("mean time:", torch.mean(mimic_train_t).item(), torch.mean(mimic_valid_t).item(), torch.mean(mimic_test_t).item())
    print("train:", mimic_train_x.shape, "validation:", mimic_valid_x.shape, "test:", mimic_test_x.shape)
    print("MIMIC-III split by", mimic_train.shape[0], mimic_valid.shape[0], mimic_test.shape[0])
    print("training censoring rate:", 1 - (mimic_train_e.sum()/mimic_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (mimic_valid_e.sum()/mimic_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (mimic_test_e.sum()/mimic_test_e.shape[0]).item())
    print("training labels:", pd.DataFrame(mimic_train_labels).value_counts())
    print("validation labels:", pd.DataFrame(mimic_valid_labels).value_counts())
    print("test labels:", pd.DataFrame(mimic_test_labels).value_counts())
    
    torch.save(mimic_train_t, './data/mimic/' + str(args.seed) + '/mimic_train_t.pt')
    torch.save(mimic_valid_t, './data/mimic/' + str(args.seed) + '/mimic_valid_t.pt')
    torch.save(mimic_test_t, './data/mimic/' + str(args.seed) + '/mimic_test_t.pt')

    torch.save(mimic_train_e, './data/mimic/' + str(args.seed) + '/mimic_train_e.pt')
    torch.save(mimic_valid_e, './data/mimic/' + str(args.seed) + '/mimic_valid_e.pt')
    torch.save(mimic_test_e, './data/mimic/' + str(args.seed) + '/mimic_test_e.pt')

    torch.save(mimic_train_x, './data/mimic/' + str(args.seed) + '/mimic_train_x.pt')
    torch.save(mimic_valid_x, './data/mimic/' + str(args.seed) + '/mimic_valid_x.pt')
    torch.save(mimic_test_x, './data/mimic/' + str(args.seed) + '/mimic_test_x.pt')

    torch.save(torch.tensor(mimic_train_labels), './data/mimic/' + str(args.seed) + '/mimic_train_labels.pt')
    torch.save(torch.tensor(mimic_valid_labels), './data/mimic/' + str(args.seed) + '/mimic_valid_labels.pt')
    torch.save(torch.tensor(mimic_test_labels), './data/mimic/' + str(args.seed) + '/mimic_test_labels.pt')

elif args.dataset in ['liver', 'stomach', 'bladder']:
    # seer = pd.read_csv("./data/seer/seer_new.csv")
    # seer = seer[seer['Time'] >= 3]
    # seer['Race'] = seer['Race'].replace({'White':'White',
    #                                      'Black':'Black',
    #                                      'Asian or Pacific Islander':'Asian',
    #                                      'American Indian/Alaska Native':'Asian'})

    # best_k = 3
    # labels = pd.Categorical(seer['Race']).codes
    # seer['labels'] = labels

    # seer = pd.get_dummies(seer, columns=['Sex', 'Race', 'Grade', 'Derived.AJCC.Stage.Group', 'Derived.AJCC.T', 'Derived.AJCC.N', 'Derived.AJCC.M',
    #                                      'Marital.status.at.diagnosis', 'Median.household.income', 'Surgery'])

    # if args.dataset == 'liver':
    #     seer = seer[seer['Site'] == 'Liver']

    # elif args.dataset == 'stomach':
    #     seer = seer[seer['Site'] == 'Stomach']

    # else:
    #     seer = seer[seer['Site'] == 'Lung and Bronchus']
    seer = pd.read_csv("./data/seer/seer_liver_stomach_bladder.csv")

    best_k = 3
    labels = pd.Categorical(seer['Race.recode..White..Black..Other.']).codes
    seer['Labels'] = labels

    seer = pd.get_dummies(seer, columns=['Sex', 'Grade.Recode..thru.2017.', 'Derived.AJCC.Stage.Group..6th.ed..2004.2015.', 
                                         'Derived.AJCC.T..6th.ed..2004.2015.', 'Derived.AJCC.N..6th.ed..2004.2015.', 
                                         'Derived.AJCC.M..6th.ed..2004.2015.', 'Race.recode..White..Black..Other.',
                                         'Marital.status.at.diagnosis', 'Median.household.income.inflation.adj.to.2023', 'Surgery'])

    if args.dataset == 'liver':
        seer = seer[seer['Site.recode.ICD.O.3.WHO.2008'] == 'Liver']

    elif args.dataset == 'stomach':
        seer = seer[seer['Site.recode.ICD.O.3.WHO.2008'] == 'Stomach']

    else:
        seer = seer[seer['Site.recode.ICD.O.3.WHO.2008'] == 'Urinary Bladder']

    seer['Vital.status'] = (seer['Vital.status.recode..study.cutoff.used.'] == "Dead")
    seer = seer.drop(columns=['Vital.status.recode..study.cutoff.used.', 'Site.recode.ICD.O.3.WHO.2008'])

    from sklearn.model_selection import train_test_split

    seer["Strata"] = seer["Vital.status"].astype(str) + "_" + seer["Labels"].astype(str)

    seer_train, seer_test_temp = train_test_split(seer, test_size=0.4, stratify=seer['Strata'], random_state=args.seed)
    seer_valid, seer_test = train_test_split(seer_test_temp, test_size=0.5, stratify=seer_test_temp['Strata'], random_state=args.seed)

    seer_train_t = torch.tensor(seer_train['Survival.months'].values, dtype=torch.float32)
    seer_train_e = torch.tensor(seer_train['Vital.status'].values, dtype=torch.float32)
    seer_train_x = seer_train.drop(columns=['Survival.months', 'Vital.status', 'Labels', 'Strata'])
    seer_train_labels = torch.tensor(seer_train['Labels'].values)

    scaler = StandardScaler()
    seer_train_x_cont = torch.tensor(scaler.fit_transform(seer_train_x[['Age.recode.with.single.ages.and.90.']]))
    seer_train_x_cate = torch.tensor(seer_train_x.drop(columns=['Age.recode.with.single.ages.and.90.']).values)
    seer_train_x = torch.concat([seer_train_x_cont, seer_train_x_cate], dim=1)
    
    seer_valid_t = torch.tensor(seer_valid['Survival.months'].values, dtype=torch.float32)
    seer_valid_e = torch.tensor(seer_valid['Vital.status'].values, dtype=torch.float32)
    seer_valid_x = seer_valid.drop(columns=['Survival.months', 'Vital.status', 'Labels', 'Strata'])
    seer_valid_x_cont = torch.tensor(scaler.transform(seer_valid_x[['Age.recode.with.single.ages.and.90.']]))
    seer_valid_x_cate = torch.tensor(seer_valid_x.drop(columns=['Age.recode.with.single.ages.and.90.']).values)
    seer_valid_x = torch.concat([seer_valid_x_cont, seer_valid_x_cate], dim=1)
    seer_valid_labels = torch.tensor(seer_valid['Labels'].values)
    
    seer_test_t = torch.tensor(seer_test['Survival.months'].values, dtype=torch.float32)
    seer_test_e = torch.tensor(seer_test['Vital.status'].values, dtype=torch.float32)
    seer_test_x = seer_test.drop(columns=['Survival.months', 'Vital.status', 'Labels', 'Strata'])
    seer_test_x_cont = torch.tensor(scaler.transform(seer_test_x[['Age.recode.with.single.ages.and.90.']]))
    seer_test_x_cate = torch.tensor(seer_test_x.drop(columns=['Age.recode.with.single.ages.and.90.']).values)
    seer_test_x = torch.concat([seer_test_x_cont, seer_test_x_cate], dim=1)
    seer_test_labels = torch.tensor(seer_test['Labels'].values)

    print("Dataset:", args.seed)
    print("mean time:", torch.mean(seer_train_t).item(), torch.mean(seer_valid_t).item(), torch.mean(seer_test_t).item())
    print("train:", seer_train_x.shape, "validation:", seer_valid_x.shape, "test:", seer_test_x.shape)
    print(f"SEER {args.dataset} split by", seer_train.shape[0], seer_valid.shape[0], seer_test.shape[0])
    print("training censoring rate:", 1 - (seer_train_e.sum()/seer_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (seer_valid_e.sum()/seer_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (seer_test_e.sum()/seer_test_e.shape[0]).item())
    print("training labels:", pd.DataFrame(seer_train_labels).value_counts())
    print("validation labels:", pd.DataFrame(seer_valid_labels).value_counts())
    print("test labels:", pd.DataFrame(seer_test_labels).value_counts())
    
    torch.save(seer_train_t, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_train_t.pt')
    torch.save(seer_valid_t, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_valid_t.pt')
    torch.save(seer_test_t, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_test_t.pt')

    torch.save(seer_train_e, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_train_e.pt')
    torch.save(seer_valid_e, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_valid_e.pt')
    torch.save(seer_test_e, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_test_e.pt')

    torch.save(seer_train_x, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_train_x.pt')
    torch.save(seer_valid_x, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_valid_x.pt')
    torch.save(seer_test_x, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_test_x.pt')

    torch.save(seer_train_labels, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_train_labels.pt')
    torch.save(seer_valid_labels, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_valid_labels.pt')
    torch.save(seer_test_labels, f'./data/seer/{args.dataset}/{args.seed}/{args.dataset}_test_labels.pt')

else:
    assert False