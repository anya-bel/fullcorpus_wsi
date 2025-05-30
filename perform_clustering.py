import argparse
import pickle
from ast import literal_eval

import pandas as pd
from transformers import AutoModel, AutoTokenizer

from src.clustering import WSIClustering
from src.utils import compute_embeddings

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=False, default='datasets/semcorwsi_dataset_all_dev.csv')
parser.add_argument('--algorithm', type=str, required=False, default='agglomerative')
parser.add_argument('--n_clusters_compute', type=str, required=False, default='silhouette')
parser.add_argument('--constraint', type=str, required=False, default=False)
parser.add_argument('--add_data_path', type=str, required=False, default=False)
parser.add_argument('--modelname', type=str, required=False, default='bert-large-uncased')
parser.add_argument('--layer', type=int, required=False, default=24)
parser.add_argument('--device', type=str, required=False, default='cpu')
parser.add_argument('--output_file', type=str, required=False, default='out')

args = parser.parse_args()

print(
    f"Starting {args.algorithm} clustering with {args.n_clusters_compute} on {args.dataset} ({args.modelname} (layer {args.layer})).",
    end=' ')
if args.constraint:
    print(f"{args.constraint} constraint is used")
else:
    print('no constraint is used')
if args.add_data_path:
    print(f"{args.add_data_path} is added to the examples")
else:
    print('no additional data is used')

border_col = 'position'
example_col = 'sentence'

modelname = args.modelname
model = AutoModel.from_pretrained(modelname).to(args.device)
tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)

dataset = pd.read_csv(args.dataset)
dataset[border_col] = dataset[border_col].apply(literal_eval)

out = compute_embeddings(dataset, tokenizer, model, args.device, border_col, example_col)

if args.add_data_path:
    add_df = pd.read_csv(args.add_data_path)
    add_df[border_col] = add_df[border_col].apply(literal_eval)
    add_out = compute_embeddings(add_df, tokenizer, model, args.device, border_col, example_col)

if args.constraint:
    wikt_df = pd.read_csv('datasets/wiktionary_all_train_filtered_on_semcorwsi_dev.csv')
    wikt_df[border_col] = wikt_df[border_col].apply(literal_eval)
    wikt_out = compute_embeddings(wikt_df, tokenizer, model, args.device, border_col, example_col)
    wsi = WSIClustering(algorithm=args.algorithm, n_cluster_compute=args.n_clusters_compute, constraint=args.constraint)
    if args.add_data_path:

        result = wsi.fit(dataset, out[args.layer], constraint_dataframe=wikt_df,
                         constraint_embeddings=wikt_out[args.layer],
                         additional_data=add_df, additional_embeddings=add_out[args.layer])
    else:
        result = wsi.fit(dataset, out[args.layer], constraint_dataframe=wikt_df,
                         constraint_embeddings=wikt_out[args.layer])
else:
    wsi = WSIClustering(algorithm=args.algorithm, n_cluster_compute=args.n_clusters_compute, constraint=False)
    if args.add_data_path:
        result = wsi.fit(dataset, out[args.layer], additional_data=add_df, additional_embeddings=add_out[args.layer])
    else:
        result = wsi.fit(dataset, out[args.layer])

with open(f'{args.output_file}.pkl', 'wb') as f:
    pickle.dump(result, f)
