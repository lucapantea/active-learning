import torch
import wandb
import argparse
import numpy as np

from utils import get_device, seed_everything, get_model, get_dataset, get_strategy

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=54)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='lenet')

    # TODO create optimizer args object
    parser.add_argument('--lr', type=float, default=0.001)


    # Active Lenaring specific arguments
    parser.add_argument('--strategy', type=str, default='random', help='Active learning strategy')
    parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")

    return parser

def main():
    parser = get_default_parser()
    args = parser.parse_args()
    model_args = vars(args)

    device = get_device()
    seed_everything(args.seed)
    dataset = get_dataset(args.dataset, args.data_dir)
    model = get_model(args.model, device)
    strategy = get_strategy(args.strategy)

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # round 0 accuracy
    print("Round 0")
    strategy.fit()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

    for rd in range(1, args.n_round+1):
        print(f"Round {rd}")

        # query
        query_idxs = strategy.query(args.n_query)

        # update labels
        strategy.update(query_idxs)
        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        acc = (dataset.Y_test == preds).sum().item() / len(dataset.Y_test)
        print(f"Round {rd} testing accuracy: {acc}")