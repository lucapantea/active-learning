import torch
import wandb
import argparse
import numpy as np
import logging
import config
from config import logger

from utils import seed_everything, get_model, get_dataset, \
                  get_strategy, wandb_run_name

def get_default_parser():
    parser = argparse.ArgumentParser()
    for arg, default_value in config.DEFAULT_CONFIG.items():
        if type(default_value) is bool:
            parser.add_argument(f'--{arg}', action=f'store_{not default_value}'.lower())
        else:
            parser.add_argument(f'--{arg}', type=type(default_value), default=default_value)
    return parser

def main(args):
    # TODO: Different Datasets
    # TODO: Different Models
    # TODO: Different n_init_labelled
    # TODO: Different n_query
    # TODO: 3 different datasets: ImageNet, CIFAR10, MNIST
    # TODO: 3 different models: LeNet, ResNet18, GoogLeNet

    # Possible Experiment:
    # - experiment the effect of different active learning strategies under different levels of noise
    # - experiment with malicious labelling and how different active learning strategies handle it
        # interesting - here it's important to define what sort of malicious labelling we're talking about
        # - define possible attacks (https://www.usenix.org/conference/usenixsecurity21/presentation/vicarte)
        # experiment with other types of accuracy: (https://arxiv.org/pdf/2107.01622.pdf)

    params = vars(args)

    # Training parameters
    params['optimizer_args'] = {'lr': args.lr}
    params['train_args'] = {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    params['test_args'] = {'batch_size': args.batch_size, 'num_workers': args.num_workers}

    # Dataset parameters
    if args.dataset == 'mnist':
        params['image_channels'] = 1
        params['num_classes'] = 10


    seed_everything(args.seed)
    dataset = get_dataset(args.dataset, args.data_dir)
    model = get_model(args.model, params)
    strategy = get_strategy(args.strategy, {'dataset': dataset, 'model': model})

    # wandb init
    run_name = wandb_run_name(args)
    if args.wandb:
        wandb.init(project='active-learning', name=run_name,
                   config=params, reinit=True, entity='msc-ai')
        wandb.watch(model.clf)

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    logger.debug(f"Initial number of labeled pool: {args.n_init_labeled}")
    logger.debug(f"Initial number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    logger.debug(f"Initial number of testing pool: {dataset.n_test}")
    logger.debug(f"Model parameters: {sum(p.numel() for p in model.clf.parameters())}")
    logger.debug(f"Model architecture: \\{model.clf}")

    best_acc = .0

    # Active Learning Rounds
    for rd in range(1, args.n_round+1):
        logger.info(f"Starting Round {rd}")

        # Query unlabeled pool
        query_idxs = strategy.query(args.n_query)

        # Update the dataset with queried indices
        strategy.update(query_idxs)

        # Retrain the model with updated dataset
        strategy.fit()

        # Compute the test accuracy
        preds = strategy.predict(dataset.get_test_data())
        acc = (dataset.Y_test == preds).sum().item() / len(dataset.Y_test)

        # Save the best accuracy
        if acc > best_acc:
            best_acc = acc

        logger.info(f"Round {rd} testing accuracy: {acc}")
        logger.debug(f"Round {rd} labeled pool: {dataset.labeled_idxs.sum()}")
        logger.debug(f"Round {rd} unlabeled pool: {dataset.n_pool-dataset.labeled_idxs.sum()}")

        if args.wandb:
            wandb.log({'Test Accuracy': acc, 'Round': rd})
            wandb.run.summary['Best Accuracy'] = best_acc
    

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.debug('Debug mode on')

    if args.experiment:
        for seed in np.random.randint(0, 1000, 3):
            logger.info(f"Running experiment with seed {seed}")
            args.seed = seed
    else:
        logger.info(f"Running experiment with seed {args.seed}")
    main(args)
