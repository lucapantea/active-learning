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
    params['num_classes'] = 10
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        params['image_channels'] = 1
        params['feature_map_dim'] = 16 * 4 * 4 * params['image_channels']
    if args.dataset == 'cifar10':
        params['image_channels'] = 3
        params['feature_map_dim'] = 16 * 5 * 5 * params['image_channels']

    # Initialize the dataset
    dataset = get_dataset(args.dataset, args.data_dir, args.num_valid)

    # Initialize the model
    model = get_model(args.model, params)

    # Initialize the active learning strategy
    strategy = get_strategy(args.strategy, {'dataset': dataset, 'model': model})

    # Remove useless parameters
    del params['feature_map_dim']
      
    # Initialize cloud logging through wandb
    run_name = wandb_run_name(args)
    if args.wandb:
        wandb.init(project='active-learning', name=run_name,
                   config=params, reinit=True, entity='msc-ai')
        wandb.watch(model.clf)

    # Use the same seed for reproducibility
    seed_everything(args.seed)

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    logger.debug(f"Initial number of labeled pool: {args.n_init_labeled}")
    logger.debug(f"Initial number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    logger.debug(f"Validation set: {dataset.n_valid}")
    logger.debug(f"Testing set: {dataset.n_test}")
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
        preds = strategy.predict(dataset.get_validation_data())
        acc = (dataset.Y_valid == preds).sum().item() / len(dataset.Y_valid)

        # Save the best accuracy
        if acc > best_acc:
            best_acc = acc

        logger.info(f"Round {rd} validation accuracy: {acc}")
        logger.debug(f"Round {rd} labeled pool: {dataset.labeled_idxs.sum()}")
        logger.debug(f"Round {rd} unlabeled pool: {dataset.n_pool-dataset.labeled_idxs.sum()}")

        if args.wandb:
            wandb.log({'Validation Accuracy': acc, 'Round': rd})
            wandb.run.summary['Best Accuracy'] = best_acc
    
    # Final test error
    preds = strategy.predict(dataset.get_test_data())
    test_acc = (dataset.Y_test == preds).sum().item() / len(dataset.Y_test)
    test_error = 1 - test_acc
    logger.info(f"Final test accuracy: {test_acc}")
    logger.info(f"Final test error: {test_error}")

    if args.wandb:
        wandb.run.summary['Test Accuracy'] = test_acc
        wandb.run.summary['Test Error'] = test_error

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
            main(args)
    else:
        logger.info(f"Running experiment with seed {args.seed}")
        main(args)
