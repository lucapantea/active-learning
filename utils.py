import random, os
import numpy as np
import torch
from config import logger
from models import Model

def get_device():
    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    if mps:
        device = 'mps'
    elif cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info('Training on' + f' {device}'.upper())
    return torch.device(device)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.backends.mps.is_available():
        torch.backends.mps.deterministic = True
        torch.backends.mps.benchmark = True
    
def get_model(model_name, params):
    if model_name == 'lenet':
        from models import LeNet
        logger.info('Using LeNet')
        model = Model(LeNet, params=params, device=get_device())
    elif model_name == 'resnet18':
        from models import ResNet
        logger.info('Using ResNet18')
        params['layers'] = [2, 2, 2, 2]
        model = Model(ResNet, params=params, device=get_device())
    elif model_name == 'resnet34':
        from models import ResNet
        logger.info('Using ResNet34')
        params['layers'] = [3, 4, 6, 3]
        model = Model(ResNet, params=params, device=get_device())
    else:
        raise NotImplementedError(f'Model {model_name} not implemented')
    return model

def get_noise(noise, noise_rate):
    if noise == 'none':
        noise_transform = None
    elif noise == 'gaussian':
        from datasets import AddGaussianNoise
        logger.info(f'Adding Gaussian Noise with mean=0 and std=1, noise rate={noise_rate}')  
        noise_transform = AddGaussianNoise(mean=0., std=1.)
    elif noise == 'salt_and_pepper':
        from datasets import AddSaltAndPepperNoise
        logger.info(f'Adding Salt and Pepper Noise with noise rate={noise_rate}')  
        noise_transform = AddSaltAndPepperNoise(noise_rate=0.1)
    else:
        raise NotImplementedError(f'Noise {noise} not implemented')
    return noise_transform

def get_dataset(dataset_name, data_dir, num_valid, noise='none', noise_rate=0.):
    noise_transform = get_noise(noise, noise_rate)

    if dataset_name == 'mnist':
        from datasets import get_mnist_al_dataset
        logger.info(f'Loading MNIST dataset from \'./{data_dir}/\'')
        dataset = get_mnist_al_dataset(data_dir=data_dir, num_valid=num_valid, noise_transform=noise_transform, noise_rate=noise_rate)
    elif dataset_name == 'cifar10':
        from datasets import get_cifar10_al_dataset
        logger.info(f'Loading CIFAR10 dataset from \'./{data_dir}/\'')
        dataset = get_cifar10_al_dataset(data_dir=data_dir, num_valid=num_valid, noise_transform=noise_transform, noise_rate=noise_rate)
    elif dataset_name == 'fashion_mnist':
        from datasets import get_fashion_mnist_al_dataset
        logger.info(f'Loading FashionMNIST dataset from \'./{data_dir}/\'')
        dataset = get_fashion_mnist_al_dataset(data_dir=data_dir, num_valid=num_valid, noise_transform=noise_transform, noise_rate=noise_rate)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
    return dataset

def get_strategy(strategy_name, strategy_args):
    if strategy_name == 'random':
        from strategies import RandomSampling
        logger.info('Using Random Sampling Strategy')
        strategy = RandomSampling(**strategy_args)
    elif strategy_name == 'entropy':
        from strategies import EntropySampling
        logger.info('Using Entropy Sampling Strategy')
        strategy = EntropySampling(**strategy_args)
    elif strategy_name == 'bald':
        from strategies import BALDDropout
        logger.info('Using BALD Strategy')
        strategy = BALDDropout(n_drop=10, **strategy_args)
    else:
        raise NotImplementedError(f'Strategy {strategy_name} not implemented')
    return strategy

def wandb_run_name(args):
    run_name = f"{args.dataset}_{args.model}_{args.strategy}_nlabelled-{args.n_init_labeled}_nquery-{args.n_query}_nround-{args.n_round}"
    return run_name