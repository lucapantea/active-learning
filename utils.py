import random, os
import numpy as np
import torch

from models import Model

def get_device():
    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    if mps:
        return torch.device('mps')
    elif cuda:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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
    
def get_model(model_name, model_args):
    if model_name == 'lenet':
        from models import LeNet
        model = Model(LeNet, model_args, device=get_device())
    return model

def get_dataset(dataset_name, data_dir):
    print('Loading datasets...')
    if dataset_name == 'mnist':
        from datasets import get_mnist_al_dataset
        dataset = get_mnist_al_dataset(data_dir)
    
    return dataset

def get_strategy(strategy_name, strategy_args):
    if strategy_name == 'random':
        from strategies import RandomSampling
        strategy = RandomSampling(**strategy_args)

    return strategy
