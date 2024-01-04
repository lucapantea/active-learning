import logging

# Argument parser settings
DEFAULT_CONFIG = {
    'dataset': 'mnist',
    'data_dir': 'data',
    'batch_size': 64,
    'epochs': 10,
    'num_workers': 0,
    'seed': 42,
    'model': 'lenet',
    'lr': 0.001,
    'strategy': 'random',
    'n_init_labeled': 10000,
    'n_query': 1000,
    'n_round': 10,
    'wandb': True,
    'experiment': False,
    'debug': False
}

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

def get_logger(name, debug=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else LOG_LEVEL)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else LOG_LEVEL) 

    # Create formatter and add it to the handler
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(ch)

    return logger

# Singleton logger
logger = get_logger('ProjectLogger')