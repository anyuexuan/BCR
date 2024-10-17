import numpy as np
import torch
import random
import logging
from logging import handlers
import torch.multiprocessing
import warnings
from data.dataset import MLLDataset, EpisodeSampler
from torch.utils.data import DataLoader
import os

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
base_path = os.path.dirname(__file__).replace('\\', '/') + '/..'


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def get_dataloader(dataset_name, phase, n_way, n_shot, n_query, transform, num_iter, num_workers, image_size):
    dataset = MLLDataset(dataset_name, phase=phase, transform=transform, image_size=image_size)
    sampler = EpisodeSampler(dataset_name=dataset_name, n_way=n_way, n_shot=n_shot, n_query=n_query, phase=phase,
                             iter=num_iter, max_idx=dataset.images.shape[0])
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    return dataloader


def get_baseline_dataloader(dataset_name, phase, batch_size, transform, num_workers, image_size):
    dataset = MLLDataset(dataset_name, phase=phase, transform=transform, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader


def get_idx_dataloader(dataset_name, phase, batch_size, transform, num_workers, image_size):
    dataset = MLLDataset(dataset_name, phase=phase, transform=transform, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader
