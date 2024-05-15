import numpy as np
import torch
import random

from dataset import TubeDataset
from model import MPNN

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def get_dataset(args):
    if args["flow_name"] == "tube":
        train_dataset = TubeDataset(filename="tube_train.hdf5", **args["dataset"])
        val_dataset = TubeDataset(filename="tube_dev.hdf5", **args["dataset"])
        test_dataset = TubeDataset(filename="tube_test.hdf5", **args["dataset"])
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset


def get_model(args):
    if args["model_name"] == "mpnn":
        model = MPNN(**args["model"])
    elif args["model_name"] == "mpnn":
        pass
    else:
        raise NotImplementedError
    return model