import numpy as np
import torch
import random

from dataset import TubeDataset, NSCHDataset, PDEDarcyDataset
from model import MPNN, GNOT

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def get_dataset(args):
    dataset_args = args["dataset"]
    if args["flow_name"] == "tube":
        train_dataset = TubeDataset(filename="tube_train.hdf5", **dataset_args)
        val_dataset = TubeDataset(filename="tube_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = TubeDataset(filename="tube_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = TubeDataset(filename="tube_test.hdf5", **dataset_args)
    elif args["flow_name"] == "NSCH":
        train_dataset = NSCHDataset(filename="train.hdf5", **dataset_args)
        val_dataset = NSCHDataset(filename="val.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = NSCHDataset(filename="test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = NSCHDataset(filename="test.hdf5", **dataset_args)
    elif args["flow_name"] == "Darcy":
        if dataset_args["case_name"] == "PDEBench":
            dataset_args.pop("case_name")
            train_dataset = PDEDarcyDataset(split="train", **dataset_args)
            val_dataset = PDEDarcyDataset(split="val", **dataset_args)
            test_dataset = PDEDarcyDataset(split="test", **dataset_args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset


def get_model(args):
    if args["model_name"] == "mpnn":
        model = MPNN(**args["model"])
    elif args["model_name"] == "gnot":
        model = GNOT(**args["model"])
    else:
        raise NotImplementedError
    return model