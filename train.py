import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from timeit import default_timer
from utils import *


def train_loop(model, train_loader, optimizer, loss_fn, args):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    step = 0
    for x, y, mask, case_params, grid, _ in train_loader:
        # preprocess data
        step += 1
        batch_size = x.shape[0]
        x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
        y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        mask = mask.to(device) # mask [b, x1, ..., xd, 1]
        case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
        y = y * mask

        # forward
        x = x[..., 0].unsqueeze(-1)
        y = y[..., 0].unsqueeze(-1)
        if train_loader.dataset.multi_step_size > 1:
            preds=[]
            for i in range(train_loader.dataset.multi_step_size):
                pred = model(x, case_params, grid)
                preds.append(pred)
                x = pred
            preds=torch.stack(preds, dim=1)
        else:
            preds = model(x, case_params, grid)
        loss = loss_fn(preds.reshape([batch_size, -1]), y.reshape([batch_size, -1]))
        train_l2 += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_l2 /= step
    t2 = default_timer()
    return train_l2, t2 - t1


def main(args):
    # init
    setup_seed(args["seed"])
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    saved_model_name = (args["model_name"] + 
                        f"_lr{args['optimizer']['lr']}" + 
                        f"_bs{args['dataloader']['batch_size']}" + 
                        f"_wd{args['optimizer']['weight_decay']}" +
                        f"_{args['training_type']}")
    saved_dir = os.path.join(args["output_dir"], args["flow_name"], args["dataset"]["case_name"])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # data
    train_data, val_data, test_data = get_dataset(args)
    train_loader = DataLoader(train_data, shuffle=True, **args["dataloader"])
    val_loader = DataLoader(val_data, shuffle=False, **args["dataloader"])
    test_loader = DataLoader(test_data, shuffle=False, **args["dataloader"])

    # model
    model = get_model(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of model parameters to train:", total_params)
    if not args["if_training"]:
        return
    if args["continue_training"]:
        checkpoint = torch.load(args["model_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # optimizer
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")
    # if continue training, resume optimizer and scheduler from checkpoint
    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)
    
    # scheduler
    start_epoch = 0
    min_val_loss = torch.inf
    if args["continue_training"]:
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['loss']
    sched_args = args["scheduler"]
    sched_name = sched_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, last_epoch=start_epoch-1, **sched_args)

    # loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # train
    print(f"Start training from epoch {start_epoch}")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        train_l2, time = train_loop(model, train_loader, optimizer, loss_fn, args)
        scheduler.step()
        total_time += time
        print(f"[Epoch {epoch}] train_l2: {train_l2}, time_spend: {time:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # can be accessed globally
    main(args)