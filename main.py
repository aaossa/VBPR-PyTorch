"""This is a demo training script"""

import random
from os import cpu_count
from pathlib import Path
from typing import Dict, List, TypedDict

import numpy as np
import torch
import wandb
from torch import nn, optim

from vbpr import VBPR, Trainer
from vbpr.datasets import TradesyDataset

RANDOM_SEED = 0
TRANSACTIONS_PATH = Path("data", "Tradesy", "tradesy.json.gz")
FEATURES_PATH = Path("data", "Tradesy", "image_features_tradesy.b")


class SchedulerArgsDict(TypedDict):
    factor: float
    patience: int


class ConfigDict(TypedDict):
    model: str
    dataset: str
    n_epochs: int
    model_params: Dict[str, int]
    optimizer: Dict[str, float]
    scheduler: SchedulerArgsDict


if __name__ == "__main__":
    config: ConfigDict = {
        "model": "VBPR",
        "dataset": "Tradesy",
        # Hyperparameters
        "n_epochs": 100,
        # Model
        "model_params": {
            "dim_gamma": 20,
            "dim_theta": 20,
        },
        # Optimizer
        "optimizer": {
            "lr": 5e-04,
            "lambda_theta": 1.0,
            "lambda_beta": 0.01,
            "lambda_E": 0.0,
        },
        # Scheduler
        "scheduler": {
            "factor": 0.3,
            "patience": 5,
        },
    }
    wandb.init(project="vbpr-tradesy", config=config)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = config["n_epochs"]
    num_workers = cpu_count() or 0

    dataset, np_features = TradesyDataset.from_files(
        TRANSACTIONS_PATH, FEATURES_PATH, random_seed=RANDOM_SEED
    )

    n_users = dataset.n_users
    n_items = dataset.n_items
    features = torch.tensor(np_features, device=device)
    dim_gamma = config["model_params"]["dim_gamma"]
    dim_theta = config["model_params"]["dim_theta"]
    model = VBPR(n_users, n_items, features, dim_gamma, dim_theta)
    model = model.to(device)
    model = model.float()
    model = model.train()

    params_groups: Dict[str, List[nn.parameter.Parameter]] = {
        "lambda_theta": [],
        "lambda_beta": [],
        "lambda_E": [],
    }
    for name, params in model.named_parameters():
        if name.startswith("gamma_") or name.startswith("theta_"):
            params_groups["lambda_theta"].append(params)
        elif name.startswith("beta_"):
            params_groups["lambda_beta"].append(params)
        elif name.startswith("embedding") or name.startswith("visual_bias"):
            params_groups["lambda_E"].append(params)
        else:
            print(f"Parameter '{name}' is not being optimized")
    optimizer = optim.SGD(
        [
            {
                "params": params_groups["lambda_theta"],
                "weight_decay": config["optimizer"]["lambda_theta"],
            },
            {
                "params": params_groups["lambda_beta"],
                "weight_decay": config["optimizer"]["lambda_beta"],
            },
            {
                "params": params_groups["lambda_E"],
                "weight_decay": config["optimizer"]["lambda_E"],
            },
        ],
        lr=config["optimizer"]["lr"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
        verbose=True,
    )

    trainer = Trainer(model, optimizer, scheduler=scheduler, device=device)
    trainer.fit(
        dataset, n_epochs=n_epochs, num_workers=num_workers, wandb_callback=wandb.log
    )
