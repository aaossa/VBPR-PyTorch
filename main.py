"""This is a demo training script"""

import random
from os import cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn, optim

from vbpr import VBPR, Trainer
from vbpr.datasets import TradesyDataset

RANDOM_SEED = 0
TRANSACTIONS_PATH = Path("data", "Tradesy", "tradesy.json.gz")
FEATURES_PATH = Path("data", "Tradesy", "image_features_tradesy.b")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 10
    num_workers = cpu_count() or 0

    dataset, np_features = TradesyDataset.from_files(
        TRANSACTIONS_PATH, FEATURES_PATH, random_seed=RANDOM_SEED
    )

    n_users = dataset.n_users
    n_items = dataset.n_items
    features = torch.tensor(np_features, device=device)
    dim_gamma = 10
    dim_theta = 20
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
            {"params": params_groups["lambda_theta"], "weight_decay": 1.0},
            {"params": params_groups["lambda_beta"], "weight_decay": 0.01},
            {"params": params_groups["lambda_E"], "weight_decay": 0.0},
        ],
        lr=0.001,
    )

    trainer = Trainer(model, optimizer, device=device)
    trainer.fit(dataset, n_epochs=n_epochs, num_workers=num_workers)
