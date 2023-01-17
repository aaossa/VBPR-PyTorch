"""This is a demo training script"""

import random
from os import cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm.autonotebook import tqdm

from vbpr import VBPR
from vbpr.datasets import TradesyDataset, TradesySample

RANDOM_SEED = 0
N_EPOCHS = 50
TRANSACTIONS_PATH = Path("data", "Tradesy", "tradesy.json.gz")
FEATURES_PATH = Path("data", "Tradesy", "image_features_tradesy.b")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    training_subset: Subset[TradesySample]
    validation_subset: Subset[TradesySample]
    evaluation_subset: Subset[TradesySample]
    training_subset, validation_subset, evaluation_subset = dataset.split()
    training_dataloader: DataLoader[TradesySample] = DataLoader(
        training_subset,
        batch_size=100,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_dataloader: DataLoader[TradesySample] = DataLoader(
        validation_subset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
    )
    dataloaders = {
        "train": training_dataloader,
        "valid": validation_dataloader,
    }

    for epoch in range(1, N_EPOCHS + 1):
        for phase, dataloader in dataloaders.items():
            # Set correct model mode
            if phase == "train":
                model = model.train()
            else:
                model = model.eval()

            # Tensor to accumulate results
            running_acc = torch.tensor(0, dtype=torch.int, device=device)
            running_loss = torch.tensor(0.0, dtype=torch.double, device=device)

            for uid, iid, jid in tqdm(
                dataloader, desc=f"Phase: {phase} (epoch={epoch})"
            ):
                # Prepare inputs
                uid = uid.to(device).squeeze()
                iid = iid.to(device).squeeze()
                jid = jid.to(device).squeeze()
                # Clear gradients
                optimizer.zero_grad()
                # Forward step
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(uid, iid, jid)
                    outputs = torch.reshape(outputs, (1, -1))
                    loss = -torch.nn.functional.logsigmoid(outputs).sum()
                    if phase == "train":
                        # Backward
                        loss.backward()
                        # Update parameters
                        optimizer.step()

                # Accumulate batch results
                running_acc.add_((outputs > 0).sum())
                running_loss.add_(loss.detach() * outputs.size(0))

            # Display epoch results
            dataset_size: int = len(dataloader.dataset)  # type: ignore
            epoch_acc = running_acc.item() / dataset_size
            epoch_loss = running_loss.item() / dataset_size
            print(f"Epoch {epoch}/{phase}: ACC={epoch_acc:.6f} LOSS={epoch_loss:.6f}")

        model.eval()
        cache = model.generate_cache()
        AUC_valid = torch.zeros(dataset.n_users, device=device)
        AUC_eval = torch.zeros(dataset.n_users, device=device)
        for user in tqdm(range(dataset.n_users), desc="AUC on All Items"):
            items_train = (
                training_subset.dataset.get_user_items(  # type: ignore[attr-defined]
                    user, training_subset.indices
                )
            )
            items_valid = (
                validation_subset.dataset.get_user_items(  # type: ignore[attr-defined]
                    user, validation_subset.indices
                )
            )
            items_eval = (
                evaluation_subset.dataset.get_user_items(  # type: ignore[attr-defined]
                    user, evaluation_subset.indices
                )
            )
            user_tensor = torch.tensor([user], device=device)

            x_u_valid = model.recommend(
                user_tensor, torch.tensor(items_valid, device=device)
            ).item()
            x_u_eval = model.recommend(
                user_tensor, torch.tensor(items_eval, device=device)
            ).item()

            user_recommendations = model.recommend_all(user_tensor, cache=cache)
            left_out_items = np.concatenate([items_train, items_valid, items_eval])
            max_possible = dataset.n_items - left_out_items.shape[0]
            seen_recommendations = user_recommendations[left_out_items]
            count_valid = (
                (x_u_valid > user_recommendations).sum()
                - (x_u_valid > seen_recommendations).sum()
            ).item()
            count_eval = (
                (x_u_eval > user_recommendations).sum()
                - (x_u_eval > seen_recommendations).sum()
            ).item()

            AUC_valid[user] = 1.0 * count_valid / max_possible
            AUC_eval[user] = 1.0 * count_eval / max_possible

        print(f"AUC_valid={AUC_valid.mean():.6f} | AUC_eval={AUC_eval.mean():.6f}")
        print()
