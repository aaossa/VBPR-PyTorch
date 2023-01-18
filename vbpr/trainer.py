from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from .datasets import TradesyDataset, TradesySample
from .vbpr import VBPR


class Trainer:
    def __init__(
        self,
        model: VBPR,
        optimizer: optim.Optimizer,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def setup_dataloaders(
        self, dataset: TradesyDataset, batch_size: int = 100, num_workers: int = 0
    ) -> Tuple[
        DataLoader[TradesySample], DataLoader[TradesySample], DataLoader[TradesySample]
    ]:
        training_subset, validation_subset, evaluation_subset = dataset.split()
        training_dl = DataLoader(
            training_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        validation_dl = DataLoader(
            validation_subset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        evaluation_dl = DataLoader(
            evaluation_subset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return training_dl, validation_dl, evaluation_dl

    def fit(
        self, dataset: TradesyDataset, n_epochs: int = 1, **dataloaders_kwargs: int
    ) -> nn.Module:
        training_dl, validation_dl, evaluation_dl = self.setup_dataloaders(
            dataset, **dataloaders_kwargs
        )

        best_auc_valid = 0.0
        best_epoch = 0

        for epoch in range(1, n_epochs + 1):
            self.training_step(epoch, training_dl)

            if epoch % 10 == 0:
                auc_valid = self.evaluation(dataset, validation_dl, phase="Validation")

                if best_auc_valid < auc_valid:
                    best_auc_valid = auc_valid
                    best_epoch = epoch
                    # save_model()
                elif epoch >= (best_epoch + 20):
                    print("Overfitted maybe...")
                    break

        # save_model()
        auc_eval = self.evaluation(dataset, evaluation_dl)

        print(f"[Validation] AUC = {best_auc_valid:.6f} (best epoch = {best_epoch})")
        print(f"[Evaluation] AUC = {auc_eval:.6f} (final)")

        return self.model

    def training_step(self, epoch: int, dataloader: DataLoader[TradesySample]) -> None:
        # Set correct model mode
        self.model = self.model.train()

        # Tensor to accumulate results
        running_acc = torch.tensor(0, dtype=torch.int, device=self.device)
        running_loss = torch.tensor(0.0, dtype=torch.double, device=self.device)

        for uid, iid, jid in tqdm(dataloader, desc=f"Training (epoch={epoch})"):
            # Prepare inputs
            uid = uid.to(self.device).squeeze()
            iid = iid.to(self.device).squeeze()
            jid = jid.to(self.device).squeeze()
            # Clear gradients
            self.optimizer.zero_grad()
            # Forward step
            with torch.set_grad_enabled(True):
                outputs = self.model(uid, iid, jid)
                outputs = torch.reshape(outputs, (1, -1))
                loss = -torch.nn.functional.logsigmoid(outputs).sum()
                # Backward
                loss.backward()
                # Update parameters
                self.optimizer.step()

            # Accumulate batch results
            running_acc.add_((outputs > 0).sum())
            running_loss.add_(loss.detach() * outputs.size(0))

        # Display epoch results
        dataset_size: int = len(dataloader.dataset)  # type: ignore
        epoch_acc = running_acc.item() / dataset_size
        epoch_loss = running_loss.item() / dataset_size
        print(
            f"Training (epoch={epoch}) ACC = {epoch_acc:.6f} (LOSS = {epoch_loss:.6f})"
        )

    def evaluation(
        self,
        full_dataset: TradesyDataset,
        dataloader: DataLoader[TradesySample],
        phase: str = "Evaluation",
    ) -> float:
        # Set correct model mode
        self.model = self.model.eval()

        # Generate cache to speed-up recommendations
        cache = self.model.generate_cache()
        indices_eval = dataloader.dataset.indices  # type: ignore[attr-defined]
        src_dataset = dataloader.dataset.dataset  # type: ignore[attr-defined]

        # Tensor to accumulate results
        AUC_eval = torch.zeros(full_dataset.n_users, device=self.device)

        for user in tqdm(range(full_dataset.n_users), desc="AUC on All Items"):
            # Prepare inputs
            left_out_items = torch.from_numpy(full_dataset.get_user_items(user))
            items_eval = src_dataset.get_user_items(user, indices=indices_eval)
            user_tensor = torch.tensor([user], device=self.device)

            # Retrieve recommendations
            x_u_eval = self.model.recommend(
                user_tensor, torch.tensor(items_eval, device=self.device)
            ).item()
            user_recommendations = self.model.recommend_all(user_tensor, cache=cache)
            max_possible = full_dataset.n_items - left_out_items.shape[0]
            seen_recommendations = user_recommendations[left_out_items]

            # Prepare user results
            count_eval = (
                (x_u_eval > user_recommendations).sum()
                - (x_u_eval > seen_recommendations).sum()
            ).item()

            # Accumulate user results
            AUC_eval[user] = 1.0 * count_eval / max_possible

        # Display evaluation results
        auc = AUC_eval.mean().item()
        print(f"{phase.title()} AUC = {auc:.6f}")
        return auc
