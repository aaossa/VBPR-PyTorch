from typing import Any, Dict, Optional, Tuple

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
        scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def setup_dataloaders(
        self, dataset: TradesyDataset, batch_size: int = 64, num_workers: int = 0
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

        epoch_pbar = tqdm(
            range(1, n_epochs + 1),
            desc="Epochs",
            unit="epoch",
            postfix=dict(best_auc=None, best_epoch=None),
        )
        train_pbar = tqdm(
            desc="Training", total=len(training_dl), postfix=dict(acc=None, loss=None)
        )
        valid_pbar = tqdm(
            desc="Validation", total=len(validation_dl), postfix=dict(auc=None)
        )
        eval_all_pbar = tqdm(
            desc="Evaluation (All Items)",
            total=len(evaluation_dl),
            postfix=dict(auc=None),
        )
        eval_cold_pbar = tqdm(
            desc="Evaluation (Cold Start)",
            total=len(evaluation_dl),
            postfix=dict(auc=None),
        )

        for epoch in epoch_pbar:
            training_metrics = self.training_step(training_dl, pbar=train_pbar)
            train_pbar.set_postfix(
                acc=training_metrics["accuracy"], loss=training_metrics["loss"]
            )

            auc_valid = self.evaluation(dataset, validation_dl, pbar=valid_pbar)
            valid_pbar.set_postfix(auc=auc_valid)

            if epoch % 10 == 0:
                auc_eval = self.evaluation(dataset, evaluation_dl, pbar=eval_all_pbar)
                eval_all_pbar.set_postfix(auc=auc_eval)
                auc_eval_cold = self.evaluation(
                    dataset,
                    evaluation_dl,
                    cold_only=True,
                    pbar=eval_cold_pbar,
                )
                eval_cold_pbar.set_postfix(auc=auc_eval_cold)

            if best_auc_valid < auc_valid:
                best_auc_valid = auc_valid
                best_epoch = epoch
                epoch_pbar.set_postfix(best_auc=best_auc_valid, best_epoch=best_epoch)
                # save_model()
            elif epoch >= (best_epoch + 20):
                print("Overfitted maybe...")
                break

            if self.scheduler is not None:
                self.scheduler.step(auc_valid)

        # save_model()
        auc_eval = self.evaluation(dataset, evaluation_dl, pbar=eval_all_pbar)
        auc_eval_cold = self.evaluation(
            dataset, evaluation_dl, cold_only=True, pbar=eval_cold_pbar
        )

        print(f"[Validation] AUC = {best_auc_valid:.6f} (best epoch = {best_epoch})")
        print(f"[Evaluation] AUC = {auc_eval:.6f} (All Items)")
        print(f"[Evaluation] AUC = {auc_eval_cold:.6f} (Cold Start)")

        return self.model

    def training_step(
        self, dataloader: DataLoader[TradesySample], pbar: Optional[tqdm[Any]] = None
    ) -> Dict[str, float]:
        # Set correct model mode
        self.model = self.model.train()

        # Tensor to accumulate results
        running_acc = torch.tensor(0, dtype=torch.int, device=self.device)
        running_loss = torch.tensor(0.0, dtype=torch.double, device=self.device)

        # Reset/create progress bar
        if pbar is None:
            pbar = tqdm(desc="Training")
        pbar.reset(total=len(dataloader))

        for uid, iid, jid in dataloader:
            # Prepare inputs
            uid = uid.to(self.device).squeeze()
            iid = iid.to(self.device).squeeze()
            jid = jid.to(self.device).squeeze()
            # Clear gradients
            self.optimizer.zero_grad()
            # Forward step
            with torch.set_grad_enabled(True):
                outputs = self.model(uid, iid, jid)
                outputs = outputs.unsqueeze(0)
                loss = -torch.nn.functional.logsigmoid(outputs).sum()
                # Backward
                loss.backward()
                # Update parameters
                self.optimizer.step()

            # Accumulate batch results
            running_acc.add_((outputs > 0).sum())
            running_loss.add_(loss.detach() * outputs.size(0))

            # Update progress bar
            pbar.update()

        # Complete progress bar
        pbar.refresh()

        # Display epoch results
        dataset_size: int = len(dataloader.dataset)  # type: ignore
        epoch_acc = running_acc.item() / dataset_size
        epoch_loss = running_loss.item() / dataset_size
        return {"accuracy": epoch_acc, "loss": epoch_loss}

    def evaluation(
        self,
        full_dataset: TradesyDataset,
        dataloader: DataLoader[TradesySample],
        cold_only: bool = False,
        pbar: Optional[tqdm[Any]] = None,
    ) -> float:
        # Set correct model mode
        self.model = self.model.eval()

        # Generate cache to speed-up recommendations
        cache = self.model.generate_cache()

        # Tensor to accumulate results
        AUC_eval = torch.zeros(full_dataset.n_users, device=self.device)

        # Reset/create progress bar
        if pbar is None:
            pbar = tqdm(desc="Evaluation")
        pbar.reset(total=len(dataloader))

        for ui, pi, _ in dataloader:
            # Prepare inputs
            ui = ui.to(self.device)
            pi = pi.to(self.device)

            # Retrieve recommendations
            x_u_eval = self.model.recommend(ui, pi)
            user_recommendations = self.model.recommend(ui, cache=cache)

            ui_array = ui.squeeze().cpu().numpy()
            pi_array = pi.squeeze().cpu().numpy()

            for i, (ui_item, pi_item) in enumerate(zip(ui_array, pi_array)):
                # Skip evaluation if item is not "cold"
                if cold_only and len(full_dataset.get_item_users(pi_item)) > 5:
                    AUC_eval[ui_item] = -1.0
                    continue

                # Additional data
                left_out_items = torch.from_numpy(full_dataset.get_user_items(ui_item))
                max_possible = full_dataset.n_items - left_out_items.shape[0]

                # Prepare batch results
                count_eval = (x_u_eval[i] > user_recommendations[i]).sum() - (
                    x_u_eval[i] > user_recommendations[i, left_out_items]
                ).sum()

                # Accumulate batch results
                AUC_eval[ui_item] = 1.0 * count_eval / max_possible

            # Update progress bar
            pbar.update()

        # Complete progress bar
        pbar.refresh()

        # Display evaluation results
        auc = AUC_eval[AUC_eval >= 0].mean().item()
        return auc
