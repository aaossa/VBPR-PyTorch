"""This module contains a VBPR implementation in PyTorch."""

from typing import Optional, Tuple, cast

import torch
from torch import nn


class VBPR(nn.Module):
    """This class represents the VBPR model as a PyTorch nn.Module.

    The implementation follows the specifications of the original paper:
    'VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback'

    NOTE: The model contains the (pretrained)  visual features as a layer to
    improve performance. Another possible implementation of this would be to
    store the features in the Dataset class and pass the emebddings to the
    forward method."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        features: torch.Tensor,
        dim_gamma: int,
        dim_theta: int,
    ):
        super().__init__()

        # Image features
        self.features = nn.Embedding.from_pretrained(
            features, freeze=True
        )  # type: ignore

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(n_users, dim_gamma)
        self.gamma_items = nn.Embedding(n_items, dim_gamma)

        # Visual factors (theta)
        self.theta_users = nn.Embedding(n_users, dim_theta)
        self.embedding = nn.Embedding(features.size(1), dim_theta)

        # Biases (beta)
        # self.beta_users = nn.Embedding(n_users, 1)
        self.beta_items = nn.Embedding(n_items, 1)
        self.visual_bias = nn.Embedding(features.size(1), 1)

        # Random weight initialization
        self.reset_parameters()

    def forward(
        self, ui: torch.Tensor, pi: torch.Tensor, ni: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Feed forward a given input (batch). Each object is expected
        to be a Tensor.

        Args:
            ui: User index, as a Tensor.
            pi: Positive item index, as a Tensor.
            ni: Negative item index, as a Tensor.

        Returns:
            Network output (scalar) for each input.
        """
        # User
        ui_latent_factors = self.gamma_users(ui)  # Latent factors of user u
        ui_visual_factors = self.theta_users(ui)  # Visual factors of user u
        # Items
        pi_bias = self.beta_items(pi)  # Pos. item bias
        ni_bias = self.beta_items(ni)  # Neg. item bias
        pi_latent_factors = self.gamma_items(pi)  # Pos. item visual factors
        ni_latent_factors = self.gamma_items(ni)  # Neg. item visual factors
        pi_features = self.features(pi)  # Pos. item visual features
        ni_features = self.features(ni)  # Neg. item visual features

        # Precompute differences
        diff_features = pi_features - ni_features
        diff_latent_factors = pi_latent_factors - ni_latent_factors

        # x_uij
        x_uij = (
            pi_bias
            - ni_bias
            + (ui_latent_factors * diff_latent_factors).sum(dim=1).unsqueeze(-1)
            + (ui_visual_factors * diff_features.mm(self.embedding.weight))
            .sum(dim=1)
            .unsqueeze(-1)
            + diff_features.mm(self.visual_bias.weight)
        )

        return cast(torch.Tensor, x_uij.unsqueeze(-1))

    def recommend_all(
        self,
        user: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        """Predict score for every item, for the given user(s)"""
        with torch.set_grad_enabled(grad_enabled):
            # User
            u_latent_factors = self.gamma_users(user)  # Latent factors of user u
            u_visual_factors = self.theta_users(user)  # Visual factors of user u

            # Items
            i_bias = self.beta_items.weight  # Items bias
            i_latent_factors = self.gamma_items.weight  # Items visual factors
            i_features = self.features.weight  # Items visual features
            if cache is not None:
                visual_rating_space, opinion_visual_appearance = cache
            else:
                visual_rating_space = i_features.mm(self.embedding.weight)
                opinion_visual_appearance = i_features.mm(self.visual_bias.weight)

            # x_ui
            x_ui = (
                i_bias
                + (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(-1)
                + (u_visual_factors * visual_rating_space).sum(dim=1).unsqueeze(-1)
                + opinion_visual_appearance
            )

            return cast(torch.Tensor, x_ui)

    def recommend(
        self,
        user: torch.Tensor,
        items: Optional[torch.Tensor] = None,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        """Predict score for the given items, for the given user(s)"""
        with torch.set_grad_enabled(grad_enabled):
            # User
            u_latent_factors = self.gamma_users(user)  # Latent factors of user u
            u_visual_factors = self.theta_users(user)  # Visual factors of user u

            # Items
            i_bias = self.beta_items(items)  # Items bias
            i_latent_factors = self.gamma_items(items)  # Items visual factors
            i_features = self.features(items)  # Items visual features
            visual_rating_space = i_features.mm(self.embedding.weight)
            opinion_visual_appearance = i_features.mm(self.visual_bias.weight)

            # x_ui
            x_ui = (
                i_bias
                + (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(-1)
                + (u_visual_factors * visual_rating_space).sum(dim=1).unsqueeze(-1)
                + opinion_visual_appearance
            )

            return cast(torch.Tensor, x_ui)

    def reset_parameters(self) -> None:
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Latent factors (gamma)
        nn.init.xavier_uniform_(self.gamma_users.weight)
        nn.init.xavier_uniform_(self.gamma_items.weight)

        # Visual factors (theta)
        nn.init.xavier_uniform_(self.theta_users.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Biases (beta)
        nn.init.xavier_uniform_(self.beta_items.weight)
        nn.init.xavier_uniform_(self.visual_bias.weight)

    def generate_cache(
        self, grad_enabled: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precalculate intermediate values before calculating recommendations"""
        with torch.set_grad_enabled(grad_enabled):
            i_features = self.features.weight  # Items visual features
            visual_rating_space = i_features.mm(self.embedding.weight)
            opinion_visual_appearance = i_features.mm(self.visual_bias.weight)
        return visual_rating_space, opinion_visual_appearance
