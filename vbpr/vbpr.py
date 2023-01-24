"""This module contains a VBPR implementation in PyTorch."""

from typing import Optional, Tuple, Union, cast

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

        return cast(torch.Tensor, x_uij.squeeze())

    @torch.no_grad()  # type: ignore[misc]
    def recommend(
        self,
        users: torch.Tensor,
        items: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict score for interactions.

        Calculates the score for the given user-item interactions.
        Output shape matches the shape of `users * items`

        Args:
            users: Users indices, as a Tensor.
            items: Items indices, as a Tensor.
            cache: Optional. A precalculated tuple of Tensors.

        Returns:
            Prediction score for each user-item pair.
        """

        def check_input_tensor(tensor: torch.Tensor, name: str) -> None:
            if tensor.dim() != 2:
                raise ValueError(f"{name} tensor must have exactly two dimensions")
            elif not any(size == 1 for size in tensor.size()):
                raise ValueError(f"{name} tensor must contain a singleton dimension")

        check_input_tensor(users, "users")
        if items is not None:
            check_input_tensor(items, "items")

        use_matmul = False
        if items is None or (
            users.size(0) == items.size(1) == 1 or users.size(1) == items.size(0) == 1
        ):
            use_matmul = True
        elif users.size() != items.size():
            raise ValueError(
                "users and items must have equal shape or different singleton dimension"
            )

        items_selector: Union[slice, torch.Tensor] = (
            slice(None) if items is None else items
        )
        repeat_factors = (max(users.size()), 1) if use_matmul else (1, 1)

        u_latent_factors = self.gamma_users(users).squeeze()
        u_visual_factors = self.theta_users(users).squeeze()
        if u_latent_factors.dim() == 1:
            u_latent_factors = u_latent_factors.unsqueeze(0)
            u_visual_factors = u_visual_factors.unsqueeze(0)

        i_bias = self.beta_items.weight[items_selector]
        i_bias = i_bias.squeeze().repeat(*repeat_factors)
        i_latent_factors = self.gamma_items.weight[items_selector]
        i_latent_factors = i_latent_factors.squeeze()
        if i_latent_factors.dim() == 1:
            i_latent_factors = i_latent_factors.unsqueeze(0)

        if cache is None:
            i_features = self.features.weight[items_selector]
            visual_rating_space = i_features.matmul(self.embedding.weight)
            opinion_visual_appearance = i_features.matmul(self.visual_bias.weight)
        else:
            visual_rating_space, opinion_visual_appearance = cache
            visual_rating_space = visual_rating_space[items_selector]
            opinion_visual_appearance = opinion_visual_appearance[items_selector]
        opinion_visual_appearance = opinion_visual_appearance.squeeze().repeat(
            *repeat_factors
        )
        visual_rating_space = visual_rating_space.squeeze()
        if visual_rating_space.dim() == 1:
            visual_rating_space = visual_rating_space.unsqueeze(0)

        if use_matmul:
            latent_component = torch.matmul(u_latent_factors, i_latent_factors.T)
            visual_component = torch.matmul(u_visual_factors, visual_rating_space.T)
        else:
            latent_component = (
                (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(0)
            )
            visual_component = (
                (u_visual_factors * visual_rating_space).sum(dim=1).unsqueeze(0)
            )

        x_ui = i_bias + latent_component + visual_component + opinion_visual_appearance
        if items is None:
            if x_ui.size() != (users.size(0), i_bias.size(1)):
                x_ui = x_ui.T
        elif x_ui.size() == (users.size(1), items.size(0)):
            x_ui = x_ui.T
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
