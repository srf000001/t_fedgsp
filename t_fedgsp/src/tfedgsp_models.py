from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _diverse_factor_anchors(rank: int, order: int) -> torch.Tensor:
    width = order + 1
    low_pass = torch.tensor([1.0 / (index + 1) for index in range(width)])
    identity = torch.zeros(width)
    identity[0] = 1.0
    first_difference = identity.clone()
    if width > 1:
        first_difference[1] = -1.0
    delayed_difference = torch.zeros(width)
    if width > 2:
        delayed_difference[1] = 1.0
        delayed_difference[2] = -1.0
    elif width > 1:
        delayed_difference[1] = 1.0
    else:
        delayed_difference[0] = 1.0
    candidates = [low_pass, identity, first_difference, delayed_difference]
    return torch.stack([candidates[index % len(candidates)] for index in range(rank)])


def _masked_pool(sequence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked mean and last-observed state for [B,T,...] sequences."""
    expand = mask
    while expand.ndim < sequence.ndim:
        expand = expand.unsqueeze(-1)
    denom = expand.sum(dim=1).clamp_min(1.0)
    mean = (sequence * expand).sum(dim=1) / denom
    positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
    last_index = torch.where(mask > 0, positions, torch.zeros_like(positions)).max(dim=1).values
    gather_shape = [sequence.shape[0], 1] + [1] * (sequence.ndim - 2)
    gather_index = last_index.view(*gather_shape).expand(sequence.shape[0], 1, *sequence.shape[2:])
    last = sequence.gather(1, gather_index).squeeze(1)
    return mean, last


def _observation_summary(observations: torch.Tensor) -> torch.Tensor:
    values = torch.log1p(observations)
    return torch.cat([values.mean(dim=1), values.amax(dim=1)], dim=-1)


class BasisModel(nn.Module):
    def __init__(self, rms: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.register_buffer("rms", rms.float().clamp_min(1e-6))
        self.gamma = float(gamma)

    def normalize(self, basis: torch.Tensor, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        density = observations.sum(dim=-1)
        mask = (density > 0).float()
        denominator = torch.pow(1.0 + torch.log1p(density), self.gamma).clamp_min(1.0)
        scaled = basis / self.rms.view(1, 1, *self.rms.shape)
        scaled = scaled / denominator[:, :, None, None]
        return scaled, mask


class StaticMLP(BasisModel):
    def __init__(self, rms: torch.Tensor, num_labels: int, gamma: float, hidden: int, dropout: float) -> None:
        super().__init__(rms, gamma)
        dimension = int(rms.shape[1]) + 8
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        x, mask = self.normalize(basis, observations)
        mean, _ = _masked_pool(x[:, :, 0, :], mask)
        return self.network(torch.cat([mean, _observation_summary(observations)], dim=-1))


class GraphTemporalGRU(BasisModel):
    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        dropout: float,
        use_graph: bool,
    ) -> None:
        super().__init__(rms, gamma)
        self.use_graph = bool(use_graph)
        num_orders = int(rms.shape[0]) if use_graph else 1
        self.graph_logits = nn.Parameter(torch.zeros(num_orders)) if use_graph else None
        input_dim = int(rms.shape[1]) + 4
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_labels))

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        x, mask = self.normalize(basis, observations)
        if self.use_graph:
            weights = torch.softmax(self.graph_logits, dim=0)
            concept = torch.einsum("btkd,k->btd", x, weights)
        else:
            concept = x[:, :, 0, :]
        sequence = torch.cat([concept, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        return self.head(torch.cat([mean, last], dim=-1))


class LowRankJointFilter(BasisModel):
    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        temporal_order: int,
        graph_order: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__(rms[: graph_order + 1], gamma)
        self.temporal_order = int(temporal_order)
        self.graph_order = int(graph_order)
        self.rank = int(rank)
        temporal_init = torch.tensor([1.0 / (lag + 1) for lag in range(temporal_order + 1)])
        graph_init = torch.tensor([1.0 / (order + 1) for order in range(graph_order + 1)])
        self.temporal_raw = nn.Parameter(temporal_init.repeat(rank, 1) + 0.02 * torch.randn(rank, temporal_order + 1))
        self.graph_raw = nn.Parameter(graph_init.repeat(rank, 1) + 0.02 * torch.randn(rank, graph_order + 1))
        projection_dim = int(rms.shape[1])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * rank * projection_dim + 8, num_labels),
        )

    @staticmethod
    def _signed_l1(raw: torch.Tensor) -> torch.Tensor:
        values = torch.tanh(raw)
        return values / values.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def coefficient_surface(self) -> torch.Tensor:
        temporal = self._signed_l1(self.temporal_raw)
        graph = self._signed_l1(self.graph_raw)
        return torch.einsum("rl,rk->rlk", temporal, graph)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        basis = basis[:, :, : self.graph_order + 1, :]
        x, mask = self.normalize(basis, observations)
        batch, steps, _, dimension = x.shape
        coefficients = self.coefficient_surface()
        filtered = x.new_zeros((batch, steps, self.rank, dimension))
        for lag in range(self.temporal_order + 1):
            if lag >= steps:
                break
            contribution = torch.einsum(
                "btkd,rk->btrd", x[:, : steps - lag], coefficients[:, lag, :]
            )
            filtered[:, lag:] = filtered[:, lag:] + contribution
        filtered = F.gelu(filtered)
        mean, last = _masked_pool(filtered, mask)
        pooled = torch.cat(
            [mean.flatten(1), last.flatten(1), _observation_summary(observations)], dim=-1
        )
        return self.head(pooled)


class JointResidualGRU(BasisModel):
    """Graph-GRU with a gated low-rank causal time-graph polynomial residual."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__(rms, gamma)
        self.temporal_order = int(temporal_order)
        self.graph_order = int(graph_order)
        self.rank = int(rank)
        self.base_graph_logits = nn.Parameter(torch.zeros(int(rms.shape[0])))
        projection_dim = int(rms.shape[1])
        self.gru = nn.GRU(projection_dim + 4, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_labels))
        self.temporal_raw = nn.Parameter(_diverse_factor_anchors(rank, temporal_order))
        self.graph_raw = nn.Parameter(_diverse_factor_anchors(rank, graph_order))
        self.rank_logits = nn.Parameter(torch.zeros(rank))
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.residual_channel_scale = nn.Parameter(torch.ones(projection_dim))
        self.residual_norm = nn.LayerNorm(projection_dim, elementwise_affine=False)

    @staticmethod
    def _signed_l1(raw: torch.Tensor) -> torch.Tensor:
        values = torch.tanh(raw)
        return values / values.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def coefficient_surface(self) -> torch.Tensor:
        temporal = self._signed_l1(self.temporal_raw)
        graph = self._signed_l1(self.graph_raw)
        return torch.einsum("rl,rk->rlk", temporal, graph)

    def residual_strength(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit)

    def _base_and_joint(
        self, basis: torch.Tensor, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mask = self.normalize(basis, observations)
        batch, steps, _, dimension = x.shape
        base_weights = torch.softmax(self.base_graph_logits, dim=0)
        base = torch.einsum("btkd,k->btd", x, base_weights)

        coefficients = self.coefficient_surface()
        filtered = x.new_zeros((batch, steps, self.rank, dimension))
        residual_basis = x[:, :, : self.graph_order + 1, :]
        for lag in range(self.temporal_order + 1):
            if lag >= steps:
                break
            contribution = torch.einsum(
                "btkd,rk->btrd", residual_basis[:, : steps - lag], coefficients[:, lag, :]
            )
            filtered[:, lag:] = filtered[:, lag:] + contribution
        rank_weights = torch.softmax(self.rank_logits, dim=0)
        joint = torch.einsum("btrd,r->btd", filtered, rank_weights)
        joint = self.residual_norm(joint) * self.residual_channel_scale
        return base, joint, mask

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        base, joint, mask = self._base_and_joint(basis, observations)
        concept = base + self.residual_strength() * F.gelu(joint)

        sequence = torch.cat([concept, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        return self.head(torch.cat([mean, last], dim=-1))


class JointLogitResidualGRU(JointResidualGRU):
    """Frozen-compatible graph-GRU plus an additive joint-filter evidence head.

    Zero initialization makes the warm-start prediction exactly equal to the
    source graph-GRU. The joint branch then learns only evidence not already
    captured by the matched backbone.
    """

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__(
            rms,
            num_labels,
            gamma,
            hidden,
            temporal_order,
            graph_order,
            rank,
            dropout,
        )
        projection_dim = int(rms.shape[1])
        self.residual_head = nn.Linear(2 * projection_dim, num_labels)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        base, joint, mask = self._base_and_joint(basis, observations)
        base_sequence = torch.cat([base, torch.log1p(observations)], dim=-1)
        base_output, _ = self.gru(base_sequence)
        base_mean, base_last = _masked_pool(base_output, mask)
        base_logits = self.head(torch.cat([base_mean, base_last], dim=-1))

        joint = F.gelu(joint)
        joint_mean, joint_last = _masked_pool(joint, mask)
        residual_logits = self.residual_head(torch.cat([joint_mean, joint_last], dim=-1))
        return base_logits + self.residual_strength() * residual_logits


class FullJointLogitResidualGRU(JointLogitResidualGRU):
    """Capacity-matched direct joint surface without a separability constraint."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        dropout: float,
    ) -> None:
        super().__init__(
            rms,
            num_labels,
            gamma,
            hidden,
            temporal_order,
            graph_order,
            1,
            dropout,
        )
        # A rank-one factorization contains (Kt+1)+(Kg+1)+1 scalars. For the
        # selected Kt=1, Kg=2 case this equals the six entries of a full 2x3
        # surface, yielding an exactly capacity-matched nonseparable control.
        del self.temporal_raw
        del self.graph_raw
        del self.rank_logits
        temporal = torch.tensor(
            [1.0 / (lag + 1) for lag in range(temporal_order + 1)]
        )
        graph = torch.tensor(
            [1.0 / (order + 1) for order in range(graph_order + 1)]
        )
        self.joint_raw = nn.Parameter(torch.outer(temporal, graph))

    def coefficient_surface(self) -> torch.Tensor:
        values = torch.tanh(self.joint_raw)
        normalized = values / values.abs().sum().clamp_min(1e-6)
        return normalized.unsqueeze(0)

    def _base_and_joint(
        self, basis: torch.Tensor, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mask = self.normalize(basis, observations)
        batch, steps, _, dimension = x.shape
        base_weights = torch.softmax(self.base_graph_logits, dim=0)
        base = torch.einsum("btkd,k->btd", x, base_weights)

        coefficients = self.coefficient_surface()[0]
        filtered = x.new_zeros((batch, steps, dimension))
        residual_basis = x[:, :, : self.graph_order + 1, :]
        for lag in range(self.temporal_order + 1):
            if lag >= steps:
                break
            contribution = torch.einsum(
                "btkd,k->btd", residual_basis[:, : steps - lag], coefficients[lag]
            )
            filtered[:, lag:] = filtered[:, lag:] + contribution
        joint = self.residual_norm(filtered) * self.residual_channel_scale
        return base, joint, mask


class JointInputResidualGRU(JointResidualGRU):
    """Zero-start time--graph adapter acting through a frozen Graph-GRU.

    Unlike the logit-residual model, this adapter does not learn a separate
    label head.  A scalar gate and channel scales inject the filtered graph
    signal into the frozen backbone input.  The tanh gate is initialized at
    zero, so transferring a Graph-GRU produces bit-exactly identical logits.
    """

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__(
            rms,
            num_labels,
            gamma,
            hidden,
            temporal_order,
            graph_order,
            rank,
            dropout,
        )
        nn.init.zeros_(self.residual_logit)

    def residual_strength(self) -> torch.Tensor:
        return torch.tanh(self.residual_logit)


class FullJointInputResidualGRU(JointInputResidualGRU):
    """Direct causal lag-by-graph surface for input-side adaptation."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        dropout: float,
    ) -> None:
        super().__init__(
            rms,
            num_labels,
            gamma,
            hidden,
            temporal_order,
            graph_order,
            1,
            dropout,
        )
        del self.temporal_raw
        del self.graph_raw
        del self.rank_logits
        temporal = torch.tensor(
            [1.0 / (lag + 1) for lag in range(temporal_order + 1)]
        )
        graph = torch.tensor(
            [1.0 / (order + 1) for order in range(graph_order + 1)]
        )
        self.joint_raw = nn.Parameter(torch.outer(temporal, graph))

    def coefficient_surface(self) -> torch.Tensor:
        values = torch.tanh(self.joint_raw)
        normalized = values / values.abs().sum().clamp_min(1e-6)
        return normalized.unsqueeze(0)

    def _base_and_joint(
        self, basis: torch.Tensor, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mask = self.normalize(basis, observations)
        batch, steps, _, dimension = x.shape
        base_weights = torch.softmax(self.base_graph_logits, dim=0)
        base = torch.einsum("btkd,k->btd", x, base_weights)

        coefficients = self.coefficient_surface()[0]
        filtered = x.new_zeros((batch, steps, dimension))
        residual_basis = x[:, :, : self.graph_order + 1, :]
        for lag in range(self.temporal_order + 1):
            if lag >= steps:
                break
            contribution = torch.einsum(
                "btkd,k->btd", residual_basis[:, : steps - lag], coefficients[lag]
            )
            filtered[:, lag:] = filtered[:, lag:] + contribution
        joint = self.residual_norm(filtered) * self.residual_channel_scale
        return base, joint, mask


class GenericInputResidualGRU(BasisModel):
    """Parameter-matched input adapter without temporal or graph filtering."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__(rms, gamma)
        projection_dim = int(rms.shape[1])
        self.base_graph_logits = nn.Parameter(torch.zeros(int(rms.shape[0])))
        self.gru = nn.GRU(projection_dim + 4, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_labels))
        self.residual_logit = nn.Parameter(torch.tensor(0.0))
        self.residual_channel_scale = nn.Parameter(torch.ones(projection_dim))
        self.residual_norm = nn.LayerNorm(projection_dim, elementwise_affine=False)

    def residual_strength(self) -> torch.Tensor:
        return torch.tanh(self.residual_logit)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        x, mask = self.normalize(basis, observations)
        base_weights = torch.softmax(self.base_graph_logits, dim=0)
        base = torch.einsum("btkd,k->btd", x, base_weights)
        generic = self.residual_norm(base) * self.residual_channel_scale
        concept = base + self.residual_strength() * F.gelu(generic)
        sequence = torch.cat([concept, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        return self.head(torch.cat([mean, last], dim=-1))


class FullJointHeadInputResidualGRU(FullJointInputResidualGRU):
    """Input-side time--graph adapter plus a zero-start hidden-state head."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        dropout: float,
    ) -> None:
        super().__init__(
            rms,
            num_labels,
            gamma,
            hidden,
            temporal_order,
            graph_order,
            dropout,
        )
        self.residual_head = nn.Linear(hidden * 2, num_labels)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        base, joint, mask = self._base_and_joint(basis, observations)
        concept = base + self.residual_strength() * F.gelu(joint)
        sequence = torch.cat([concept, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        pooled = torch.cat([mean, last], dim=-1)
        return self.head(pooled) + self.residual_head(pooled)


class GenericHeadInputResidualGRU(GenericInputResidualGRU):
    """Near-parameter-matched head/input adapter without time--graph filtering."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__(rms, num_labels, gamma, hidden, dropout)
        self.residual_head = nn.Linear(hidden * 2, num_labels)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        x, mask = self.normalize(basis, observations)
        base_weights = torch.softmax(self.base_graph_logits, dim=0)
        base = torch.einsum("btkd,k->btd", x, base_weights)
        generic = self.residual_norm(base) * self.residual_channel_scale
        concept = base + self.residual_strength() * F.gelu(generic)
        sequence = torch.cat([concept, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        pooled = torch.cat([mean, last], dim=-1)
        return self.head(pooled) + self.residual_head(pooled)


class FullJointHybridAdapterGRU(FullJointInputResidualGRU):
    """Hidden-state head adaptation plus a low-rank time--graph evidence path."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        temporal_order: int,
        graph_order: int,
        dropout: float,
        bottleneck: int = 4,
    ) -> None:
        super().__init__(
            rms,
            num_labels,
            gamma,
            hidden,
            temporal_order,
            graph_order,
            dropout,
        )
        projection_dim = int(rms.shape[1])
        self.residual_head = nn.Linear(hidden * 2, num_labels)
        self.residual_joint_down = nn.Linear(2 * projection_dim, bottleneck)
        self.residual_joint_up = nn.Linear(bottleneck, hidden * 2, bias=False)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.residual_joint_up.weight)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        base, joint, mask = self._base_and_joint(basis, observations)
        sequence = torch.cat([base, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        pooled = torch.cat([mean, last], dim=-1)

        joint = F.gelu(joint)
        joint_mean, joint_last = _masked_pool(joint, mask)
        joint_pooled = torch.cat([joint_mean, joint_last], dim=-1)
        joint_hidden = self.residual_joint_up(
            F.gelu(self.residual_joint_down(joint_pooled))
        )
        frozen_classifier = self.head[-1]
        joint_logits = F.linear(joint_hidden, frozen_classifier.weight)
        gate = 1.0 + torch.tanh(self.residual_logit)
        return self.head(pooled) + self.residual_head(pooled) + gate * joint_logits


class GenericHybridAdapterGRU(GenericInputResidualGRU):
    """Near-parameter-matched low-rank adapter without time--graph filtering."""

    def __init__(
        self,
        rms: torch.Tensor,
        num_labels: int,
        gamma: float,
        hidden: int,
        dropout: float,
        bottleneck: int = 4,
    ) -> None:
        super().__init__(rms, num_labels, gamma, hidden, dropout)
        projection_dim = int(rms.shape[1])
        self.residual_head = nn.Linear(hidden * 2, num_labels)
        self.residual_joint_down = nn.Linear(2 * projection_dim, bottleneck)
        self.residual_joint_up = nn.Linear(bottleneck, hidden * 2, bias=False)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.residual_joint_up.weight)

    def forward(self, basis: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        x, mask = self.normalize(basis, observations)
        base_weights = torch.softmax(self.base_graph_logits, dim=0)
        base = torch.einsum("btkd,k->btd", x, base_weights)
        sequence = torch.cat([base, torch.log1p(observations)], dim=-1)
        output, _ = self.gru(sequence)
        mean, last = _masked_pool(output, mask)
        pooled = torch.cat([mean, last], dim=-1)

        generic = F.gelu(self.residual_norm(base) * self.residual_channel_scale)
        generic_mean, generic_last = _masked_pool(generic, mask)
        generic_pooled = torch.cat([generic_mean, generic_last], dim=-1)
        generic_hidden = self.residual_joint_up(
            F.gelu(self.residual_joint_down(generic_pooled))
        )
        frozen_classifier = self.head[-1]
        generic_logits = F.linear(generic_hidden, frozen_classifier.weight)
        gate = 1.0 + torch.tanh(self.residual_logit)
        return self.head(pooled) + self.residual_head(pooled) + gate * generic_logits


@dataclass(frozen=True)
class ModelConfig:
    name: str
    gamma: float = 0.5
    dropout: float = 0.2
    hidden: int = 42
    temporal_order: int = 2
    graph_order: int = 2
    rank: int = 4


def create_model(config: ModelConfig, rms: torch.Tensor, num_labels: int) -> nn.Module:
    if config.name == "static_mlp":
        return StaticMLP(rms, num_labels, config.gamma, hidden=144, dropout=config.dropout)
    if config.name == "temporal_gru":
        return GraphTemporalGRU(
            rms, num_labels, config.gamma, hidden=config.hidden, dropout=config.dropout, use_graph=False
        )
    if config.name == "graph_gru":
        return GraphTemporalGRU(
            rms, num_labels, config.gamma, hidden=config.hidden, dropout=config.dropout, use_graph=True
        )
    if config.name == "temporal_only":
        return LowRankJointFilter(
            rms, num_labels, config.gamma, config.temporal_order, 0, config.rank, config.dropout
        )
    if config.name == "graph_only":
        return LowRankJointFilter(
            rms, num_labels, config.gamma, 0, config.graph_order, config.rank, config.dropout
        )
    if config.name == "separable":
        return LowRankJointFilter(
            rms, num_labels, config.gamma, config.temporal_order, config.graph_order, 1, config.dropout
        )
    if config.name == "tfedgsp":
        return LowRankJointFilter(
            rms,
            num_labels,
            config.gamma,
            config.temporal_order,
            config.graph_order,
            config.rank,
            config.dropout,
        )
    if config.name == "separable_resgru":
        return JointResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            1,
            config.dropout,
        )
    if config.name == "tfedgsp_resgru":
        return JointResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            config.rank,
            config.dropout,
        )
    if config.name in {"separable_logitres", "tfedgsp_logitres"}:
        return JointLogitResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            1 if config.name == "separable_logitres" else config.rank,
            config.dropout,
        )
    if config.name == "tfedgsp_fulljoint_logitres":
        return FullJointLogitResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            config.dropout,
        )
    if config.name == "tfedgsp_fulljoint_inputres":
        return FullJointInputResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            config.dropout,
        )
    if config.name == "generic_inputres":
        return GenericInputResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.dropout,
        )
    if config.name == "tfedgsp_head_inputres":
        return FullJointHeadInputResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            config.dropout,
        )
    if config.name == "generic_head_inputres":
        return GenericHeadInputResidualGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.dropout,
        )
    if config.name == "tfedgsp_hybrid_adapter":
        return FullJointHybridAdapterGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.temporal_order,
            config.graph_order,
            config.dropout,
        )
    if config.name == "generic_hybrid_adapter":
        return GenericHybridAdapterGRU(
            rms,
            num_labels,
            config.gamma,
            config.hidden,
            config.dropout,
        )
    raise ValueError(f"unknown model {config.name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
