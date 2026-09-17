from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from tfedgsp_models import ModelConfig, create_model, trainable_parameter_count


class ModelForwardTest(unittest.TestCase):
    def test_all_registered_models_are_finite(self) -> None:
        torch.manual_seed(1)
        batch, steps, orders, dimension = 5, 6, 3, 8
        basis = torch.randn(batch, steps, orders, dimension)
        observations = torch.poisson(torch.ones(batch, steps, 4) * 2)
        observations[:, -1] = 0
        rms = torch.ones(orders, dimension)
        for name in [
            "static_mlp",
            "temporal_gru",
            "temporal_only",
            "graph_only",
            "graph_gru",
            "separable",
            "tfedgsp",
            "separable_resgru",
            "tfedgsp_resgru",
            "separable_logitres",
            "tfedgsp_logitres",
            "tfedgsp_fulljoint_logitres",
            "tfedgsp_fulljoint_inputres",
            "generic_inputres",
            "tfedgsp_head_inputres",
            "generic_head_inputres",
            "tfedgsp_hybrid_adapter",
            "generic_hybrid_adapter",
        ]:
            model = create_model(ModelConfig(name=name), rms, 7)
            logits = model(basis, observations)
            self.assertEqual(list(logits.shape), [batch, 7], name)
            self.assertTrue(torch.isfinite(logits).all(), name)
            self.assertGreater(trainable_parameter_count(model), 0, name)

    def test_joint_surface_has_requested_rank_components(self) -> None:
        rms = torch.ones(3, 8)
        model = create_model(ModelConfig(name="tfedgsp", rank=4), rms, 7)
        surface = model.coefficient_surface()
        self.assertEqual(list(surface.shape), [4, 3, 3])

        residual_model = create_model(ModelConfig(name="tfedgsp_resgru", rank=4), rms, 7)
        residual_surface = residual_model.coefficient_surface()
        self.assertEqual(list(residual_surface.shape), [4, 3, 3])
        effective_surface = residual_surface.mean(dim=0)
        self.assertGreaterEqual(int(torch.linalg.matrix_rank(effective_surface)), 2)
        residual_strength = float(residual_model.residual_strength().detach())
        self.assertGreater(residual_strength, 0.0)
        self.assertLess(residual_strength, 0.5)

        logit_residual = create_model(ModelConfig(name="tfedgsp_logitres", rank=4), rms, 7)
        logit_surface = logit_residual.coefficient_surface()
        self.assertEqual(list(logit_surface.shape), [4, 3, 3])
        self.assertTrue(torch.count_nonzero(logit_residual.residual_head.weight) == 0)

        full_joint = create_model(
            ModelConfig(name="tfedgsp_fulljoint_logitres", temporal_order=1, graph_order=2),
            rms,
            7,
        )
        self.assertEqual(list(full_joint.coefficient_surface().shape), [1, 2, 3])
        with torch.no_grad():
            full_joint.joint_raw.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]))
        self.assertEqual(
            int(torch.linalg.matrix_rank(full_joint.coefficient_surface()[0])), 2
        )

        reduced_order = create_model(
            ModelConfig(name="tfedgsp_resgru", temporal_order=1, graph_order=1, rank=4), rms, 7
        )
        reduced_logits = reduced_order(torch.randn(2, 5, 3, 8), torch.ones(2, 5, 4))
        self.assertEqual(list(reduced_logits.shape), [2, 7])

    def test_zero_logit_residual_matches_transferred_graph_gru(self) -> None:
        torch.manual_seed(11)
        rms = torch.ones(3, 8)
        baseline = create_model(ModelConfig(name="graph_gru", dropout=0.0), rms, 7)
        basis = torch.randn(4, 6, 3, 8)
        observations = torch.poisson(torch.ones(4, 6, 4) * 2)
        baseline.eval()
        for name in [
            "tfedgsp_logitres",
            "tfedgsp_fulljoint_logitres",
            "tfedgsp_fulljoint_inputres",
            "generic_inputres",
            "tfedgsp_head_inputres",
            "generic_head_inputres",
            "tfedgsp_hybrid_adapter",
            "generic_hybrid_adapter",
        ]:
            proposed = create_model(ModelConfig(name=name, dropout=0.0), rms, 7)
            target = proposed.state_dict()
            for source_name, value in baseline.state_dict().items():
                target_name = "base_graph_logits" if source_name == "graph_logits" else source_name
                target[target_name] = value.clone()
            proposed.load_state_dict(target)
            proposed.eval()
            torch.testing.assert_close(
                baseline(basis, observations), proposed(basis, observations), rtol=0.0, atol=0.0
            )


if __name__ == "__main__":
    unittest.main()
