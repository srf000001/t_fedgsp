from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = ROOT / "t_fedgsp" / "configs" / "templates"
OUTPUT_ROOT = ROOT / "t_fedgsp" / "configs"

FAMILIES = {
    "common_graphgru": {
        "template": "federated_confirm_backbone.yaml",
        "output": "federated_confirm_seed{seed}_backbone",
        "checkpoint": False,
    },
    "graph_adamw_fedavg": {
        "template": "federated_graph_extra30_seed0.yaml",
        "output": "federated_strong_seed{seed}_graph",
        "checkpoint": True,
    },
    "residual_sgd_fedadam": {
        "template": "federated_fulljoint_strongbase_seed0.yaml",
        "output": "federated_strong_seed{seed}_fulljoint",
        "checkpoint": True,
    },
    "residual_adamw_fedavg": {
        "template": "review_control_fulljoint_adamwfedavg_seed0.yaml",
        "output": "review_control_fulljoint_adamwfedavg_seed{seed}",
        "checkpoint": True,
    },
    "graph_sgd_fedadam": {
        "template": "review_control_graph_sgdfedadam_seed0.yaml",
        "output": "review_control_graph_sgdfedadam_seed{seed}",
        "checkpoint": True,
    },
    "hybrid_sgd_fedadam": {
        "template": "review_control_hybrid_adapters_seed0.yaml",
        "output": "review_control_hybrid_adapters_seed{seed}",
        "checkpoint": True,
    },
}


def instantiate(template: str, seed: int, output: str, checkpoint: bool) -> str:
    text = (TEMPLATE_ROOT / template).read_text(encoding="utf-8")
    text, seed_count = re.subn(r"(?m)^  seed: \d+\s*$", f"  seed: {seed}", text)
    text, root_count = re.subn(
        r"(?m)^  root: t_fedgsp/results/\S+\s*$",
        f"  root: t_fedgsp/results/{output}",
        text,
    )
    if seed_count != 1 or root_count != 1:
        raise RuntimeError(f"unexpected template structure: {template}")
    if checkpoint:
        replacement = (
            "  pretrained_backbone_checkpoint: "
            f"t_fedgsp/results/federated_confirm_seed{seed}_backbone/"
            "checkpoints/fedavg_graph_continue.pt"
        )
        text, checkpoint_count = re.subn(
            r"(?m)^  pretrained_backbone_checkpoint: \S+\s*$",
            replacement,
            text,
        )
        if checkpoint_count != 1:
            raise RuntimeError(f"missing checkpoint field: {template}")
    return text.rstrip() + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix = []
    for seed in (1, 2, 3):
        for label, specification in FAMILIES.items():
            output = specification["output"].format(seed=seed)
            filename = f"{label}_seed{seed}.yaml"
            rendered = instantiate(
                specification["template"], seed, output, specification["checkpoint"]
            )
            (OUTPUT_ROOT / filename).write_text(rendered, encoding="utf-8")
            matrix.append(
                {
                    "condition": label,
                    "seed": seed,
                    "config": f"t_fedgsp/configs/{filename}",
                    "result_root": f"t_fedgsp/results/{output}",
                    "test_evaluated": False,
                }
            )
    (OUTPUT_ROOT / "seed_matrix.json").write_text(
        json.dumps({"status": "PASS", "runs": matrix}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(matrix)} explicit seed configurations")


if __name__ == "__main__":
    main()
