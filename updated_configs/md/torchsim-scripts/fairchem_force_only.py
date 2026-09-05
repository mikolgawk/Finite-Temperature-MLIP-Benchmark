"""Disable stress inside the two supported legacy FairChem checkpoints.

Apply after checkpoint loading, before inference. This changes only the live
model and inference targets; checkpoint files and training metadata stay intact.
"""


def disable_legacy_stress(calculator, family: str) -> None:
    """Configure an eqV2 or eSEN FairChemV1Model for energy/forces only."""
    if family not in {"eqv2", "esen"}:
        raise ValueError(f"Unsupported legacy model family: {family}")
    trainer = calculator.trainer
    models = [m for m in trainer.model.modules() if hasattr(m, "output_heads")]
    if len(models) != 1:
        raise RuntimeError("Expected exactly one legacy Hydra model")
    model = models[0]
    heads = model.output_heads
    required = {"energy", "forces"}
    for outputs in (trainer.output_targets, trainer.config["outputs"], calculator.config["outputs"]):
        if not required.issubset(outputs):
            raise RuntimeError("Checkpoint does not expose energy and forces")

    if family == "eqv2":
        if set(heads) != {"energy", "forces", "stress"}:
            raise RuntimeError(f"Unexpected eqV2 heads: {list(heads)}")
        # Remove the actual network head, including its decomposed stress outputs.
        del heads["stress"]
    else:
        if set(heads) != {"mptrj"}:
            raise RuntimeError(f"Unexpected eSEN heads: {list(heads)}")
        backbone, head = model.backbone, heads["mptrj"]
        if head.__class__.__name__ != "MLP_EFS_Head":
            raise RuntimeError("Expected eSEN MLP_EFS_Head")
        for component in (backbone, head):
            if not component.regress_forces or not component.regress_stress:
                raise RuntimeError("Expected joint eSEN energy/force/stress model")
        if backbone.direct_forces:
            raise RuntimeError("Expected conservative eSEN backbone")
        # Both flags matter: the backbone constructs the strain graph, while
        # the head chooses positions-only versus joint position/strain gradients.
        backbone.regress_stress = False
        head.regress_stress = False

    # predict() and _forward() each consult a different target dictionary.
    # Whitelisting also removes eqV2's stress_isotropic/stress_anisotropic targets.
    trainer.output_targets = {k: v for k, v in trainer.output_targets.items() if k in required}
    trainer.config["outputs"] = {k: v for k, v in trainer.config["outputs"].items() if k in required}
    calculator.config["outputs"] = {k: v for k, v in calculator.config["outputs"].items() if k in required}
    calculator.implemented_properties = ["energy", "forces"]
    # The 0.5.2 legacy adapter exposes read-only public properties.
    calculator._compute_stress = False
    calculator._compute_forces = True


def load_force_only_legacy(checkpoint, family: str, device, seed: int = 42):
    """Load original weights with the existing legacy adapter, then disable stress."""
    from torch_sim.models.fairchem_legacy import FairChemV1Model

    calculator = FairChemV1Model(
        model=str(checkpoint), device=device, compute_stress=False,
        seed=seed, pbc=True, disable_amp=True,
    )
    disable_legacy_stress(calculator, family)
    return calculator
