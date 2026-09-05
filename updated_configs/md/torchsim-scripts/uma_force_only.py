"""Force-only inference for UMA 1p1 with fairchem-core 2.21.0."""


def disable_uma_stress(calculator):
    """Disable derivative stress and remove its predictor tasks before first use."""
    predictor = calculator.predictor
    if predictor.lazy_model_intialized:
        raise RuntimeError("Disable UMA stress before the first prediction/compilation")
    model = predictor.model.module
    config = model.backbone.regress_config
    if config.direct_forces or config.direct_stress or not config.forces or config.hessian:
        raise RuntimeError("Expected conservative UMA energy/forces without Hessians")
    if not config.stress:
        raise RuntimeError("Expected the original stress-enabled UMA model")
    tasks = {name: task for name, task in model.tasks.items() if task.property != "stress"}
    omat_properties = {task.property for task in tasks.values() if "omat" in task.datasets}
    if not {"energy", "forces"}.issubset(omat_properties):
        raise RuntimeError("UMA checkpoint is missing OMat energy/force tasks")

    # Task routing/normalization must agree with the outputs the heads produce.
    # Preserve the existing Task objects, including normalizers and atom references.
    from fairchem.core.models.base import _get_dataset_to_tasks_map

    model._tasks = tasks
    model._dataset_to_tasks = _get_dataset_to_tasks_map(tasks.values())
    model.backbone.validate_tasks(model._dataset_to_tasks)
    predictor.inference_settings.auto_add_default_untrained_tasks = False
    predictor.inference_settings.predict_untrained_stress = set()
    # EFS heads share this config with the backbone. Also check every head so a
    # package/API change cannot silently leave joint differentiation enabled.
    config.stress = False
    for module in model.modules():
        regression = getattr(module, "regress_config", None)
        if regression is not None and regression.stress:
            raise RuntimeError("UMA contains a head with an unshared stress configuration")
    calculator._compute_stress = False
    calculator.implemented_properties = ["energy", "forces"]
    return calculator


def load_force_only_uma(model_name, device):
    """Keep the standard loader, precision and inference settings; disable stress."""
    import torch
    from torch_sim.models.fairchem import FairChemModel

    calculator = FairChemModel(
        model=model_name, task_name="omat", device=device,
        dtype=torch.float32, compute_stress=False,
    )
    return disable_uma_stress(calculator)
