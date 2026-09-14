"""Disable MACE's unconditional stress request, including compiled calls."""


def disable_mace_stress(calculator):
    def force_only(module, args, kwargs):
        kwargs.update(
            compute_stress=False, compute_virials=False,
            compute_displacement=False, compute_atomic_stresses=False,
            compute_edge_forces=False,
        )
        return args, kwargs

    # The hook runs before the compiled module receives its forward arguments.
    # Install before the first evaluation so stress never enters the graph.
    for model in calculator.models:
        model.register_forward_pre_hook(force_only, with_kwargs=True)
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator
