"""Remove GRACE's virial branch from a loaded TensorFlow SavedModel graph."""


def prune_grace_stress(restored_compute):
    """Return a callable that computes only total energy and atomic forces.

    GRACE SavedModels expose one XLA-compiled ``compute`` function whose normal
    output dictionary also contains virial, atomic-energy, and pair-force
    tensors. Selecting two entries after calling that function is too late to
    avoid those output branches. TensorFlow's graph-pruning machinery instead
    traces backwards from just the two outputs required by NVT MD.

    This uses TensorFlow internals, so fail loudly if the pinned TensorFlow API
    or the GRACE artifact layout changes rather than silently benchmarking the
    unpruned graph.
    """
    import tensorflow as tf
    from tensorflow.python.eager.wrap_function import VariableHolder, WrappedFunction

    concrete_functions = restored_compute.concrete_functions
    if len(concrete_functions) != 1:
        raise RuntimeError(
            f"Expected one GRACE compute graph, found {len(concrete_functions)}"
        )
    concrete = concrete_functions[0]
    outputs = tf.nest.pack_sequence_as(concrete.structured_outputs, concrete.outputs)
    required = {"total_energy", "total_f"}
    if not required.issubset(outputs) or "virial" not in outputs:
        raise RuntimeError(
            f"Unexpected GRACE outputs: {sorted(outputs)}"
        )

    # WrappedFunction.prune works for this SavedModel wrapper but expects the
    # holder attribute normally installed by tf.compat.v1.wrap_function. All
    # model variables remain existing external captures; no variables are made.
    concrete._variable_holder = VariableHolder()  # noqa: SLF001
    pruned = WrappedFunction.prune(
        concrete,
        feeds=concrete.inputs,
        fetches={name: outputs[name] for name in sorted(required)},
        name="grace_energy_forces",
        input_signature=concrete.structured_input_signature,
    )
    # prune() does not propagate function attributes. Re-wrap the reduced graph
    # with the original XLA attributes so this remains the same compiled
    # inference workload rather than falling back to uncompiled TensorFlow.
    reduced = pruned
    pruned = WrappedFunction(
        reduced.graph,
        variable_holder=VariableHolder(),
        attrs=dict(concrete.function_def.attr),
    )
    pruned._num_positional_args = reduced._num_positional_args  # noqa: SLF001
    pruned._arg_keywords = reduced._arg_keywords  # noqa: SLF001
    if set(pruned.structured_outputs) != required:
        raise RuntimeError("GRACE force-only graph pruning failed")
    if (concrete.function_def.attr.get("_XlaMustCompile") is not None
            and not pruned.function_def.attr["_XlaMustCompile"].b):
        raise RuntimeError("GRACE force-only graph lost XLA compilation")
    return pruned
