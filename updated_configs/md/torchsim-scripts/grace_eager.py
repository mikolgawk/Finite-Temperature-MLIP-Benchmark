"""Reconstruct GRACE Python instructions for genuine TensorFlow eager inference.

This does not invoke a SavedModel graph, tf.function, or XLA during inference.
GRACE-MP deliberately remains on its stress-pruned XLA SavedModel path because
its public artifact lacks the Python model definition required here.
"""

import os
from pathlib import Path


def load_eager_grace(family):
    import tensorflow as tf
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.tpmodel import TPModel

    if family != "grace_oam":
        raise ValueError("Only GRACE-OAM has the artifacts required for eager inference")
    artifact = "GRACE-2L-OMAT-large-ft-AM"
    cache = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache/grace"))
    root = cache / "checkpoints" / artifact
    potential = Path(os.environ.get("GRACE_OAM_POTENTIAL", root / "model-single.yaml"))
    checkpoint = Path(os.environ.get(
        "GRACE_OAM_CHECKPOINT", root / "checkpoint-single-fp32",
    ))
    if not potential.is_file() or not Path(str(checkpoint) + ".index").is_file():
        raise FileNotFoundError(
            f"True eager {artifact} needs its Python instruction YAML and matching "
            f"fp32 checkpoint. Set GRACE_OAM_POTENTIAL and GRACE_OAM_CHECKPOINT "
            f"(checkpoint prefix without .index). Looked for "
            f"{potential} and {checkpoint}.index. A SavedModel is not an eager fallback."
        )
    model = TPModel(load_instructions(str(potential)))
    model.build(tf.float32)
    status = tf.train.Checkpoint(model=model).read(str(checkpoint))
    status.assert_existing_objects_matched()
    status.expect_partial()  # Training step/optimizer are intentionally unused.

    def compute(input_data):
        from tensorpotential import constants as C
        from tensorpotential.tpmodel import execute_instructions

        if not tf.executing_eagerly():
            raise RuntimeError("GRACE eager inference was called inside a traced graph")
        data = dict(input_data)
        # Match TensorPotential's single-structure force convention exactly,
        # but never construct its virial reduction.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(data[C.BOND_VECTOR])
            execute_instructions(data, model.instructions, training=False)
            atomic_energy = tf.reshape(data[C.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_forces = -tape.gradient(atomic_energy, data[C.BOND_VECTOR])
        n_atoms = tf.reshape(data[C.N_ATOMS_BATCH_TOTAL], [])
        forces = tf.math.unsorted_segment_sum(
            pair_forces, data[C.BOND_IND_J], n_atoms
        ) - tf.math.unsorted_segment_sum(pair_forces, data[C.BOND_IND_I], n_atoms)
        return {
            "total_energy": tf.reduce_sum(atomic_energy, axis=0, keepdims=True),
            "total_f": forces,
        }

    return compute
