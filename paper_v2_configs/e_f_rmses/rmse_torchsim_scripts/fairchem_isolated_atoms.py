"""Native FairChem energy evaluation for nonperiodic isolated-atom corrections."""


def _support_empty_eqv2_graphs(backbone):
    """Define the zero-neighbour case in the native eqV2 calculator only.

    FairChem 1.10.0 reduces an empty edge tensor in its rotation builder and
    reshapes it with an ambiguous inferred dimension in attention. With no
    incoming edges, the message sum is zero; the learned projection (including
    its bias), residuals and feed-forward layers must still run normally.
    """
    from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding

    original_rotation = backbone._init_edge_rot_mat

    def rotation(data, edge_index, vectors):
        if vectors.shape[0] == 0:
            return vectors.new_empty((0, 3, 3))
        return original_rotation(data, edge_index, vectors)

    def attention_forward(attention):
        original_forward = attention.forward

        def forward(x, atomic_numbers, edge_distance, edge_index, node_offset=0):
            if edge_index.shape[1] == 0:
                messages = SO3_Embedding(
                    len(x.embedding), attention.lmax_list.copy(),
                    attention.num_heads * attention.attn_value_channels,
                    device=x.embedding.device, dtype=x.embedding.dtype,
                )
                return attention.proj(messages)
            return original_forward(x, atomic_numbers, edge_distance, edge_index, node_offset)

        return forward

    backbone._init_edge_rot_mat = rotation
    for block in backbone.blocks:
        block.ga.forward = attention_forward(block.ga)


def make_fairchem_isolated_atom_calculator(checkpoint, family, device, seed=42):
    """Load the matching legacy checkpoint without force or stress calculations.

    OCPCalculator passes the input's PBC flags through to FairChem, whereas the
    TorchSim legacy adapter requires them to match its initialization setting.
    This separate calculator leaves the periodic TorchSim model untouched.
    """
    if family not in {'eqv2', 'esen'}:
        raise ValueError(f'Unsupported legacy model family: {family}')

    from fairchem.core import OCPCalculator

    calculator = OCPCalculator(
        checkpoint_path=str(checkpoint), cpu=device.type == 'cpu',
        seed=seed, disable_amp=True,
    )
    trainer = calculator.trainer
    models = [m for m in trainer.model.modules() if hasattr(m, 'output_heads')]
    if len(models) != 1:
        raise RuntimeError('Expected exactly one legacy Hydra model')
    model = models[0]
    heads = model.output_heads

    if family == 'eqv2':
        if set(heads) != {'energy', 'forces', 'stress'}:
            raise RuntimeError(f'Unexpected eqV2 heads: {list(heads)}')
        del heads['forces']
        del heads['stress']
        _support_empty_eqv2_graphs(model.backbone)
    else:
        if set(heads) != {'mptrj'}:
            raise RuntimeError(f'Unexpected eSEN heads: {list(heads)}')
        backbone, head = model.backbone, heads['mptrj']
        if head.__class__.__name__ != 'MLP_EFS_Head' or backbone.direct_forces:
            raise RuntimeError('Expected conservative eSEN MLP_EFS_Head')
        for component in (backbone, head):
            component.regress_forces = False
            component.regress_stress = False

    # Filtering trainer targets alone does not stop the network's derivative
    # heads from running; disable those above as well as narrowing the outputs.
    for outputs in (trainer.output_targets, trainer.config['outputs'], calculator.config['outputs']):
        if 'energy' not in outputs:
            raise RuntimeError('Checkpoint does not expose energy')
        for key in list(outputs):
            if key != 'energy':
                del outputs[key]
    calculator.implemented_properties = ['energy']
    return calculator
