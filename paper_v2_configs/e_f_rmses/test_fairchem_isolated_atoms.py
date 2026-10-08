"""Check eqV2's empty-graph extension against ordinary disconnected nodes."""

import copy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent / 'rmse_torchsim_scripts'))
from fairchem_isolated_atoms import _support_empty_eqv2_graphs

try:
    LEGACY_FAIRCHEM = version('fairchem-core') == '1.10.0'
except PackageNotFoundError:
    LEGACY_FAIRCHEM = False


@unittest.skipUnless(LEGACY_FAIRCHEM, 'requires fairchem-core 1.10.0')
class EqV2EmptyGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from fairchem.core.common.utils import setup_imports
        setup_imports()
        from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone

        torch.manual_seed(42)
        cls.original = EquiformerV2Backbone(
            use_pbc=True, num_layers=2, sphere_channels=8, attn_hidden_channels=8,
            num_heads=2, attn_alpha_channels=2, attn_value_channels=2,
            ffn_hidden_channels=8, edge_channels=8, lmax_list=[2], mmax_list=[1],
            alpha_drop=0, drop_path_rate=0,
        ).eval()
        # Exercise the learned projection bias, not only zero-biased modules.
        with torch.no_grad():
            for block in cls.original.blocks:
                block.ga.proj.bias.fill_(0.25)
        cls.patched = copy.deepcopy(cls.original)
        _support_empty_eqv2_graphs(cls.patched)

    def embeddings(self, backbone, atoms):
        import torch
        from fairchem.core.preprocessing import AtomsToGraphs
        from torch_geometric.data import Batch

        data = Batch.from_data_list([AtomsToGraphs(r_pbc=True, r_edges=False).convert(atoms)])
        # eqV2 uses random edge-coordinate frames; match them for comparisons.
        torch.manual_seed(42)
        with torch.no_grad():
            return backbone(data)['node_embedding'].embedding.clone()

    def test_empty_graph_matches_disconnected_atom_in_original_graph(self):
        import torch
        from ase import Atoms

        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]], pbc=False)
        for symbol in ('C', 'H'):
            with self.subTest(symbol=symbol):
                isolated = Atoms(symbol, positions=[[30, 0, 0]], pbc=False)
                alone = self.embeddings(self.patched, isolated)[0]
                disconnected = self.embeddings(self.original, h2 + isolated)[-1]
                torch.testing.assert_close(alone, disconnected, rtol=1e-5, atol=1e-6)

    def test_nonempty_graph_preserves_original_forward(self):
        import torch
        from ase import Atoms

        atoms = Atoms('CH', positions=[[0, 0, 0], [0, 0, 1.1]], cell=[10, 10, 10], pbc=True)
        original = self.embeddings(self.original, atoms)
        patched = self.embeddings(self.patched, atoms)
        torch.testing.assert_close(patched, original, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
