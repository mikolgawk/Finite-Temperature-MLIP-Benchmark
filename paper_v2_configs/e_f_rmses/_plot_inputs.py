"""Select one source's RMSE tables and plot destinations."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metric_sources import SOURCES
from system_filters import add_molecular_crystal_option, filter_molecular_crystals


INCLUDE_MOLECULAR_CRYSTALS = False


def plot_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=SOURCES, default='mlip-trajs-torchsim-eager')
    parser.add_argument('--results-dir', type=Path, default=here / 'results')
    parser.add_argument('--plots-dir', type=Path, default=here / 'plots')
    add_molecular_crystal_option(parser)
    args = parser.parse_args()
    global INCLUDE_MOLECULAR_CRYSTALS
    INCLUDE_MOLECULAR_CRYSTALS = args.include_molecular_crystals
    return args.results_dir.resolve() / args.source, args.plots_dir.resolve() / args.source


def read_metric_csv(path):
    import pandas as pd
    return filter_molecular_crystals(pd.read_csv(path), INCLUDE_MOLECULAR_CRYSTALS)
