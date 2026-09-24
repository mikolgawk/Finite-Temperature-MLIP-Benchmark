"""Select one source's RMSE tables and plot destinations."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metric_sources import SOURCES


def plot_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=SOURCES, default='mlip-trajs-torchsim-eager')
    parser.add_argument('--results-dir', type=Path, default=here / 'results')
    parser.add_argument('--plots-dir', type=Path, default=here / 'plots')
    args = parser.parse_args()
    return args.results_dir.resolve() / args.source, args.plots_dir.resolve() / args.source
