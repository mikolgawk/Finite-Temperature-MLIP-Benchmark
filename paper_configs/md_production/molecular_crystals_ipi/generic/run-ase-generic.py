'''
Generic i-PI driver: builds an ASE calculator selected at runtime from
model_calculators.json and serves it to i-PI over a unix-domain socket.

Replaces the per-(system, model) run-ase.py copies; the calculator is the
only thing that ever varied between them.

Usage:
    MODEL_NAME=mace-mpa-0 IPI_ADDRESS=driver python3 run-ase-generic.py

Run from inside the run directory (the one holding init.xyz), which is what
submit-generic.sh does.
'''

import os
import json
import contextlib

from ase.io import read
from ase.calculators.socketio import SocketClient

# Shared with md_script-generic.py: paper_configs/md_production/model_calculators.json
MODEL_CATALOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'model_calculators.json')
)


def load_model_catalog(path=MODEL_CATALOG_PATH):
    '''Loads the model calculator catalog from JSON.'''
    with open(path) as f:
        catalog = json.load(f)
    return {model['name']: model for model in catalog['models']}


@contextlib.contextmanager
def working_directory(path):
    '''Temporarily changes the process working directory.'''
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_calculator(model_entry):
    '''
    Executes a model's import statements and evaluates its calculator
    expression, returning the constructed ASE calculator instance.

    Evaluated with the working directory set to the catalog's own directory,
    so the relative checkpoint paths in calculator_expr ('../models/...')
    resolve exactly as they do for md_script-generic.py, which runs from
    md_production/.
    '''
    namespace = {}
    for import_line in model_entry['imports']:
        exec(import_line, namespace)
    with working_directory(os.path.dirname(MODEL_CATALOG_PATH)):
        return eval(model_entry['calculator_expr'], namespace)


def main():
    model_name = os.environ.get('MODEL_NAME')
    catalog = load_model_catalog()
    if model_name not in catalog:
        raise SystemExit(f"Set MODEL_NAME to one of: {', '.join(sorted(catalog))}")

    address = os.environ.get('IPI_ADDRESS', 'driver')

    print("Reading atoms object.")
    atoms = read("init.xyz", 0)

    print(f"Initializing calculator '{model_name}'...")
    atoms.calc = build_calculator(catalog[model_name])
    print(f"  Loaded {model_name}")

    print(f"Setting up socket on unix address '{address}'.")
    client = SocketClient(unixsocket=address)

    print("Running socket.")
    client.run(atoms, use_stress=True)


if __name__ == "__main__":
    main()
