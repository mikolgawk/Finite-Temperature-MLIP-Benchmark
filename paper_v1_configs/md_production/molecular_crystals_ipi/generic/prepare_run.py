#!/usr/bin/env python3
'''
Materializes one i-PI run directory for a (system, model) pair.

Renders input.xml from input.xml.template using the per-system settings in
ipi_settings_ref.csv, and stages the system's init.xyz. Replaces the 85
hand-maintained copies of input.xml/init.xyz.

    python3 prepare_run.py --system naphthalene_295K_Sharma_S \
                           --model mace-mpa-0 \
                           --rundir runs/naphthalene_295K_Sharma_S/mace-mpa-0 \
                           --address ipi_naphthalene_mace-mpa-0_12345

If the run directory already holds a RESTART file, input.xml is left alone
and the RESTART file's <address> is rewritten to the supplied address
instead -- i-PI resumes from the checkpoint, and the checkpoint carries its
own (now stale) socket address.

--model is validated against model_calculators.json so a typo fails here
rather than after the job has queued. Choosing and activating a Python
environment for that model is the caller's business.
'''

from __future__ import annotations

import argparse
import csv
import re
import json
import shutil
from pathlib import Path
from string import Template

GENERIC_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = GENERIC_DIR / 'ipi_settings_ref.csv'
TEMPLATE_PATH = GENERIC_DIR / 'input.xml.template'
SYSTEMS_DIR = GENERIC_DIR / 'systems'
MODEL_CATALOG_PATH = GENERIC_DIR.parent.parent / 'model_calculators.json'

ADDRESS_RE = re.compile(r'(<address>\s*)(.*?)(\s*</address>)', re.DOTALL)


def load_settings() -> dict[str, dict[str, str]]:
    '''Loads per-system i-PI settings keyed by system name.'''
    with SETTINGS_PATH.open(newline='') as f:
        return {row['system']: row for row in csv.DictReader(f)}


def load_catalog() -> dict[str, dict]:
    '''Loads the shared model calculator catalog keyed by model name.'''
    with MODEL_CATALOG_PATH.open() as f:
        return {model['name']: model for model in json.load(f)['models']}


def validate_model(model: str) -> None:
    '''Fails early if a model name is not in the shared catalog.'''
    catalog = load_catalog()
    if model not in catalog:
        raise SystemExit(f"Unknown model {model!r}. Known: {', '.join(sorted(catalog))}")


def render_input_xml(system: str, address: str) -> str:
    '''Renders input.xml for a system, substituting its settings and address.'''
    settings = load_settings()
    if system not in settings:
        raise SystemExit(f"Unknown system {system!r}. Known: {', '.join(sorted(settings))}")

    fields = dict(settings[system])
    fields.pop('system')
    fields['address'] = address

    template = Template(TEMPLATE_PATH.read_text())
    return template.substitute(fields)


def rewrite_restart_address(restart_path: Path, address: str) -> None:
    '''Points an existing i-PI RESTART checkpoint at the current socket address.'''
    text = restart_path.read_text()
    text, n = ADDRESS_RE.subn(lambda m: f"{m.group(1)}{address}{m.group(3)}", text)
    if n == 0:
        raise SystemExit(f"No <address> element found in {restart_path}")
    restart_path.write_text(text)
    print(f"  Rewrote {n} <address> element(s) in {restart_path} -> {address}")


def prepare(system: str, model: str, rundir: Path, address: str) -> None:
    '''Creates and populates a run directory for one (system, model) pair.'''
    validate_model(model)
    rundir.mkdir(parents=True, exist_ok=True)

    init_source = SYSTEMS_DIR / system / 'init.xyz'
    if not init_source.is_file():
        raise SystemExit(f"Missing initial structure: {init_source}")
    shutil.copyfile(init_source, rundir / 'init.xyz')
    print(f"  Staged init.xyz from {init_source}")

    restart_path = rundir / 'RESTART'
    if restart_path.is_file():
        rewrite_restart_address(restart_path, address)
        print(f"  RESTART present; leaving input.xml untouched.")
    else:
        (rundir / 'input.xml').write_text(render_input_xml(system, address))
        print(f"  Wrote {rundir / 'input.xml'} (address {address})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--system', required=True, help='System name, as listed in ipi_settings_ref.csv')
    parser.add_argument('--model', required=True, help='Model name, as listed in model_calculators.json')
    parser.add_argument('--rundir', type=Path, required=True, help='Run directory to create/populate')
    parser.add_argument('--address', required=True, help='Unix socket address for this run')
    args = parser.parse_args()

    prepare(args.system, args.model, args.rundir, args.address)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
