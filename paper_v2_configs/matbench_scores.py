"""Match MD model identifiers to the repository's available Matbench scores.

Execution variants keep separate MD metrics and plot points, but use their base
model's Matbench scores. Aliases are explicit so that similarly named models,
especially conservative and direct Orb-v3 models, cannot share scores by accident.
"""

from model_display_names import normalize_display_key


MATBENCH_MODEL_ALIASES = {
    "orb-v3-cons": "orb-v3-omat",
    "orb-v3-conservative-inf-omat": "orb-v3-omat",
    "orb-v3-direct-20-omat": "orb-v3-direct-omat",
    "mace-mp-0-compile": "mace-mp-0",
    "mace-mpa-0-compile": "mace-mpa-0",
    "mace-mh-omat-compile": "mace-mh-omat",
    "mattersim-v1-5m-compile": "mattersim-v1-5m",
    "grace-oam-compiled": "grace-oam",
    "pet-oam-xl-torchscript": "pet-oam-xl",
    "pet-omat-xl-torchscript": "pet-omat-xl",
    "uma-s-omat-compile": "uma-s-omat",
    "uma-s-omat-turbo": "uma-s-omat",
    "uma-m-omat-compile": "uma-m-omat",
    "uma-m-omat-turbo": "uma-m-omat",
}


def matbench_model_key(name: object) -> str:
    """Return the score-table identity without changing the plotted model name."""
    key = normalize_display_key(name)
    return MATBENCH_MODEL_ALIASES.get(key, key)
