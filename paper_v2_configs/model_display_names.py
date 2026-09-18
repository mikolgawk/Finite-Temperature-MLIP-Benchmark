"""Authoritative model names used by every paper-v2 plot.

Edit ``MODEL_DISPLAY_NAMES`` to change a label everywhere. Keys are normalized
model identifiers (lowercase, without ``-force-only``, ``-stress``, or a final
``-eager`` tag). Accelerated execution variants have their own entries so their
suffix labels can also be controlled here.
"""

from __future__ import annotations


MODEL_DISPLAY_NAMES: dict[str, str] = {
    # Tier 1
    "chgnet": "CHGNet",
    "mace-mp-0": "MACE-MP-0",
    "mace-mp-0-compile": "MACE-MP-0",
    "grace-mp": "GRACE-2L-MPtrj",
    # Tier 2
    "mace-mpa-0": "MACE-MPA-0",
    "mace-mpa-0-compile": "MACE-MPA-0",
    "orb-v2": "orb-v2",
    # Tier 3
    "mattersim-v1-5m": "MatterSim-v1.0.0-5M",
    "mattersim-v1-5m-compile": "MatterSim-v1.0.0-5M",
    "grace-oam": "GRACE-2L-OAM",
    "grace-oam-compiled": "GRACE-2L-OAM",
    "orb-v3": "orb-v3-conservative-inf-mpa",
    "orb-v3-direct": "orb-v3-direct-20-mpa",
    "esen-30m-oam": "eSEN-30M-OAM",
    "nequip": "NequIP-OAM-XL",
    "eq-v2-m-omat": "EquiformerV2",
    "pet-oam-xl": "PET-OAM-XL",
    "pet-oam-xl-torchscript": "PET-OAM-XL",
    "pet-omat-xl": "PET-OMAT-XL",
    "pet-omat-xl-torchscript": "PET-OMAT-XL",
    # Tier 4
    "mace-mh-omat": "MACE-MH-1-OMAT",
    "mace-mh-omat-compile": "MACE-MH-1-OMAT",
    "uma-s-omat": "UMA-S-P1",
    "uma-s-omat-compile": "UMA-S-P1 (compile)",
    "uma-s-omat-turbo": "UMA-S-P1 (turbo)",
    "uma-m-omat": "UMA-M-P1",
    "uma-m-omat-compile": "UMA-M-P1 (compile)",
    "uma-m-omat-turbo": "UMA-M-P1 (turbo)",
}


MODEL_KEY_ALIASES = {
    "nequip-oam-l": "nequip",
}


def normalize_display_key(name: object) -> str:
    """Normalize only tags that do not identify a distinct plotted variant."""
    key = str(name).strip().lower()
    key = key.replace("-force-only", "").replace("-stress", "")
    if key.endswith("-eager"):
        key = key.removesuffix("-eager")
    return MODEL_KEY_ALIASES.get(key, key)


def display_model_name(name: object) -> str:
    """Return the configured plot label, falling back to the input text."""
    text = str(name).strip()
    return MODEL_DISPLAY_NAMES.get(normalize_display_key(text), text)

