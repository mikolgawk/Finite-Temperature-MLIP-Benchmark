#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    rmse_chgnet.py
    rmse_eq_v2_M_omat.py
    rmse_esen_30M_OAM.py
    rmse_grace_mp.py
    rmse_grace_oam.py
    rmse_mace_mh_omat.py
    rmse_mace_mp_0.py
    rmse_mace_mpa_0.py
    rmse_mattersim_v1_5M.py
    rmse_nequip.py
    rmse_orb_v2.py
    rmse_orb_v3.py
    rmse_orb_v3_direct.py
    rmse_pet_oam_xl.py
    rmse_pet_omat_xl.py
    rmse_uma_s_omat.py
    rmse_uma_m_omat.py
)

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not available on PATH." >&2
    exit 127
fi

failures=()

for script in "${scripts[@]}"; do
    echo
    echo "============================================================"
    echo "Running ${script}"
    echo "============================================================"

    if uv run "${script_dir}/${script}" "$@"; then
        echo "Completed ${script}"
    else
        status=$?
        echo "Failed ${script} (exit ${status})" >&2
        failures+=("${script}:${status}")
    fi
done

echo
if ((${#failures[@]} > 0)); then
    echo "RMSE sweep finished with ${#failures[@]} failure(s):" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "RMSE sweep completed successfully for all ${#scripts[@]} calculators."
