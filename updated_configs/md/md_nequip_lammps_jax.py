# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26",
#   "h5py>=3.12",
#   "numpy>=2.0",
# ]
# ///
"""LAMMPS-JAX NVT production MD — NequIP-OAM-XL.

The NequIP model is exported through
``lammps-jax/examples/torch/nequip_kernel_export.py``. That exporter replaces
the interpreted tensor products with the fused OpenEquivariance kernels. The
resulting bundle is evaluated by LAMMPS ``pair_style jax/kk`` on the GPU.

Set ``LAMMPS_BIN``, ``LAMMPS_PLUGIN_PATH``, and ``PJRT_PLUGIN`` as described in
``lammps-jax/README.md``, then run ``uv run md_nequip_lammps_jax.py``. The LAMMPS build
needs CUDA Kokkos and the EXTRA-FIX package (for the CSVR reference systems).

Per-system parameters come from ``md_metadata.json``. LAMMPS dumps are
streamed into the HDF5 arrays consumed by the repository's TorchSim analysis:

    <OUT_ROOT>/<system>/nvt_nequip.h5
"""

import csv
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from ase.data import atomic_masses, chemical_symbols
from ase.io import read, write


MODEL_NAME = "nequip"
MODEL_ID = os.environ.get("NEQUIP_MODEL_ID", "nequip.net:mir-group/NequIP-OAM-XL:0.1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LAMMPS_JAX = HERE / "lammps-jax"
EXPORTER = LAMMPS_JAX / "examples" / "torch" / "nequip_kernel_export.py"
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-lammps-jax-matched"
BUNDLE = Path(
    os.environ.get(
        "NEQUIP_LAMMPS_JAX_BUNDLE",
        REPO
        / "updated_configs"
        / "data"
        / "models"
        / "nequip-oam-xl-oeq.lammps-jax.json",
    )
).resolve()

# One bundle covers every element currently present in the benchmark. LAMMPS
# atom type i maps to TYPE_Z[i - 1] in the exported NequIP model.
TYPE_Z = (
    1,
    3,
    6,
    7,
    8,
    12,
    13,
    16,
    22,
    27,
    29,
    30,
    34,
    35,
    40,
    42,
    47,
    50,
    53,
    55,
    78,
    79,
    82,
)
TYPE_SYMBOLS = tuple(chemical_symbols[z] for z in TYPE_Z)
TYPE_TO_Z = np.asarray(TYPE_Z, dtype=np.int32)

# Static JAX shapes. The edge capacity includes headroom over the benchmark's
# densest initial structure (H at 1050 K: about 230k directed edges at 6 A).
MAX_ATOMS = 8192  # H cell reaches about 7k owned plus ghost rows
MAX_OWNED = 1024  # benchmark maximum is currently 512 owned atoms
MAX_EDGES = 300_000

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SEED = 42
NEIGHBOR_SKIN_ANGSTROM = 1.0


class HDF5TrajectoryWriter:
    """Write the state arrays used by this repository's TorchSim readers."""

    def __init__(self, filename: Path, atomic_numbers: np.ndarray, pbc: np.ndarray):
        self.file = h5py.File(filename, "w")
        header = self.file.create_group("header")
        header.attrs["program"] = np.bytes_("LAMMPS-JAX")
        header.attrs["title"] = np.bytes_("NequIP-OAM-XL trajectory")
        self.file.create_group("metadata")
        self.data = self.file.create_group("data")
        self.steps = self.file.create_group("steps")

        numbers = np.asarray(atomic_numbers, dtype=np.int32)
        masses = atomic_masses[numbers].astype(np.float32)
        self.data.create_dataset("atomic_numbers", data=numbers[None, :])
        self.data.create_dataset("masses", data=masses[None, :])
        self.data.create_dataset("pbc", data=np.asarray(pbc, dtype=bool))
        self.steps.create_dataset("atomic_numbers", data=np.asarray([0], np.int32))
        self.steps.create_dataset("masses", data=np.asarray([0], np.int32))
        self.steps.create_dataset("pbc", data=np.asarray([0], np.int32))

        n_atoms = len(numbers)
        self.arrays = {
            name: self.data.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                chunks=(1,) + shape,
                compression="gzip",
                compression_opts=1,
                dtype=np.float32,
            )
            for name, shape in {
                "positions": (n_atoms, 3),
                "velocities": (n_atoms, 3),
                "cell": (3, 3),
            }.items()
        }
        self.step_arrays = {
            name: self.steps.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype=np.int32,
            )
            for name in self.arrays
        }

    def write(
        self, step: int, positions: np.ndarray, velocities: np.ndarray, cell: np.ndarray
    ) -> None:
        values = {
            "positions": positions,
            "velocities": velocities,
            # TorchSim stores cell vectors as columns; ASE/LAMMPS use rows.
            "cell": cell.T,
        }
        for name, value in values.items():
            array = self.arrays[name]
            step_array = self.step_arrays[name]
            index = len(array)
            array.resize(index + 1, axis=0)
            step_array.resize(index + 1, axis=0)
            array[index] = np.asarray(value, dtype=np.float32)
            step_array[index] = step

    def close(self) -> None:
        self.file.close()


def benchmark_type_z(metadata: dict) -> tuple[int, ...]:
    """Return the element union in available benchmark initial structures."""
    numbers: set[int] = set()
    for meta in metadata.values():
        init_file = REPO / meta["initfile_path"]
        if init_file.is_file():
            numbers.update(int(z) for z in read(init_file, index=0).numbers)
    return tuple(sorted(numbers))


def bundle_configuration() -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "type_z": list(TYPE_Z),
        "max_atoms": MAX_ATOMS,
        "max_owned": MAX_OWNED,
        "max_edges": MAX_EDGES,
        "kernel": "OpenEquivariance",
    }


def ensure_bundle() -> Path:
    """Export the fused-kernel bundle once, regenerating stale configurations."""
    sidecar = Path(str(BUNDLE) + ".env")
    config_file = Path(str(BUNDLE) + ".benchmark-config.json")
    configuration = bundle_configuration()
    if BUNDLE.is_file() and sidecar.is_file() and config_file.is_file():
        try:
            if json.loads(config_file.read_text()) == configuration:
                return sidecar
        except json.JSONDecodeError:
            pass

    if not EXPORTER.is_file():
        raise FileNotFoundError(f"NequIP kernel exporter not found: {EXPORTER}")
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to export the LAMMPS-JAX NequIP bundle")

    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary_bundle = Path(str(BUNDLE) + ".inprogress")
    temporary_sidecar = Path(str(temporary_bundle) + ".env")
    temporary_bundle.unlink(missing_ok=True)
    temporary_sidecar.unlink(missing_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "LAMMPS_JAX_NEQUIP_OUTPUT": str(temporary_bundle),
            "LAMMPS_JAX_NEQUIP_MODEL": MODEL_ID,
            "LAMMPS_JAX_NEQUIP_TYPE_Z": ",".join(map(str, TYPE_Z)),
            "LAMMPS_JAX_NEQUIP_MAX_ATOMS": str(MAX_ATOMS),
            "LAMMPS_JAX_NEQUIP_MAX_OWNED": str(MAX_OWNED),
            "LAMMPS_JAX_NEQUIP_MAX_EDGES": str(MAX_EDGES),
        }
    )
    print(f"[{MODEL_NAME}] exporting OpenEquivariance bundle {BUNDLE}")
    try:
        subprocess.run(["uv", "run", str(EXPORTER)], check=True, env=environment)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "NequIP kernel export failed. The OpenEquivariance pip wheel does not "
            "contain openequivariance_extjax; build that extension as described in "
            "lammps-jax/examples/torch/nequip_kernel_export.py."
        ) from error

    if not temporary_bundle.is_file() or not temporary_sidecar.is_file():
        raise RuntimeError(
            "NequIP exporter did not produce its bundle and .env sidecar"
        )
    temporary_bundle.replace(BUNDLE)
    temporary_sidecar.replace(sidecar)
    config_file.write_text(json.dumps(configuration, indent=2) + "\n")
    return sidecar


def load_export_environment(sidecar: Path) -> dict[str, str]:
    """Read the exporter's trusted ``export NAME=value`` runtime sidecar."""
    result: dict[str, str] = {}
    for line in sidecar.read_text().splitlines():
        if not line.startswith("export "):
            raise ValueError(f"unexpected line in {sidecar}: {line!r}")
        name, raw_value = line.removeprefix("export ").split("=", 1)
        values = shlex.split(raw_value)
        if len(values) != 1:
            raise ValueError(f"invalid value for {name} in {sidecar}")
        result[name] = values[0]
    return result


def runtime_paths() -> tuple[str, str, str]:
    lammps_value = os.environ.get("LAMMPS_BIN", "lmp")
    lammps_bin = shutil.which(lammps_value)
    if lammps_bin is None:
        raise RuntimeError(f"LAMMPS executable not found: {lammps_value}")

    plugin_path = os.environ.get("LAMMPS_PLUGIN_PATH")
    pjrt_plugin = os.environ.get("PJRT_PLUGIN")
    missing = [
        name
        for name, value in (
            ("LAMMPS_PLUGIN_PATH", plugin_path),
            ("PJRT_PLUGIN", pjrt_plugin),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "set " + ", ".join(missing) + " for the LAMMPS-JAX runtime; see "
            f"{LAMMPS_JAX / 'README.md'}"
        )
    assert plugin_path is not None and pjrt_plugin is not None
    plugin_directories = [Path(value) for value in plugin_path.split(os.pathsep)]
    if not any(path.is_dir() for path in plugin_directories):
        raise RuntimeError(
            f"LAMMPS_PLUGIN_PATH does not contain a directory: {plugin_path}"
        )
    if not Path(pjrt_plugin).is_file():
        raise RuntimeError(f"PJRT_PLUGIN does not exist: {pjrt_plugin}")
    return lammps_bin, plugin_path, pjrt_plugin


def thermostat_commands(thermostat: str, temperature_k: float, tau_fs: float) -> str:
    tau_ps = tau_fs / 1000.0
    if thermostat == "Nose-Hoover":
        return (
            f"fix thermostat all nvt temp {temperature_k:.16g} "
            f"{temperature_k:.16g} {tau_ps:.16g} "
            f"tchain {CHAIN_LENGTH} tloop {CHAIN_STEPS}"
        )
    if thermostat == "Langevin":
        return (
            f"fix integrate all nve\n"
            f"fix thermostat all langevin {temperature_k:.16g} "
            f"{temperature_k:.16g} {tau_ps:.16g} {SEED}"
        )
    if thermostat.lower().startswith("velocity"):
        return (
            f"fix integrate all nve\n"
            f"fix thermostat all temp/csvr {temperature_k:.16g} "
            f"{temperature_k:.16g} {tau_ps:.16g} {SEED}"
        )
    raise ValueError(f"unsupported thermostat type {thermostat!r}")


def make_lammps_input(
    temperature_k: float,
    timestep_fs: float,
    tau_fs: float,
    thermostat: str,
    stride: int,
    n_steps: int,
) -> str:
    return f"""\
units metal
atom_style atomic
boundary p p p
newton on

read_data ${{data}}

neighbor {NEIGHBOR_SKIN_ANGSTROM:.16g} bin
neigh_modify every 1 delay 0 check yes one 1024 page 1000000

pair_style jax/kk ${{pjrt}}
pair_coeff * * ${{bundle}}

velocity all create {temperature_k:.16g} {SEED} mom yes rot no dist gaussian
{thermostat_commands(thermostat, temperature_k, tau_fs)}
timestep {timestep_fs / 1000.0:.16g}

thermo {max(1, stride)}
thermo_style custom step atoms temp pe ke etotal press
thermo_modify norm no format float %.16g

dump trajectory all custom {stride} ${{dump_path}} id type x y z vx vy vz
dump_modify trajectory sort id first yes format float %.16g

run_style verlet/kk
run {n_steps}
"""


def dump_frames(
    filename: Path,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Stream sorted frames from a LAMMPS custom text dump."""
    with filename.open() as file:
        while True:
            marker = file.readline()
            if not marker:
                return
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"invalid LAMMPS dump marker: {marker.rstrip()!r}")
            step = int(file.readline())
            if file.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("LAMMPS dump is missing NUMBER OF ATOMS")
            n_atoms = int(file.readline())
            box_header = file.readline().split()
            if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("LAMMPS dump is missing BOX BOUNDS")
            bounds = [list(map(float, file.readline().split())) for _ in range(3)]

            if {"xy", "xz", "yz"}.issubset(box_header):
                xlo_b, xhi_b, xy = bounds[0]
                ylo_b, yhi_b, xz = bounds[1]
                zlo, zhi, yz = bounds[2]
                xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                xhi = xhi_b - max(0.0, xy, xz, xy + xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                cell = np.asarray(
                    [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]],
                    dtype=np.float64,
                )
            else:
                cell = np.diag([row[1] - row[0] for row in bounds]).astype(np.float64)

            atom_header = file.readline().split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("LAMMPS dump is missing ATOMS")
            columns = atom_header[2:]
            required = ("id", "type", "x", "y", "z", "vx", "vy", "vz")
            indices = {name: columns.index(name) for name in required}
            rows = np.asarray(
                [np.fromstring(file.readline(), sep=" ") for _ in range(n_atoms)]
            )
            order = np.argsort(rows[:, indices["id"]].astype(np.int64))
            rows = rows[order]
            types = rows[:, indices["type"]].astype(np.int32)
            if np.any(types < 1) or np.any(types > len(TYPE_Z)):
                raise ValueError("LAMMPS dump contains an atom type outside the bundle")
            numbers = TYPE_TO_Z[types - 1]
            positions = rows[:, [indices["x"], indices["y"], indices["z"]]]
            velocities = rows[:, [indices["vx"], indices["vy"], indices["vz"]]]
            yield step, numbers, positions, velocities, cell


def convert_dump(
    dump_file: Path,
    output_file: Path,
    expected_numbers: np.ndarray,
    pbc: np.ndarray,
    expected_last_step: int,
) -> int:
    temporary = Path(str(output_file) + ".inprogress")
    temporary.unlink(missing_ok=True)
    writer = HDF5TrajectoryWriter(temporary, expected_numbers, pbc)
    frame_count = 0
    last_step = -1
    try:
        for step, numbers, positions, velocities, cell in dump_frames(dump_file):
            if not np.array_equal(numbers, expected_numbers):
                raise ValueError("LAMMPS changed atom ordering or type mapping")
            writer.write(step, positions, velocities, cell)
            frame_count += 1
            last_step = step
    finally:
        writer.close()
    if frame_count == 0 or last_step != expected_last_step:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"incomplete LAMMPS dump: {frame_count} frames, last step {last_step}; "
            f"expected last reported step {expected_last_step}"
        )
    temporary.replace(output_file)
    return frame_count


def run_system(
    name: str,
    meta: dict,
    sidecar: Path,
    lammps_bin: str,
    plugin_path: str,
    pjrt_plugin: str,
) -> None:
    init_file = REPO / meta["initfile_path"]
    if not init_file.is_file():
        print(f"[{MODEL_NAME}] {name}: init file missing, skipping ({init_file})")
        return

    out_dir = OUT_ROOT / name
    out_h5 = out_dir / f"nvt_{MODEL_NAME}.h5"
    out_csv = out_dir / f"md_timing_{MODEL_NAME}.csv"
    if out_h5.exists() and out_csv.exists():
        print(f"[{MODEL_NAME}] {name}: output exists, skipping")
        return

    temperature_k = float(meta["temperature"])
    timestep_fs = float(meta["timestep"])
    tau_fs = float(meta["thermostat_coupling_constant"])
    thermostat = meta["thermostat_type"]
    stride = int(meta["position_print_stride"] or 1)
    n_steps = round(float(meta["trajectory_length_ps"]) * 1000.0 / timestep_fs)

    atoms = read(init_file, index=0)
    if not np.all(atoms.pbc):
        raise ValueError(f"{name}: LAMMPS-JAX runner currently requires 3D periodicity")
    unsupported = sorted(set(int(z) for z in atoms.numbers) - set(TYPE_Z))
    if unsupported:
        raise ValueError(
            f"{name}: elements not present in NequIP bundle: {unsupported}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{MODEL_NAME}] {name}: T={temperature_k:.0f} K, dt={timestep_fs:g} fs, "
        f"{thermostat} tau={tau_fs:g} fs, {n_steps} steps, {len(atoms)} atoms, "
        f"stride {stride}"
    )

    with tempfile.TemporaryDirectory(prefix="nequip-lammps-jax-", dir=out_dir) as work:
        work_dir = Path(work)
        data_file = work_dir / "system.data"
        input_file = work_dir / "in.nequip"
        dump_file = work_dir / "trajectory.dump"
        log_file = work_dir / "log.lammps"
        write(
            data_file,
            atoms,
            format="lammps-data",
            atom_style="atomic",
            specorder=list(TYPE_SYMBOLS),
            masses=True,
        )
        input_file.write_text(
            make_lammps_input(
                temperature_k, timestep_fs, tau_fs, thermostat, stride, n_steps
            )
        )

        environment = os.environ.copy()
        environment.update(load_export_environment(sidecar))
        environment["LAMMPS_PLUGIN_PATH"] = plugin_path
        environment.setdefault("LAMMPS_JAX_MEM_FRACTION", "0.85")
        environment.setdefault("OMP_NUM_THREADS", "1")
        command = [
            lammps_bin,
            "-k",
            "on",
            "g",
            "1",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
            "-sf",
            "kk",
            "-var",
            "pjrt",
            pjrt_plugin,
            "-var",
            "bundle",
            str(BUNDLE),
            "-var",
            "data",
            str(data_file),
            "-var",
            "dump_path",
            str(dump_file),
            "-log",
            str(log_file),
            "-in",
            str(input_file),
        ]

        start = time.perf_counter()
        subprocess.run(command, check=True, env=environment)
        elapsed = time.perf_counter() - start
        expected_last_step = (n_steps // stride) * stride
        frame_count = convert_dump(
            dump_file,
            out_h5,
            atoms.numbers.astype(np.int32),
            atoms.pbc,
            expected_last_step,
        )

    with out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "calculator",
                "system",
                "temperature_K",
                "n_steps",
                "time_step_fs",
                "thermostat",
                "tau_fs",
                "record_interval",
                "elapsed_seconds",
                "seconds_per_step",
                "engine",
                "kernel",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "calculator": MODEL_NAME,
                "system": name,
                "temperature_K": temperature_k,
                "n_steps": n_steps,
                "time_step_fs": timestep_fs,
                "thermostat": thermostat,
                "tau_fs": tau_fs,
                "record_interval": stride,
                "elapsed_seconds": f"{elapsed:.2f}",
                "seconds_per_step": f"{elapsed / n_steps:.6f}",
                "engine": "lammps-jax",
                "kernel": "OpenEquivariance",
                "seed": SEED,
            }
        )
    print(
        f"[{MODEL_NAME}] {name}: saved {out_h5} ({frame_count} frames, "
        f"{elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)"
    )


def main() -> None:
    metadata = json.loads(METADATA_FILE.read_text())
    actual_type_z = benchmark_type_z(metadata)
    if actual_type_z != TYPE_Z:
        raise RuntimeError(
            f"benchmark element union changed: expected {TYPE_Z}, found {actual_type_z}; "
            "update TYPE_Z so the bundle and LAMMPS atom types remain aligned"
        )

    lammps_bin, plugin_path, pjrt_plugin = runtime_paths()
    sidecar = ensure_bundle()
    for name, meta in metadata.items():
        run_system(name, meta, sidecar, lammps_bin, plugin_path, pjrt_plugin)
    print(f"[{MODEL_NAME}] done.")


if __name__ == "__main__":
    main()
