import argparse
import sys
import math

script_description = """
------------------------------------------------------------------------------------------------------------
Get VDOS (Vibrational Density of States) from Molecular Dynamics Trajectories

## Overview
This Python script, `get_VDOS.py`, calculates the Vibrational Density of States (VDOS)
from molecular dynamics trajectory files by computing the Fourier transform of the velocity
autocorrelation function.

## Features
- Supports XYZ and NETCDF file formats.
- Full and bond-specific modes.
- Windowing functions for Fourier Transform: Gaussian, Blackman-Harris, Hamming, Hann.
- Option for numerical or file-based velocity calculations.
- Configurable zero padding for FFT.

## Usage
python get_VDOS.py -i [input file] -o [output file] -dt [delta time] [other optional arguments]
------------------------------------------------------------------------------------------------------------
"""

# Universal constants
c = 2.9979245899e10  # speed of light in vacuum in [cm/s]

def print_welcome_message():
    print("""
    ************************************************************
    *                                                          *
    *          Welcome to the get_VDOS Python Script!          *
    *                                                          *
    * This tool calculates the Vibrational Density of States   *
    * (VDOS) from molecular dynamics trajectory files.         *
    *                                                          *
    ************************************************************
    """)

def check_mode(args):
    if args.mode not in ['full', 'bond']:
        raise argparse.ArgumentTypeError("Mode must be 'full' or 'bond'")
    if args.mode == 'bond':
        if args.bond is None or len(args.bond) != 2:
            raise argparse.ArgumentTypeError("When mode is 'bond', -b must be provided with two integers")
    if args.mode == 'full' and args.bond is not None:
        print("WARNING: --mode is 'full' but --bond was provided")
    return args.mode

def check_window_kind(window_kind):
    if window_kind not in ['Gaussian', 'Blackman-Harris', 'Hamming', 'Hann']:
        sys.exit("Error: Window kind (-w) must be 'Gaussian', 'Blackman-Harris', 'Hamming', or 'Hann'")

def check_bool(inputvariable, rightvariable):
    if inputvariable not in ['True', 'False']:
        sys.exit(f"Error: {rightvariable} must be 'True' or 'False'")

def check_libraries_and_file(input_name):
    try:
        from scipy import signal  # noqa: F401
    except ImportError:
        sys.exit("Error: scipy is required but not installed.")
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        sys.exit("Error: numpy is required but not installed.")

    if input_name.endswith('.xyz') or input_name.endswith('.XYZ'):
        try:
            from ase.io import read  # noqa: F401
        except ImportError:
            sys.exit("Error: ASE is required for XYZ files but is not installed.")
    else:
        try:
            from scipy.io import netcdf_file  # noqa: F401
        except ImportError:
            sys.exit("Error: scipy.io.netcdf_file is required for NETCDF files but is not installed.")

def calc_derivative(array_1D, delta_t):
    dy = np.gradient(array_1D)
    return dy / delta_t

def choose_window(nsteps, window_kind):
    if window_kind == "Gaussian":
        sigma = 2 * math.sqrt(2 * math.log(2))
        std = nsteps / 8 #4000.0
        window_function = signal.windows.gaussian(nsteps, std / sigma, sym=False)
    elif window_kind == "Blackman-Harris":
        window_function = signal.windows.blackmanharris(nsteps, sym=False)
    elif window_kind == "Hamming":
        window_function = signal.windows.hamming(nsteps, sym=False)
    elif window_kind == "Hann":
        window_function = signal.windows.hann(nsteps, sym=False)
    return window_function

def zero_padding(sample_data, pad_factor=1):
    """
    FFT length with zero padding.

    pad_factor = 1 -> next power of two
    pad_factor = 2 -> double that
    pad_factor = 4 -> 4x that
    """
    n = len(sample_data)
    base = int(2 ** math.ceil(math.log2(n)))
    return pad_factor * base

def calc_FFT(array_1D, window, pad_factor=1):
    """
    Calculates FFT intensity of the ACF with optional zero padding.
    """
    WE = np.sum(window) / len(array_1D)
    wf = window / WE
    sig = array_1D * wf
    N = zero_padding(sig, pad_factor=pad_factor)
    yfft = np.fft.fft(sig, n=N, axis=0) / len(sig)
    return np.square(np.abs(yfft))

def calc_ACF(array_1D):
    yunbiased = array_1D - np.mean(array_1D, axis=0)
    ynorm = np.sum(np.power(yunbiased, 2), axis=0)

    if ynorm == 0:
        return np.zeros_like(array_1D, dtype=float)

    autocor = signal.fftconvolve(
        yunbiased,
        yunbiased[::-1],
        mode='full'
    )[len(array_1D) - 1:] / ynorm

    return autocor

# -------------------------------------------
# Argument parser
parser = argparse.ArgumentParser(
    description=script_description,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument('-i', '--input', required=True, help='Input file name')
parser.add_argument('-o', '--output', required=True, help='Output file name')
parser.add_argument('-m', '--mode', default='full', help="Mode: 'full' (default) or 'bond'")
parser.add_argument('-dt', '--delta_t', type=float, required=True, help='Delta time in femtoseconds')
parser.add_argument('-b', '--bond', nargs=2, type=int, help='Bond indices (two integers)')
parser.add_argument('-w', '--window_kind', default='Hann',
                    help="Window: 'Gaussian' (default), 'Blackman-Harris', 'Hamming', or 'Hann'")
parser.add_argument('-f', '--force_numerical', default='False',
                    help="Force numerical calculation of velocities (default: False)")
parser.add_argument('-n', '--use_normalized_vectors', default='False',
                    help="Use norms of coordinates/velocities instead of xyz components (default: False)")
parser.add_argument('-p', '--pad_factor', type=int, default=1,
                    help='Zero-padding factor for FFT (default: 1)')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

input_name = args.input
output_name = args.output
mode = args.mode
delta_t = args.delta_t * 1e-15  # fs -> s
bond_indices = args.bond if mode == 'bond' else None
window_kind = args.window_kind

check_libraries_and_file(input_name)
check_mode(args)
check_window_kind(window_kind)

check_bool(args.force_numerical, "--force_numerical (-f)")
force_numerical = args.force_numerical == "True"

check_bool(args.use_normalized_vectors, "--use_normalized_vectors (-n)")
use_normalized_vectors = args.use_normalized_vectors == "True"

pad_factor = args.pad_factor
if pad_factor < 1:
    sys.exit("Error: --pad_factor (-p) must be a positive integer")

# Conditional imports
if input_name.endswith('.xyz') or input_name.endswith('.XYZ'):
    from ase.io import read

from scipy import signal
from scipy.io import netcdf_file
import numpy as np

if __name__ == "__main__":
    print_welcome_message()

contains_velocities = False

# Read data
if input_name.endswith('.xyz') or input_name.endswith('.XYZ'):
    print("\nCoordinates from the xyz file will be read using ASE\n")
    print("\nReading file...\n")
    trajectory = read(input_name, index=':')
    nsteps = len(trajectory)
    natoms = len(trajectory[0])

    coordinates = np.empty((nsteps, natoms, 3))
    for i, frame in enumerate(trajectory):
        coordinates[i] = frame.get_positions()

    if mode == "full":
        print("\nThe VDOS will be obtained considering all atoms\n")
        print("\nVelocities will be calculated numerically\n")
        normal_vectors = np.linalg.norm(coordinates, axis=-1)
    else:
        print("\nThe VDOS associated with the stretching of two atoms will be obtained\n")
        print("\nDerivatives will be calculated numerically\n")
        distances = np.linalg.norm(
            coordinates[:, bond_indices[0], :] - coordinates[:, bond_indices[1], :],
            axis=1
        )

else:
    print("\nCoordinates/velocities from the netcdf file will be read using scipy\n")
    print("\nReading file...\n")
    trajectory = netcdf_file(input_name, 'r')

    if mode == "full":
        print("\nThe VDOS will be obtained considering all atoms\n")
        contains_velocities = "velocities" in trajectory.variables

        if contains_velocities and not force_numerical:
            print("\nVelocities will be read from the trajectory file\n")
            velocities = np.array(trajectory.variables['velocities'].data)
            nsteps = len(velocities)
            natoms = len(velocities[0])
            normal_vectors = np.linalg.norm(velocities, axis=-1)
        else:
            if contains_velocities and force_numerical:
                print("\nFound velocities but numerical calculation is forced\n")
            print("\nVelocities will be calculated numerically\n")
            coordinates = np.array(trajectory.variables['coordinates'].data)
            nsteps = len(coordinates)
            natoms = len(coordinates[0])
            normal_vectors = np.linalg.norm(coordinates, axis=-1)

        print("\nThe program will deal with all atoms one by one\n")

    else:
        print("\nThe VDOS associated with the stretching of two atoms will be obtained\n")
        print("\nDerivatives will be calculated numerically\n")
        coordinates = np.array(trajectory.variables['coordinates'].data)
        nsteps = len(coordinates)
        distances = np.linalg.norm(
            coordinates[:, bond_indices[0], :] - coordinates[:, bond_indices[1], :],
            axis=1
        )

window = choose_window(nsteps, window_kind)

if mode == "full":
    if use_normalized_vectors:
        for i in range(natoms):
            if contains_velocities and not force_numerical:
                atom_velocities = normal_vectors[:, i]
            else:
                atom_velocities = calc_derivative(normal_vectors[:, i], delta_t)

            ACF = calc_ACF(atom_velocities)
            yfft_i = calc_FFT(ACF, window, pad_factor=pad_factor)

            if i == 0:
                yfft = yfft_i
            else:
                yfft += yfft_i

    else:
        for i in range(natoms):
            for j in range(3):
                if contains_velocities and not force_numerical:
                    atom_velocities = velocities[:, i, j]
                else:
                    atom_velocities = calc_derivative(coordinates[:, i, j], delta_t)

                ACF = calc_ACF(atom_velocities)
                yfft_i = calc_FFT(ACF, window, pad_factor=pad_factor)

                if i == 0 and j == 0:
                    yfft = yfft_i
                else:
                    yfft += yfft_i

else:
    distances_velocities = calc_derivative(distances, delta_t)
    ACF = calc_ACF(distances_velocities)
    yfft = calc_FFT(ACF, window, pad_factor=pad_factor)

half = len(yfft) // 2
wavenumber = np.fft.fftfreq(len(yfft), delta_t * c)[:half]
intensity = yfft[:half]

print(f"\nVDOS saved to {output_name}\n")
print("\nUnits are cm-1 for wavenumber and arbitrary units for intensity\n")
header = "# Wavenumber(cm-1)   Intensity(a.u.)"
np.savetxt(output_name, np.column_stack((wavenumber, intensity)), header=header, comments='')
