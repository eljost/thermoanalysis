#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import h5py
import pandas as pd
from tabulate import tabulate

from thermoanalysis.config import p_DEFAULT, ROTOR_CUT_DEFAULT, T_DEFAULT
from thermoanalysis.thermo import thermochemistry
from thermoanalysis.QCData import QCData


def print_thermos(thermos):
    fields = "T p U_el U_therm U_tot H TS_tot G dG".split()
    filtered = list()
    for thermo in thermos:
        _ = [getattr(thermo, f) for f in fields]
        filtered.append(_)

    headers = fields
    thermos_arr = np.array(filtered)
    float_fmts = [".6f"] * len(fields)
    float_fmts[0] = ".2f"
    float_fmts[1] = ".4e"
    table = tabulate(thermos_arr, headers=headers, floatfmt=float_fmts)
    print("Wavenumbers in cm⁻¹")
    for i, nu in enumerate(thermo.wavenumbers):
        print(f"{i: >05d}: {nu:>12.4f}")
    print(f"Total mass = {thermos[0].M:.2f} amu")
    print(f"ZPE = {thermos[0].ZPE:.6f} au / particle (independent of T)")
    print("U_vib and U_tot already include the ZPE.")
    print("All quantities given in au / particle except T (in K) and p (in Pa).")
    print(table)


def dump_thermos(log_fn, thermos):
    log_path = Path(log_fn)
    # columns =
    df = pd.DataFrame(
        thermos,
    ).drop(columns=["wavenumbers", "point_group"])
    h5_fn = f"{log_path.stem}_thermo.h5"
    df.to_hdf(h5_fn, key=f"thermo")
    print(f"Dumped thermo-data to '{h5_fn}'.")


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "inp_fn",
        help="Path to log file containing the frequency calculation or "
        "an HDF5 hessian from pysisyphus.",
    )
    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument(
        "--temp",
        default=T_DEFAULT,
        type=float,
        help="Temperature for the thermochemistry analysis in K.",
    )
    temp_group.add_argument(
        "--temps",
        nargs=3,
        type=float,
        default=None,
        metavar=("T_start", "T_end", "steps"),
        help="Determine thermochemistry data for a range of temperatures.",
    )
    parser.add_argument(
        "--pg",
        default="c1",
        help="Point group of the molecule. Important for the correct "
        "determination of the symmetry number in the calculation "
        "of rotational terms.",
    )
    parser.add_argument(
        "--scale",
        default=1.0,
        type=float,
        help="Scaling factor for vibrational frequencies.",
    )
    parser.add_argument(
        "--vibs",
        choices="rrho qrrho".split(),
        default="qrrho",
        help="Wether to use Grimmes QRRHO approach ('qrrho') or an purely "
        "harmonic approach ('rrho') for the calculation of vibrational entropy.",
    )
    parser.add_argument(
        "--qrrho-cutoff",
        default=ROTOR_CUT_DEFAULT,
        type=float,
        help="Rotor cutoff value for Grimmes modified QRRHO approach.",
    )
    parser.add_argument(
        "--pressure", "-p", type=float, default=p_DEFAULT, help="Pressure in Pascal."
    )
    parser.add_argument(
        "--invert-imags",
        type=float,
        default=0.0,
        help="Invert imaginary frequencies bigger or equal to the threshold.",
    )

    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])

    inp_fn = args.inp_fn
    T = args.temp
    pressure = args.pressure
    point_group = args.pg
    scale = args.scale
    vib_kind = args.vibs
    rotor_cutoff = args.qrrho_cutoff
    invert_imags = args.invert_imags

    print(
        f"Using {vib_kind.upper()}-approach for vibrational entropies"
        + (f" with cutoff {rotor_cutoff:.2f} cm⁻¹" if vib_kind == "qrrho" else "")
        + "."
    )

    qc = QCData(
        inp_fn,
        point_group=point_group,
    )
    if args.temps:
        T_init, T_final, steps = args.temps
        temps = np.linspace(T_init, T_final, int(steps))
    else:
        temps = [
            T,
        ]
    thermos = [
        thermochemistry(
            qc,
            T,
            pressure=pressure,
            kind=vib_kind,
            rotor_cutoff=rotor_cutoff,
            scale_factor=scale,
            invert_imags=invert_imags,
        )
        for T in temps
    ]

    print_thermos(thermos)
    # dump_thermos(inp_fn, thermos)


if __name__ == "__main__":
    run()
