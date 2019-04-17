#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import h5py
import pandas as pd
from tabulate import tabulate

from thermoanalysis.thermo import thermochemistry, print_thermo_results
from thermoanalysis.QCData import QCData


def print_thermos(thermos):
    fields = ("T U_trans U_rot U_vib U_tot "
              "TS_el TS_trans TS_rot TS_vib TS_tot".split()
    )
    filtered = list()
    for thermo in thermos:
        _ = [getattr(thermo, f) for f in fields]
        filtered.append(_)

    headers = fields
    thermos_arr = np.array(filtered)
    thermos_arr[:,1:] /= 1000
    table = tabulate(thermos_arr, headers=headers, floatfmt=".2f")
    print(f"ZPE = {thermos[0].ZPE / 1000:.2f} kJ/mol (independent of T)")
    print("U_vib and U_tot already include the ZPE.")
    print("All quantities given in kJ/mol except T (given in K).")
    print(table)


def dump_thermos(log_fn, thermos):
    log_path = Path(log_fn)
    df = pd.DataFrame(thermos)
    h5_fn = f"{log_path.stem}_thermo.h5"
    with h5py.File(h5_fn, "w") as handle:
        for thermo in thermos:
            handle.create_dataset(name=f"{thermo.T:.4f}", dtype=float, data=thermo)
    print(f"Dumped thermo-data to '{h5_fn}'.")


def parse_args(args):
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("log",
        help="Path to the log file containing the frequency calculation."
    )
    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument("--temp", default=298.15, type=float,
        help="Temperature for the thermochemistry analysis in K."
    )
    temp_group.add_argument("--temps", nargs=3, type=float, default=None,
        metavar=("T_start", "T_end", "steps"),
        help="Determine thermochemistry data for a range of temperatures.",
    )
    parser.add_argument("--pg", default="c1",
        help="Point group of the molecule. Important for the correct "
             "determination of the symmetry number in the calculation "
             "of rotational terms.")
    parser.add_argument("--scale", default=1.0, type=float,
        help="Scaling factor for vibrational frequencies."
    )
    parser.add_argument("--vibs", choices="rrho qrrho".split(), default="qrrho",
        help="Wether to use Grimmes QRRHO approach ('qrrho') or an purely "
             "harmonic approach ('rrho') for the calculation of vibrational entropy."
    )

    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])

    log = args.log
    point_group = args.pg
    scale = args.scale
    vib_kind = args.vibs

    print(f"Using {vib_kind.upper()}-approach for vibrational entropies.")

    if args.temps:
        temps = np.linspace(*args.temps)
        qc = QCData(log, point_group=point_group, scale_factor=scale)
        thermos = [thermochemistry(qc, T, kind=vib_kind) for T in temps]
    else:
        T = args.temp
        qc = QCData(log, point_group=point_group, scale_factor=scale)
        thermo = thermochemistry(qc, T, kind=vib_kind)
        thermos = [thermo, ]

    print_thermos(thermos)
    dump_thermos(log, thermos)


if __name__ == "__main__":
    run()
