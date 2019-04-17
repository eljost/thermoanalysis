#!/usr/bin/env python3


import argparse
import sys

from thermoanalysis.thermo import thermochemistry, print_thermo_results
from thermoanalysis.QCData import QCData


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("log",
        help="Path to the log file containing the frequency calculation."
    )
    parser.add_argument("--temp", default=298.15, type=float,
        help="Temperature for the thermochemistry analysis in K."
    )
    parser.add_argument("--pg", default="c1",
        help="Point group of the molecule. Important for the correct "
             "determination of the symmetry number in the calculation "
             "of rotational terms.")
    parser.add_argument("--scale", default=1.0, type=float,
        help="Scaling factor for vibrational frequencies (default=1.0)."
    )
    parser.add_argument("--vibs", choices="rrho qrrho".split(), default="qrrho",
        help="Wether to use Grimmes QRRHO approach ('qrrho') or an purely "
             "harmonic approach for the calculation of vibrational entropy."
    )

    return parser.parse_args(args)


def run():
    args = parse_args(sys.argv[1:])

    log = args.log
    point_group = args.pg
    T = args.temp
    scale = args.scale
    vib_kind = args.vibs

    qc = QCData(log, point_group=point_group, scale_factor=scale)
    thermo = thermochemistry(qc, T, kind=vib_kind)
    print_thermo_results(thermo)


if __name__ == "__main__":
    run()
