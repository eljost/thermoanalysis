#!/usr/bin/env python3

import numpy as np
import re

from pysisyphus.helpers import geom_from_xyz_file
from pysisyphus.Geometry import Geometry


C = 299_792_458 # m/s
PLANCK = 6.626_0700_40e-34 # J/s
KB = 1.380_648_52e-23 # J/K
NA = 6.022_140_857e23 # 1/mol


def get_symmetry_number(point_group):
    symm_dict = {
        "c1": 1,
        "ci": 1,
        "cs": 1,
        "cinf": 1,
        "dinfh": 2,
        "t": 12,
        "td": 12,
        "oh": 24,
        "ih": 60,
    }
    pg = point_group.lower()
    try:
        return symm_dict[pg]
    except KeyError:
        pass
    regex = "[cds](\d+)"
    mobj = re.match(regex, pg)
    try:
        sym_num = int(mobj[1])
    except TypeError:
        raise Exception(f"Specified point group '{pg}' is invalid!")

    if pg.startswith("d"):
        sym_num *= 2
    elif pg.startswith("s"):
        sym_num /= 2
    assert sym_num == int(sym_num), "Check your point group! Did you " \
        "specify some 'Sn' group with n ∈ (1, 3, 5, ...)? Please use " \
        "the corresponding 'Cnm' groups instead!"
    return sym_num


def rotational_temperature(Rm):
    """Rotational temperature in K.

    Arguments: rotational constants in 1/m."""
    return PLANCK*C/KB*np.array(Rm, dtype=float)


def get_V_free(solvent="chloroform", C_free=8):
    solvents = {
        # (Concentration in mol/l, molecular volume in Å^3)
        "chloroform": (12.5, 97),
        "dioxane": (11.72, 115),
    }

    try:
        concentration, V_molec = solvents[solvent]
    except KeyError:
        valid_solvents = ", ".join(solvents.keys())
        raise Exception(f"Invalid solvent! Valid solvents are: {valid_solvents}.")
    return C_free * ((1e27/(concentration*NA))**(1/3) - V_molec**(1/3))**3


def run():
    Rcm= (10.067986, 0.877991, 0.840280)
    Rm= np.array(Rcm) * 100
    t = rotational_temperature(Rm)


    sn = get_symmetry_number
    c2v = sn("C2v")
    V = get_V_free()
    # import pdb; pdb.set_trace()
    v = get_V_free("chloroform")



if __name__ == "__main__":
    run()
