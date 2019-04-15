#!/usr/bin/env python3

import math
import numpy as np
import re

from pysisyphus.helpers import geom_from_xyz_file
from pysisyphus.Geometry import Geometry


from nicevibes.constants import C, KB, NA, R, PLANCK


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


def translation_energy(temperature):
    """Kinectic energy of an ideal gas."""
    return 3/2 * R * temperature


def electronic_entropy(multiplicity):
    """Considering only the ground state."""
    return R * math.log(multiplicity)


def sackur_tetrode(molecular_mass, temperature):
    """Translational entropy for a monoatomic ideal gas."""
    return (  3/2 * R * np.log(molecular_mass)
            + 5/2 * R * np.log(temperature)
            - 2.315)


def translational_entropy(molecular_mass, temperature, kind="sackur"):
    funcs = {
        "sackur": sackur_tetrode,
    }
    return funcs[kind](molecular_mass, temperature)


def thermochemistry(qc, temperature):
    U_trans = translation_energy(temperature)
    print("U_trans", U_trans)
    S_el = electronic_entropy(qc.mult)
    print("S_el", S_el)
    S_trans = translational_entropy(qc.M, temperature)
    print("S_trans", S_trans)


def run():
    Rcm= (10.067986, 0.877991, 0.840280)
    Rm= np.array(Rcm) * 100
    t = rotational_temperature(Rm)


    V = get_V_free()
    # import pdb; pdb.set_trace()
    v = get_V_free("chloroform")



if __name__ == "__main__":
    run()
