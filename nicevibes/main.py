#!/usr/bin/env python3

import math
import numpy as np
import re

from pysisyphus.helpers import geom_from_xyz_file
from pysisyphus.Geometry import Geometry


from nicevibes.constants import C, KB, NA, R, PLANCK, J2AU, J2CAL


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
    return 3/2 * KB * temperature


def electronic_entropy(multiplicity):
    """Considering only the ground state."""
    return KB * math.log(multiplicity)


def sackur_tetrode(molecular_mass, temperature):
    """Translational entropy for a monoatomic ideal gas."""
    print("Wrong unit in sackur tetrode! Convert to single particle energy!")
    return (  3/2 * R * np.log(molecular_mass)
            + 5/2 * R * np.log(temperature)
            - 2.315)


def translational_entropy(molecular_mass, temperature, kind="sackur"):
    funcs = {
        "sackur": sackur_tetrode,
    }
    return funcs[kind](molecular_mass, temperature)


def zero_point_energy(frequencies):
    return (PLANCK * frequencies  / 2).sum()


def rotational_energy(temperature, is_linear, is_atom):
    if is_atom:
        rot_energy = 0
    elif is_linear:
        rot_energy = KB * temperature

    rot_energy = 3/2 * KB * temperature
    return rot_energy


def vibrational_energy(temperature, frequencies):
    vib_temperatures = PLANCK * frequencies / KB
    return KB * np.sum(vib_temperatures
                       * (1/2 + 1 / (np.exp(vib_temperatures/temperature) - 1))
    )


def rotational_entropy(temperature, rot_temperatures, symmetry_number,
                       is_linear, is_atom):
    if is_atom:
        S_rot = 0
    if is_linear:
        q_rot = temperature / (rot_temperatures * symmetry_number)
        S_rot = KB * (np.log(q_rot) + 1)
    # Polyamtomic, non-linear case
    q_rot = (np.pi**(1/2) / symmetry_number
             * (temperature**(3/2) / np.product(rot_temperatures)**(1/2))
    )
    S_rot = KB * (np.log(q_rot) + 3/2)
    return S_rot


def thermochemistry(qc, temperature):
    print(f"Thermochemistry with {temperature:.2f} K")
    J2au = lambda J: f"{J*J2AU:.8f} au"
    J2kcalmol = lambda J: f"{J*J2CAL*NA/1000:.8f} kcal/mol"
    S2kcalmol = lambda S, T: f"{S*T*J2CAL*NA/1000:.8f} kcal/mol"
    S2calmol = lambda S: f"{S*J2CAL*NA:.8f} cal/(mol*K)"

    zpe = zero_point_energy(qc.vib_frequencies)
    print("ZPE", zpe, "J", f"{zpe*J2AU:.6f} au")
    U_trans = translation_energy(temperature)
    print("U_trans", U_trans, J2au(U_trans))
    U_rot = rotational_energy(temperature, qc.is_linear, qc.is_atom)
    print("U_rot", U_rot, J2au(U_rot))
    U_vib = vibrational_energy(temperature, qc.vib_frequencies)
    print("U_vib", U_vib, J2au(U_vib))

    # ZPE isn't included here as it is already included in the U_vib term
    therm_corr = U_rot + U_vib + U_trans
    print("thermal_corr", therm_corr, J2au(therm_corr))

    S_el = electronic_entropy(qc.mult)
    print("S_el", S_el)
    S_trans = translational_entropy(qc.M, temperature)
    print("S_trans", S_trans)
    rts = np.array((0.30314, 0.29984, 0.18372))
    # S_rot = rotational_entropy(temperature, qc.rot_temperatures, qc.symmetry_number,
                               # qc.is_linear, qc.is_atom)
    S_rot = rotational_entropy(temperature, rts, qc.symmetry_number,
                               qc.is_linear, qc.is_atom)
    print("S_rot", S_rot, S2kcalmol(S_rot, temperature), S2calmol(S_rot))
    print(f"S*T: {S_rot*temperature*J2AU:.6f}")


def run():
    Rcm= (10.067986, 0.877991, 0.840280)
    Rm= np.array(Rcm) * 100
    t = rotational_temperature(Rm)


    V = get_V_free()
    v = get_V_free("chloroform")



if __name__ == "__main__":
    run()
