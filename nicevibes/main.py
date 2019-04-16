#!/usr/bin/env python3

import numpy as np
import re

from nicevibes.constants import C, KB, NA, R, PLANCK, J2AU, J2CAL, AMU2KG


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
    return KB * np.log(multiplicity)


def sackur_tetrode(molecular_mass, temperature):
    # Just using 1e5 instead of a "true" atmosphere of 1.01325e5 seems to
    # agree better with the results Gaussian and ORCA produce.
    # pressure = 1.01325e5
    pressure = 1e5
    q_trans = ((2*np.pi*molecular_mass*AMU2KG*KB*temperature/PLANCK**2)**(3/2)
                * KB * temperature / pressure
    )
    S_trans = KB*(np.log(q_trans) + 1 + 3/2)
    return S_trans


def sackur_tetrode_simplified(molecular_mass, temperature):
    """Translational entropy for 1 mol of a monoatomic ideal gas."""
    return (  3/2 * R * np.log(molecular_mass)
            + 5/2 * R * np.log(temperature)
            - 2.315)


def translational_entropy(molecular_mass, temperature, kind="sackur"):
    funcs = {
        "sackur": sackur_tetrode,
        "sackur_simple": sackur_tetrode_simplified,
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


def harmonic_vibrational_entropies(temperature, frequencies):
    # This is the formula as given in the Grimme paper in Eq. (3).
    # As given in the paper the first term misses a T in the
    # denominator.
    # hnu = PLANCK * frequencies
    # hnu_kt = hnu / (KB * temperature)
    # S_vib = KB * (hnu / (KB*(np.exp(hnu_kt) - 1)*temperature)
                  # - np.log(1 - np.exp(-hnu_kt))
    # ).sum()

    # As given in the Gaussian whitepaper "Thermochemistry in Gaussian."
    vib_temps = frequencies * PLANCK / KB
    S_vibs = KB * (
                (vib_temps / temperature) / (np.exp(vib_temps/temperature) - 1)
                 - np.log(1- np.exp(-vib_temps/temperature))
    )
    return S_vibs


def quasi_harmonic_vibrational_entropies(temperature, frequencies, B_av=1e-44):
    inertia_moments = PLANCK / (8 * np.pi**2 * frequencies)
    eff_inertia_moments = (inertia_moments * B_av) / (inertia_moments + B_av)
    S_vibs = KB * (
        1/2 + np.log((8*np.pi**3*eff_inertia_moments*KB*temperature/PLANCK**2)**(1/2))
    )
    return S_vibs


def vibrational_entropies(temperature, frequencies, cutoff, alpha):
    """cutoff in cm^-1"""
    wavenumbers = (frequencies / C) / 100
    # print("wavenumbers")
    # print(wavenumbers)
    weights = 1 / (1 + (cutoff/wavenumbers)**alpha)
    # print("weights")
    # print(weights)
    S_harmonic = harmonic_vibrational_entropies(temperature, frequencies)
    S_quasi_harmonic = quasi_harmonic_vibrational_entropies(temperature, frequencies)
    S_vibs = weights*S_harmonic + (1 - weights)*S_quasi_harmonic
    return S_vibs


def vibrational_entropy(temperature, frequencies, cutoff=100, alpha=4):
    return vibrational_entropies(temperature, frequencies, cutoff, alpha).sum()


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
    # rts = np.array((0.30314, 0.29984, 0.18372))
    S_rot = rotational_entropy(temperature, qc.rot_temperatures, qc.symmetry_number,
                               qc.is_linear, qc.is_atom)
    print("S_rot", S_rot, S2kcalmol(S_rot, temperature), S2calmol(S_rot))
    S_trans = translational_entropy(qc.M, temperature)
    print("S_trans", S_trans, S2kcalmol(S_trans, temperature), S2calmol(S_trans))

    S_hvibs = harmonic_vibrational_entropies(temperature, qc.vib_frequencies)
    S_hvib = S_hvibs.sum()
    print("S_hvib", S_hvib, S2kcalmol(S_hvib, temperature), S2calmol(S_hvib))

    S_qvibs = quasi_harmonic_vibrational_entropies(temperature, qc.vib_frequencies)
    S_qvib = S_qvibs.sum()
    print("S_qvib", S_qvib, S2kcalmol(S_qvib, temperature), S2calmol(S_qvib))

    S_vib = vibrational_entropy(temperature, qc.vib_frequencies)
    print("S_vib", S_vib, S2kcalmol(S_vib, temperature), S2calmol(S_vib))
