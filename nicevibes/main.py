#!/usr/bin/env python3

# [1] http://gaussian.com/thermo/
# [2] https://doi.org/10.1002/chem.201200497
# [3] https://doi.org/10.1021/acs.organomet.8b00456

from collections import namedtuple

import numpy as np

from nicevibes.constants import C, KB, NA, R, PLANCK, J2AU, J2CAL, AMU2KG


ThermoResults = namedtuple(
                    "ThermoResults",
                    ("ZPE U_trans U_rot U_vib therm_corr "
                     "S_trans S_rot S_vib S_el S_tot "
                     "T"),
)


##############
# ELECTRONIC #
##############


def electronic_entropy(multiplicity):
    """Electronic entropy.

    Only the ground state is considered. See [1] for reference.

    Parameters
    ----------
    multiplicity : int
        Multiplicity of the molecule.

    Returns
    -------
    S_el : float
        Electronic entropy in J/mol.
    """
    S_el = R * np.log(multiplicity)
    return S_el


#######################
# TRANSLATIONAL TERMS #
#######################


def translation_energy(temperature):
    """Kinectic energy of an ideal gas.

    See [1] for reference.

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.

    Returns
    -------
    U_trans : float
        Kinetic energy in J/mol.
    """
    U_trans = 3/2 * R * temperature
    return U_trans


def sackur_tetrode(molecular_mass, temperature):
    """Translational entropy of an ideal gas.

    See [1] for reference.

    Parameters
    ----------
    molecular_mass : float
        Molecular mass in atomic mass units (amu).
    temperature : float
        Absolute temperature in Kelvin.

    Returns
    -------
    S_trans : float
        Translational entropy in J/(mol*K).
    """
    # Just using 1e5 instead of a "true" atmosphere of 1.01325e5 seems to
    # agree better with the results Gaussian and ORCA produce.
    # pressure = 1.01325e5
    pressure = 1e5
    q_trans = ((2*np.pi*molecular_mass*AMU2KG*KB*temperature/PLANCK**2)**(3/2)
                * KB * temperature / pressure
    )
    S_trans = R*(np.log(q_trans) + 1 + 3/2)
    return S_trans


def sackur_tetrode_simplified(molecular_mass, temperature):
    """Translational entropy of a monoatomic ideal gas.

    See [3] for reference.

    Parameters
    ----------
    molecular_mass : float
        Molecular mass in atomic mass units (amu).
    temperature : float
        Absolute temperature in Kelvin.

    Returns
    -------
    S_trans : float
        Translational entropy in J/(mol*K).
    """
    S_trans = (  3/2 * R * np.log(molecular_mass)
               + 5/2 * R * np.log(temperature)
               - 2.315
    )
    return S_trans


def translational_entropy(molecular_mass, temperature, kind="sackur"):
    """Wrapper for translational entropy calculation.

    Parameters
    ----------
    molecular_mass : float
        Molecular mass in atomic mass units (amu).
    temperature : float
        Absolute temperature in Kelvin.
    kind : str, ("sackur", "sackur_simple")
        Type of calculation method.

    Returns
    -------
    S_trans : float
        Translational entropy in J/(mol*K).
    """
    funcs = {
        "sackur": sackur_tetrode,
        "sackur_simple": sackur_tetrode_simplified,
    }
    return funcs[kind](molecular_mass, temperature)


####################
# ROTATIONAL TERMS #
####################


def rotational_energy(temperature, is_linear, is_atom):
    """Rotational energy.

    See [1] for reference.

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    is_linear : bool
        Wether the molecule is linear.
    is_atom : bool
        Wether the molcule is an atom.

    Returns
    -------
    U_rot : float
        Rotational energy in J/mol.
    """
    if is_atom:
        rot_energy = 0
    elif is_linear:
        rot_energy = R * temperature

    U_rot = 3/2 * R * temperature
    return U_rot


def rotational_entropy(temperature, rot_temperatures, symmetry_number,
                       is_linear, is_atom):
    """Rotational entropy.

    See [1] for reference.

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    rot_temperatures : np.array of size 3
        Rotational temperatures in Kelvin.
    symmetry_number : int
        Symmetry number.
    is_linear : bool
        Wether the molecule is linear.
    is_atom : bool
        Wether the molcule is an atom.

    Returns
    -------
    S_rot : float
        Rotational entropy in J/(mol*K).
    """
    if is_atom:
        S_rot = 0
    if is_linear:
        q_rot = temperature / (rot_temperatures * symmetry_number)
        S_rot = R * (np.log(q_rot) + 1)
    # Polyamtomic, non-linear case
    q_rot = (np.pi**(1/2) / symmetry_number
             * (temperature**(3/2) / np.product(rot_temperatures)**(1/2))
    )
    S_rot = R * (np.log(q_rot) + 3/2)
    return S_rot


#####################
# VIBRATIONAL TERMS #
#####################


def zero_point_energy(frequencies):
    """Vibrational zero point energies.

    See [1] for reference.

    Parameters
    ----------
    frequencies : array-like
        Vibrational frequencies in 1/s.

    Returns
    -------
    ZPE : float
        Vibrational zero point energy for the given frequencies in J/mol.
    """

    ZPE = NA * (PLANCK * frequencies / 2).sum()
    return ZPE


def vibrational_energy(temperature, frequencies):
    """Vibrational energy.

    See [1] for reference.

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    frequencies : array-like
        Vibrational frequencies in 1/s.

    Returns
    -------
    U_vib : float
        Vibrational energy in J/mol.
    """
    vib_temperatures = PLANCK * frequencies / KB
    U_vib =  R * np.sum(vib_temperatures
                        * (1/2 + 1 / (np.exp(vib_temperatures/temperature) - 1))
    )
    return U_vib


def harmonic_vibrational_entropies(temperature, frequencies):
    """Vibrational entropy of a harmonic oscillator.

    See [1] and [2] for reference. Eq. (3) in the Grimme paper
    is lacking a T in the denominator of the first term. It is given as
    h*w/(k(e^(hw/kt) -1)) but it must be h*w(kT(e^(hw/kT)-1)) instead.
    Here the calculation is done as presented in [1].

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    frequencies : array-like
        Vibrational frequencies in 1/s.

    Returns
    -------
    S_vib : array-like
        Array containing vibrational entropies in J/(mol*K).
    """

    # Correct formula from the Grimme paper [3].
    # hnu = PLANCK * frequencies
    # hnu_kt = hnu / (KB * temperature)
    # S_vib = KB * (hnu / (KB*(np.exp(hnu_kt) - 1)*temperature)
                  # - np.log(1 - np.exp(-hnu_kt))
    # ).sum()

    # As given in [1].
    vib_temps = frequencies * PLANCK / KB
    S_vibs = R * (
                (vib_temps / temperature) / (np.exp(vib_temps/temperature) - 1)
                 - np.log(1 - np.exp(-vib_temps/temperature))
    )
    return S_vibs


def free_rotor_entropies(temperature, frequencies, B_av=1e-44):
    """Entropy of a free rotor.

    See [2] for reference.

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    frequencies : array-like
        Vibrational frequencies in 1/s.
    B_av : float
        Limiting value for effective moment of inertia in kg*m².

    Returns
    -------
    S_free_rots : array-like
        Array containing free-rotor entropies in J/(mol*K).
    """
    inertia_moments = PLANCK / (8 * np.pi**2 * frequencies)
    eff_inertia_moments = (inertia_moments * B_av) / (inertia_moments + B_av)
    S_free_rots = R * (
        1/2 + np.log((8*np.pi**3*eff_inertia_moments*KB*temperature/PLANCK**2)**(1/2))
    )
    return S_free_rots


def vibrational_entropies(temperature, frequencies, cutoff=100, alpha=4):
    """Weighted vibrational entropy.

    As given in Eq. (7) of [2].

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    frequencies : array-like
        Vibrational frequencies in 1/s.
    cutoff : float
        Wavenumber cutoff in cm⁻¹. Vibrations below this threshold will mostly
        be described by free-rotor entropies.
    alpha : float
        Exponent alpha in the damping function (Eq. (8) in [2]).

    Returns
    -------
    S_vibs : array-like
        Array containing vibrational entropies in J/(mol*K).
    """
    wavenumbers = (frequencies / C) / 100
    weights = 1 / (1 + (cutoff/wavenumbers)**alpha)
    S_harmonic = harmonic_vibrational_entropies(temperature, frequencies)
    S_quasi_harmonic = free_rotor_entropies(temperature, frequencies)
    S_vibs = weights*S_harmonic + (1 - weights)*S_quasi_harmonic
    return S_vibs


def vibrational_entropy(temperature, frequencies, cutoff=100, alpha=4):
    """Vibrational entropy.

    Wrapper function. As given in Eq. (7) of [2].

    Parameters
    ----------
    temperature : float
        Absolute temperature in Kelvin.
    frequencies : array-like
        Vibrational frequencies in 1/s.
    cutoff : float
        Wavenumber cutoff in cm⁻¹. Vibrations below this threshold will mostly
        be described by free-rotor entropies.
    alpha : float
        Exponent alpha in the damping function (Eq. (8) in [2]).

    Returns
    -------
    S_vib : float
        Vibrational entropy in J/(mol*K).
    """
    return vibrational_entropies(temperature, frequencies, cutoff, alpha).sum()


def thermochemistry(qc, temperature, kind="qrrho"):
    assert kind in "qrrho rrho".split()

    zpe = zero_point_energy(qc.vib_frequencies)
    U_trans = translation_energy(temperature)
    U_rot = rotational_energy(temperature, qc.is_linear, qc.is_atom)
    U_vib = vibrational_energy(temperature, qc.vib_frequencies)

    # ZPE isn't included here as it is already included in the U_vib term
    therm_corr = U_rot + U_vib + U_trans

    S_el = electronic_entropy(qc.mult)
    S_rot = rotational_entropy(temperature, qc.rot_temperatures, qc.symmetry_number,
                               qc.is_linear, qc.is_atom)
    S_trans = translational_entropy(qc.M, temperature)

    if kind == "rrho":
        S_hvibs = harmonic_vibrational_entropies(temperature, qc.vib_frequencies)
        S_vib = S_hvibs.sum()
    elif kind == "qrrho":
        S_vib = vibrational_entropy(temperature, qc.vib_frequencies)
    else:
        raise Exception("You should never get here!")
    S_tot = S_el + S_trans + S_rot + S_vib

    thermo = ThermoResults(
                zpe,
                U_trans,
                U_rot,
                U_vib,
                therm_corr,
                S_trans,
                S_rot,
                S_vib,
                S_el,
                S_tot,
                temperature,

    )
    return thermo


def print_thermo_results(thermo_results):
    J2KJ = lambda J: f"{J/1000:.2f} kJ/mol"
    S2KJ = lambda S, T: f"{S*T/1000:.2f} kJ/mol"

    tr = thermo_results
    T = tr.T
    print(f"Thermochemistry @ {T:.2f} K")

    print("ZPE", J2KJ(tr.ZPE))
    print("U_trans", J2KJ(tr.U_trans))
    print("U_rot", J2KJ(tr.U_rot))
    print("U_vib", J2KJ(tr.U_vib))

    print("thermal_corr", J2KJ(tr.therm_corr))

    print("S_el", S2KJ(tr.S_el, T))
    print("S_trans", S2KJ(tr.S_trans, T))
    print("S_rot", S2KJ(tr.S_rot, T))
    print("S_vib", S2KJ(tr.S_vib, T))
    print("S_tot", S2KJ(tr.S_tot, T))
