# [1] http://gaussian.com/thermo/
# [2] https://doi.org/10.1002/chem.201200497
# [3] https://doi.org/10.1021/acs.organomet.8b00456
# [4] https://cccbdb.nist.gov/thermo.asp
# [5] https://pubs.acs.org/doi/10.1021/acs.jctc.0c01306
#     Single-Point Hessian Calculations
#     Spicher, Grimme, 2021
# [6] https://doi.org/10.1063/5.0061187
#     Calculation of improved enthalpy and entropy of vaporization by a modified
#     partition function in quantum cluster equilibrium theory
#     Ingenmey, Grimme, 2021


from collections import namedtuple
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from thermoanalysis.constants import (
    C,
    KB,
    KBAU,
    NA,
    R,
    PLANCK,
    J2AU,
    J2CAL,
    AMU2KG,
)
from thermoanalysis.config import p_DEFAULT, ROTOR_CUT_DEFAULT
from thermoanalysis.QCData import QCData


TRANS_ENTROPY_KINDS = Literal["sackur", "sackur_simple"]
VIB_KINDS = Literal["qrrho", "rrho"]


ThermoResults = namedtuple(
    "ThermoResults",
    (
        "T kBT M p org_wavenumbers wavenumbers "
        "scale_factor zpe_scale_factor invert_imag cutoff kind "
        "atom_num linear point_group sym_num "
        "Q_el Q_trans Q_rot Q_vib Q_vib_V0 Q_tot Q_tot_V0 "
        "U_el U_trans U_rot U_vib U_therm U_tot ZPE H "
        "S_trans S_rot S_vib S_el S_tot "
        "c_el c_trans c_rot c_vib c_tot "
        "TS_trans TS_rot TS_vib TS_el TS_tot G dG"
    ),
)


def chai_head_gordon_weights(
    frequencies: NDArray[np.float64], cutoff: float, alpha: int = 4
) -> NDArray[np.float64]:
    """Chai-Head-Gordon damping function.

    Used for interpolating between harmonic oscillator and hindered rotor
    approximations. See eq. (8) in [2], or Eq. (10) in [7].

    Parameters
    ----------
    frequencies
        Vibrational frequencies in 1/s.
    cutoff
        Wavenumber cutoff in cm⁻¹. Vibrations below this threshold will mostly
        be treated as hindered rotors.
    alpha
        Exponent alpha in the damping function.

    Returns
    -------
    weights
        Weights of the damping function.
    """

    wavenumbers = (frequencies / C) / 100  # in cm⁻¹
    weights = 1 / (1 + (cutoff / wavenumbers) ** alpha)
    return weights


##############
# ELECTRONIC #
##############


def electronic_part_func(
    multiplicity: int, electronic_energies: ArrayLike, temperature: float
) -> float:
    """Electronic partition function.

    Parameters
    ----------
    multiplicity
        Multiplicity of the molecule.
    electronic_energies
        Electronic energy/energies in Hartree.
    temperature : float
        Absolute temperature in Kelvin.

    Returns
    -------
    Q_el
        Electronic partition function.
    """

    electronic_energies = np.atleast_1d(electronic_energies)
    energy_diffs = electronic_energies - electronic_energies.min()
    g = multiplicity
    q_el = g * (np.exp(-energy_diffs / (KBAU * temperature))).sum()
    return q_el


def electronic_entropy(multiplicity: int) -> float:
    """Electronic entropy.

    Currently, only one electronic state is considered. See [1] for reference.

    Parameters
    ----------
    multiplicity
        Multiplicity of the molecule.

    Returns
    -------
    S_el
        Electronic entropy in Hartree / particle.
    """
    S_el = KBAU * np.log(multiplicity)
    return S_el


#######################
# TRANSLATIONAL TERMS #
#######################


def translational_part_func(
    molecular_mass: float, temperature: float, pressure: float
) -> float:
    """Translational partition function Q_trans.

    Parameters
    ----------
    molecular_mass
        Molecular mass in atomic mass units (amu).
    temperature
        Absolute temperature in Kelvin.
    pressure
        Pressure in Pascal.

    Returns
    -------
    q_trans
        Translational partition function.
    """
    # Volume of an atom of an ideal gas
    volume = KB * temperature / pressure
    molecular_mass_kg = molecular_mass * AMU2KG
    q_trans = (
        (2 * np.pi * molecular_mass_kg * KB * temperature / PLANCK**2) ** 1.5
    ) * volume
    return q_trans


def translational_energy(temperature: float) -> float:
    """Kinectic energy of an ideal gas.

    See [1] for reference.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.

    Returns
    -------
    U_trans
        Kinetic energy in Hartree / particle.
    """
    U_trans = 3 / 2 * KBAU * temperature
    return U_trans


def sackur_tetrode(molecular_mass: float, temperature: float, pressure: float) -> float:
    """Translational entropy of an ideal gas.

    See [1] for reference.

    Parameters
    ----------
    molecular_mass
        Molecular mass in atomic mass units (amu).
    temperature
        Absolute temperature in Kelvin.
    pressure
        Pressure in Pascal.

    Returns
    -------
    S_trans
        Translational entropy in Hartree / (particle * K).
    """
    # Just using 1e5 instead of a "true" atmosphere of 1.01325e5 seems to
    # agree better with the results Gaussian and ORCA produce.
    q_trans = (
        (2 * np.pi * molecular_mass * AMU2KG * KB * temperature / PLANCK**2) ** (3 / 2)
        * KB
        * temperature
        / pressure
    )
    S_trans = KBAU * (np.log(q_trans) + 1 + 3 / 2)
    return S_trans


def sackur_tetrode_simplified(molecular_mass: float, temperature: float) -> float:
    """Translational entropy of a monoatomic ideal gas.

    See [3] for reference.

    Parameters
    ----------
    molecular_mass
        Molecular mass in atomic mass units (amu).
    temperature
        Absolute temperature in Kelvin.

    Returns
    -------
    S_trans
        Translational entropy in J/(mol*K).
    """
    S_trans = (
        3 / 2 * R * np.log(molecular_mass) + 5 / 2 * R * np.log(temperature) - 2.315
    )
    return S_trans


def translational_entropy(
    molecular_mass: float,
    temperature: float,
    pressure: float,
    kind: TRANS_ENTROPY_KINDS = "sackur",
) -> float:
    """Wrapper for translational entropy calculation.

    Parameters
    ----------
    molecular_mass
        Molecular mass in atomic mass units (amu).
    temperature
        Absolute temperature in Kelvin.
    kind
        Type of calculation method.

    Returns
    -------
    S_trans
        Translational entropy.
    """
    funcs = {
        "sackur": lambda M, T: sackur_tetrode(
            molecular_mass=M, temperature=T, pressure=pressure
        ),
        "sackur_simple": sackur_tetrode_simplified,
    }
    return funcs[kind](molecular_mass, temperature)


def translational_heat_capacity() -> float:
    """Constant volume heat capacity.

    Returns
    -------
    c_trans
        Constant volume heat capacity in J / (K mol).
    """
    c_trans = 1.5 * R
    return c_trans


####################
# ROTATIONAL TERMS #
####################


def rotational_part_func(
    temperature: float,
    rot_temperatures: NDArray[np.float64],
    symmetry_number: int,
    is_atom: bool,
    is_linear: bool,
) -> float:
    """Rotational partition function Q_rot.

    See [1] for reference.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    rot_temperatures
        Rotational temperatures in Kelvin.
    symmetry_number
        Symmetry number.
    is_atom
        Wether the molcule is an atom.
    is_linear
        Wether the molecule is linear.

    Returns
    -------
    Q_rot
        Rotational partition function.
    """

    if is_atom:
        q_rot = 1
    elif is_linear:
        # First rot_temperature will be infinite, and the last two components
        # will be identical. Only use the last one.
        q_rot = temperature / (rot_temperatures[-1] * symmetry_number)
    else:
        # Polyamtomic, non-linear case
        q_rot = (
            np.pi ** (1 / 2)
            / symmetry_number
            * (temperature ** (3 / 2) / np.prod(rot_temperatures) ** (1 / 2))
        )
    return q_rot


def rotational_energy(temperature: float, is_linear: bool, is_atom: bool) -> float:
    """Rotational energy.

    See [1] for reference.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    is_linear
        Wether the molecule is linear.
    is_atom
        Wether the molcule is an atom.

    Returns
    -------
    U_rot
        Rotational energy in Hartree / particle.
    """
    if is_atom:
        factor = 0.0
    elif is_linear:
        factor = 1.0
    else:
        factor = 1.5

    U_rot = factor * KBAU * temperature
    return U_rot


def rotational_entropy(
    temperature: float,
    rot_temperatures: NDArray[np.float64],
    symmetry_number: int,
    is_atom: bool,
    is_linear: bool,
) -> float:
    """Rotational entropy.

    See [1] for reference.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    rot_temperatures
        Rotational temperatures in Kelvin.
    symmetry_number
        Symmetry number.
    is_atom
        Wether the molcule is an atom.
    is_linear
        Wether the molecule is linear.

    Returns
    -------
    S_rot
        Rotational entropy in Hartree /(particle * K).
    """

    q_rot = rotational_part_func(
        temperature, rot_temperatures, symmetry_number, is_atom, is_linear=is_linear
    )
    if is_atom:
        plus = 0.0
    elif is_linear:
        plus = 1.0
    # Polyamtomic, non-linear case
    else:
        plus = 1.5
    S_rot = KBAU * (np.log(q_rot) + plus)
    return S_rot


def rotational_heat_capacity(is_atom: bool, is_linear: bool) -> float:
    """Rotational heat capacity.

    Parameters
    ----------
    is_atom
        Wether the molcule is an atom.
    is_linear
        Wether the molecule is linear.

    Returns
    -------
    c_rot
        Rotational contributions to the heat capacity.
    """
    if is_atom:
        c_rot = 0
    elif is_linear:
        c_rot = R
    else:
        c_rot = 1.5 * R
    return c_rot


#####################
# VIBRATIONAL TERMS #
#####################


def zero_point_energy(frequencies: NDArray[np.float64]) -> float:
    """Vibrational zero point energies.

    See [1] for reference.

    Parameters
    ----------
    frequencies
        Vibrational frequencies in 1/s.

    Returns
    -------
    ZPE
        Vibrational ZPE for the given frequencies in Hartree / particle.
    """

    ZPE = J2AU * (PLANCK * frequencies / 2).sum()
    return ZPE


def vibrational_part_funcs(
    temperature: float, frequencies: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vibrational partition functions for harmonic oscillators.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.

    Returns
    -------
    q_vibs : ArrayLike
        Vibrational partition functions at the bottom of the well.
    q_vibs_V0  : ArrayLike
        Vibrational partition functions with first vibrational energy level
        as reference.
    """
    frequencies = np.array(frequencies, dtype=float)
    quot = -PLANCK * frequencies / (KB * temperature)
    denom = 1 / (1 - np.exp(quot))
    quot_half = quot / 2
    # Bottom of the well as reference
    q_vibs = np.exp(quot_half) * denom
    # First vibrational energy level as reference
    q_vibs_V0 = denom
    return q_vibs, q_vibs_V0


def qrrho_vibrational_part_func(
    temperature: float,
    frequencies: NDArray[np.float64],
    I_mean: Optional[float],
    cutoff: float,
    alpha: int = 4,
) -> Tuple[float, float]:
    """QRRHO vibrational partition function.

    As given in Eq. (7) in [6]. Mix between hindererd rotor
    and harmonic oscillator partition function.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.
    I_mean
        Average moment of inertia of the molecule in Angstrom² * amu.
    cutoff
        Wavenumber cutoff in cm⁻¹. Vibrations below this threshold will mostly
        be treated as hindered rotors.
    alpha
        Exponent alpha in the damping function (Eq. (8) in [2], or Eq. (10) in [7])

    Returns
    -------
    q_qrrho
        QRRHO Vibrational partition function, unitless.
    q_qrrho_V0
        QRRHO Vibrational partition function, evaluated at first vibrational energy level.
    """

    wavenumbers = (frequencies / C) / 100  # in cm⁻¹
    q_vibs, q_vibs_V0 = vibrational_part_funcs(temperature, frequencies)

    if cutoff > 0.0 and I_mean:
        # Normal mode moments of inertia
        wavenumbers_m = 100 * wavenumbers  # in m⁻¹
        mu = PLANCK / (
            2 * np.pi * 4 * np.pi * wavenumbers_m
        )  # Eq. (9) in [6], in kg m³ s⁻¹
        # Convert average moment of inertia from Å² AMU to SI units (m² kg)
        I_mean_SI = I_mean * 1e-20 * AMU2KG
        mu_ = mu * I_mean_SI / (mu + I_mean_SI)
        q_hr = np.sqrt(
            2 * mu_ / (np.pi * (PLANCK / (2 * np.pi)) ** 2 / (temperature * KB))
        )
    # Without cutoff this partition function reduces to the partition function
    # of the harmonic oscillator.
    else:
        q_hr = np.ones_like(q_vibs)

    weights = chai_head_gordon_weights(frequencies, cutoff, alpha)

    def prod(q_vibs):
        return np.prod((q_vibs**weights) * (q_hr ** (1 - weights)))

    q_qrrho = prod(q_vibs)
    q_qrrho_V0 = prod(q_vibs_V0)
    return q_qrrho, q_qrrho_V0


def vibrational_part_func(
    temperature: float, frequencies: NDArray[np.float64]
) -> Tuple[float, float]:
    """Vibrational partition function.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.

    Returns
    -------
    q_qrrho
        QRRHO Vibrational partition function, unitless.
    q_qrrho_V0
        QRRHO Vibrational partition function, evaluated at first vibrational energy level,
        unitless.
    """
    return qrrho_vibrational_part_func(
        temperature, frequencies, I_mean=None, cutoff=0.0
    )


def vibrational_energy(temperature: float, frequencies: NDArray[np.float64]) -> float:
    """Vibrational energy.

    See [1] for reference.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.

    Returns
    -------
    U_vib
        Vibrational energy in Hartree / particle.
    """
    vib_temperatures = PLANCK * frequencies / KB
    U_vib = KBAU * np.sum(
        vib_temperatures * (1 / 2 + 1 / (np.exp(vib_temperatures / temperature) - 1))
    )
    return U_vib


def harmonic_vibrational_entropies(
    temperature: float, frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
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
        Array containing vibrational entropies in Hartree / (particle * K).
    """

    # Correct formula from the Grimme paper [3].
    # hnu = PLANCK * frequencies
    # hnu_kt = hnu / (KB * temperature)
    # S_vib = KB * (hnu / (KB*(np.exp(hnu_kt) - 1)*temperature)
    # - np.log(1 - np.exp(-hnu_kt))
    # ).sum()

    # As given in [1].
    vib_temps = frequencies * PLANCK / KB
    S_vibs = KBAU * (
        (vib_temps / temperature) / (np.exp(vib_temps / temperature) - 1)
        - np.log(1 - np.exp(-vib_temps / temperature))
    )
    return S_vibs


def free_rotor_entropies(
    temperature: float, frequencies: NDArray[np.float64], B_av: float = 1e-44
) -> NDArray[np.float64]:
    """Entropy of a free rotor.

    See [2] for reference.

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.
    B_av
        Limiting value for effective moment of inertia in kg*m².

    Returns
    -------
    S_free_rots
        Array containing free-rotor entropies in Hartree / (particle * K).
    """
    inertia_moments = PLANCK / (8 * np.pi**2 * frequencies)
    eff_inertia_moments = (inertia_moments * B_av) / (inertia_moments + B_av)
    S_free_rots = KBAU * (
        1 / 2
        + np.log(
            (8 * np.pi**3 * eff_inertia_moments * KB * temperature / PLANCK**2)
            ** (1 / 2)
        )
    )
    return S_free_rots


def vibrational_entropies(
    temperature: float, frequencies: NDArray[np.float64], cutoff: float, alpha: int = 4
) -> NDArray[np.float64]:
    """Weighted vibrational entropy.

    As given in Eq. (7) of [2].

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.
    cutoff
        Wavenumber cutoff in cm⁻¹. Vibrations below this threshold will mostly
        be described by free-rotor entropies.
    alpha
        Exponent alpha in the damping function (Eq. (8) in [2]).

    Returns
    -------
    S_vibs
        Array containing vibrational entropies in Hartree / (particle * K).
    """
    weights = chai_head_gordon_weights(frequencies, cutoff, alpha)
    S_harmonic = harmonic_vibrational_entropies(temperature, frequencies)
    S_quasi_harmonic = free_rotor_entropies(temperature, frequencies)
    S_vibs = weights * S_harmonic + (1 - weights) * S_quasi_harmonic
    return S_vibs


def vibrational_entropy(
    temperature: float, frequencies: NDArray[np.float64], cutoff: float, alpha: int = 4
) -> float:
    """Vibrational entropy.

    Wrapper function. As given in Eq. (7) of [2].

    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.
    cutoff
        Wavenumber cutoff in cm⁻¹. Vibrations below this threshold will mostly
        be described by free-rotor entropies.
    alpha
        Exponent alpha in the damping function (Eq. (8) in [2]).

    Returns
    -------
    S_vib
        Vibrational entropy in Hartree / (particle * K).
    """
    return vibrational_entropies(temperature, frequencies, cutoff, alpha).sum()


def vibrational_heat_capacity(
    temperature: float, frequencies: NDArray[np.float64]
) -> float:
    """
    Parameters
    ----------
    temperature
        Absolute temperature in Kelvin.
    frequencies
        Vibrational frequencies in 1/s.

    Returns
    -------
    c_vib
        Vibrational contributions to the heat capacity in J / (K mol).
    """
    quot = PLANCK * frequencies / (KB * temperature)

    c_vib = R * (np.exp(quot) * (quot / (np.exp(quot) - 1)) ** 2).sum()
    return c_vib


def thermochemistry(
    qc: QCData,
    temperature: float,
    pressure: float = p_DEFAULT,
    kind: VIB_KINDS = "qrrho",
    rotor_cutoff: float = ROTOR_CUT_DEFAULT,
    scale_factor: float = 1.0,
    zpe_scale_factor: float = 1.0,
    invert_imags: float = 0.0,
    cutoff: float = 0.0,
) -> ThermoResults:
    assert kind in "qrrho rrho".split()
    assert invert_imags <= 0.0
    assert cutoff >= 0.0

    T = temperature
    pressure = pressure

    org_wavenumbers = qc.wavenumbers.copy()
    wavenumbers = org_wavenumbers.copy()
    # If given a full set of 3N normal mode wavenumbers, exclude N wavenumbers
    # with the smallest absolute wavenumbers.
    if wavenumbers.size == 3 * qc.atom_num:
        abs_wavenumbers = np.abs(wavenumbers)
        inds = np.argsort(abs_wavenumbers)
        drop_first = 6 - int(qc.is_linear)
        # Drop first N wavenumbers with smallest absolute values
        keep_mask = np.ones_like(wavenumbers, dtype=bool)
        keep_mask[inds[:drop_first]] = False
        wavenumbers = wavenumbers[keep_mask]

    """
    Remaining imaginary frequencies with small absolute values can be inverted.
    Afterwards, all frequencies are scaled by the given factor. Finally, positive
    frequencies below a given cutoff are set to this cutoff, e.g. 50 cm⁻¹.

    [5] suggests inverting imaginary frequencies above -20 cm⁻¹.
    """
    if invert_imags < 0.0:
        to_invert = np.logical_and(invert_imags <= wavenumbers, wavenumbers <= 0.0)
        if to_invert.sum() > 0:
            print(f"Inverted {to_invert.sum()} imaginary frequencies.")
        wavenumbers[to_invert] *= -1

    wavenumbers *= scale_factor

    if cutoff > 0.0:
        to_cut = np.logical_and(wavenumbers < cutoff, wavenumbers > 0.0)
        wavenumbers[to_cut] = cutoff

    wavenumbers = wavenumbers[wavenumbers > 0.0]
    vib_frequencies = C * wavenumbers * 100

    # Partition functions
    Q_el = electronic_part_func(qc.mult, qc.scf_energy, temperature=T)
    Q_trans = translational_part_func(qc.M, temperature=T, pressure=pressure)
    Q_rot = rotational_part_func(
        temperature=T,
        rot_temperatures=qc.rot_temperatures,
        symmetry_number=qc.symmetry_number,
        is_atom=qc.is_atom,
        is_linear=qc.is_linear,
    )

    # Zero point energy
    zpe_org = zpe_scale_factor * zero_point_energy(vib_frequencies)
    zpe = zpe_scale_factor * zpe_org

    # Internal energies
    U_el = qc.scf_energy
    U_trans = translational_energy(T)
    U_rot = rotational_energy(T, qc.is_linear, qc.is_atom)
    # U_vib already includes ZPE
    U_vib = vibrational_energy(T, vib_frequencies)
    # Replace ZPE in U_vib w/ scaled ZPE
    U_vib = U_vib - zpe_org + zpe
    U_therm = U_rot + U_vib + U_trans
    U_tot = U_el + U_therm

    H = U_tot + KBAU * T

    # Entropies
    S_el = electronic_entropy(qc.mult)
    S_rot = rotational_entropy(
        T,
        qc.rot_temperatures,
        qc.symmetry_number,
        is_atom=qc.is_atom,
        is_linear=qc.is_linear,
    )
    S_trans = translational_entropy(qc.M, T, pressure=pressure)

    # Heat capacities
    c_el = 0.0  # always zero
    c_trans = translational_heat_capacity()
    c_rot = rotational_heat_capacity(is_atom=qc.is_atom, is_linear=qc.is_linear)
    c_vib = vibrational_heat_capacity(T, vib_frequencies)
    c_tot = c_el + c_trans + c_rot + c_vib

    if kind == "rrho":
        S_hvibs = harmonic_vibrational_entropies(T, vib_frequencies)
        S_vib = S_hvibs.sum()
        # Harmonic oscillator partition function
        Q_vib, Q_vib_V0 = vibrational_part_func(
            frequencies=vib_frequencies, temperature=T
        )
    elif kind == "qrrho":
        S_vib = vibrational_entropy(T, vib_frequencies, cutoff=rotor_cutoff)
        I_mean = qc.average_moment_of_inertia
        Q_vib, Q_vib_V0 = qrrho_vibrational_part_func(
            T, vib_frequencies, I_mean, cutoff=rotor_cutoff
        )
    else:
        raise Exception("You should never get here!")

    Q_tot = Q_el * Q_trans * Q_rot * Q_vib
    Q_tot_V0 = Q_el * Q_trans * Q_rot * Q_vib_V0

    S_tot = S_el + S_trans + S_rot + S_vib
    G = H - T * S_tot
    dG = G - U_el

    thermo = ThermoResults(
        T=temperature,
        kBT=KBAU * temperature,
        M=qc.M,
        p=pressure,
        point_group=qc.point_group,
        sym_num=qc.symmetry_number,
        scale_factor=scale_factor,
        zpe_scale_factor=zpe_scale_factor,
        invert_imag=invert_imags,
        cutoff=cutoff,
        kind=kind,
        org_wavenumbers=org_wavenumbers,
        wavenumbers=wavenumbers,
        atom_num=qc.atom_num,
        linear=qc.is_linear,
        Q_el=Q_el,
        Q_trans=Q_trans,
        Q_rot=Q_rot,
        Q_vib=Q_vib,
        Q_vib_V0=Q_vib_V0,
        Q_tot=Q_tot,
        Q_tot_V0=Q_tot_V0,
        U_el=U_el,
        U_trans=U_trans,
        U_rot=U_rot,
        U_vib=U_vib,
        U_therm=U_therm,
        U_tot=U_tot,
        ZPE=zpe,
        H=H,
        S_trans=S_trans,
        S_rot=S_rot,
        S_vib=S_vib,
        S_el=S_el,
        S_tot=S_tot,
        c_el=c_el,
        c_trans=c_trans,
        c_rot=c_rot,
        c_vib=c_vib,
        c_tot=c_tot,
        TS_trans=T * S_trans,
        TS_rot=T * S_rot,
        TS_vib=T * S_vib,
        TS_el=T * S_el,
        TS_tot=T * S_tot,
        G=G,
        dG=dG,
    )
    return thermo


def print_thermo_results(thermo_results: ThermoResults):
    au2CalMol = 1 / J2AU * NA * J2CAL

    def toCalMol(E):
        return f"{E*au2CalMol: >14.2f} cal/mol"

    def StoCalKMol(S):
        return f"{S*au2CalMol: >12.2f} cal/(K mol)"

    def CtoCalKMol(c):
        return f"{c*J2CAL: >12.4f} cal/(K mol)"

    def part_func(Q):
        return f"{Q: >12.8e}"

    def print_line(key, fmt_func, num):
        print(f"{key: >18s} = {fmt_func(num)}")

    def pQ(key, num):
        print_line(f"Q_{key}", part_func, num)

    def pU(key, num):
        print_line(f"U_{key}", toCalMol, num)

    def pS(key, num):
        print_line(f"S_{key}", StoCalKMol, num)

    def pC(key, num):
        print_line(f"C_{key}", CtoCalKMol, num)

    tr = thermo_results
    T = tr.T
    print(
        f"Thermochemistry @ {T:.2f} K and {tr.p:.6e} Pa, '{tr.kind.upper()}' analysis.\n"
    )

    print("Partition functions:")
    pQ("el", tr.Q_el)
    pQ("trans", tr.Q_trans)
    pQ("rot", tr.Q_rot)
    pQ("vib BOT", tr.Q_vib)
    pQ("vib V0", tr.Q_vib_V0)
    print()

    print("Zero-point energy")
    print_line("ZPE", toCalMol, tr.ZPE)
    print()

    print("Internal energy")
    pU("el", tr.U_el)
    pU("trans", tr.U_trans)
    pU("rot", tr.U_rot)
    pU("vib (incl. ZPE)", tr.U_vib)
    pU("therm", tr.U_therm)
    print("U_tot = U_el + U_trans + U_rot + U_vib")
    pU("tot", tr.U_tot)
    print()

    print("Entropy contributions:")
    pS("el", tr.S_el)
    pS("trans", tr.S_trans)
    pS("rot", tr.S_rot)
    pS("vib", tr.S_vib)
    print("S_tot = S_el + S_trans + S_rot + S_vib")
    pS("tot", tr.S_tot)
    print()

    print("Heat capacities:")
    pC("el", tr.c_el)
    pC("trans", tr.c_trans)
    pC("rot", tr.c_rot)
    pC("vib", tr.c_vib)
    pC("tot", tr.c_tot)
