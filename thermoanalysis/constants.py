#!/usr/bin/env python3

# Just for reference, these values are right now not used here:
#     http://gaussian.com/constants/

AU2EV = 27.21138602 # Hartree to eV
CAL2J = 4.1868
J2CAL = 1/CAL2J # Joule to calorie
J2AU = 2.293710449e+17 # Joule to Hartree
ANG2AU = 1.8897259886 # Angstrom to Bohr
AMU2KG = 1.660539e-27 # Atomic mass units to g
ANG2M = 1e-10 # Angstrom to meter

R = 8.3144598 # J/(K*mol), ideal gas constant
C = 299_792_458 # m/s, speed of light
PLANCK = 6.626_0700_40e-34 # J/s, Planck constant
KB = 1.380_648_52e-23 # J/K, Boltzmann constant
KBAU = KB * J2AU
NA = 6.022_140_857e23 # 1/mol, Avogadro constant

AU2J_MOL = 1/J2AU*NA
KCAL2J = CAL2J * 1000
CAL_MOL2AU = CAL2J * J2AU / NA
KCAL_MOL2AU = 1000 * CAL_MOL2AU
