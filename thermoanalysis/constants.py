import scipy.constants as spc


AU2EV = spc.value("Hartree energy in eV")  # Hartree to eV
CAL2J = spc.calorie
J2CAL = 1 / CAL2J  # Joule to calorie
J2AU = 1 / spc.value("Hartree energy")  # Joule to Hartree
ANG2M = 1e-10  # Angstrom to meter
ANG2AU = 1 / spc.value("Bohr radius") * ANG2M  # Angstrom to Bohr
AMU2KG = spc.value("unified atomic mass unit")  # Atomic mass units to g
R = spc.R  # J/(K*mol), ideal gas constant
C = spc.c  # Speed of light in m/s
PLANCK = spc.Planck  # J/s, Planck constant
KB = spc.Boltzmann  # J/K, Boltzmann constant
KBAU = KB * J2AU
NA = spc.N_A  # 1/mol, Avogadro constant
AU2J_MOL = 1 / J2AU * NA
KCAL2J = CAL2J * 1000
CAL_MOL2AU = CAL2J * J2AU / NA
KCAL_MOL2AU = 1000 * CAL_MOL2AU
