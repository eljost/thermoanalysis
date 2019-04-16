#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from nicevibes.constants import NA, C
from nicevibes.main import (thermochemistry, sackur_tetrode_simplified,
                            harmonic_vibrational_entropies,
                            quasi_harmonic_vibrational_entropies,
                            vibrational_entropies)
from nicevibes.QCData import QCData


np.set_printoptions(precision=6)


def run():
    # log = "logs/02_dmso_pm6_freq.log"
    log = "logs/04_dmso_hf_freq.log"
    q = QCData(log, point_group="c1")
    print(q)
    rt = q.rot_temperatures
    print("rot_temps", rt)

    temp = 298.15

    thermochemistry(q, temp)

    # Rcm= (10.067986, 0.877991, 0.840280)
    # Rm= np.array(Rcm) * 100
    # t = rotational_temperature(Rm)
    # V = get_V_free()
    # v = get_V_free("chloroform")


def plot_s_trans():
    Ms = np.linspace(1, 300, 100)
    temps = (298.15, 373.15, 573.15)

    fig, ax = plt.subplots()
    for T in temps:
        S_trans = sackur_tetrode_simplified(Ms, T)
        TS = -T*S_trans
        TS /= 1000
        TS /= 4.1868
        ax.plot(Ms, TS, label=f"T={T:.2f} K")
    ax.set_xlim(0, Ms.max())
    ax.set_xlabel("$M \quad / \quad g \cdot mol^{-1}$")
    ax.set_ylabel("$-TS \quad / \quad kcal \cdot mol^{-1}$")
    ax.legend()
    plt.show()


def plot_vibrational_entropies():
    fig, ax = plt.subplots()

    T = 298.15
    wavenumbers = np.linspace(0, 350, 100) * 100 # in m^-1
    freqs = C*wavenumbers

    S_hvibs = harmonic_vibrational_entropies(T, freqs)
    S_qvibs = quasi_harmonic_vibrational_entropies(T, freqs)
    cutoff = 100
    alpha = 4
    S_vibs = vibrational_entropies(T, freqs, cutoff, alpha)

    ax.plot(wavenumbers/100, S_hvibs*NA, label="harmonic")
    ax.plot(wavenumbers/100, S_qvibs*NA, label="quasi")
    ax.plot(wavenumbers/100, S_vibs*NA, label="weighted")
    ax.set_xlabel("mode frequency / cm$^{-1}$")
    ax.set_ylabel("entropy / J mol$^{-1}$ K$^{-1}$")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run()
    # plot_s_trans()
    # plot_vibrational_entropies()
