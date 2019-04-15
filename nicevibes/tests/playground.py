#!/usr/bin/env python3

from nicevibes.QCData import QCData


def run():
    log = "logs/02_dmso_freq.log"
    q = QCData(log, point_group="c2v")
    # import pdb; pdb.set_trace()
    print(q)

    temp = 298.15

    from nicevibes.main import thermochemistry
    thermochemistry(q, temp)


def plot_s_trans():
    from nicevibes.main import sackur_tetrode
    from nicevibes.constants import NA
    import matplotlib.pyplot as plt
    import numpy as np

    Ms = np.linspace(0, 300)
    temps = (298.15, 373.15, 573.15)

    fig, ax = plt.subplots()
    for T in temps:
        S_trans = sackur_tetrode(Ms, T)
        TS = -T*S_trans
        TS /= 1000
        TS /= 4.1868
        ax.plot(Ms, TS, label=f"T={T:.2f} K")
    ax.set_xlabel("$M \quad / \quad g \cdot mol^{-1}$")
    ax.set_ylabel("$-TS \quad / \quad kcal \cdot mol^{-1}$")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run()
    # plot_s_trans()
