#!/usr/bin/env python3

import cclib

class Thermo:

    def __init__(self, log_fn,
                 point_group="c1", scale_factor=1.0, solvent=None):
        self.log_fn = log_fn

        self.point_group = point_group
        self.scale_factor = scale_factor
        self.solvent = solvent

        parser = cclib.io.ccopen(self.log_fn)
        data = parser.parse()
        self.data = data
        self.vibfreqs = self.scale_factor * self.data.vibfreqs
        self.scf_energy = self.data.scfenergies[-1]
