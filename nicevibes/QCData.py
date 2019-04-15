#!/usr/bin/env python3

import re

import cclib

class QCData:

    def __init__(self, log_fn,
                 point_group="c1", scale_factor=1.0, solvent=None):
        self.log_fn = log_fn

        self.point_group = point_group
        self.scale_factor = scale_factor
        self.solvent = solvent

        self.symmetry_number = self.get_symmetry_number()

        parser = cclib.io.ccopen(self.log_fn)
        data = parser.parse()
        self.data = data
        self.vibfreqs = self.scale_factor * self.data.vibfreqs
        self.scf_energy = self.data.scfenergies[-1]
        self.atom_masses = self.data.atommasses
        self._mult = self.data.mult

    @property
    def M(self):
        """Molecular mass."""
        return self.atom_masses.sum()

    @property
    def mult(self):
        """Multiplicity."""
        return self._mult

    def get_symmetry_number(self, point_group=None):
        symm_dict = {
            "c1": 1,
            "ci": 1,
            "cs": 1,
            "cinf": 1,
            "dinfh": 2,
            "t": 12,
            "td": 12,
            "oh": 24,
            "ih": 60,
        }
        if point_group is None:
            point_group = self.point_group
        pg = point_group.lower()
        try:
            return symm_dict[pg]
        except KeyError:
            pass
        regex = "[cds](\d+)"
        mobj = re.match(regex, pg)
        try:
            sym_num = int(mobj[1])
        except TypeError:
            raise Exception(f"Specified point group '{pg}' is invalid!")

        if pg.startswith("d"):
            sym_num *= 2
        elif pg.startswith("s"):
            sym_num /= 2
        assert sym_num == int(sym_num), "Check your point group! Did you " \
            "specify some 'Sn' group with n ∈ (1, 3, 5, ...)? Please use " \
            "the corresponding 'Cnm' groups instead!"
        return sym_num
