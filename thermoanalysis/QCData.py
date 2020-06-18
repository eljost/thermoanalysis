#!/usr/bin/env python3

import re

import cclib
import numpy as np

from thermoanalysis.constants import C, ANG2M, AMU2KG, PLANCK, KB, AU2EV


class QCData:

    def __init__(self, log_fn, point_group="c1", scale_factor=1.0):
        self.log_fn = str(log_fn)

        self.point_group = point_group.lower()
        self.scale_factor = scale_factor

        self.symmetry_number = self.get_symmetry_number()

        self.set_data(self.log_fn)

        assert self.coords3d.all()
        assert self.wavenumbers.all()
        assert self.scf_energy
        assert self.masses.all()
        assert self.mult

    def set_data(self, log_fn):
        parser = cclib.io.ccopen(log_fn)
        data = parser.parse()
        self.data = data
        coords3d = self.data.atomcoords  # in Angstrom
        # assert coords3d.shape[0] == 1
        self.coords3d = coords3d[-1]
        self.wavenumbers = self.scale_factor * self.data.vibfreqs
        # TODO: report imaginary frequencies
        self.wavenumbers = self.wavenumbers[self.wavenumbers > 0]
        self.scf_energy = self.data.scfenergies[-1] / AU2EV
        assert self.data.scfenergies.shape[0] == self.data.atomcoords.shape[0]
        self.masses = self.data.atommasses
        self._mult = self.data.mult

    @property
    def M(self):
        """Molecular mass.

        Returns
        -------
        M : float
            Total molecular mass in amu.
        """
        return self.masses.sum()

    @property
    def mult(self):
        """Multiplicity.

        Returns
        -------
        2S+1 : int
            Multiplicity.
        """
        return self._mult

    @property
    def vib_frequencies(self):
        """Vibrational frequencies.

        Returns
        -------
        vibfreqs : np.array
            Vibrational frequencies in 1/s.
        """
        return C * self.wavenumbers * 100

    @property
    def is_linear(self):
        """Wether the molecule is linear.

        Returns
        -------
        is_linear : bool
            Wether the molecule is linear.
        """
        return self.point_group in ("cinf", "dinfh")

    @property
    def is_atom(self):
        """Wether the 'molecule' consists of only an atom.

        Returns
        -------
        is_atoms : bool
            Wether the molecule is only one atom.
        """
        return len(self.masses) == 1

    @property
    def rot_temperatures(self):
        """Rotational temperatures in K.

        Returns
        -------
        rot_temps : np.array
            Rotational temperatures in K.
        """
        self.standard_orientation()
        I = self.inertia_tensor() * ANG2M**2 * AMU2KG
        w, v = np.linalg.eigh(I)
        rot_temps = PLANCK**2/(8*np.pi**2*w*KB)
        return rot_temps

    def get_symmetry_number(self, point_group=None):
        """Symmetry number for rotatioanl partiton function.

        Returns
        -------
        symmetry_number : int
            Symmetry number for calculation of rotational terms.
        """
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
        regex = r"[cds](\d+)"
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

    def inertia_tensor(self):
        """Inertita tensor.

                              | x² xy xz |
        (x y z)^T . (x y z) = | xy y² yz |
                              | xz yz z² |
        Returns
        -------
        I : np.array, shape (3, 3)
            Ineratia tensor  in units of Angstrom² * amu.
        """
        x, y, z = self.coords3d.T
        squares = np.sum(self.coords3d**2 * self.masses[:, None], axis=0)
        I_xx = squares[1] + squares[2]
        I_yy = squares[0] + squares[2]
        I_zz = squares[0] + squares[1]
        I_xy = -np.sum(self.masses*x*y)
        I_xz = -np.sum(self.masses*x*z)
        I_yz = -np.sum(self.masses*y*z)
        I = np.array((
                (I_xx, I_xy, I_xz),
                (I_xy, I_yy, I_yz),
                (I_xz, I_yz, I_zz)
        ))
        return I

    @property
    def center_of_mass(self):
        """Returns the center of mass.

        Returns
        -------
        R : np.array, shape (3, )
            Center of mass in Angstrom.
        """
        return 1/self.M * np.sum(self.coords3d*self.masses[:, None],
                                 axis=0)

    def principal_axes_are_aligned(self):
        """Check if the principal axes are aligned with the cartesian axes.

        Returns
        -------
        aligned : bool
            Wether the principal axes are aligned or not.
        """
        w, v = np.linalg.eigh(self.inertia_tensor())
        return np.allclose(v, np.eye(3)), v

    def align_principal_axes(self):
        """Align the principal axes to the cartesian axes.

        https://math.stackexchange.com/questions/145023
        """
        I = self.inertia_tensor()
        w, v = np.linalg.eigh(I)
        self.coords3d = v.T.dot(self.coords3d.T).T

    def standard_orientation(self):
        """Bring molecule in standard orientation."""
        # Translate center of mass to cartesian origin
        self.coords3d -= self.center_of_mass
        # Try to rotate the principal axes onto the cartesian axes
        for i in range(5):
            self.align_principal_axes()
            aligned, vecs = self.principal_axes_are_aligned()
            if aligned:
                break
