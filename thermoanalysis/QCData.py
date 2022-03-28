from pathlib import Path
import re

import cclib
import h5py
import numpy as np

from thermoanalysis.constants import C, ANG2M, AMU2KG, PLANCK, KB, AU2EV, ANG2AU
from thermoanalysis.parser import parse


class CCLibParserError(Exception):
    pass


class QCData:
    def __init__(
        self,
        inp,
        point_group="c1",
    ):

        self.point_group = point_group.lower()
        self.symmetry_number = self.get_symmetry_number()

        try:
            inp_path = Path(inp)
        except TypeError:
            inp_path = None

        if inp_path and inp_path.exists():
            inp = str(inp)
            self.fn = inp
            # Try to read as pysisyphus HDF5 Hessian
            if inp.endswith(".h5"):
                data = self.from_pysis_hdf5_hessian(inp)

            # Try cclib
            try:
                data = self.from_cclib(inp)
            # Try own parser
            except CCLibParserError:
                with open(inp) as handle:
                    text = handle.read()
                data = self.from_parser(text)
        # Treat inp as dict
        else:
            data = inp

        # Actually set data
        self.set_data(data)

        self.standard_orientation()
        I = self.inertia_tensor()
        w, v = np.linalg.eigh(I)
        self._linear = (abs(w[0]) < 1e-8) and (abs(w[1] - w[2]) < 1e-8)

    def from_pysis_hdf5_hessian(self, fn):
        with h5py.File(fn, "r") as handle:
            data = {
                "masses": handle["masses"][:],
                "wavenumbers": handle["vibfreqs"][:],
                "coords": handle["coords3d"][:] / ANG2AU,  # in Angstrom
                "scf_energy": handle.attrs["energy"],
                "mult": handle.attrs["mult"],
            }
        return data

    def from_cclib(self, fn):
        parser = cclib.io.ccopen(fn)
        try:
            data = parser.parse()
        except:
            raise CCLibParserError

        try:
            wavenumbers = data.vibfreqs
        # Single atoms don't vibrate
        except AttributeError:
            wavenumbers = list()
        results = {
            "coords3d": data.atomcoords[-1],  # in Angstrom
            "wavenumbers": wavenumbers,
            "scf_energy": data.scfenergies[-1] / AU2EV,
            "masses": data.atommasses,
            "mult": data.mult,
        }
        return results

    def from_parser(self, fn):
        return parse(fn)

    def set_data(self, data):
        expect = set("coords3d wavenumbers scf_energy masses mult".split())
        present = set(data.keys())
        missing = expect - present
        assert len(missing) == 0, f"Keys '{missing}' are missing!"

        self.masses = np.array(data["masses"], dtype=float)
        self.wavenumbers = np.array(data["wavenumbers"], dtype=float)
        self.coords3d = np.array(data["coords3d"], dtype=float).reshape(-1, 3)
        assert self.coords3d.size == 3 * len(self.masses)
        self.scf_energy = float(data["scf_energy"])
        self.mult = int(data["mult"])

    @property
    def atom_num(self):
        return len(self.masses)

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

    @mult.setter
    def mult(self, mult):
        self._mult = mult

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
        # return self.point_group in ("cinf", "dinfh")
        return self._linear

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
        I = self.inertia_tensor() * ANG2M ** 2 * AMU2KG
        w, v = np.linalg.eigh(I)
        rot_temps = PLANCK ** 2 / (8 * np.pi ** 2 * w * KB)
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
        assert sym_num == int(sym_num), (
            "Check your point group! Did you "
            "specify some 'Sn' group with n ∈ (1, 3, 5, ...)? Please use "
            "the corresponding 'Cnm' groups instead!"
        )
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
        squares = np.sum(self.coords3d ** 2 * self.masses[:, None], axis=0)
        I_xx = squares[1] + squares[2]
        I_yy = squares[0] + squares[2]
        I_zz = squares[0] + squares[1]
        I_xy = -np.sum(self.masses * x * y)
        I_xz = -np.sum(self.masses * x * z)
        I_yz = -np.sum(self.masses * y * z)
        I = np.array(((I_xx, I_xy, I_xz), (I_xy, I_yy, I_yz), (I_xz, I_yz, I_zz)))
        return I

    @property
    def average_moment_of_inertia(self):
        """Average moment of inertia in Angstrom² * amu."""
        w, _ = np.linalg.eigh(self.inertia_tensor())
        I_avg = np.mean(w)
        return I_avg

    @property
    def center_of_mass(self):
        """Returns the center of mass.

        Returns
        -------
        R : np.array, shape (3, )
            Center of mass in Angstrom.
        """
        return 1 / self.M * np.sum(self.coords3d * self.masses[:, None], axis=0)

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
        for _ in range(5):
            self.align_principal_axes()
            aligned, vecs = self.principal_axes_are_aligned()
            if aligned:
                break
