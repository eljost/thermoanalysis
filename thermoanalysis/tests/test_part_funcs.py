import json

import pytest

from thermoanalysis.QCData import QCData
from thermoanalysis.thermo import thermochemistry


@pytest.fixture
def dmso_thermo(this_dir):
    """Gaussian HF / 3-21G DMSO frequency."""
    log = this_dir / "logs/04_dmso_hf_freq.log"
    qc = QCData(log, point_group="c1")
    T = 298.15
    thermo = thermochemistry(qc, T, kind="rrho")
    return thermo


def test_el_part_func(dmso_thermo):
    q_el = dmso_thermo.Q_el
    assert q_el == pytest.approx(1.0)


def test_trans_part_func(dmso_thermo):
    q_trans = dmso_thermo.Q_trans
    assert q_trans == pytest.approx(0.270840e8, abs=33)


def test_rot_part_func(dmso_thermo):
    q_rot = dmso_thermo.Q_rot
    assert q_rot == pytest.approx(0.706132e5)


def test_vib_part_func(dmso_thermo):
    q_vib = dmso_thermo.Q_vib
    assert q_vib == pytest.approx(0.275364e-37)


def test_vib_part_func_V0(dmso_thermo):
    q_vib_V0 = dmso_thermo.Q_vib_V0
    assert q_vib_V0 == pytest.approx(0.113299e2)


@pytest.fixture
def dmso_xtb(this_dir):
    with open(this_dir / "logs/dmso_641_xtbout.json") as handle:
        data = json.load(handle)

    results = {
        "coords3d": data["coords"],
        "wavenumbers": data["vibrational frequencies/rcm"],
        "scf_energy": data["total energy"],
        "masses": data["masses"],
        "mult": 1,
    }
    qc = QCData(inp=results, point_group="cs")
    thermo = thermochemistry(qc, temperature=298.15, kind="rrho", rotor_cutoff=50)
    return thermo


def test_xtb_rot_part_func(dmso_xtb):
    q_rot = dmso_xtb.Q_rot
    assert q_rot == pytest.approx(0.624e05, abs=31)


def test_xtb_vib_V0_part_func(dmso_xtb):
    q_vib_V0 = dmso_xtb.Q_vib_V0
    assert q_vib_V0 == pytest.approx(10.0, abs=0.03)
