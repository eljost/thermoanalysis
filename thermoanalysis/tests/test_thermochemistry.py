import pytest
from pytest import approx

from thermoanalysis.constants import CAL_MOL2AU, KCAL_MOL2AU
from thermoanalysis.thermo import thermochemistry, print_thermo_results
from thermoanalysis.QCData import QCData


def test_g16_thermochemistry(this_dir):
    log = this_dir / "logs/04_dmso_hf_freq.log"
    qc = QCData(log, point_group="c1")
    T = 298.15
    thermo = thermochemistry(qc, T, kind="rrho")

    zpe_ref = 0.083950
    assert thermo.ZPE == approx(zpe_ref)

    u_trans_ref = 0.889 * KCAL_MOL2AU
    assert thermo.U_trans == approx(u_trans_ref, rel=1e-3)
    u_rot_ref = 0.889 * KCAL_MOL2AU
    assert thermo.U_rot == approx(u_rot_ref, rel=1e-3)
    u_vib_ref = 54.676 * KCAL_MOL2AU
    assert thermo.U_vib == approx(u_vib_ref, rel=1e-3)

    s_el_ref = 0.0 * CAL_MOL2AU
    assert thermo.S_el == approx(s_el_ref)
    s_trans_ref = 38.978 * CAL_MOL2AU
    assert (thermo.S_trans) == approx(s_trans_ref, rel=3e-3)
    s_rot_ref = 25.168 * CAL_MOL2AU
    assert (thermo.S_rot) == approx(s_rot_ref, rel=1e-3)
    s_vib_ref = 11.519 * CAL_MOL2AU
    assert (thermo.S_vib) == approx(s_vib_ref, rel=1e-3)


def test_orca_thermochemistry(this_dir):
    log = this_dir / "logs/05_dmso_hf_orca_freq.out"
    qc = QCData(log, point_group="c1")
    T = 298.15
    thermo = thermochemistry(qc, T, kind="qrrho")

    zpe_ref = 0.08393782
    assert thermo.ZPE == approx(zpe_ref)

    u_trans_ref = 0.00141627
    assert thermo.U_trans == approx(u_trans_ref, rel=1e-5)
    u_rot_ref = 0.00141627
    assert thermo.U_rot == approx(u_rot_ref, rel=1e-5)

    s_el_ref = 0.0
    assert thermo.S_el == approx(s_el_ref)
    s_trans_ref = 0.01852169
    assert (thermo.S_trans * T) == approx(s_trans_ref, rel=1e-3)
    s_rot_ref = 0.01092172
    assert (thermo.S_rot * T) == approx(s_rot_ref, rel=1e-1)
    s_vib_ref = 0.00546508
    assert (thermo.S_vib * T) == approx(s_vib_ref, rel=1e-4)


def test_orca42_thermochemistry(this_dir):
    log = this_dir / "logs/01_alkin_000.007.orca.out"
    qc = QCData(log, point_group="c1")
    T = 298.15
    thermo = thermochemistry(qc, T, kind="qrrho")

    zpe_ref = 0.15505609
    assert thermo.ZPE == approx(zpe_ref, rel=1e-5)

    u_trans_ref = 0.00141627
    assert thermo.U_trans == approx(u_trans_ref, rel=1e-5)
    u_rot_ref = 0.00141627
    assert thermo.U_rot == approx(u_rot_ref, rel=1e-5)

    s_el_ref = 0.0
    assert thermo.S_el == approx(s_el_ref)
    s_trans_ref = 0.01953842
    assert (thermo.S_trans * T) == approx(s_trans_ref, rel=1e-3)
    s_rot_ref = 0.01479918
    assert (thermo.S_rot * T) == approx(s_rot_ref, rel=1e-5)
    s_vib_ref = 0.01433560
    assert (thermo.S_vib * T) == approx(s_vib_ref, rel=1e-3)


def test_orca42_benzaldehyde(this_dir):
    log = this_dir / "logs/06_benzaldehyde_b973c_orca_gas.out"
    qc = QCData(log, point_group="c1")
    T = 298.15
    thermo = thermochemistry(qc, T, kind="qrrho")

    dG_ref = 0.10897713
    assert thermo.dG == approx(dG_ref, abs=2e-5)


def test_orca_bigger(this_dir):
    log = this_dir / "logs/high_high_000.010.orca.out"
    qc = QCData(log, point_group="c1")
    thermo = thermochemistry(qc, temperature=298.15, kind="qrrho")

    dG_ref = 0.25625328
    assert thermo.dG == approx(dG_ref, abs=2e-5)


@pytest.mark.parametrize("invert_imags, dG_ref", ((0.0, 0.25709996), (-20, 0.25522808)))
def test_orca_invert(invert_imags, dG_ref, this_dir):
    log = this_dir / "logs/gas_000.017.orca.out"
    qc = QCData(log, point_group="c1")
    thermo = thermochemistry(
        qc, temperature=298.15, kind="qrrho", invert_imags=invert_imags
    )

    assert thermo.dG == approx(dG_ref, abs=2e-5)


def test_orca_ts(this_dir):
    log = this_dir / "logs/orca_hf_321g_hcn_iso_ts.out"
    qc = QCData(log, point_group="c1")
    thermo = thermochemistry(qc, temperature=298.15, kind="qrrho")

    assert thermo.dG == approx(-0.01058726, abs=2e-5)


@pytest.mark.parametrize(
    "id_, dG_ref",
    (
        (24, 0.62709533),
        (63, 0.62781152),
        (84, 0.62876245),
    ),
)
def test_irc_logs(this_dir, id_, dG_ref):
    log = this_dir / f"logs/irc_000.0{id_}.orca.out"
    qc = QCData(log, point_group="c1")
    thermo = thermochemistry(qc, temperature=298.15, kind="qrrho")
    assert thermo.dG == approx(dG_ref, abs=1e-5)


def test_print_thermo_results(this_dir):
    log = this_dir / "logs/06_benzaldehyde_b973c_orca_gas.out"
    qc = QCData(log, point_group="c1")
    T = 298.15
    thermo = thermochemistry(qc, T, kind="qrrho")
    print_thermo_results(thermo)
