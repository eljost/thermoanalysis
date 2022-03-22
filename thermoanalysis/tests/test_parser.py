import pytest

from thermoanalysis.parser import get_parse_func, parse_orca, parse


@pytest.fixture
def dmso_orca(this_dir):
    with open(this_dir / "logs" / "02_tpss_freq.log") as handle:
        text = handle.read()
    return text


def test_get_parse_func(dmso_orca):
    parse_func = get_parse_func(dmso_orca)
    assert parse_func == parse_orca


def test_parse_orca(dmso_orca):
    results = parse_orca(dmso_orca)
    assert results["scf_energy"] == pytest.approx(-552.993458333088)
    assert len(results["wavenumbers"]) == 3 * len(results["atoms"])


def test_parse(dmso_orca):
    results = parse(dmso_orca)
    assert results["scf_energy"] == pytest.approx(-552.993458333088)
