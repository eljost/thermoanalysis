import re

import pyparsing as pp
import numpy as np


from thermoanalysis.constants import ANG2AU


def parse_orca(text):
    real = pp.common.real
    int_ = pp.common.integer

    coord_line = pp.Group(
        int_.set_results_name("no")
        + pp.Word(pp.alphas).set_results_name("lb")
        + real.set_results_name("za")
        + int_.set_results_name("frag")
        + real.set_results_name("mass")
        + pp.Group(real + real + real).set_results_name("xyz")
    )

    nu_line = (
        pp.Suppress(int_ + pp.Literal(":"))
        + real.set_results_name("nu")
        + pp.Suppress(pp.Literal("cm**-1"))
    )

    parser = (
        # Coordinates and masses
        pp.SkipTo(
            pp.Literal("MASS") + pp.Literal("X") + pp.Literal("Y") + pp.Literal("Z"),
            include=True,
        )
        + pp.OneOrMore(coord_line).set_results_name("coords")
        # Multiplicity
        + pp.SkipTo(
            pp.Literal("Multiplicity") + pp.Literal("Mult") + pp.Literal("...."),
            include=True,
        )
        + int_.set_results_name("mult")
        # Single point energy
        + pp.SkipTo(
            pp.Literal("FINAL SINGLE POINT ENERGY"),
            include=True,
        )
        + real.set_results_name("scf_energy")
        # Wavenumbers
        + pp.SkipTo(pp.Literal("Scaling factor for frequencies ="), include=True)
        + real.set_results_name("scaling_factor")
        + pp.Literal("(already applied!)")
        + pp.OneOrMore(nu_line).set_results_name("nus")
    )
    res = parser.parse_string(text)

    atoms = list()
    masses = list()
    coords = list()
    for line in res.coords:
        atoms.append(line["lb"])
        masses.append(line["mass"])
        coords.append(line["xyz"].as_list())
    masses = np.array(masses)
    coords = np.array(coords).reshape(-1, 3) / ANG2AU
    # "Unscale" wavenumbers
    nus = np.array(res.nus.as_list()) / res.scaling_factor
    results = {
        "atoms": tuple(atoms),
        "masses": masses,
        "wavenumbers": nus,
        "coords3d": coords,
        "mult": res.mult,
        "scf_energy": res.scf_energy,
    }
    return results


PARSE_FUNCS = {
    "orca": parse_orca,
}
PATTERNS = {
    "orca": r"\* O   R   C   A \*",
}
REGEXS = {key: re.compile(p) for key, p in PATTERNS.items()}


def get_parse_func(text):
    for key, regex in REGEXS.items():
        mobj = regex.search(text)
        if mobj:
            parse_func = PARSE_FUNCS[key]
            break
    return parse_func


def parse(text, key=None):
    if key is None:
        parse_func = get_parse_func(text)
    else:
        parse_func = PARSE_FUNCS[key]

    results = parse_func(text)
    return results
