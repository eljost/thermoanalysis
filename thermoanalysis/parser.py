import re
import warnings

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
        + pp.Optional(pp.Literal("*"))
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
            pp.Literal(
                "NO LB      ZA    FRAG     MASS         X           Y           Z"
            ),
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


def parse_orca_hess(text):
    integer = pp.common.integer
    line_integer = integer.copy().setWhitespaceChars(" \t")
    real = pp.common.real
    sci_real = pp.common.sci_real

    int_line = pp.Suppress(pp.OneOrMore(line_integer) + pp.LineEnd())
    hess_block_line = pp.Suppress(integer) + pp.OneOrMore(sci_real)
    hess_block = integer + int_line + pp.OneOrMore(hess_block_line)

    vib_line = pp.Suppress(integer) + real
    comment_line = (
        pp.pythonStyleComment
        + pp.ZeroOrMore(pp.Word(pp.printables).setWhitespaceChars(" \t"))
        + pp.LineEnd()
    )
    atom_line = pp.Group(
        pp.Word(pp.alphas).set_results_name("atom")
        + real.set_results_name("mass")
        + pp.Group(real + real + real).set_results_name("xyz")
    )
    three_sci_reals = sci_real + sci_real + sci_real
    ir_line = real + real + real + real + real + real
    parser = (
        pp.Literal("$orca_hessian_file")
        + pp.Literal("$act_atom")
        + integer.set_results_name("act_atom")
        + pp.Literal("$act_coord")
        + integer.set_results_name("act_coord")
        + pp.Literal("$act_energy")
        + real.set_results_name("act_energy")
        + pp.Literal("$hessian")
        + integer
        + pp.OneOrMore(hess_block).set_results_name("hessian")
        + pp.Literal("$vibrational_frequencies")
        + integer
        + pp.OneOrMore(vib_line).set_results_name("vibrational_frequencies")
        + pp.Literal("$normal_modes")
        + integer
        + integer
        + pp.OneOrMore(hess_block).set_results_name("normal_modes")
        + pp.OneOrMore(comment_line)
        + pp.Literal("$atoms")
        + integer.set_results_name("atom_num")
        + pp.OneOrMore(atom_line).set_results_name("atom_lines")
        + pp.Literal("$actual_temperature")
        + real
        + pp.Literal("$frequency_scale_factor")
        + real.set_results_name("frequency_scale_factor")
        + pp.Literal("$dipole_derivatives")
        + integer
        + pp.OneOrMore(three_sci_reals)
        + pp.OneOrMore(comment_line)
        + pp.Literal("$ir_spectrum")
        + integer
        + pp.OneOrMore(ir_line)
        + pp.Literal("$end")
    )
    res = parser.parse_string(text)
    res_dict = res.as_dict()

    atom_num = res_dict["atom_num"]
    atoms = list()
    masses = np.zeros(atom_num)
    coords3d = np.zeros((atom_num, 3))
    for i, atom_line in enumerate(res_dict["atom_lines"]):
        atoms.append(atom_line["atom"])
        masses[i] = atom_line["mass"]
        coords3d[i] = atom_line["xyz"]
    coords3d /= ANG2AU

    wavenumbers = np.array(res_dict["vibrational_frequencies"])
    scf_energy = res.act_energy
    warnings.warn(
        "As of ORCA <= 5.0.3, the multiplicity is not found in the *.hess file. "
        "Currently, it is hardcoded to 1!"
    )
    results = {
        "atoms": tuple(atoms),
        "masses": masses,
        "wavenumbers": wavenumbers,
        "coords3d": coords3d,
        "mult": 1,
        "scf_energy": scf_energy,
    }
    return results


PARSE_FUNCS = {
    "orca": parse_orca,
    "orca_hess": parse_orca_hess,
}
PATTERNS = {
    "orca": r"\* O   R   C   A \*",
    "orca_hess": r"\$orca_hessian_file",
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
