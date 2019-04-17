# thermoanalysis

External thermochemistry analysis implemented in pure python for all
frequency calculations that can be parsed by `cclib` and provide the
required attributes (`atomcoords`, `vibfreqs`, `scfenergies`, `atommasses`, `mult`).

The program is inspired by the nice [GoodVibes](https://github.com/bobbypaton/GoodVibes) script from the Patton group.

## Requirements
```
python3.6+
cclib
numpy
```

## Installation
```
python setup.py install
```
Until cclib 1.6.2 is released this patch [Correct parsing of vibrational frequencies with ORCA 4.1.x](https://github.com/cclib/cclib/pull/706/commits/18a3945ed6eaa82f418e2150eb5307be9697c238) has to be applied to correctly parse frequency calculations done with recent ORCA versions (4.1+).

## Running tests
```
# Can be executed from any directory after installation of thermoanalysis
pytest --pyargs thermoanalysis
```

## Usage
```
thermo [log] --temp [298.15] --pg [c1]
```

## Some remarks
Even though it does not (yet) implement all features of the `GoodVibes` the present implementation is not restricted to only Gaussian logs as `GoodVibes`. In principle all frequency calculations that can be parsed by cclib should be supported. Parsing of files that can't be handled by cclib could be implemented by deriving a new class from `QCData` and overwriting the `set_data(log_fn)` method to handle the specific format.

Imaginary frequencies are neglected in the thermochemistry analysis.
