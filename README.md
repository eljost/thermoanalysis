# thermoanalysis

External thermochemistry analysis implemented in pure python for all
frequency calculations that can be parsed by `cclib` and provide the
required attributes (`atomcoords`, `vibfreqs`, `scfenergies`, `atommasses`, `mult`).

The program is inspired by the nice [GoodVibes](https://github.com/bobbypaton/GoodVibes) script from the Patton group.

## Requirements
```
python3.6+
cclib
h5py
numpy
pytest
tabulate
```

## Installation
```
python setup.py install
```
Until `cclib 1.6.2` is released a patch ([Correct parsing of vibrational frequencies with ORCA 4.1.x](https://github.com/cclib/cclib/pull/706/commits/18a3945ed6eaa82f418e2150eb5307be9697c238)) has to be applied to `cclib` correctly parse frequency calculations done with recent ORCA versions (4.1+).

## Running tests
```
# Can be executed from any directory after installation of thermoanalysis
pytest --pyargs thermoanalysis
```

## Usage
```
thermo [log] --temp [298.15] --pg [c1]
```

## Sample output
```
$ thermo 05_dmso_hf_orca_freq.out --temps 298.15 398.15 11
All quantities given in kJ/mol except T (given in K).
ZPE = 220.38 kJ/mol (independent of T)
     T    U_trans    U_rot    U_vib    U_tot    TS_el    TS_trans    TS_rot    TS_vib    TS_tot
------  ---------  -------  -------  -------  -------  ----------  --------  --------  --------
298.15       3.72     3.72   228.74   236.17     0.00       48.66     31.40     14.35     94.41
308.15       3.84     3.84   229.30   236.98     0.00       50.50     32.58     15.38     98.47
318.15       3.97     3.97   229.87   237.81     0.00       52.36     33.76     16.45    102.57
328.15       4.09     4.09   230.46   238.65     0.00       54.21     34.95     17.56    106.72
338.15       4.22     4.22   231.07   239.50     0.00       56.07     36.14     18.69    110.91
348.15       4.34     4.34   231.69   240.37     0.00       57.94     37.34     19.87    115.15
358.15       4.47     4.47   232.33   241.26     0.00       59.82     38.54     21.07    119.43
368.15       4.59     4.59   232.98   242.16     0.00       61.70     39.74     22.31    123.75
378.15       4.72     4.72   233.65   243.08     0.00       63.59     40.94     23.58    128.11
388.15       4.84     4.84   234.33   244.02     0.00       65.48     42.15     24.88    132.52
398.15       4.97     4.97   235.04   244.97     0.00       67.38     43.37     26.22    136.96
```

## Some remarks
Even though it does not (yet) implement all features of the `GoodVibes` the present implementation is not restricted to only Gaussian logs as `GoodVibes`. In principle all frequency calculations that can be parsed by cclib should be supported. Parsing of files that can't be handled by cclib could be implemented by deriving a new class from `QCData` and overwriting the `set_data(log_fn)` method to handle the specific format.

Imaginary frequencies are neglected in the thermochemistry analysis.
