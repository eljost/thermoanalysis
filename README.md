# thermoanalysis

External thermochemistry analysis implemented in pure python for all
frequency calculations that can be parsed by `cclib` and provide the
required attributes (`atomcoords`, `vibfreqs`, `scfenergies`, `atommasses`, `mult`).

The program is inspired by the nice [GoodVibes](https://github.com/bobbypaton/GoodVibes) script from the Patton group.

## Installation
```
git clone git@github.com:eljost/thermoanalysis.git
cd thermoanalysis
python -m pip install .
```

## Verifying the installation
```
# Can be executed from any directory after installation of thermoanalysis
pytest --pyargs thermoanalysis
```

## Usage
```
thermo [log] --temp [298.15] --pg [c1] [--verbose]
```

## Sample output
```
$ thermo 05_dmso_hf_orca_freq.out --temps 298.15 398.15 11
Using QRRHO-approach for vibrational entropies.
ZPE = 0.083938 au / particle (independent of T)
U_vib and U_tot already include the ZPE.
All quantities given in au / particle except T (given in K).
     T         U_el    U_therm        U_tot            H    TS_tot            G
------  -----------  ---------  -----------  -----------  --------  -----------
298.15  -548.679124   0.089954  -548.589170  -548.589170  0.035958  -548.625128
308.15  -548.679124   0.090262  -548.588863  -548.588863  0.037504  -548.626367
318.15  -548.679124   0.090576  -548.588549  -548.588549  0.039067  -548.627616
328.15  -548.679124   0.090895  -548.588229  -548.588229  0.040647  -548.628876
338.15  -548.679124   0.091221  -548.587903  -548.587903  0.042244  -548.630147
348.15  -548.679124   0.091553  -548.587571  -548.587571  0.043857  -548.631428
358.15  -548.679124   0.091891  -548.587233  -548.587233  0.045487  -548.632720
368.15  -548.679124   0.092235  -548.586889  -548.586889  0.047133  -548.634022
378.15  -548.679124   0.092585  -548.586539  -548.586539  0.048794  -548.635334
388.15  -548.679124   0.092941  -548.586184  -548.586184  0.050472  -548.636656
398.15  -548.679124   0.093303  -548.585822  -548.585822  0.052166  -548.637988
Dumped thermo-data to '05_dmso_hf_orca_freq_thermo.h5'.
```

## Some remarks
Even though it does not (yet) implement all features of the original `GoodVibes` the present implementation is not restricted to Gaussian logs as `GoodVibes`. In principle all frequency calculations that can be parsed by cclib are supported. Parsing of files that can't be handled by cclib can easily be achived by reimplenting the `set_data(log_fn)`-method in subclass of QCData.

Imaginary frequencies are neglected in the thermochemistry calculations.
