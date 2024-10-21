{ buildPythonPackage
, lib
, setuptools
, setuptools-scm
, cclib
, numpy
, h5py
, pandas
, tabulate
, pytestCheckHook
}:

buildPythonPackage rec {
  pname = "thermoanalysis";
  version = "0.1.0";

  src = lib.cleanSource ../.;

  pyproject = true;

  build-system = [
    setuptools
    setuptools-scm
  ];

  dependencies = [
    cclib
    numpy
    h5py
    pandas
    tabulate
  ];

  nativeCheckInputs = [ pytestCheckHook ];

  doCheck = true;

  meta = with lib; {
    description = "Stand-alone thermochemistry in python for ORCA and Gaussian";
    license = licenses.gpl3Plus;
    homepage = "https://github.com/eljost/thermoanalysis";
    maintainers = [ maintainers.sheepforce ];
  };
}