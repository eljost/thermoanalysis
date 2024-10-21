{
  description = "Stand-alone thermochemistry in python for ORCA and Gaussian";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    qchem.url = "github:nix-qchem/nixos-qchem";
  };

  outputs = { self, qchem, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          overlay = import ./nix/overlay.nix;
          pkgs = import qchem.inputs.nixpkgs {
            inherit system;
            overlays = [
              qchem.overlays.default
              overlay
            ];
          };
        in
        {
          packages.default = pkgs.python3.pkgs.thermoanalysis;

          formatter = pkgs.nixpkgs-fmt;
        }) // {
      overlays.default = import ./nix/overlay.nix;
    };
}