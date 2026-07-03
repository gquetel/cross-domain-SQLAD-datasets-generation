# Common, pinned, python environment to use in shell / Dockerfiles.
# This file only manages the python interpreter + uv, not runtime system libs.
let
  sources = import ../npins;
  pkgs = import sources.nixpkgs { config.allowUnfree = true; };
in {
  inherit (pkgs) python313 uv;
}
