let
  inputs = import ./npins;
  pkgs = import inputs.nixpkgs {
    config.allowUnfree = true;
  };
  pyEnv = import ./nix/python-env.nix;

  sqlmap = pkgs.python313Packages.sqlmap.overridePythonAttrs (oldAttrs: {
    propagatedBuildInputs = (oldAttrs.propagatedBuildInputs or [ ]) ++ [
      pkgs.python313Packages.sqlalchemy
      pkgs.python313Packages.pymysql
    ];
  });

  fontsConf = pkgs.makeFontsConf {
    fontDirectories = [
      pkgs.corefonts # Times New Roman (and other MS core fonts) for paper figures
      pkgs.cm_unicode # Computer Modern Unicode (CMU Serif / CMU Sans Serif)
    ];
  };
in
# buildFHSEnv is the most reliable way to make pip-installed wheels (torch,
# transformers, ...) link against CUDA and system libs across hosts (NixOS, Ubuntu +
# Nix, the AlmaLinux cluster). Python itself is managed by uv -- the Nix env
# only provides the interpreter, uv, and the system tools below.
(pkgs.buildFHSEnv {
  name = "sqlia-dataset-generator";

  targetPkgs =
    pkgs: with pkgs; [
      pyEnv.python313
      pyEnv.uv

      # Formatting tools
      treefmt
      black
      nixpkgs-fmt
      taplo
      mdformat

      # System tools used during generation
      percona-toolkit
      perl # required by pt-kill (missing Sys/Hostname.pm)
      mysql84
      metasploit
      sqlmap

      # Chromium for kaleido/plotly static image export (experiments/embedding_viz.py).
      # Kaleido v1 needs a real Chrome; the one `plotly_get_chrome` downloads is a
      # generic dynamically-linked binary that won't run on NixOS, so we provide
      # Chromium here and point kaleido at it via BROWSER_PATH below.
      chromium

      # System-level packages
      procps # if not available breaks venv
      git # uv reads the git SHA when locking
      fish

      # Shared libs that pip-installed wheels link against
      zlib
      stdenv.cc.cc.lib # libstdc++, libgcc_s

      # Fonts for matplotlib paper figures (experiments/diversity_metric.py)
      corefonts
      cm_unicode
    ];

  runScript = ''
    bash -c '
      # Fonts available to matplotlib when rendering figures.
      export FONTCONFIG_FILE=${fontsConf}

      # Browser used by kaleido/plotly for static image export. Pinned to the
      # Nix-provided Chromium so it never falls back to a downloaded binary.
      export BROWSER_PATH=${pkgs.chromium}/bin/chromium

      # uv must use the Nix-provided interpreter, never download its own.
      export UV_PYTHON_DOWNLOADS=never
      export UV_PYTHON_PREFERENCE=only-system

      # MySQL environment variables used by mysql-start / mysql-stop scripts.
      # Data lives in /tmp (local to each machine) to avoid NFS conflicts.
      export MYSQL_HOME="/tmp/mysql-dev-sqlia"
      export MYSQL_DATADIR="$MYSQL_HOME/data"
      export MYSQL_UNIX_PORT="$MYSQL_HOME/mysql.sock"
      export MYSQL_PORT=61337

      # Put project scripts on PATH
      export PATH="$PWD/scripts:$PATH"

      # Build/refresh .venv from uv.lock. Generate the lock on first entry so a
      # fresh checkout works without a separate bootstrap step.
      if [ ! -f uv.lock ]; then uv lock; fi
      uv sync --frozen

      echo "Run mysql-start to start MySQL, mysql-stop to stop it."
      exec fish -C "source .venv/bin/activate.fish"
    '
  '';
}).env
