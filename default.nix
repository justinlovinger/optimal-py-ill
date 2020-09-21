{ pkgs ? (import (builtins.fetchGit {
  url = https://github.com/NixOS/nixpkgs-channels.git;
  ref = "nixos-19.09";
  rev = "2de9367299f325c2b2021a44c2f63c810f8ad023";
}) { }) }:

let
  learning = import (pkgs.fetchFromGitHub {
    owner = "JustinLovinger";
    repo = "learning";
    rev = "4a40e39989f5234affeddd3ef7d5fd6a59784041";
    sha256 = "1yh7y09k6q8c54r4gz31dccbc5vizl1na7fvpah043kzfj8klf5a";
  }) { inherit pkgs; };
in pkgs.python2Packages.buildPythonPackage {
  pname = "ill";
  version = "1.0.0";
  src = ./.;

  checkInputs = with pkgs.python2Packages; [ pytest ];
  propagatedBuildInputs = with pkgs.python2Packages; [ learning numpy scipy ];

  checkPhase = "pytest";

  meta = with pkgs.stdenv.lib; {
    description = "Ensemble that enables supervised learning algorithms to effectively learn incrementally";
    homepage = "https://github.com/justinlovinger/ill";
    license = licenses.mit;
    maintainers = [ ];
  };
}
