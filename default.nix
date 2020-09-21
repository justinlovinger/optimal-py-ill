{ pkgs ? (import (builtins.fetchGit {
  url = https://github.com/NixOS/nixpkgs-channels.git;
  ref = "nixos-19.09";
  rev = "2de9367299f325c2b2021a44c2f63c810f8ad023";
}) { }) }:

let
  learning = import (pkgs.fetchFromGitHub {
    owner = "JustinLovinger";
    repo = "learning";
    rev = "5ae4136173d0fd0e903301bb27be87459512cd40";
    sha256 = "13nf3xjd48h4n15i4i5vz3cqvpzsj0zlvn6g1qr1l5jmk6h6n1a3";
  }) { inherit pkgs; };
in pkgs.python2Packages.buildPythonPackage {
  pname = "ill";
  version = "1.0.0";
  src = ./.;

  checkInputs = with pkgs.python2Packages; [ pytest ];
  propagatedBuildInputs = with pkgs.python2Packages; [ learning numpy scipy ];

  meta = with pkgs.stdenv.lib; {
    description = "Ensemble that enables supervised learning algorithms to effectively learn incrementally";
    homepage = "https://github.com/justinlovinger/ill";
    license = licenses.mit;
    maintainers = [ ];
  };
}
