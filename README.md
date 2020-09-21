# Infinite Lattice Learner (ILL)

An ensemble learning algorithm
that enables supervised learning algorithms
to effectively learn incrementally.
Introduced in the paper,
[Infinite Lattice Learner: an ensemble for incremental learning](https://www.justinlovinger.com/file/infinite-lattice-learner.pdf).

Although the algorithm itself can be very efficient,
this code was developed primarily for research
and is not as efficient as it could be.

## Installation

### Nix

Add to Nix script like

```
...
let
  ill = (import (pkgs.fetchFromGitHub {
    owner = "JustinLovinger";
    repo = "ill";
    rev = "LATEST_VERSION_TAG";
    sha256 = "SHA25_OF_LATEST_VERSION_TAG";
  }) { inherit pkgs; });
in pkgs.python2Packages.buildPythonPackage {
  ...
  propagatedBuildInputs = [ ill ];
  ...
}
```

Get `sha256`
with `nix run nixpkgs.nix-prefetch-github -c nix-prefetch-github --rev "LATEST_VERSION_TAG" JustinLovinger ill`.

### Other

Obtain dependencies listed in default.nix `propagatedBuildInputs`,
and copy ill.py to your project.

## Example Usage

```py
import functools

from learning import datasets, validation, LinearRegressionModel
from ill import ILL, get_neighborhood_k_nearest

# Grab the popular iris dataset, from 'learning'
dataset = datasets.get_iris()

# Make an underlying model for ILL
# See 'learning' for more details
underlying_model = LinearRegressionModel(4, 3)

# Make an ILL ensemble of our underlying model
# See code for more options
model = ILL(
    underlying_model,
    grid_spacing=0.5,
    neighborhood_func=functools.partial(
        get_neighborhood_k_nearest, k_nearest=5))

# Lets train our ILL
# First, we'll split our dataset into training and testing sets
# Our training set will contain 30 samples from each class
training_set, testing_set = validation.make_train_test_sets(
    *dataset, train_per_class=30)

# We could customize training and stopping criteria through
# the arguments of train, but the defaults should be sufficient here
model.train(*training_set)

# Our ILL should converge in a few moments
# Lets see how our ILL does on the testing set
print 'Testing accuracy:', validation.get_accuracy(model, *testing_set)
```

## Development

Enter a Nix shell with `nix-shell`.
