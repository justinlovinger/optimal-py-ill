import functools

from learning import datasets, validation, LinearRegressionModel
from ill import ILL, get_neighborhood_k_nearest

# Grab the popular iris dataset, from 'learning'
dataset = datasets.get_iris()

# Make an underlying model for ILL
# See 'learning' library for more details
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
