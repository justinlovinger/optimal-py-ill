"""Infinite Lattice Learner (ILL)"""
import math
import logging
import copy
import functools
import heapq

import numpy
from scipy import optimize

from learning import Model
from learning import preprocess, calculate

# Finite precision math can cause neighborhood functions
# to return points that virtually overlap, when they should be identical
# ILL treats such points as unique, and affects performance
# To avoid this issue, we round to the max precision
MAXIMUM_PRECISION = 10  # decimal places

class ILL(Model):
    """An ensemble that extends the given model to incremental learning.

    Infinite Lattice Learner (ILL)

    Args:
        model: A learning.Model object. ILL creates an ensemble of this model.
        grid_spacing: Step size betwen each point in cartesian grid.
        neighborhood_func: f(grid_spacing, center) -> Matrix of points around center,
            on a grid with grid_spacing spacing
        similarity_func: f(matrix_a, vec_b) -> vector of similarities between
            rows in matrix_a and vec_b
        normalize_similarity: Boolean; Whether or not to normalize similarity to a sum of 1.
        learn_exact: Boolean; Either calculate the optimal targets for each model during learning,
            or approximate.
        dim_reduction: (int, int)/func: Dimensionality reduction applied to the ILL neighborhood
            (but not the given model).
            If 2 tuple of ints, a simple projection to the given number of dimensions is constructed and applied.
            If function, the function is applied to each input vector, and is expected to return a
            vector with fewer dimensions.
    """
    def __init__(self,
                 model,
                 grid_spacing=1.0,
                 neighborhood_func=None,
                 similarity_func=None,
                 normalize_similarity=True,
                 learn_exact=False,
                 dim_reduction=None):
        super(ILL, self).__init__()

        if grid_spacing < (10.0**(-MAXIMUM_PRECISION)):
            raise ValueError('grid_spacing must be >= %s decimal places ' \
                             'to avoid precision errors' % MAXIMUM_PRECISION)

        self._hash_model = copy.deepcopy(model)

        # Hash field variables
        self._grid_spacing = grid_spacing
        self._normalize_similarity = normalize_similarity

        # Neighborhood function
        if neighborhood_func is None:
            self._neighborhood_func = get_neighborhood_adaptive_radius
        else:
            self._neighborhood_func = neighborhood_func

        # Similarity function
        if similarity_func is None:
            self._similarity_func = scaled_gaussian_similarity
        else:
            self._similarity_func = similarity_func

        if learn_exact is True:
            self._model_target_func = _solve_model_target_matrix
        elif learn_exact is False:
            self._model_target_func = _adjust_model_target_matrix_one_step
        elif learn_exact == 'adjust_to_target':  # Legacy option (originally a bug, but has merits)
            # adjust_to_target does extremely poorly on basic learning tests, despite having low retention error
            self._model_target_func = _adjust_model_target_matrix_to_target
        elif learn_exact == 'one_step':  # Experimental option (currently default approx)
            # one_step does very well on basic learning,
            # and has decent performance on retention error (and is well justified)
            self._model_target_func = _adjust_model_target_matrix_one_step
        elif learn_exact == 'legacy_nf':  # Learning rule from Neural Field
            # This approx does well on basic learning, but extremely poorly on retention error
            self._model_target_func = _adjust_model_target_matrix
        else:
            raise ValueError('Unknown option for learn_exact')

        # Dimensionality reduction
        self._dim_reduce_func = None
        if dim_reduction is None:
            # Don't reduce dimensions
            self._dim_reduce_shape = None
            self._dim_reduce_func = _identity
        elif isinstance(dim_reduction, (tuple, list)):
            if len(dim_reduction) != 2:
                raise ValueError('dim_reduction must be 2 tuple, or function')
            if not (isinstance(dim_reduction[0], int)
                    and isinstance(dim_reduction[1], int)):
                raise ValueError('dim_reduction tuple must contain integers')
            if dim_reduction[1] < 1:
                raise ValueError('Reduced # dimensions must be > 0')
            if dim_reduction[1] >= dim_reduction[0]:
                raise ValueError('dim_reduction must reduce dimensions')

            # Make the reduction function
            # Reduce using a random projection
            self._dim_reduce_shape = dim_reduction  # To reset random projection
            self._set_random_projection()
        else:
            self._dim_reduce_shape = None
            self._dim_reduce_func = dim_reduction

        # Variables to remember for learning
        self._active_models = None
        self._active_similarities = None
        self._active_outputs = None

        # Field maps indexes to predictors or neurons
        self._field = {}

    def _set_random_projection(self):
        """Set random projection for dimensionality reduction."""
        self._dim_reduce_func = functools.partial(
            numpy.dot,
            b=(preprocess.normalize(
                numpy.random.random(self._dim_reduce_shape)) /
               self._dim_reduce_shape[0]))

    def reset(self):
        self._hash_model.reset()

        self._active_models = None
        self._active_similarities = None
        self._active_outputs = None

        self._field = {}

        # Make new random projection for dimensionality reduction
        if self._dim_reduce_shape is not None:
            self._set_random_projection()

    def activate(self, input_vec):
        # Get nearby model points in neighborhood
        # Dimensionality reduction is only applied to neighborhood
        reduced_input = self._dim_reduce_func(input_vec)
        points = self._get_points(reduced_input)
        if len(points) == 0:
            logging.warning('No models within neighborhood of input_vec')

        if not isinstance(reduced_input, numpy.ndarray):
            reduced_input = numpy.array(reduced_input)

        # Similarity is calculated between reduced input and each point
        similarities = self._similarity_func(points, reduced_input)
        if self._normalize_similarity:
            # Normalize similarities to sum of 1
            similarities /= numpy.sum(similarities)

        # Get model for each point, and output for each model
        active_models = [self._get_model(point) for point in points]
        output_matrix = numpy.array(
            [model.activate(input_vec) for model in active_models])

        # Save for learning
        self._active_models = active_models
        self._active_similarities = similarities
        self._active_outputs = output_matrix

        # Return weighted sum of output vectors from each model,
        # weighted by similarity
        return similarities.dot(output_matrix)

    def _get_points(self, input_vec):
        return self._neighborhood_func(self._grid_spacing, input_vec)

    def _get_model(self, point):
        """Return stored or new model at point."""
        # Lazy evaluation
        # Initialize model if empty
        try:
            return self._field[point]
        except KeyError:
            new_model = copy.deepcopy(self._hash_model)
            # Re-randomize, assuming model has random initialization
            new_model.reset()
            new_model.logging = False
            self._field[point] = new_model
            return new_model

    def train_step(self, input_matrix, target_matrix):
        # Make target matrix for each activated model
        errors = []
        models = set()  # Train each activated model once
        model_input_matrices = {}  # Maps model to input_matrix for that model
        model_target_matrices = {}  # Maps model to target_matrix for that model
        for input_vec, target_vec in zip(input_matrix, target_matrix):
            error, active_models, model_target_matrix = self._get_model_targets(
                input_vec, target_vec)

            errors.append(error)

            # Add each activated model, without duplicates
            models.update(active_models)

            # Add input vector for each activated model
            for model in active_models:
                try:
                    model_input_matrices[model].append(input_vec)
                except KeyError:
                    model_input_matrices[model] = [input_vec]

            # Add target vector for each activated model
            for model, target_vec in zip(active_models, model_target_matrix):
                try:
                    model_target_matrices[model].append(target_vec)
                except KeyError:
                    model_target_matrices[model] = [target_vec]

        # Train activated models
        for model in models:
            # Update model (1 iteration, for non-batch models)
            model.train(
                numpy.array(model_input_matrices[model]),
                numpy.array(model_target_matrices[model]),
                iterations=1,
                retries=0,
                error_break=None)

        return numpy.mean(errors)

    def _get_model_targets(self, input_vec, target_vec):
        output_vec = self.activate(input_vec)

        # Find target vector for each model, that leads to better error
        # Note: each row corresponds to a model
        model_target_matrix = self._model_target_func(
            self._active_similarities, self._active_outputs, output_vec,
            target_vec)

        return (numpy.mean((target_vec - output_vec)**2),
                self._active_models, model_target_matrix)


###########################
# Variance
###########################
def _distance_at_cutoff(variance, similarity_cutoff):
    """Return distance that gives similarity_cutoff for a gaussian kernel with given variance."""
    return math.sqrt(
        variance * math.log(1.0 / similarity_cutoff))  # log is natural


def _variance_to_cutoff(distance_cutoff, similarity_cutoff):
    """Return variance that gives similarity_cutoff at distance_cutoff."""
    # log is natural
    return distance_cutoff**2 / (math.log(1.0 / similarity_cutoff))


###########################
# Getting points
###########################
def get_neighborhood_k_nearest(grid_spacing, center, k_nearest, incr_rate=1.2):
    """Return the k nearest point on cartesian grid.

    Find the k nearest points by gradually increasing radius,
    and applying get_neighborhood_radius, until at least
    k points are returned.
    Then cull if |points| > k.
    """
    points = get_neighborhood_approx_k_nearest(
        grid_spacing, center, k_nearest, incr_rate=incr_rate)

    if len(points) > k_nearest:
        # Cull furthest points
        distances = [calculate.distance(p, center) for p in points]
        indices = heapq.nsmallest(
            k_nearest, range(len(distances)), key=distances.__getitem__)
        points = [points[i] for i in indices]

    return points


def get_neighborhood_approx_k_nearest(grid_spacing,
                                      center,
                                      k_nearest,
                                      incr_rate=1.02):
    """Return the approximately k nearest point on cartesian grid.

    Number of points returned is >= k, and approx k.

    Find the k nearest points by gradually increasing radius,
    and applying get_neighborhood_radius, until at least
    k points are returned.
    """
    # Initial radius
    # Half of grid spacing will not include more than 1 point
    # (unless exactly between points, in which case,
    # it would never include less than those equidistant points)
    radius = grid_spacing * 0.5

    # Gradually increase radius until k_nearest are contained
    points = []
    while len(points) < k_nearest:
        points = get_neighborhood_radius(grid_spacing, center, radius)
        radius *= incr_rate

    # Return at least k points
    return points


def get_neighborhood_normalized_adaptive_radius(grid_spacing,
                                                center,
                                                initial_radius=1.0):
    """Return all points within radius slightly farther than closest to center.

    A wrapper for get_neighborhood_adaptive_radius that adjusts radius_scale based on
    dimensionality.

    NOTE: initial_radius must be a floating point number for proper usage.
    """
    return get_neighborhood_adaptive_radius(
        grid_spacing, center, radius_scale=initial_radius / len(center))


def get_neighborhood_adaptive_radius(grid_spacing, center, radius_scale=1.0):
    """Obtain all points within radius slightly farther than closest to center.

    A wrapper for get_neighborhood_radius that adjusts radius based on closest point.
    """
    # Get distance from center to closest point
    distance = _distance(
        center,
        # Closest point
        # Obtained by rounding each coordinate to nearest grid coordinate
        [_round_partial(v, grid_spacing) for v in center])

    # Radius is a little more than distance to closest
    return get_neighborhood_radius(
        grid_spacing, center,
        distance + radius_scale * math.log(1.0000000001 + distance))


def _distance(vec_a, vec_b):
    """Return euclidian distance between vec_a and vec_b."""
    return numpy.linalg.norm(numpy.subtract(vec_a, vec_b))


def get_neighborhood_radius(grid_spacing, center, radius):
    """Obtain all points within radius of center."""
    return _get_neighborhood_radius(grid_spacing, center, radius)


def _get_neighborhood_radius(grid_spacing, center, radius):
    """Recursively traverse each dimension to obtain points in a radius.

    By constraining the search of each lower dimension by the distance from the higher
    dimension to the edge of the hypersphere, we efficiently search the sphere.
    """
    if len(center) == 1:
        x = _ceil_partial(center[0] - radius, grid_spacing)
        end = center[0] + radius

        points = []
        while x <= end:
            points.append((round(x, MAXIMUM_PRECISION), ))
            x += grid_spacing

        return points

    points = []

    # Cache some values used in loop
    r_squared = radius**2
    center_0 = center[0]
    center_1plus = center[1:]

    # Start at furthest point in circle
    x = _ceil_partial(center_0 - radius, grid_spacing)
    end = center_0 + radius
    while x <= end:
        # Constrain next dimension, based on distance to edge of circle
        try:
            r2 = math.sqrt(r_squared - (x - center_0)**2)
        except ValueError:
            # Floating point rounding errors can result in the
            # subtraction being negative
            r2 = math.sqrt(
                round(radius, MAXIMUM_PRECISION)**2 - round(
                    x - center_0, MAXIMUM_PRECISION)**2)

        # Get all points in the next dimension
        ys = _get_neighborhood_radius(grid_spacing, center_1plus, r2)
        # Add point from this dimension
        x_rounded_tuple = (round(x, MAXIMUM_PRECISION), )
        points.extend([x_rounded_tuple + y for y in ys])

        x += grid_spacing

    return points


def _ceil_partial(value, grid_spacing):
    """Ceil to next point on a grid with a given grid_spacing."""
    return math.ceil(value / grid_spacing) * grid_spacing


def _round_partial(value, grid_spacing):
    """Round to nearest point on a grid with a given grid_spacing."""
    return round(value / grid_spacing) * grid_spacing


#########################
# Similarity
#########################
def scaled_gaussian_similarity(matrix_a, vec_b, furthest_similarity=0.01):
    """Gaussian similarity metric, with variance scaled by distance of furthest row in matrix_a."""
    squared_distances = _squared_distances(matrix_a, vec_b)

    # Adjust variance based on furthest distance
    # Similarity at at furthest is furthest_similarity
    variance = max(
        _variance_to_cutoff(
            numpy.sqrt(numpy.max(squared_distances)), furthest_similarity),
        # Don't let variance be 0, happens when max(squared_distances) == 0
        1e-10)

    # Calculates Gaussian similarity for each squared distance
    return _partial_gaussian(squared_distances, variance)


def gaussian_similarity(matrix_a, vec_b, variance=1.0):
    """Gaussian similarity metric."""
    return _partial_gaussian(_squared_distances(matrix_a, vec_b), variance)


def _squared_distances(matrix_a, vec_b):
    diffs = numpy.subtract(matrix_a, vec_b)
    return numpy.einsum('ij,ij->i', diffs, diffs)  # Dot each row with itself


def _partial_gaussian(squared_distances, variance):
    """Return Gaussian similarity for each squared distance."""
    return numpy.exp(-(squared_distances / variance))


##########################
# Learning target
##########################
def _solve_model_target_matrix(similarities, model_output_matrix, output_vec,
                               target_vec):
    """Return output values that result in the given target.

    We also want to minimize the distance between current output_vec and new output_vec.
    """
    # Minimize target error + mean squared error between current and new output_vec
    return optimize.minimize(
        _output_change_objective,
        jac=_output_change_jacobian,
        # Use approximate solver as heuristic for initial position
        x0=model_output_matrix.ravel(),  # ravel() == Flatten
        args=(similarities, model_output_matrix),
        constraints={
            'type': 'eq',
            'fun': _output_correct_objective,
            'jac': _output_correct_jacobian,
            'args': (similarities, target_vec)
        }).x.reshape(model_output_matrix.shape)  # Un-flatten


def _adjust_model_target_matrix(similarities, model_output_matrix, output_vec,
                                target_vec):
    """Return output values that are closer to resulting in the given target."""
    # Use error gradient of ILL model, abstracting away details of underlying model
    # Step size for each underlying model is corresponding similarity
    # (each row is multiplied by similarity in corresponding col)
    return model_output_matrix + similarities[:, None] * (
        target_vec - output_vec)


def _adjust_model_target_matrix_to_target(similarities, model_output_matrix,
                                          output_vec, target_vec):
    """Return output values that are closer to the given target."""
    # Use error gradient of ILL model, abstracting away details of underlying model
    # Step size for each underlying model is corresponding similarity
    # (each row is multiplied by similarity in corresponding col)
    return model_output_matrix + similarities[:, None] * (
        target_vec - model_output_matrix)


def _adjust_model_target_matrix_one_step(similarities, model_output_matrix,
                                         output_vec, target_vec):
    """Take one step towards the exact model target matrix."""
    # Minimize target error + mean squared error between current and new output_vec
    return optimize.minimize(
        _output_change_objective,
        jac=_output_change_jacobian,
        # Use approximate solver as heuristic for initial position
        x0=model_output_matrix.ravel(),  # ravel() == Flatten
        args=(similarities, model_output_matrix),
        constraints={
            'type': 'eq',
            'fun': _output_correct_objective,
            'jac': _output_correct_jacobian,
            'args': (similarities, target_vec)
        },
        options={'maxiter': 1}  # 1 step
    ).x.reshape(model_output_matrix.shape)  # Un-flatten


def _output_change_objective(flat_model_target_vec, similarities,
                             model_output_matrix):
    """Return how much output_vec will change, weighed by similarity."""
    # Unflatten flat_model_target_vec, into model_target_matrix
    num_rows = similarities.shape[0]
    model_target_matrix = _unflatten(flat_model_target_vec, num_rows)

    # We want the distance of each
    # row in model_output_matrix to each row in model_target_matrix
    # scaled by similarity
    # The MSE (or sum squared error), is another distance metric, that is more
    # efficient than norm, because it doesn't require a sqrt

    # Mean of
    return numpy.mean(
        # Distance between corresponding rows
        numpy.mean((model_output_matrix - model_target_matrix)**2, axis=1)
        # Scaled by corresponding similarities
        / similarities)


def _output_change_jacobian(flat_model_target_vec, similarities,
                            model_output_matrix):
    """Return flat jocobian of _output_change_objective."""
    # Unflatten flat_model_target_vec, into model_target_matrix
    num_rows = similarities.shape[0]
    model_target_matrix = _unflatten(flat_model_target_vec, num_rows)

    # d/dx (y - x)**2 = 2*(x - y)
    # d/dx mean((y - x)**2, axis=1) = 2*(x - y) / rows(x)
    # d/dx (mean((y - x)**2, axis=1) / s) = (2*(x - y) / rows(x)) / s.T
    # d/dx mean(mean((y - x)**2, axis=1) / s) = ((2*(x - y) / rows(x)) / s.T) / cols(x)

    # Reorder to minimize multiplications and divisions with matrices

    return ((model_target_matrix - model_output_matrix) /
            ((0.5 * model_target_matrix.shape[0] * model_target_matrix.shape[1]
              ) * similarities)[:, None]).ravel()


def _output_correct_objective(flat_model_target_vec, similarities, target_vec):
    """Return how close proposed model targets are to equaling target when combined."""
    # Unflatten model_target_vec, into model_target_matrix
    num_rows = similarities.shape[0]
    model_target_matrix = _unflatten(flat_model_target_vec, num_rows)

    # similarities.dot(model_target_matrix) == ILL output
    # numpy.mean((ILL output - target_vec)**2) == distance to target_vec
    return numpy.mean((similarities.dot(model_target_matrix) - target_vec)**2)


def _output_correct_jacobian(flat_model_target_vec, similarities, target_vec):
    """Return flat jocobian of _output_correct_objective."""
    # Unflatten model_target_vec, into model_target_matrix
    num_rows = similarities.shape[0]
    model_target_matrix = _unflatten(flat_model_target_vec, num_rows)

    # d/dx s.dot(x) = tile(s.T, (1, cols(x))) # Repeat row of s as cols until shape of x
    # d/dx s.dot(x) - t = tile(s.T, (1, cols(x)))
    # d/dx (s.dot(x) - t)**2 = 2*(s.dot(x) - t) * tile(s.T, (1, cols(x)))
    # d/dx mean((s.dot(x) - t)**2) = (2*(s.dot(x) - t) * tile(s.T, (1, cols(x)))) / cols(x)

    # Reorder to minimize multiplications and divisions with matrices

    return (
        (similarities.dot(model_target_matrix) - target_vec)
        # Do the scalar operations on this smaller vector, instead of whole matrix
        # No need to tile, matrix * col_vec effectively tiles
        * (similarities[:, None] /
           (0.5 * model_target_matrix.shape[1]))).ravel()


def _unflatten(flat_matrix, num_rows):
    return flat_matrix.reshape((num_rows, flat_matrix.shape[0] / num_rows))


############################
# Dimensionality Reduction
############################
def _identity(x):
    return x


#########################
# Neural Field
#########################
class LearnOutput(Model):
    def __init__(self, num_outputs, learn_rate):
        super(LearnOutput, self).__init__()

        self._num_outputs = num_outputs
        self._output = None

        self.learn_rate = learn_rate

        self.reset()

    def reset(self):
        self._output = numpy.zeros(self._num_outputs)

    def activate(self, input_vec):
        return self._output

    def _train_increment(self, input_vec, target_vec):
        # TODO: Use BFGS optimizer?
        self._output += self.learn_rate * (
            target_vec - self.activate(input_vec))


def make_neuralfield(num_outputs, learn_rate=1.0, *args, **kwargs):
    """Create a neural field network."""
    return ILL(LearnOutput(num_outputs, learn_rate), *args, **kwargs)
