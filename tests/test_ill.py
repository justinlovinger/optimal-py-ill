import random
import itertools

import pytest
import numpy

from learning.data import datasets
from learning import MLP
from learning import calculate
from learning import validation

from learning.testing import helpers

import ill

XOR_SPACING = 1.5

def test_ill_mlp_approx_target():
    # Run for a couple of iterations
    # assert that new error is less than original
    dataset = datasets.get_xor()

    model = ill.ILL(MLP((2, 2, 2)), grid_spacing=XOR_SPACING, learn_exact=False)

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error, "Training decreases error"

@pytest.mark.slowtest()
def test_ill_mlp_convergence_approx_target():
    # Run until convergence
    # assert that network can converge
    dataset = datasets.get_xor()

    model = ill.ILL(MLP((2, 2, 2)), grid_spacing=XOR_SPACING, learn_exact=False)

    error = validation.get_error(model, *dataset)
    model.train(*dataset, retries=5, error_break=0.002)
    assert validation.get_error(model, *dataset) < 0.02, "Training should reach low error"


def test_ill_mlp_exact_target():
    # Run for a couple of iterations
    # assert that new error is less than original
    dataset = datasets.get_xor()

    model = ill.ILL(MLP((2, 2, 2)), grid_spacing=XOR_SPACING, learn_exact=True)

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error, "Training decreases error"

@pytest.mark.slowtest()
def test_ill_mlp_convergence_exact_target():
    # Run until convergence
    # assert that network can converge
    dataset = datasets.get_xor()

    model = ill.ILL(MLP((2, 2, 2)), grid_spacing=XOR_SPACING, learn_exact=True)

    error = validation.get_error(model, *dataset)
    model.train(*dataset, retries=5, error_break=0.002)
    assert validation.get_error(model, *dataset) < 0.02, "Training should reach low error"


def test_ill_mlp_dim_reduction_tuple(monkeypatch):
    REDUCED_DIMENSIONS = 1

    dataset = datasets.get_xor()

    model = ill.ILL(MLP((2, 2, 2)), grid_spacing=XOR_SPACING, dim_reduction=(2, REDUCED_DIMENSIONS))

    # Points should have reduced dimensions
    points = _get_neighborhood_points(model, dataset, monkeypatch)
    for point in points:
        assert len(point) == REDUCED_DIMENSIONS

    # Should be able to train
    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error, "Training decreases error"

def test_ill_mlp_dim_reduction_tuple_reset(monkeypatch):
    REDUCED_DIMENSIONS = 1

    dataset = datasets.get_xor()

    model = ill.ILL(MLP((2, 2, 2)), grid_spacing=XOR_SPACING, dim_reduction=(2, REDUCED_DIMENSIONS))

    # Points should have reduced dimensions
    points = _get_neighborhood_points(model, dataset, monkeypatch)
    for point in points:
        assert len(point) == REDUCED_DIMENSIONS

    # Points should be different after reset
    model.reset()
    new_points = _get_neighborhood_points(model, dataset, monkeypatch)
    for point in new_points[1:]: # Ignore (0, 0) point, it will always have same reduced dims
        assert point not in points
    # Should still have reduced dimensions
    for point in points:
        assert len(point) == REDUCED_DIMENSIONS

def _get_neighborhood_points(model, dataset, monkeypatch):
    # Monkeypatch to extract points
    points = [None]
    orig_get_points = ill.ILL._get_points
    def _get_points(*args):
        points[0] = orig_get_points(*args)
        return points[0]
    monkeypatch.setattr(ill.ILL, '_get_points', _get_points)
    model.activate(dataset[0][0])
    monkeypatch.undo()

    return points[0]

##############################
# Target getting
##############################
def test_output_change_jacobian():
    rows = random.randint(1, 10)
    cols = random.randint(1, 10)
    flat_model_target_vec = numpy.random.random(rows*cols)
    similarities = numpy.random.random(rows)
    model_output_matrix = numpy.random.random((rows, cols))

    helpers.check_gradient(
        lambda x: ill._output_change_objective(x, similarities, model_output_matrix),
        lambda x: ill._output_change_jacobian(x, similarities, model_output_matrix),
        f_arg_tensor=flat_model_target_vec,
        f_shape='scalar'
    )

def test_output_correct_jacobian():
    rows = random.randint(1, 10)
    cols = random.randint(1, 10)
    flat_model_target_vec = numpy.random.random(rows*cols)
    similarities = numpy.random.random(rows)
    target_vec = numpy.random.random(cols)

    helpers.check_gradient(
        lambda x: ill._output_correct_objective(x, similarities, target_vec),
        lambda x: ill._output_correct_jacobian(x, similarities, target_vec),
        f_arg_tensor=flat_model_target_vec,
        f_shape='scalar'
    )


##############################
# Neighborhood
##############################
def test_get_neighborhood_k_nearest():
    dims = random.randint(1, 6)
    k_nearest = random.randint(1, 40)

    grid_spacing = random.uniform(0.0001, 4.0)
    print 'g:', grid_spacing
    center = (numpy.random.random(dims) - 0.5) * 4.0
    print 'c:', center

    points = ill.get_neighborhood_k_nearest(grid_spacing, center, k_nearest)

    assert len(points) == k_nearest

def test_get_neighborhood_approx_k_nearest():
    dims = 2

    grid_spacing = random.uniform(0.0001, 4.0)
    print 'g:', grid_spacing
    center = (numpy.random.random(dims) - 0.5) * 4.0
    print 'c:', center

    points = ill.get_neighborhood_approx_k_nearest(grid_spacing, center, 3)

    # May return more than 3, but not by much
    assert 3 <= len(points) <= 5

def test_get_neighborhood_radius_consistent():
    """Neighborhood functions should return the same points, irrespective of finite precision."""
    grid_spacing = random.uniform(1e-6, 10.0)
    center = numpy.random.random(random.randint(1, 3))

    # Find points with radius neighborhood
    radius = random.uniform(_distance_to_nearest(grid_spacing, center), grid_spacing*5)
    points = ill.get_neighborhood_radius(grid_spacing, center, radius)

    # Every points found within this radius, should be in the points of a larger radius
    outer_points = ill.get_neighborhood_radius(grid_spacing, center,
                                               radius+random.uniform(0.0, grid_spacing*5))

    for point in points:
        assert point in outer_points

def _distance_to_nearest(grid_spacing, center):
    closest_point = [ill._round_partial(v, grid_spacing) for v in center]
    distance = ill._distance(center, closest_point)
    return max(distance, 1e-10)

def test_get_neighborhood_radius_correct():
    """Neighborhood functions should return all points on grid in range."""
    grid_spacing = random.uniform(1e-6, 4.0)
    dimensionality = random.randint(1, 3)

    center = numpy.random.random(dimensionality)*2 - 1.0
    radius = random.uniform(1e-6, grid_spacing*2)

    # Find all points on grid in range with exhaustive search
    grid = _make_grid(grid_spacing, dimensionality,
                      numpy.min(center)-radius, numpy.max(center)+radius)
    expected_neighborhood = [point for point in grid if calculate.distance(point, center) <= radius]

    assert (sorted(ill.get_neighborhood_radius(grid_spacing, center, radius))
            == sorted(expected_neighborhood))

def test_make_grid():
    assert sorted(_make_grid(1.0, 1, -4, 4)) == [(-4.0,), (-3.0,), (-2.0,), (-1.0,), (0.0,),
                                                 (1.0,), (2.0,), (3.0,), (4.0,)]
    assert sorted(_make_grid(1.0, 2, -1, 1)) == [(-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
                                                 (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
                                                 (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)]

def _make_grid(grid_spacing, dimensionality, min_, max_):
    coords = []
    x = 0.0
    # Walk negative
    while x >= min_:
        coords.append(round(x, ill.MAXIMUM_PRECISION))
        x -= grid_spacing
    # Walk positive
    x = grid_spacing
    while x <= max_:
        coords.append(round(x, ill.MAXIMUM_PRECISION))
        x += grid_spacing

    # Each combination of coords is a point
    return list(itertools.product(coords, repeat=dimensionality))


##############################
# Neural Field
##############################
def test_neuralfield():
    # Run for a couple of iterations
    # assert that new error is less than original
    dataset = datasets.get_xor()

    model = ill.make_neuralfield(2, grid_spacing=XOR_SPACING)

    error = validation.get_error(model, *dataset)
    model.train(*dataset, iterations=10)
    assert validation.get_error(model, *dataset) < error, "Training decreases error"

@pytest.mark.slowtest()
def test_neuralfield_convergence():
    # Run until convergence
    # assert that network can converge
    dataset = datasets.get_xor()

    model = ill.make_neuralfield(2, grid_spacing=XOR_SPACING)

    error = validation.get_error(model, *dataset)
    model.train(*dataset, error_break=0.002)
    assert validation.get_error(model, *dataset) < 0.02, "Training should reach low error"

###################
# Other functions
###################
def test_squared_distances():
    cols = random.randint(1, 20)
    matrix_a = numpy.random.random([random.randint(1, 20), cols])
    vec_b = numpy.random.random(cols)

    # Compare efficient implementation to basic norm on each row
    # All must be approx equal
    for d_1, d_2 in zip(ill._squared_distances(matrix_a, vec_b),
                        [numpy.linalg.norm(row-vec_b)**2 for row in matrix_a]):
        assert (d_1-d_2) < 0.000001
