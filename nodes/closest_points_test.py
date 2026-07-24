import math

from gen.messages_pb2 import Body, ClosestPointsInput, Scene, Shape
from nodes.closest_points import closest_points
from nodes._test_helpers import _TestContext


def test_separated_spheres_matches_distance_formula():
    """Independent oracle: for two non-overlapping spheres, the closest
    distance is the closed-form `center_distance - r1 - r2`. Computed from
    scratch here, not by calling into pybullet.
    """
    c1 = (0.0, 0.0, 0.0)
    c2 = (5.0, 0.0, 0.0)
    r1, r2 = 1.0, 1.5
    expected = math.dist(c1, c2) - r1 - r2  # 2.5

    scene = Scene(
        bodies=[
            Body(id="a", shape=Shape(type="sphere", radius=r1), mass=0.0,
                 position={"x": c1[0], "y": c1[1], "z": c1[2]}),
            Body(id="b", shape=Shape(type="sphere", radius=r2), mass=0.0,
                 position={"x": c2[0], "y": c2[1], "z": c2[2]}),
        ]
    )
    result = closest_points(
        _TestContext(),
        ClosestPointsInput(scene=scene, body_a_id="a", body_b_id="b", max_distance=100.0),
    )
    assert result.error.code == ""
    assert result.found is True
    assert abs(result.distance - expected) < 1e-6
    # Point on A should be on the +x side of sphere A, i.e. x == r1.
    assert abs(result.point_on_a.x - r1) < 1e-6


def test_pair_farther_than_max_distance_reports_not_found():
    scene = Scene(
        bodies=[
            Body(id="a", shape=Shape(type="sphere", radius=0.5), mass=0.0, position={"x": 0, "y": 0, "z": 0}),
            Body(id="b", shape=Shape(type="sphere", radius=0.5), mass=0.0, position={"x": 100, "y": 0, "z": 0}),
        ]
    )
    result = closest_points(
        _TestContext(),
        ClosestPointsInput(scene=scene, body_a_id="a", body_b_id="b", max_distance=1.0),
    )
    assert result.error.code == ""
    assert result.found is False


def test_unknown_body_id_returns_structured_error_not_crash():
    scene = Scene(bodies=[Body(id="a", shape=Shape(type="sphere", radius=0.5), mass=0.0)])
    result = closest_points(
        _TestContext(),
        ClosestPointsInput(scene=scene, body_a_id="a", body_b_id="does-not-exist", max_distance=10.0),
    )
    assert result.error.code == "NOT_FOUND"


def test_non_positive_max_distance_returns_structured_error_not_crash():
    scene = Scene(bodies=[Body(id="a", shape=Shape(type="sphere", radius=0.5), mass=0.0)])
    result = closest_points(
        _TestContext(),
        ClosestPointsInput(scene=scene, body_a_id="a", body_b_id="a", max_distance=0.0),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_non_finite_max_distance_returns_structured_error_not_crash():
    """Regression: max_distance used to be checked only for `<= 0`, so
    NaN (which fails every ordering comparison) slipped past validation
    and silently produced found=False instead of a structured error."""
    scene = Scene(bodies=[Body(id="a", shape=Shape(type="sphere", radius=0.5), mass=0.0)])
    result = closest_points(
        _TestContext(),
        ClosestPointsInput(scene=scene, body_a_id="a", body_b_id="a", max_distance=float("nan")),
    )
    assert result.error.code == "NON_FINITE_VALUE"
